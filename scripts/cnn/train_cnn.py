"""
train_cnn.py
------------
Train a CNN on patch features with given data and splits.

Inputs:
  - features/<ver>/{X.npy, y.npy, meta.json}
  - data/splits/<default_split>.json
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from videre.evals.metrics import compute_metrics
from videre.evals.plots import plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from videre.models.torch_models import PatchCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN classifier on patch-token features.")
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()

def load_data(feature_dir: str, split_file: str):
    X = np.load(os.path.join(feature_dir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(feature_dir, "y.npy"), mmap_mode="r")
    with open(os.path.join(feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    with open(split_file, "r") as f:
        split = json.load(f)
    return X, y, meta, split

class PatchFeatureDataset(Dataset):
    def __init__(self, X, y, patch_dim, H, W):
        self.X = X
        self.y = y
        self.patch_dim = patch_dim
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32).reshape(self.patch_dim, self.H, self.W)
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def evaluate(loader, model, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_labels.append(yb.numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )

def run_and_score(loader, model, device):
    y_true, y_pred, y_proba = evaluate(loader, model, device)
    y_class1 = y_proba[:, 1]
    metrics = compute_metrics(y_true, y_pred, y_class1)
    return metrics, y_true, y_pred, y_class1

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = {
        "lr": 1e-5,
        "epochs": 80,
        "weight_decay": 0.0,
        "patience": 10,
        "batch_size": 8
    }

    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    X, y, meta, split = load_data(args.feature_dir, args.split_file)
    patch_dim = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

    train_idx = np.array(split["train"])
    val_idx = np.array(split["val"])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_dataset = PatchFeatureDataset(X_train, y_train, patch_dim, H, W)
    val_dataset = PatchFeatureDataset(X_val, y_val, patch_dim, H, W)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=False)

    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = PatchCNN(embedding_dim=patch_dim, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])

    best_val_loss = float("inf")
    wait = 0
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        total_train_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        # Evaluate
        train_metrics, y_train_true, y_train_pred, y_train_prob = run_and_score(
            train_loader, model, device
        )
        val_metrics, y_val_true, y_val_pred, y_val_prob = run_and_score(
            val_loader, model, device
        )
        val_loss = val_metrics["loss"] if "loss" in val_metrics else val_metrics["log_loss"]
        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUROC: {val_metrics.get('roc_auc', 0):.4f}"
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        })

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"{args.run_name}_best.pt"))
        else:
            wait += 1

        if wait >= cfg["patience"]:
            logger.info("Early stopping triggered")
            break

    torch.save(model.state_dict(), os.path.join(model_dir, f"{args.run_name}.pt"))

    # Final metrics + plots
    train_metrics, y_train_true, y_train_pred, y_train_prob = run_and_score(
        train_loader, model, device
    )

    val_metrics, y_val_true, y_val_pred, y_val_prob = run_and_score(
        val_loader, model, device
    )

    with open(os.path.join(results_dir, "metrics_train.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)

    with open(os.path.join(results_dir, "metrics_val.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_roc_curve(y_train_true, y_train_prob, os.path.join(results_dir, "roc_curve_train.png"))
    plot_pr_curve(y_train_true, y_train_prob, os.path.join(results_dir, "pr_curve_train.png"))
    plot_confusion_matrix(y_train_true, y_train_pred,
                          os.path.join(results_dir, "confusion_matrix_train.png"))

    plot_roc_curve(y_val_true, y_val_prob, os.path.join(results_dir, "roc_curve_val.png"))
    plot_pr_curve(y_val_true, y_val_prob, os.path.join(results_dir, "pr_curve_val.png"))
    plot_confusion_matrix(y_val_true, y_val_pred,
                          os.path.join(results_dir, "confusion_matrix_val.png"))

    logger.info("Training + evaluation complete")


if __name__ == "__main__":
    main()
