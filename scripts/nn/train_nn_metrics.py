"""
train_nn.py
------------
Train a NN on patch features with given data and splits.

Inputs:
  - features/<ver>/{X.npy, y.npy, meta.json}
  - data/splits/<default_split>.json
"""

import os
import json
import argparse
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from videre.evals.metrics import compute_metrics
from videre.evals.plots import plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from videre.models.torch_models import PatchNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", type=str, required=True)
    p.add_argument("--split-file", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="artifacts")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def load_data(feature_dir, split_file):
    X = np.load(os.path.join(feature_dir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(feature_dir, "y.npy"), mmap_mode="r")
    with open(os.path.join(feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    with open(split_file, "r") as f:
        split = json.load(f)
    return X, y, meta, split


class PatchFeatureDataset(Dataset):
    def __init__(self, X, y, patch_dim, H, W, indices=None):
        self.X = X
        self.y = y
        self.indices = indices
        self.patch_dim = patch_dim
        self.H = H
        self.W = W

        if self.indices is None:
            self.mode = "slice"
        else:
            self.mode = "index"

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.X)

    def __getitem__(self, idx):
        if self.mode == "index":
            real_idx = self.indices[idx]
        else:
            real_idx = idx
        x = self.X[real_idx].astype(np.float32).reshape(self.patch_dim, self.H, self.W)
        y = int(self.y[real_idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def evaluate(loader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if len(all_labels) == 0:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = {
        "lr": 1e-5,
        "epochs": 80,
        "weight_decay": 0.0,
        "patience": 10,
        "batch_size": args.batch_size
    }

    X, y, meta, split = load_data(args.feature_dir, args.split_file)
    patch_dim = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

    train_idx = np.array(split["train"])
    val_idx = np.array(split["val"])

    train_dataset = PatchFeatureDataset(X, y, patch_dim, H, W, indices=train_idx)
    val_dataset = PatchFeatureDataset(X, y, patch_dim, H, W, indices=val_idx)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=False)

    model = PatchNN(patch_dim, H, W, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_val_loss = float("inf")
    wait = 0
    history = []

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss_epoch = total_loss / max(1, n_batches)
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss_epoch = val_loss / max(1, n_val_batches)

        logger.info(f"Epoch {epoch+1}/{cfg['epochs']}  train loss={train_loss_epoch:.6f}  val loss={val_loss_epoch:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss_epoch),
            "val_loss": float(val_loss_epoch)
        })

        if val_loss_epoch < best_val_loss - 1e-6:
            best_val_loss = val_loss_epoch
            wait = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"{args.run_name}_best.pt"))
        else:
            wait += 1
            if wait >= cfg["patience"]:
                logger.info("Early stopping triggered")
                break

    torch.save(model.state_dict(), os.path.join(model_dir, f"{args.run_name}.pt"))

    y_train_true, y_train_pred, y_train_proba = evaluate(train_loader, model, device)
    y_val_true, y_val_pred, y_val_proba = evaluate(val_loader, model, device)

    train_proba_class1 = y_train_proba[:, 1] if y_train_proba.size else np.array([])
    val_proba_class1 = y_val_proba[:, 1] if y_val_proba.size else np.array([])

    metrics_train = compute_metrics(y_train_true, y_train_pred, train_proba_class1) if y_train_true.size else {}
    metrics_val = compute_metrics(y_val_true, y_val_pred, val_proba_class1) if y_val_true.size else {}

    with open(os.path.join(results_dir, "metrics_train.json"), "w") as f:
        json.dump(metrics_train, f, indent=2)
    with open(os.path.join(results_dir, "metrics_val.json"), "w") as f:
        json.dump(metrics_val, f, indent=2)
    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if y_train_true.size:
        plot_roc_curve(y_train_true, train_proba_class1, os.path.join(results_dir, "roc_curve_train.png"))
        plot_pr_curve(y_train_true, train_proba_class1, os.path.join(results_dir, "pr_curve_train.png"))
        plot_confusion_matrix(y_train_true, y_train_pred, os.path.join(results_dir, "confusion_matrix_train.png"))

    if y_val_true.size:
        plot_roc_curve(y_val_true, val_proba_class1, os.path.join(results_dir, "roc_curve_val.png"))
        plot_pr_curve(y_val_true, val_proba_class1, os.path.join(results_dir, "pr_curve_val.png"))
        plot_confusion_matrix(y_val_true, y_val_pred, os.path.join(results_dir, "confusion_matrix_val.png"))

    logger.info("Training + final evaluation complete")


if __name__ == "__main__":
    main()

