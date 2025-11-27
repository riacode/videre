"""
train_cnn.py
------------
CLI to train a CNN classifier on patch-token features for a given split.

Inputs:
  - features/<ver>/{X.npy, y.npy, meta.json}
  - data/splits/<split>.json
  - (optional) config file with hyperparameters

Outputs:
  - artifacts/models/<run_name>.pth
  - artifacts/results/<run_name>/metrics_train.json
  - artifacts/results/<run_name>/metrics_val.json
  - artifacts/results/<run_name>/run_config.json

Flow:
  load features -> reshape -> slice by split -> build torch datasets ->
  train CNN -> evaluate -> save artifacts
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
from videre.models.torch_models import get_cnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN classifier on patch-token features.")
    parser.add_argument("--feature-dir", type=str, required=True, help="Path to feature directory")
    parser.add_argument("--split-file", type=str, required=True, help="Path to split JSON")
    parser.add_argument("--run-name", type=str, required=True, help="Run name")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()

def load_data(feature_dir: str, split_file: str):
    X = np.load(os.path.join(feature_dir, "X.npy"))
    y = np.load(os.path.join(feature_dir, "y.npy"))

    with open(os.path.join(feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    with open(split_file, "r") as f:
        split = json.load(f)

    return X, y, meta, split

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg["model_params"]

class PatchFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def save_artifacts(model, metrics_train, metrics_val, config, model_dir, results_dir, run_name):
    model_path = os.path.join(model_dir, f"{run_name}.pth")
    torch.save(model.state_dict(), model_path)

    with open(os.path.join(results_dir, f"{run_name}_metrics_train.json"), "w") as f:
        json.dump(metrics_train, f, indent=2)

    with open(os.path.join(results_dir, f"{run_name}_metrics_val.json"), "w") as f:
        json.dump(metrics_val, f, indent=2)

    with open(os.path.join(results_dir, f"{run_name}_run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load arrays, metadata, and split indices
    X, y, meta, split = load_data(args.feature_dir, args.split_file)

    # Infer patch layout
    num_patches = meta["num_patches"]
    patch_dim = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

    X = X.reshape(-1, patch_dim, H, W)

    # Apply split
    train_idx = np.array(split["train"])
    val_idx = np.array(split["val"])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Prepare datasets
    train_dataset = PatchFeatureDataset(X_train, y_train)
    val_dataset = PatchFeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load model config
    cfg = load_config(args.config)
    model = get_cnn_model(patch_dim, **cfg)
    model = model.to("cuda")

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    epochs = cfg.get("epochs", 10)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.cuda(), yb.cuda()
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch+1}/{epochs} completed")

    # Evaluation
    def evaluate(loader):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = Xb.cuda()
                out = model(Xb)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = out.argmax(1).cpu().numpy()
                all_probs.append(prob)
                all_preds.append(pred)
                all_labels.append(yb.numpy())
        return (np.concatenate(all_labels),
                np.concatenate(all_preds),
                np.concatenate(all_probs))

    y_train_true, y_train_pred, y_train_proba = evaluate(train_loader)
    y_val_true, y_val_pred, y_val_proba = evaluate(val_loader)

    metrics_train = compute_metrics(y_train_true, y_train_pred, y_train_proba)
    metrics_val = compute_metrics(y_val_true, y_val_pred, y_val_proba)

    # Prepare directories
    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save artifacts
    save_artifacts(model, metrics_train, metrics_val, {
        "run_name": args.run_name,
        "feature_dir": args.feature_dir,
        "split_file": args.split_file,
        "model_params": cfg,
        "patch_dim": patch_dim,
        "grid_h": H,
        "grid_w": W,
        "seed": args.seed,
    }, model_dir, results_dir, args.run_name)

    logger.info("CNN training completed")


if __name__ == "__main__":
    main()
