"""
train_cnn.py
------------
CLI to train a CNN classifier on patch-token features for a given split.

Inputs:
  - features/<ver>/{X.npy, y.npy, meta.json}
  - data/splits/<default_split>.json

Outputs:
  - artifacts/models/<run_name>.pt
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
from videre.models.torch_models import PatchCNN

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
    X = np.load(os.path.join(feature_dir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(feature_dir, "y.npy"), mmap_mode="r")

    with open(os.path.join(feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    with open(split_file, "r") as f:
        split = json.load(f)

    return X, y, meta, split
    
class PatchFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def save_artifacts(model, model_dir, run_name):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{run_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = { # Just a default placeholder config (more epochs later) 
        "lr": 1e-3,
        "epochs": 10,
        "weight_decay": 0.0,
    }

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

    print(y_train)
    print(y_val)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Prepare datasets
    train_dataset = PatchFeatureDataset(X_train, y_train)
    val_dataset = PatchFeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PatchCNN(
        embedding_dim=patch_dim,   # 384
        num_classes=2
    )
    
    model = model.to("cuda")

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    epochs = cfg["epochs"]

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

    # Save artifacts
    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    save_artifacts(model, model_dir, args.run_name)

    logger.info("CNN training completed")


if __name__ == "__main__":
    main()
