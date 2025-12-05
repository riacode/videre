"""
train_nn.py
------------
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
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def save_artifacts(model, model_dir, run_name):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{run_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = {
        "lr": 1e-4,
        "epochs": 80,
        "weight_decay": 0.0,
        "patience": 10
    }

    # Load arrays, metadata, and split indices
    X, y, meta, split = load_data(args.feature_dir, args.split_file)

    # Infer patch layout
    num_patches = meta["num_patches"]
    patch_dim = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

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
    train_dataset = PatchFeatureDataset(X_train, y_train, patch_dim, H, W)
    val_dataset   = PatchFeatureDataset(X_val, y_val, patch_dim, H, W)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
        num_workers=0,
        pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0,
        pin_memory=False)

    model = PatchCNN(
        embedding_dim=patch_dim,   # 384
        num_classes=2
    )
    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    model = model.to("cuda")

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    epochs = cfg["epochs"]
    patience = cfg["patience"]
    wait = 0
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.cuda(), yb.cuda()
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.cuda(), yb.cuda()
                val_loss += criterion(model(Xb), yb).item()
        val_loss /= len(val_loader)
    
        logger.info(f"Epoch {epoch+1}/{epochs} | val_loss={val_loss:.4f}")
    
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"{args.run_name}_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping triggered")
                break
        logger.info(f"Epoch {epoch+1}/{epochs} completed")

    save_artifacts(model, model_dir, args.run_name)

    logger.info("CNN training completed")


if __name__ == "__main__":
    main()
