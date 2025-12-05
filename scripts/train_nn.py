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
from videre.models.torch_models import PatchNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args(): 
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


if __name__ == "__main__":
    main()
