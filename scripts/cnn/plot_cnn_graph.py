import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot
import torch.onnx
from videre.models.torch_models import PatchCNN

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

def load_data(feature_dir: str, split_file: str):
    X = np.load(os.path.join(feature_dir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(feature_dir, "y.npy"), mmap_mode="r")
    with open(os.path.join(feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    with open(split_file, "r") as f:
        split = json.load(f)
    return X, y, meta, split

def main():
    parser = argparse.ArgumentParser(description="Generate TorchViz graph for PatchCNN")
    parser.add_argument("--feature-dir", type=str, required=True, help="Path to features")
    parser.add_argument("--split-file", type=str, required=True, help="Path to JSON split file")
    parser.add_argument("--run-name", type=str, required=True, help="Run name for saving graph")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cpu")

    X, y, meta, split = load_data(args.feature_dir, args.split_file)
    patch_dim = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

    train_idx = np.array(split["train"])
    X_train, y_train = X[train_idx], y[train_idx]
    train_dataset = PatchFeatureDataset(X_train, y_train, patch_dim, H, W)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)  # small batch

    model = PatchCNN(embedding_dim=patch_dim, num_classes=2).to(device)
    model.eval()

    sample_X, sample_y = next(iter(train_loader))
    sample_X = sample_X.to(device)

    logits = model(sample_X)
    dot = make_dot(logits, params=dict(model.named_parameters()))
    os.makedirs(args.output_dir, exist_ok=True)
    graph_path = "graph.png"
    dot.render(graph_path, format="png")

    onnxpath = "bruh.onnx"
    torch.onnx.export(model, sample_X, onnxpath, input_names=['input'], output_names=['output'], opset_version=13, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    main()