"""
eval_cnn.py
-----------
Evaluate a saved PyTorch CNN model on the test split and generate metrics + plots.

Inputs:
  - artifacts/models/<run_name>.pt
  - features/<ver>/{X.npy, y.npy, meta.json}
  - data/splits/default.json

Outputs:
  - artifacts/results/<run_name>/metrics_test.json
  - artifacts/results/<run_name>/{roc_curve.png, pr_curve.png, confusion_matrix.png}
"""

import numpy as np
import json
import os
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader

from videre.models.torch_models import PatchCNN
from videre.evals.metrics import compute_metrics
from videre.evals.plots import plot_roc_curve, plot_pr_curve, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved PyTorch CNN model on test data.")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run (model name)")
    parser.add_argument("--feature-dir", type=str, required=True, help="Path to feature directory")
    parser.add_argument("--split-file", type=str, required=True, help="Path to split file")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    return parser.parse_args()

class PatchDataset(Dataset):
    def __init__(self, X, y, indices, patch_dim, H, W):
        self.X = X
        self.y = y
        self.indices = indices
        self.patch_dim = patch_dim
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        x = self.X[idx].astype("float32")  # just this sample
        x = x.reshape(self.patch_dim, self.H, self.W)

        y = int(self.y[idx])

        return torch.tensor(x), torch.tensor(y)

def evaluate(loader, model, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)

            logits = model(Xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load metadata
    with open(os.path.join(args.feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    patch_dim = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

    # Load model checkpoint
    model_path = os.path.join(args.output_dir, "models", f"{args.run_name}.pt")
    logger.info(f"Loading model from {model_path}")

    model = PatchCNN(embedding_dim=patch_dim, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X = np.load(os.path.join(args.feature_dir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(args.feature_dir, "y.npy"), mmap_mode="r")

    # Load split indices
    with open(args.split_file, "r") as f:
        split = json.load(f)

    test_indices = np.array(split["test"])
    logger.info(f"Test samples: {len(test_indices)}")

    # Dataset + dataloader
    test_dataset = PatchDataset(X, y, test_indices, patch_dim, H, W)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,     # safe for mmap
        pin_memory=False,
    )

    # Run evaluation
    y_true, y_pred, y_proba = evaluate(test_loader, model, device)

    # Probabilities for class 1
    y_proba_class1 = y_proba[:, 1]

    # Compute metrics
    metrics_test = compute_metrics(y_true, y_pred, y_proba_class1)

    logger.info("Test metrics:")
    for k, v in metrics_test.items():
        logger.info(f"{k}: {v:.4f}")

    # Save outputs
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f"{args.run_name}_metrics_test.json"), "w") as f:
        json.dump(metrics_test, f, indent=2)

    # Plots
    plot_roc_curve(y_true, y_proba_class1, os.path.join(results_dir, "roc_curve.png"))
    plot_pr_curve(y_true, y_proba_class1, os.path.join(results_dir, "pr_curve.png"))
    plot_confusion_matrix(y_true, y_pred, os.path.join(results_dir, "confusion_matrix.png"))

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
