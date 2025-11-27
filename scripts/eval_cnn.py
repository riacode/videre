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
from torch.utils.data import TensorDataset, DataLoader

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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load PyTorch model
    model_path = os.path.join(args.output_dir, "models", f"{args.run_name}.pt")
    logger.info(f"Loading model from {model_path}")

    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    # Load features
    logger.info(f"Loading data from {args.feature_dir}")
    X = np.load(os.path.join(args.feature_dir, "X.npy"))
    y = np.load(os.path.join(args.feature_dir, "y.npy"))

    # Ensure data is float32 for PyTorch
    X = X.astype("float32")
    y = y.astype("int64")

    with open(args.split_file, "r") as f:
        split = json.load(f)

    test_indices = np.array(split["test"])
    X_test = torch.tensor(X[test_indices])
    y_test = torch.tensor(y[test_indices])

    logger.info(f"Test set: {len(X_test)} samples")

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("Running inference...")

    preds = []
    probs = []

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            outputs = model(xb)

            # If outputs are logits for binary/multiclass classification
            if outputs.ndim == 2:
                # Softmax for multiclass, sigmoid for binary
                if outputs.shape[1] == 1:
                    p = torch.sigmoid(outputs)
                    pred = (p >= 0.5).long().cpu().numpy()
                    prob = p.cpu().numpy().flatten()
                else:
                    p = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(p, dim=1).cpu().numpy()
                    prob = p[:, 1].cpu().numpy() if outputs.shape[1] == 2 else p.cpu().numpy()
            else:
                raise ValueError("Model output shape not recognized.")

            preds.append(pred)
            probs.append(prob)

    y_test_pred = np.concatenate(preds)
    y_test_proba = np.concatenate(probs)

    # Compute metrics
    metrics_test = compute_metrics(y_test.numpy(), y_test_pred, y_test_proba)

    logger.info("Test metrics:")
    for key, value in metrics_test.items():
        logger.info(f"{key}: {value:.4f}")

    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = os.path.join(results_dir, f"{args.run_name}_metrics_test.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_test, f, indent=2)

    # ROC curve
    roc_path = os.path.join(results_dir, "roc_curve.png")
    plot_roc_curve(y_test.numpy(), y_test_proba, roc_path)

    # PR curve
    pr_path = os.path.join(results_dir, "pr_curve.png")
    plot_pr_curve(y_test.numpy(), y_test_proba, pr_path)

    # Confusion matrix
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test.numpy(), y_test_pred, cm_path)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
