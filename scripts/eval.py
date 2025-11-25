"""
eval.py
-------
Evaluate a saved model on the test split and generate metrics + plots.

Inputs:
  - artifacts/models/<run_name>.joblib
  - features/<ver>/{X.npy,y.npy,meta.json}
  - data/splits/default.json (default for now)

Outputs:
  - artifacts/results/<run_name>/metrics_test.json
  - artifacts/results/<run_name>/{roc_curve.png, pr_curve.png, confusion_matrix.png}
"""
import numpy as np
import json
import os
import argparse
import logging
from joblib import load

from videre.evals.metrics import compute_metrics
from videre.evals.plots import plot_roc_curve, plot_pr_curve, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a saved model on test data.")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run (model name)")
    parser.add_argument("--feature-dir", type=str, required=True, help="Path to feature directory")
    parser.add_argument("--split-file", type=str, required=True, help="Path to split file")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load model
    model_path = os.path.join(args.output_dir, "models", f"{args.run_name}.joblib")
    logger.info(f"Loading model from {model_path}")
    model = load(model_path)
    
    # Load data
    logger.info(f"Loading data from {args.feature_dir}")
    X = np.load(os.path.join(args.feature_dir, "X.npy"))
    y = np.load(os.path.join(args.feature_dir, "y.npy"))
    
    # Load split
    with open(args.split_file, "r") as f:
        split = json.load(f)
    
    # Get test indices
    test_indices = np.array(split["test"])
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Get predictions and probabilities
    logger.info("Computing predictions...")
    y_test_pred = model.predict(X_test)
    
    try:
        y_test_proba = model.predict_proba(X_test)
    except AttributeError:
        logger.warning("Model does not support predict_proba()")
        y_test_proba = None
    
    # Compute metrics
    metrics_test = compute_metrics(y_test, y_test_pred, y_test_proba)
    
    logger.info("Test metrics:")
    for key, value in metrics_test.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Create output directory
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(results_dir, f"{args.run_name}_metrics_test.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_test, f, indent=2)
    
    # Generate plots if probabilities available
    if y_test_proba is not None:
        logger.info("Generating plots...")
        
        # Get positive class probabilities
        if y_test_proba.ndim == 2 and y_test_proba.shape[1] == 2:
            y_test_scores = y_test_proba[:, 1]
        else:
            y_test_scores = y_test_proba
        
        # ROC curve
        roc_path = os.path.join(results_dir, "roc_curve.png")
        plot_roc_curve(y_test, y_test_scores, roc_path)
        
        # PR curve
        pr_path = os.path.join(results_dir, "pr_curve.png")
        plot_pr_curve(y_test, y_test_scores, pr_path)
        
        # Confusion matrix
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_test, y_test_pred, cm_path)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
