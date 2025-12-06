"""
train.py
---------
CLI to train a classifier on features for a given split.

Inputs:
  - features/<ver>/{X.npy,y.npy,meta.json}
  - data/splits/<split>.json
  - (optional) config file with hyperparameters

Outputs:
  - artifacts/models/<run_name>.joblib
  - artifacts/models/<run_name>_scaler.joblib
  - artifacts/results/<run_name>/metrics_train.json
  - artifacts/results/<run_name>/metrics_val.json
  - artifacts/results/<run_name>/run_config.json

Flow:
  load features -> slice by split -> fit model -> evaluate -> save artifacts
"""
import numpy as np
import json
import os
import argparse
import logging
from joblib import dump
from typing import Dict, Any, Optional

from videre.models.sklearn_models import get_model, get_model_names
from videre.evals.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a classifier on features for a given split.")
    parser.add_argument("--feature-dir", type=str, required=True, help="Path to feature directory")
    parser.add_argument("--split-file", type=str, required=True, help="Path to split file")
    parser.add_argument("--model", type=str, required=True, choices=get_model_names(), help="Model to use")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()


def load_data(feature_dir: str, split_file: str):
    """
    Load features, labels, metadata, and split indices.
    
    Returns:
        Tuple of (X, y, meta, split)
    """
    # Load feature arrays
    X = np.load(os.path.join(feature_dir, "X.npy"))
    y = np.load(os.path.join(feature_dir, "y.npy"))

    # Load train/val/test split
    with open(split_file, "r") as f:
        split = json.load(f)

    return X, y, split


def load_config(config_path: Optional[str], model_name: str) -> Dict[str, Any]:
    """
    Load model configuration from file or return defaults.
    
    Returns:
        Dictionary of model hyperparameters
    """
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # Assume config has the shape: {"model_params": {...}}
    return cfg["model_params"]


def save_artifacts(model, metrics_train, metrics_val, config, model_dir, results_dir, run_name):
    """
    Save model, scaler, metrics, and configuration.
    
    Uses joblib to save sklearn models (faster and more efficient than pickle for numpy arrays).
    """
    # Save model
    model_path = os.path.join(model_dir, f"{run_name}.joblib")
    dump(model, model_path)

    # Save metrics
    metrics_train_path = os.path.join(results_dir, f"{run_name}_metrics_train.json")
    metrics_val_path = os.path.join(results_dir, f"{run_name}_metrics_val.json")

    with open(metrics_train_path, "w") as f:
        json.dump(metrics_train, f, indent=2)
    with open(metrics_val_path, "w") as f:
        json.dump(metrics_val, f, indent=2)

    # Save run config
    config_path = os.path.join(results_dir, f"{run_name}_run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Load data and splits
    X, y, split = load_data(args.feature_dir, args.split_file)
    
    # Split into train/val
    train_indices = np.array(split["train"])
    val_indices = np.array(split["val"])
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Load model config
    model_config = load_config(args.config, args.model)
    
    # Add random_state to model config
    if "random_state" not in model_config:
        model_config["random_state"] = args.seed
    
    # Initialize and train model
    logger.info(f"Initializing {args.model} model...")
    model = get_model(args.model, **model_config)
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Training completed")
    
    # Get predictions and probabilities
    logger.info("Computing predictions...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Get probabilities if available
    try:
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)
    except AttributeError:
        logger.warning("Model does not support predict_proba()")
        y_train_proba = None
        y_val_proba = None
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics_train = compute_metrics(y_train, y_train_pred, y_train_proba)
    metrics_val = compute_metrics(y_val, y_val_pred, y_val_proba)
    
    logger.info("Training metrics:")
    for key, value in metrics_train.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("Validation metrics:")
    for key, value in metrics_val.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Output directories
    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Prepare config for saving
    saved_config = {
        "run_name": args.run_name,
        "model": args.model,
        "model_params": model_config,
        "seed": args.seed,
        "feature_dir": args.feature_dir,
        "split_file": args.split_file,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "n_features": X_train.shape[1]
    }
    
    # Save artifacts
    save_artifacts(model, metrics_train, metrics_val, saved_config, 
                   model_dir, results_dir, args.run_name)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
