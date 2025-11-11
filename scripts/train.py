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
  load features -> slice by split -> scale -> fit model -> evaluate -> save artifacts
"""
import numpy as np
import json
import os
import argparse
import logging
import joblib
from typing import Dict, Any, Optional

from videre.models.sklearn_models import get_model, get_model_names
from videre.evals.metrics import compute_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

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
    parser.add_argument("--scale", action="store_true", default=True, help="Scale features")
    parser.add_argument("--no-scale", dest="scale", action="store_false", help="Disable scaling")
    parser.add_argument("--class-weight", type=str, choices=["balanced", "none"], default="none", help="Class weight strategy")
    return parser.parse_args()


def load_data(feature_dir: str, split_file: str):
    """
    Load features, labels, metadata, and split indices.
    
    Returns:
        Tuple of (X, y, meta, split)
    """
    # TODO: Load X.npy, y.npy, meta.json, and split file
    # TODO: Log data statistics
    raise NotImplementedError


def load_config(config_path: Optional[str], model_name: str) -> Dict[str, Any]:
    """
    Load model configuration from file or return defaults.
    
    Returns:
        Dictionary of model hyperparameters
    """
    # TODO: Load config from JSON file if provided, else return defaults
    # TODO: Default configs: lr (max_iter, solver), svm (kernel, probability), mlp (hidden_layer_sizes, max_iter)
    raise NotImplementedError


def save_artifacts(model, scaler, metrics_train, metrics_val, config, model_dir, results_dir, run_name):
    """
    Save model, scaler, metrics, and configuration.
    
    Uses joblib to save sklearn models (faster and more efficient than pickle for numpy arrays).
    """
    # TODO: Save model using joblib.dump to <run_name>.joblib
    # TODO: Save scaler using joblib.dump to <run_name>_scaler.joblib (if provided)
    # TODO: Save metrics_train.json and metrics_val.json
    # TODO: Save run_config.json
    raise NotImplementedError


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Load data and splits
    X, y, meta, split = load_data(args.feature_dir, args.split_file)
    
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
