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
    """
    Main training function.
    
    Steps:
    1. Parse arguments
    2. Set random seed
    3. Load data and splits
    4. Split into train/val
    5. Scale features (if enabled)
    6. Load model config
    7. Compute class weights (if enabled)
    8. Initialize and train model
    9. Get predictions and probabilities
    10. Compute metrics
    11. Create output directories
    12. Save artifacts
    """
    # TODO: Implement main training pipeline
    raise NotImplementedError


if __name__ == "__main__":
    main()
