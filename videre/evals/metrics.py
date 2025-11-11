import numpy as np
from typing import Optional, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Minimal metricss for now
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        scores = (
            y_proba[:, 1]
            if (y_proba.ndim == 2 and y_proba.shape[1] >= 2)
            else y_proba if y_proba.ndim == 1
            else None
        )
        if scores is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics["average_precision"] = float(average_precision_score(y_true, scores))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["average_precision"] = float("nan")

    return metrics
