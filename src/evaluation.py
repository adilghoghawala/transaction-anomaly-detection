# src/evaluation.py

from typing import Dict

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from .models import scores_to_labels


def evaluate_anomaly_model(
    scores: np.ndarray,
    y_true: np.ndarray,
    fraction_anomalies: float = 0.02,
) -> Dict[str, float]:
    """
    Evaluate an anomaly detection model given its anomaly scores
    and the ground-truth fraud labels.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores (higher = more anomalous).
    y_true : np.ndarray
        True labels (0 = normal, 1 = fraud).
    fraction_anomalies : float
        Fraction of points to mark as anomalies when binarizing scores.

    Returns
    -------
    metrics : dict
        Dictionary with precision, recall, f1, ROC-AUC, and PR-AUC.
    """
    # Convert continuous scores into binary anomaly predictions
    y_pred = scores_to_labels(scores, fraction_anomalies=fraction_anomalies)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )

    # Some metrics need scores (not hard labels)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }
