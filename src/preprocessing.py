# src/models.py

from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


# ---------------------------
# Isolation Forest
# ---------------------------

def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.02,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train an Isolation Forest on the training data.
    """
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train)
    return clf


def score_isolation_forest(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return anomaly scores (higher = more anomalous).
    IsolationForest.decision_function returns higher = more normal,
    so we negate it.
    """
    return -model.decision_function(X)


# ---------------------------
# One-Class SVM
# ---------------------------

def fit_oneclass_svm(
    X_train: np.ndarray,
    nu: float = 0.02,
    gamma: str = "scale",
) -> OneClassSVM:
    """
    Train a One-Class SVM on the training data.
    nu ~ expected fraction of anomalies.
    """
    clf = OneClassSVM(
        kernel="rbf",
        nu=nu,
        gamma=gamma,
    )
    clf.fit(X_train)
    return clf


def score_oneclass_svm(
    model: OneClassSVM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return anomaly scores (higher = more anomalous).
    OneClassSVM.decision_function returns higher = more normal,
    so we negate it.
    """
    return -model.decision_function(X)


# ---------------------------
# Local Outlier Factor (LOF)
# ---------------------------

def fit_lof(
    X_train: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.02,
) -> LocalOutlierFactor:
    """
    Train Local Outlier Factor model for novelty detection.
    Set novelty=True so we can call decision_function/predict on new data.
    """
    clf = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
    )
    clf.fit(X_train)
    return clf


def score_lof(
    model: LocalOutlierFactor,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return anomaly scores (higher = more anomalous).
    LOF.decision_function returns higher = more normal,
    so we negate it.
    """
    return -model.decision_function(X)


# ---------------------------
# Optional helper: scores → labels
# ---------------------------

def scores_to_labels(
    scores: np.ndarray,
    fraction_anomalies: float = 0.02,
) -> np.ndarray:
    """
    Convert continuous anomaly scores into binary labels.
    Marks the top `fraction_anomalies` scores as anomalies (1), rest as 0.
    """
    threshold = np.quantile(scores, 1 - fraction_anomalies)
    return (scores >= threshold).astype(int)
