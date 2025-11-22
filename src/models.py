from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


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
    """
    # decision_function returns (higher = more normal), so negate it
    scores = -model.decision_function(X)
    return scores
