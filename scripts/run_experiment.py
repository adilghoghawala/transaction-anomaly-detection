# scripts/run_experiment.py

import sys
from pathlib import Path

# --- make Python see the "src" package ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------

from src.data_loading import load_raw_creditcard
from src.preprocessing import train_test_preprocess
from src.models import (
    fit_isolation_forest,
    score_isolation_forest,
)
from src.evaluation import evaluate_anomaly_model


def main():
    df = load_raw_creditcard()
    X_train, X_test, y_train, y_test = train_test_preprocess(df)

    iso = fit_isolation_forest(X_train)
    scores = score_isolation_forest(iso, X_test)

    eval_results = evaluate_anomaly_model(scores, y_test)
    print(eval_results)


if __name__ == "__main__":
    main()
