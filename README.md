# transaction-anomaly-detection

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from ULB (Université Libre de Bruxelles).

- `creditcard.csv` contains **284,807** transactions made by European cardholders in September 2013.
- Each row is a transaction with:
  - `Time` – seconds elapsed between this transaction and the first transaction in the dataset
  - `V1`–`V28` – numerical features resulting from a PCA transformation (original features are confidential)
  - `Amount` – transaction amount
  - `Class` – target label (`0` = normal, `1` = fraud)

For the anomaly detection problem, models are trained in an **unsupervised** way (using only features), and the `Class` label is used **only for evaluation** (to see how well the anomaly scores align with actual frauds).

CSV can be installed here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

Place the file at:

```text
data/raw/creditcard.csv

## Project Motivation

Financial institutions process hundreds of thousands of credit card transactions every day, but only a tiny fraction are actually fraudulent. Labels for fraud are often delayed, incomplete, or noisy, which makes it difficult to rely solely on fully supervised models. However, the business still needs a way to automatically flag **suspicious transactions** so that fraud analysts can prioritize their review time.

This project builds an **unsupervised anomaly detection pipeline** using the ULB Credit Card Fraud dataset. The goal is to learn what “normal” transaction behavior looks like and assign each transaction an **anomaly score** indicating how unusual it is compared to the majority of data. Higher scores correspond to more suspicious transactions.

## What This Project Demonstrates

- **Unsupervised fraud detection**  
  The models (Isolation Forest, optionally One-Class SVM and LOF) are trained only on the feature space (`V1–V28`, `Time`, `Amount`) without using the fraud labels. This mimics real-world scenarios where labels are limited or delayed.

- **Anomaly scoring & ranking**  
  Each transaction receives a continuous anomaly score. These scores can be used to rank transactions from “most normal” to “most suspicious,” which is exactly what a fraud review team needs to triage their workload.

- **Evaluation with ground-truth labels**  
  Although training is unsupervised, the true `Class` label is used for **evaluation**:
  - ROC-AUC and PR-AUC to measure how well anomaly scores separate fraud vs. normal transactions.
  - Precision, recall, and F1 at different thresholds to study the tradeoff between catching more fraud and generating too many false alerts.

- **Business tradeoffs**  
  By adjusting the fraction of transactions flagged as anomalies, we can explore:
  - Higher thresholds → fewer alerts, higher precision, lower recall.
  - Lower thresholds → more alerts, lower precision, higher recall.  
  This reflects a real business decision: do we want to **catch almost everything** (high recall) or **minimize false alarms** (high precision)?

Overall, the project shows how unsupervised learning can be used to bootstrap a practical fraud detection system, and how to evaluate and explain its performance in a way that aligns with real-world risk and operations.

