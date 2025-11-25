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
