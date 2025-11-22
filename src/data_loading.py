import pandas as pd
from .config import RAW_DATA_PATH


def load_raw_creditcard() -> pd.DataFrame:
    """
    Load the raw credit card transactions CSV.

    Returns
    -------
    df : pd.DataFrame
        Raw dataframe with columns: Time, V1..V28, Amount, Class.
    """
    df = pd.read_csv(RAW_DATA_PATH)
    return df
