from pathlib import Path

# Project root = one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "creditcard_processed.parquet"
