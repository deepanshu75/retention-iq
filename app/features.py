"""Load and look up user features from the subscription dataset."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "subscription_data.csv"
FEATURE_COLUMNS = [
    "tenure_days",
    "sessions_last_30d",
    "payment_failures",
    "support_tickets",
    "feature_usage_score",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def get_user_features(df: pd.DataFrame, user_id: str) -> pd.Series | None:
    row = df[df["user_id"].astype(str) == str(user_id)]
    if row.empty:
        return None
    return row[FEATURE_COLUMNS].iloc[0]
