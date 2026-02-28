"""
Generate synthetic subscription dataset with churn logic.
Writes data/subscription_data.csv.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent / "data" / "subscription_data.csv"
N_USERS = 5000
SEED = 42


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tenure_days = np.random.exponential(scale=180, size=N_USERS).astype(int)
    tenure_days = np.clip(tenure_days, 1, 2000)
    sessions_last_30d = np.random.poisson(lam=12, size=N_USERS)
    sessions_last_30d = np.clip(sessions_last_30d, 0, 60)
    payment_failures = np.random.poisson(lam=0.5, size=N_USERS)
    payment_failures = np.clip(payment_failures, 0, 8)
    support_tickets = np.random.poisson(lam=0.8, size=N_USERS)
    support_tickets = np.clip(support_tickets, 0, 10)
    feature_usage_score = np.random.beta(2, 2, size=N_USERS) * 10
    feature_usage_score = np.clip(feature_usage_score, 0, 10)

    churn_logit = (
        -0.02 * tenure_days
        - 0.08 * sessions_last_30d
        + 0.35 * payment_failures
        + 0.12 * support_tickets
        - 0.25 * feature_usage_score
        + np.random.normal(0, 0.8, size=N_USERS)
    )
    churned = (1 / (1 + np.exp(-churn_logit)) > 0.5).astype(int)

    df = pd.DataFrame({
        "user_id": np.arange(1, N_USERS + 1),
        "tenure_days": tenure_days,
        "sessions_last_30d": sessions_last_30d,
        "payment_failures": payment_failures,
        "support_tickets": support_tickets,
        "feature_usage_score": np.round(feature_usage_score, 2),
        "churned": churned,
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
