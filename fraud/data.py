"""Loading and splitting the credit-card fraud data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *V_COLUMNS, "Amount"]
TARGET = "Class"
OPENML_ID = 1597  # "creditcard" on OpenML: the same ULB dataset that is on Kaggle


def load_transactions(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load creditcard.csv from disk, or download it from OpenML when no path is given."""
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame

    missing = sorted(set(RAW_COLUMNS + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[RAW_COLUMNS + [TARGET]].copy()
    df[RAW_COLUMNS] = df[RAW_COLUMNS].astype(float)
    # OpenML stores the label as the text categories "0"/"1"; the CSV stores numbers.
    df[TARGET] = pd.to_numeric(df[TARGET].astype(str)).astype(int)
    return df


def time_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: the oldest transactions train the model, the newest test it.

    A random split lets the model learn from transactions that happen *after* the ones
    it is tested on, which flatters the score. A bank scores new transactions with a
    model trained on older ones, so the test set is the most recent slice.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1 and sum to less than 1")

    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return ordered.iloc[:train_end], ordered.iloc[train_end:val_end], ordered.iloc[val_end:]
