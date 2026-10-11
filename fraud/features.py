"""Turn raw transaction fields into model features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import V_COLUMNS

FEATURE_COLUMNS = [*V_COLUMNS, "LogAmount", "Hour"]


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep V1-V28 as they are, log the amount, and turn Time into hour of day.

    Raw `Time` is seconds since the first transaction in the dataset. With a time-based
    split every test value is larger than every training value, so the model can't use
    it sensibly. Hour of day keeps the useful part: when in the day the purchase happened.
    """
    features = raw[V_COLUMNS].astype(float).copy()
    features["LogAmount"] = np.log1p(raw["Amount"].astype(float))
    features["Hour"] = (raw["Time"].astype(float) // 3600) % 24
    return features[FEATURE_COLUMNS]
