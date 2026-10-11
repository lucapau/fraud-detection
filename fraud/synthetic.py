"""Fake transactions with the same columns as the real dataset, for tests and dry runs.

The numbers mean nothing: fraud rows just get shifted V-values and bigger amounts so the
models have something to learn. Only report results from the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET, V_COLUMNS


def make_synthetic(n: int = 20_000, fraud_rate: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    v = rng.normal(size=(n, len(V_COLUMNS)))
    shift = np.zeros(len(V_COLUMNS))
    shift[[2, 9, 11, 13]] = [-2.5, -2.0, -2.0, -3.0]  # V3, V10, V12, V14
    v[is_fraud] += shift

    df = pd.DataFrame(v, columns=V_COLUMNS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 2 * 86_400, n)))  # two days, like the real data
    df["Amount"] = np.round(rng.lognormal(3.0, 1.2, n) * np.where(is_fraud, 1.5, 1.0), 2)
    df[TARGET] = is_fraud.astype(int)
    return df
