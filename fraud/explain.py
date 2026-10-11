"""Explanations: which features pushed a transaction's fraud score up or down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def _iteration_range(model) -> tuple[int, int]:
    """Use the same trees as predict_proba (early stopping may have kept extra ones)."""
    try:
        return (0, model.best_iteration + 1)
    except AttributeError:
        return (0, 0)  # (0, 0) means "all trees"


def shap_values(model, features: pd.DataFrame, include_bias: bool = False) -> np.ndarray:
    """SHAP values from XGBoost's built-in TreeSHAP, so no extra library is needed.

    One row per transaction and one column per feature, in log-odds: positive values
    push towards "fraud", negative values towards "legitimate". With include_bias=True
    the last column is the baseline, and each row then adds up to the model's raw score.
    """
    contributions = model.get_booster().predict(
        xgb.DMatrix(features),
        pred_contribs=True,
        iteration_range=_iteration_range(model),
    )
    return contributions if include_bias else contributions[:, :-1]


def top_reasons(model, features: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """The k features with the biggest effect on each transaction's score."""
    values = shap_values(model, features)
    reasons = []
    for row in values:
        biggest = np.argsort(-np.abs(row))[:k]
        reasons.append(
            [{"feature": features.columns[i], "contribution": round(float(row[i]), 4)} for i in biggest]
        )
    return reasons


def global_importance(model, features: pd.DataFrame) -> pd.Series:
    """Average size of each feature's effect across many transactions."""
    values = shap_values(model, features)
    return pd.Series(np.abs(values).mean(axis=0), index=features.columns).sort_values(ascending=False)
