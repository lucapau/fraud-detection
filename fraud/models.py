"""The two models being compared."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def baseline_model() -> Pipeline:
    """Logistic regression: simple, fast and easy to explain.

    class_weight="balanced" makes each fraud count as much as hundreds of normal
    transactions, so the model doesn't just learn to say "not fraud" every time.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )


def boosted_model(fraud_weight: float, random_state: int = 42) -> XGBClassifier:
    """Gradient-boosted trees (XGBoost), which can pick up patterns a straight line can't.

    scale_pos_weight does the same job as class_weight above. Early stopping watches the
    validation set and stops adding trees once they stop helping.
    """
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
