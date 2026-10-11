"""A small web API that scores one transaction and explains the score.

Run it after training, so models/model.joblib exists:
    uvicorn fraud.api:app --reload
then open http://127.0.0.1:8000/docs to try it in the browser.
"""

import math
import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import Field, create_model

from .data import V_COLUMNS
from .explain import top_reasons
from .features import make_features

# The request body: the same raw fields as one row of the dataset.
Transaction = create_model(
    "Transaction",
    Time=(float, Field(ge=0, description="Seconds since the first transaction in the dataset")),
    Amount=(float, Field(ge=0, description="Transaction amount")),
    **{column: (float, Field(description="Anonymised feature supplied by the bank")) for column in V_COLUMNS},
)


def load_bundle(path: str | None = None) -> dict:
    """Load the model saved by `python -m fraud.train`."""
    return joblib.load(path or os.environ.get("MODEL_PATH", "models/model.joblib"))


def create_app(bundle: dict | None = None) -> FastAPI:
    app = FastAPI(title="Card fraud scoring API", version="1.0.0")

    @lru_cache(maxsize=1)
    def model_bundle() -> dict:
        return bundle if bundle is not None else load_bundle()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/score")
    def score(transaction: Transaction) -> dict:
        b = model_bundle()
        features = make_features(pd.DataFrame([transaction.model_dump()]))
        probability = float(b["model"].predict_proba(features)[:, 1][0])
        threshold = b["threshold"]
        return {
            "fraud_probability": round(probability, 6),
            "flagged": probability >= threshold,
            "threshold": threshold if math.isfinite(threshold) else None,
            "top_reasons": top_reasons(b["model"], features, k=3)[0],
        }

    return app


app = create_app()
