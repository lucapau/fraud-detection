import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud.api import create_app
from fraud.data import TARGET, time_split
from fraud.evaluate import pick_threshold
from fraud.explain import shap_values
from fraud.features import FEATURE_COLUMNS, make_features
from fraud.synthetic import make_synthetic
from fraud.train import run


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n=6_000, fraud_rate=0.03, seed=1)


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return run(data, out_dir=out / "reports", model_path=out / "model.joblib", verbose=False)


def a_transaction(data, **changes) -> dict:
    return data.drop(columns=TARGET).iloc[0].to_dict() | changes


def test_time_split_keeps_test_set_newest(data):
    train, val, test = time_split(data)
    assert len(train) + len(val) + len(test) == len(data)
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_features_have_expected_columns(data):
    features = make_features(data.head(10))
    assert list(features.columns) == FEATURE_COLUMNS
    assert features["Hour"].between(0, 23).all()


def test_threshold_never_costs_more_than_doing_nothing():
    rng = np.random.default_rng(0)
    is_fraud = rng.random(2_000) < 0.05
    scores = np.where(is_fraud, rng.uniform(0.4, 1.0, 2_000), rng.uniform(0.0, 0.6, 2_000))
    amounts = rng.uniform(1, 500, 2_000)

    threshold = pick_threshold(is_fraud, scores, amounts, review_cost=5)
    flagged = scores >= threshold
    cost = amounts[is_fraud & ~flagged].sum() + 5 * flagged.sum()
    assert cost <= amounts[is_fraud].sum()


def test_threshold_treats_tied_scores_alike():
    scores = np.array([0.9, 0.5, 0.5, 0.1])
    is_fraud = np.array([1, 1, 0, 0])
    amounts = np.array([100.0, 100.0, 100.0, 100.0])
    threshold = pick_threshold(is_fraud, scores, amounts, review_cost=1)
    flagged = scores >= threshold
    assert flagged[1] == flagged[2]


def test_xgboost_ranks_fraud_far_better_than_chance(trained, data):
    results, _ = trained
    xgb = next(r for r in results if r.model == "XGBoost")
    assert xgb.pr_auc > 5 * data[TARGET].mean()


def test_shap_values_add_up_to_the_model_score(trained, data):
    _, bundle = trained
    features = make_features(data.tail(50))
    contributions = shap_values(bundle["model"], features, include_bias=True)
    raw_score = bundle["model"].predict(features, output_margin=True)
    np.testing.assert_allclose(contributions.sum(axis=1), raw_score, atol=1e-3)


def test_api_scores_and_explains_a_transaction(trained, data):
    _, bundle = trained
    client = TestClient(create_app(bundle))
    response = client.post("/score", json=a_transaction(data))
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert isinstance(body["flagged"], bool)
    assert len(body["top_reasons"]) == 3


def test_api_rejects_a_negative_amount(trained, data):
    _, bundle = trained
    client = TestClient(create_app(bundle))
    response = client.post("/score", json=a_transaction(data, Amount=-1))
    assert response.status_code == 422
