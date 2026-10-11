"""Choosing the alert threshold and measuring how well a model does."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pick_threshold(y_true, scores, amounts, review_cost: float = 5.0) -> float:
    """Return the score threshold that gives the lowest total cost.

    The cost model is an assumption (change it with --review-cost):
      * every fraud we miss costs its transaction amount
      * every alert we raise costs a fixed amount for someone to review it

    Accuracy is no use here: flagging nothing at all is 99.8% accurate.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    amt = np.asarray(amounts, dtype=float)

    order = np.argsort(-s, kind="stable")
    s, y, amt = s[order], y[order], amt[order]

    # Total cost if we flag the k highest-scoring transactions, for every k from 0 to n.
    caught = np.concatenate([[0.0], np.cumsum(np.where(y, amt, 0.0))])
    k = np.arange(len(s) + 1)
    cost = (caught[-1] - caught) + review_cost * k

    # Only cut between different scores, so tied transactions are always treated alike.
    allowed = np.ones(len(s) + 1, dtype=bool)
    allowed[1:-1] = s[:-1] != s[1:]
    candidates = np.flatnonzero(allowed)
    best_k = int(candidates[np.argmin(cost[candidates])])
    return float("inf") if best_k == 0 else float(s[best_k - 1])


@dataclass
class Result:
    model: str
    pr_auc: float
    roc_auc: float
    threshold: float
    alerts: int
    precision: float
    recall: float
    fraud_caught: int
    fraud_total: int
    amount_caught: float
    amount_missed: float
    review_cost_total: float
    total_cost: float
    cost_without_model: float

    @property
    def cost_saved(self) -> float:
        """Share of fraud losses avoided, after paying for reviews, versus having no model."""
        if not self.cost_without_model:
            return 0.0
        return 1 - self.total_cost / self.cost_without_model

    def to_dict(self) -> dict:
        return {**asdict(self), "cost_saved": self.cost_saved}


def evaluate(name, y_true, scores, amounts, threshold: float, review_cost: float) -> Result:
    """Score a model on held-out data at a fixed threshold."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    amt = np.asarray(amounts, dtype=float)

    flagged = s >= threshold
    caught = flagged & y
    alerts = int(flagged.sum())
    frauds = int(y.sum())
    amount_missed = float(amt[y & ~flagged].sum())
    review_total = review_cost * alerts

    return Result(
        model=name,
        pr_auc=float(average_precision_score(y, s)),
        roc_auc=float(roc_auc_score(y, s)),
        threshold=float(threshold),
        alerts=alerts,
        precision=int(caught.sum()) / alerts if alerts else 0.0,
        recall=int(caught.sum()) / frauds if frauds else 0.0,
        fraud_caught=int(caught.sum()),
        fraud_total=frauds,
        amount_caught=float(amt[caught].sum()),
        amount_missed=amount_missed,
        review_cost_total=review_total,
        total_cost=amount_missed + review_total,
        cost_without_model=float(amt[y].sum()),
    )
