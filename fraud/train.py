"""Train both models, pick their alert thresholds, test them on the newest data, save XGBoost.

Usage:
    python -m fraud.train --data data/creditcard.csv   # the CSV from Kaggle
    python -m fraud.train --openml                      # download from OpenML, no account needed
    python -m fraud.train --synthetic                   # quick dry run on fake data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from .data import TARGET, load_transactions, time_split
from .evaluate import Result, evaluate, pick_threshold
from .explain import global_importance
from .features import FEATURE_COLUMNS, make_features
from .models import baseline_model, boosted_model
from .synthetic import make_synthetic


def run(
    df,
    review_cost: float = 5.0,
    out_dir: str | Path = "reports",
    model_path: str | Path = "models/model.joblib",
    verbose: bool = True,
) -> tuple[list[Result], dict]:
    out_dir, model_path = Path(out_dir), Path(model_path)

    train, val, test = time_split(df)
    x_train, x_val, x_test = (make_features(part) for part in (train, val, test))
    y_train, y_val, y_test = (part[TARGET].to_numpy() for part in (train, val, test))
    if min(y_train.sum(), y_val.sum(), y_test.sum()) == 0:
        raise ValueError("Every split needs at least one fraud. Use more data.")

    fraud_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    models = {
        "Logistic regression": baseline_model().fit(x_train, y_train),
        "XGBoost": boosted_model(fraud_weight).fit(
            x_train, y_train, eval_set=[(x_val, y_val)], verbose=False
        ),
    }

    results: list[Result] = []
    test_scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        # Thresholds are chosen on the validation set. The test set is only used to report.
        threshold = pick_threshold(y_val, model.predict_proba(x_val)[:, 1], val["Amount"], review_cost)
        test_scores[name] = model.predict_proba(x_test)[:, 1]
        results.append(evaluate(name, y_test, test_scores[name], test["Amount"], threshold, review_cost))

    bundle = {
        "model": models["XGBoost"],
                  "threshold": next(r for r in results if r.model == "XGBoost").threshold,
        "features": FEATURE_COLUMNS,
        "review_cost": review_cost,
    }

    splits = {
        name: {"transactions": len(part), "frauds": int(part[TARGET].sum())}
        for name, part in (("train", train), ("validation", val), ("test", test))
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"review_cost": review_cost, "splits": splits, "results": [r.to_dict() for r in results]},
            indent=2,
        )
    )
    table = results_markdown(results, splits, review_cost)
    (out_dir / "results.md").write_text(table)
    plot_pr_curves(y_test, test_scores, results, out_dir / "pr_curve.png")
    plot_top_features(models["XGBoost"], x_test, out_dir / "top_features.png")

    if verbose:
        print(table)
        print(f"Saved the XGBoost model to {model_path} and charts to {out_dir}/")
    return results, bundle


def results_markdown(results: list[Result], splits: dict, review_cost: float) -> str:
    test = splits["test"]
    lines = [
        f"Test set: the newest 20% of transactions ({test['transactions']:,} transactions, "
        f"{test['frauds']} frauds). Alert thresholds were picked on the validation set, "
        f"assuming each alert costs {review_cost:g} to review.",
        "",
        "| Model | PR-AUC | Fraud caught (recall) | Alerts that were fraud (precision) "
        "| Alerts | Fraud amount caught | Losses avoided vs no model |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        total = r.amount_caught + r.amount_missed
        lines.append(
            f"| {r.model} | {r.pr_auc:.3f} | {r.recall:.1%} ({r.fraud_caught}/{r.fraud_total}) "
            f"| {r.precision:.1%} | {r.alerts:,} | {r.amount_caught:,.0f} of {total:,.0f} "
            f"| {r.cost_saved:.1%} |"
        )
    return "\n".join(lines) + "\n"


def plot_pr_curves(y_test, test_scores: dict, results: list[Result], path: Path) -> None:
    fig = Figure(figsize=(6.5, 4.5))
    ax = fig.subplots()
    for result in results:
        precision, recall, _ = precision_recall_curve(y_test, test_scores[result.model])
        ax.plot(recall, precision, label=f"{result.model} (PR-AUC {result.pr_auc:.3f})")
    base_rate = float(np.mean(y_test))
    ax.axhline(base_rate, linestyle="--", color="grey", label=f"Random guessing ({base_rate:.2%} fraud)")
    ax.set_xlabel("Recall: share of fraud caught")
    ax.set_ylabel("Precision: share of alerts that are fraud")
    ax.set_title("Precision vs recall on the newest 20% of transactions")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_top_features(model, x_test, path: Path, top: int = 10) -> None:
    importance = global_importance(model, x_test).head(top)[::-1]
    fig = Figure(figsize=(6.5, 4.5))
    ax = fig.subplots()
    ax.barh(importance.index, importance.to_numpy())
    ax.set_xlabel("Average effect on the fraud score (mean |SHAP value|)")
    ax.set_title("Features that drive the XGBoost model")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="path to creditcard.csv")
    source.add_argument("--openml", action="store_true", help="download the dataset from OpenML")
    source.add_argument("--synthetic", action="store_true", help="dry run on generated data")
    parser.add_argument("--review-cost", type=float, default=5.0, help="assumed cost of reviewing one alert (default 5)")
    parser.add_argument("--out", type=Path, help="folder for results (default reports/, or reports/dry-run/ with --synthetic)")
    args = parser.parse_args(argv)

    if args.synthetic:
        df = make_synthetic()
        out_dir = args.out or Path("reports/dry-run")
    else:
        df = load_transactions(None if args.openml else args.data)
        out_dir = args.out or Path("reports")
    run(df, review_cost=args.review_cost, out_dir=out_dir)


if __name__ == "__main__":
    main()
