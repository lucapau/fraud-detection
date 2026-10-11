Test set: the newest 20% of transactions (56,962 transactions, 75 frauds). Alert thresholds were picked on the validation set, assuming each alert costs 5 to review.

| Model | PR-AUC | Fraud caught (recall) | Alerts that were fraud (precision) | Alerts | Fraud amount caught | Losses avoided vs no model |
|---|---|---|---|---|---|---|
| Logistic regression | 0.748 | 66.7% (50/75) | 96.2% | 52 | 4,664 of 7,729 | 57.0% |
| XGBoost | 0.792 | 74.7% (56/75) | 86.2% | 65 | 4,095 of 7,729 | 48.8% |
