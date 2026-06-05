# Hypothesis Backlog

| Priority | Hypothesis | Suggested Exp | Risk |
|---|---|---|---|
| High | Class multiplier tuning can improve official balanced_accuracy, but needs test probabilities and LB confirmation | rerun LightGBM with `test_proba.csv`, then thresholded submission | Medium-high overfitting |
| High | Advanced features should be split into groups instead of all at once | color-only, redshift-interaction-only, sky-only experiments | Runtime / overfitting |
| Medium | Model diversity can improve OOF ensemble | CatBoost / XGBoost OOF, then soft voting or stacking | Runtime |
| Medium | Public submission weighted voting can improve LB, but not local CV | exp008-style vote variants | High leakage / public LB overfit |

| Hypothesis | Priority | Status | Notes |
|---|---:|---|---|
