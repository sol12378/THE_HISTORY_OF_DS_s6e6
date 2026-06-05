# Experiment Summary

## Current Best

| Type | Experiment | CV | LB | Notes |
|---|---|---:|---:|---|
| Baseline | exp001_baseline | 0.964030 | 0.96462 | LightGBM + color indices, official balanced accuracy |
| Best CV | exp003_oof_threshold_search | 0.964788 | | OOF class multiplier diagnostic; no LB yet |
| Best LB | exp006_rule_postprocess | 0.964108 | 0.96463 | Rule postprocess improved LB by +0.00001 vs exp001 |

## Experiments

| Exp | Date | Hypothesis | CV | LB | Decision | Notes |
|---|---|---|---:|---:|---|---|
| exp001_baseline | 2026-06-05 | Establish a reproducible LightGBM baseline for official balanced accuracy. | 0.964030 | 0.96462 | submitted | CV/LB gap +0.000590; leakage risk low-medium due to ID-like columns retained |
| exp002_advanced_features | 2026-06-05 | Add advanced color/redshift/sky features to improve LightGBM CV. | | | timeout | 59 features + 5-fold LightGBM exceeded 20min fast-experiment budget; split feature groups next |
| exp003_oof_threshold_search | 2026-06-05 | Optimize class multipliers on exp001 OOF probabilities for balanced accuracy. | 0.964788 | | diagnostic_success | +0.000758 CV balanced_accuracy; overfitting risk medium-high because tuned on OOF labels |
| exp004_lgbm_basic_proba | 2026-06-05 | Rerun baseline and save test_proba.csv for threshold/ensemble work. | | | timeout | Timed out after 20min under CPU contention; no artifacts |
| exp006_rule_postprocess | 2026-06-05 | Apply simple redshift/category rules to exp001 hard predictions. | 0.964108 | 0.96463 | submitted | Small CV and LB gain; CV/LB gap +0.000522; overfitting risk medium-high |
