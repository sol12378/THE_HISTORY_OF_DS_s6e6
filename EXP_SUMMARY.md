# Experiment Summary

## Current Best

| Type | Experiment | CV | LB | Notes |
|---|---|---:|---:|---|
| Baseline | exp001_baseline | 0.964030 | 0.96462 | LightGBM + color indices, official balanced accuracy |
| Best CV | exp003_oof_threshold_search | 0.964788 | | OOF class multiplier diagnostic; no LB yet |
| Best LB | exp008_nina_weighted_vote | | 0.97074 | External public submission weighted vote; high leakage/overfitting risk |

## Experiments

| Exp | Date | Hypothesis | CV | LB | Decision | Notes |
|---|---|---|---:|---:|---|---|
| exp001_baseline | 2026-06-05 | Establish a reproducible LightGBM baseline for official balanced accuracy. | 0.964030 | 0.96462 | submitted | CV/LB gap +0.000590; leakage risk low-medium due to ID-like columns retained |
| exp002_advanced_features | 2026-06-05 | Add advanced color/redshift/sky features to improve LightGBM CV. | | | timeout | 59 features + 5-fold LightGBM exceeded 20min fast-experiment budget; split feature groups next |
| exp003_oof_threshold_search | 2026-06-05 | Optimize class multipliers on exp001 OOF probabilities for balanced accuracy. | 0.964788 | | diagnostic_success | +0.000758 CV balanced_accuracy; overfitting risk medium-high because tuned on OOF labels |
| exp004_lgbm_basic_proba | 2026-06-05 | Rerun baseline and save test_proba.csv for threshold/ensemble work. | | | timeout | Timed out after 20min under CPU contention; no artifacts |
| exp006_rule_postprocess | 2026-06-05 | Apply simple redshift/category rules to exp001 hard predictions. | 0.964108 | 0.96463 | submitted | Small CV and LB gain; CV/LB gap +0.000522; overfitting risk medium-high |
| exp007_nina_simple_vote | 2026-06-05 | Reproduce nina2025 simple vote notebook final combo. | | 0.97072 | submitted | External public submission blend; no local CV; leakage/overfit risk high |
| exp008_nina_weighted_vote | 2026-06-05 | Improve nina2025 simple vote using top-12 weighted vote by filename score. | | 0.97074 | submitted | New best LB; external public submission blend; leakage/overfit risk high |
| exp009_lgbm_color_redshift_small | 2026-06-05 | Add focused color/redshift/magnitude features to improve LightGBM CV. | 0.963628 | | completed | Worse than exp001; useful test_proba artifact produced |
| exp010_catboost_gpu_basic | 2026-06-05 | Run CatBoost GPU diagnostic for model diversity. | 0.951213 | | completed | GPU works and is fast, but this CatBoost setup is weak for balanced accuracy |
| exp011_exp009_threshold_search | 2026-06-05 | Optimize class multipliers on exp009 OOF probabilities. | 0.964604 | | diagnostic_success | +0.000976 over exp009; overfitting risk medium-high |
| exp012_exp009_threshold_submission | 2026-06-05 | Apply exp011 multipliers to exp009 test probabilities. | 0.964604 | 0.96523 | submitted | Best self-model LB before exp014 |
| exp013_catboost_threshold_search | 2026-06-05 | Optimize class multipliers on exp010 CatBoost OOF. | 0.954779 | | diagnostic_success | Still below LightGBM; do not use standalone |
| exp014_lgbm_catboost_blend | 2026-06-05 | Blend exp009 LightGBM and exp010 CatBoost probabilities. | 0.964652 | 0.96545 | submitted | CatBoost weight optimized to 0; thresholded exp009 improved self-model LB |
