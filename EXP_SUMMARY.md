# Experiment Summary

## Current Best

| Type | Experiment | CV | LB | Notes |
|---|---|---:|---:|---|
| Baseline | exp001_baseline | 0.964030 | 0.96462 | LightGBM + color indices, official balanced accuracy |
| Best CV | exp024_oof_stacker_with_lgbm_basic | 0.966112 | | Logistic OOF stacker with an additional basic LightGBM base; API submission failed likely due daily limit |
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
| exp015_sdss17_source_match | 2026-06-05 | Check whether public SDSS17 data can directly match official train/test rows. | | | diagnostic_completed | Rounded exact matches were essentially zero; sampled nearest-neighbor label balanced accuracy only ~0.81, so direct label transfer is rejected for now |
| exp016_xgboost_gpu_basic | 2026-06-05 | Train GPU XGBoost diagnostic for model diversity. | 0.964020 | | completed | Strong enough to use as blend candidate, but below best thresholded LightGBM standalone |
| exp017_lgbm_xgboost_blend | 2026-06-05 | Blend exp009 LightGBM and exp016 GPU XGBoost OOF/test probabilities. | 0.965235 | 0.96597 | submitted | New best self-model CV and LB; XGBoost adds useful diversity |
| exp018_oof_logistic_stacker | 2026-06-05 | Train a nested-CV logistic OOF stacker over available self-model probabilities. | 0.965560 | 0.96641 | submitted | New best self-model CV and LB; stacker improves over fixed two-model blend |
| exp019_xgboost_gpu_deep | 2026-06-05 | Train a deeper/longer GPU XGBoost diagnostic for stack diversity. | 0.962960 | | completed | Weak standalone balanced accuracy but useful as an additional stack base |
| exp020_oof_stacker_with_xgb_deep | 2026-06-05 | Add exp019 deep GPU XGBoost to the logistic OOF stacker. | 0.965901 | 0.96642 | submitted | New best self-model CV; LB only slightly improved, so future submissions need strong raw-CV gains |
| exp021_xgboost_gpu_shallow | 2026-06-05 | Train a shallow/regularized GPU XGBoost diagnostic for stack diversity. | 0.964128 | | completed | Stronger standalone than exp019, but only useful if stack improves |
| exp022_oof_stacker_with_xgb_shallow | 2026-06-05 | Add exp021 shallow GPU XGBoost to the exp020 stack. | 0.965864 | | no_submit | Below exp020 CV, so no API submission under confident-CV-only rule |
| exp023_lgbm_basic_fast | 2026-06-05 | Train a fast basic-feature LightGBM base with OOF/test probabilities. | 0.963713 | | completed | Not a standalone submission; useful as stack diversity |
| exp024_oof_stacker_with_lgbm_basic | 2026-06-05 | Add exp023 basic LightGBM to the exp020 stack. | 0.966112 | | api_failed | New best CV; API submission failed with 400 likely due daily submission limit |
| exp025_lgbm_color_regularized | 2026-06-05 | Train a regularized stochastic color/redshift LightGBM base. | 0.962930 | | completed | Weak standalone; tested only as stack diversity |
| exp026_oof_stacker_with_two_lgbm | 2026-06-05 | Add both exp023 and exp025 LightGBM bases to the stack. | 0.966092 | | no_submit | Below exp024; no API submission |
