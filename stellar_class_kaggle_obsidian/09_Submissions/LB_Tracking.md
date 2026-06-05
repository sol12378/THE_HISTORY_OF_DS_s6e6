# LB Tracking

| Date | Submission | Source Exp | CV | LB | Gap | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-06-05 | 53383739 | exp001_baseline | 0.964030 | 0.96462 | +0.000590 | LightGBM + color indices; official metric balanced_accuracy |
| 2026-06-05 | 53383938 | exp001_baseline | 0.964030 | 0.96462 | +0.000590 | Same file submitted through Kaggle Python API |
| 2026-06-05 | 53387393 | exp006_rule_postprocess | 0.964108 | 0.96463 | +0.000522 | Simple redshift/category rules on exp001 hard predictions |
| 2026-06-05 | 53387851 | exp007_nina_simple_vote | | 0.97072 | | Reproduced `nina2025/ps-s6e6-simple-vote`; no local CV |
| 2026-06-05 | 53387906 | exp008_nina_weighted_vote | | 0.97074 | | Top-12 weighted vote from `nina2025/ps-s6e6`; no local CV |
| 2026-06-05 | 53389973 | exp012_exp009_threshold_submission | 0.964604 | 0.96523 | +0.000626 | Thresholded exp009 LightGBM |
| 2026-06-05 | 53390199 | exp014_lgbm_catboost_blend | 0.964652 | 0.96545 | +0.000798 | Best self-model LB; CatBoost blend weight 0 |
| 2026-06-05 | 53391073 | exp017_lgbm_xgboost_blend | 0.965235 | 0.96597 | +0.000735 | New best self-model LB; XGBoost adds useful diversity |
| 2026-06-05 | 53391289 | exp018_oof_logistic_stacker | 0.965560 | 0.96641 | +0.000850 | New best self-model LB; logistic stacker over LightGBM/CatBoost/XGBoost |
