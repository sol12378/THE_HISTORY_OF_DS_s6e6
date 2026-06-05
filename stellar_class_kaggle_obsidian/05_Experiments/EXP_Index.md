# Experiment Index

| Exp | Date | Status | CV | LB | Notes |
|---|---|---|---:|---:|---|
| [[exp001_baseline]] | 2026-06-05 | submitted | 0.964030 | 0.96462 | LightGBM + color indices baseline |
| [[exp002_advanced_features]] | 2026-06-05 | timeout | | | Advanced features too slow as one 5-fold experiment |
| [[exp003_oof_threshold_search]] | 2026-06-05 | diagnostic_success | 0.964788 | | OOF class multiplier search; no submission yet |
| [[exp004_lgbm_basic_proba]] | 2026-06-05 | timeout | | | Baseline rerun for test_proba timed out |
| [[exp006_rule_postprocess]] | 2026-06-05 | submitted | 0.964108 | 0.96463 | Small rule-based postprocess gain |
| [[exp007_nina_simple_vote]] | 2026-06-05 | submitted | | 0.97072 | Reproduced specified public simple vote notebook |
| [[exp008_nina_weighted_vote]] | 2026-06-05 | submitted | | 0.97074 | Improved weighted vote over specified notebook |
| [[exp009_lgbm_color_redshift_small]] | 2026-06-05 | completed | 0.963628 | | Focused features hurt raw CV |
| [[exp010_catboost_gpu_basic]] | 2026-06-05 | completed | 0.951213 | | GPU works, CatBoost setup weak |
| [[exp011_exp009_threshold_search]] | 2026-06-05 | diagnostic_success | 0.964604 | | Threshold improved exp009 OOF |
| [[exp012_exp009_threshold_submission]] | 2026-06-05 | submitted | 0.964604 | 0.96523 | Best self-model LB before exp014 |
| [[exp013_catboost_threshold_search]] | 2026-06-05 | diagnostic_success | 0.954779 | | CatBoost still below LightGBM |
| [[exp014_lgbm_catboost_blend]] | 2026-06-05 | submitted | 0.964652 | 0.96545 | Best self-model LB; CatBoost weight 0 |

| Exp | Status | CV | LB | Decision |
|---|---|---:|---:|---|
| exp001_baseline | planned | | | |
