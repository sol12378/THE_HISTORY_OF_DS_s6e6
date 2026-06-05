# exp014_lgbm_catboost_blend

## Purpose

LightGBM (`exp009_lgbm_color_redshift_small`) と CatBoost GPU (`exp010_catboost_gpu_basic`) の probability を soft voting し、model diversity で CV balanced_accuracy が改善するか確認する。

## Result

- CV balanced_accuracy: 0.964652
- CV accuracy: 0.962783
- Left weight: 1.000
- Right weight: 0.000
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.3, 'STAR': 1.35}
- Public LB: 0.96545
- Kaggle API submission ref: 53390199
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Risks

- Leakage risk: model training への leakage はないが、blend weights と multipliers を OOF labels で選んでいる。
- Overfitting risk: medium-high。LB確認が必要。
