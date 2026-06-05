# exp012_exp009_threshold_submission

## Purpose

`exp009_lgbm_color_redshift_small` の `test_proba.csv` に OOF-optimized class multipliers を適用し、Kaggle submission を作成する。

## Hypothesis

`exp003_oof_threshold_search` で CV balanced_accuracy が改善した multiplier を test probabilities に適用すれば、Public LB でも `exp001_baseline` より改善する可能性がある。

## Result

- Multipliers: {'GALAXY': 1.0, 'QSO': 1.28, 'STAR': 1.3}
- Submission distribution: {'GALAXY': 157692, 'QSO': 51107, 'STAR': 38636}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}
- Public LB: 0.96523
- Kaggle API submission ref: 53389973

## Risks

- Leakage risk: target-derived training feature はないが、multiplier は OOF label で選んでいる。
- Overfitting risk: medium-high。LB で CV/LB gap を確認する。
