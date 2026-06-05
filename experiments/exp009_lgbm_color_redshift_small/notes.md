# exp009_lgbm_color_redshift_small

## Purpose

`exp002_advanced_features` の全盛り timeout を避け、color/redshift/magnitude 統計に絞って LightGBM CV を改善する。

## Hypothesis

feature importance 上位の color/redshift 周辺だけを追加すれば、runtime を抑えつつ `exp001_baseline` の CV balanced_accuracy 0.964030 を上回れる。

## Result

- CV balanced_accuracy: 0.963628
- CV accuracy: 0.964765
- Feature mode: color_redshift
- Feature count: 36
- Train shape: (577347, 38)
- Test shape: (247435, 37)
- Target distribution: {'GALAXY': 377480, 'QSO': 117143, 'STAR': 82724}
- Submission distribution: {'GALAXY': 158665, 'QSO': 50843, 'STAR': 37927}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_test': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Changes

- 5-fold `StratifiedKFold(random_state=42)`。
- LightGBM multiclass + `class_weight="balanced"`。
- Feature mode: `color_redshift`。
- `spectral_type` と `galaxy_population` は categorical feature として使用。
- `oof.csv`, `test_proba.csv`, `submission.csv`, `feature_importance.csv`, `result.json` を保存。

## Risks

- Leakage risk: ID-like columns は初回 baseline では残している。target encoding や test label 由来の処理は使っていないため直接 leakage はないが、ID 系が synthetic split に強く効く可能性は次回検証する。
- Overfitting risk: LightGBM 2000 estimators は early stopping 付き。提出後に CV/LB gap を確認し、validation 連動性を検証する。

## Next Actions

- Kaggle に初回提出し、LB と CV gap を記録する。
- 次フェーズでは ID 系 drop 比較、全色指数、redshift interaction を検証する。
