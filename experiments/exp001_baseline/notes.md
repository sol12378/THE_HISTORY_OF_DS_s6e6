# exp001_baseline

## Purpose

初回提出用に、公式 metric の `balanced_accuracy` に合わせた LightGBM baseline と再現可能な CV/OOF/submission pipeline を確立する。

## Hypothesis

`redshift`、測光特徴量、`spectral_type` / `galaxy_population`、色指数6個を LightGBM に入れることで、単体 baseline でも `balanced_accuracy` 0.95 前後に到達し、初回提出に十分な形式検証済み `submission.csv` を作れる。

## Result

- CV balanced_accuracy: 0.964030
- CV accuracy: 0.965139
- Public LB balanced_accuracy: 0.964620
- LB - CV gap: +0.000590
- Kaggle submission ref: 53383739
- Train shape: (577347, 18)
- Test shape: (247435, 17)
- Target distribution: {'GALAXY': 377480, 'QSO': 117143, 'STAR': 82724}
- Submission distribution: {'GALAXY': 158747, 'QSO': 50846, 'STAR': 37842}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_test': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Changes

- 5-fold `StratifiedKFold(random_state=42)`。
- LightGBM multiclass + `class_weight="balanced"`。
- 色指数: `u_g`, `g_r`, `r_i`, `i_z`, `u_r`, `g_i`。
- `spectral_type` と `galaxy_population` は categorical feature として使用。
- `oof.csv`, `submission.csv`, `feature_importance.csv`, `result.json` を保存。

## Risks

- Leakage risk: ID-like columns は初回 baseline では残している。target encoding や test label 由来の処理は使っていないため直接 leakage はないが、ID 系が synthetic split に強く効く可能性は次回検証する。
- Overfitting risk: LightGBM 2000 estimators は early stopping 付き。提出後に CV/LB gap を確認し、validation 連動性を検証する。

## Next Actions

- CV/LB gap は小さく、validation は初回 baseline として信頼できそう。
- 次フェーズでは ID 系 drop 比較、全色指数、redshift interaction を検証する。
