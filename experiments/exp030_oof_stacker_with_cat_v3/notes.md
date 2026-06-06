# exp030_oof_stacker_with_cat_v3

## Purpose

利用可能な自前モデルOOF probabilityを使い、LogisticRegressionのmeta-modelでOOF stackを試す。

## Result

- Base experiments: ['exp009_lgbm_color_redshift_small', 'exp010_catboost_gpu_basic', 'exp016_xgboost_gpu_basic', 'exp019_xgboost_gpu_deep', 'exp023_lgbm_basic_fast', 'exp029_catboost_v3_style']
- Best C: 10.0
- Raw CV balanced_accuracy: 0.966222
- Thresholded CV balanced_accuracy: 0.966222
- CV accuracy: 0.960381
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.0, 'STAR': 1.0}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

`exp017_lgbm_xgboost_blend` のCV 0.965235を超えるかどうかで採否を判断する。超えない場合も、stacker実装は次のモデル追加時に再利用できる。

## Submission Decision

- Previous best CV: `exp024_oof_stacker_with_lgbm_basic`, 0.966112
- Candidate raw/thresholded CV: 0.966222
- Improvement: +0.000110
- Multipliers are all 1.0, so the improvement is not threshold-driven.
- Decision: API提出しない。改善は本物だが、confident-CV-only ruleの提出ラインとしては小幅。次のbase追加で0.9663+を狙う。

## Risks

- Leakage risk: medium。baseはOOFだが、meta hyperparameterとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。PB狙いではfold別の安定性とLB相関を確認する。
