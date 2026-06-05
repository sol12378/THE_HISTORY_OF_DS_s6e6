# exp024_oof_stacker_with_lgbm_basic

## Purpose

利用可能な自前モデルOOF probabilityを使い、LogisticRegressionのmeta-modelでOOF stackを試す。

## Result

- Base experiments: ['exp009_lgbm_color_redshift_small', 'exp010_catboost_gpu_basic', 'exp016_xgboost_gpu_basic', 'exp019_xgboost_gpu_deep', 'exp023_lgbm_basic_fast']
- Best C: 10.0
- Raw CV balanced_accuracy: 0.966090
- Thresholded CV balanced_accuracy: 0.966112
- CV accuracy: 0.959335
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.0, 'STAR': 1.05}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

`exp017_lgbm_xgboost_blend` のCV 0.965235を超えるかどうかで採否を判断する。超えない場合も、stacker実装は次のモデル追加時に再利用できる。

## Submission Confidence

- Current self-model best before this experiment: `exp020_oof_stacker_with_xgb_deep`, CV 0.965901, LB 0.96642
- Candidate raw CV: 0.966090
- Candidate thresholded CV: 0.966112
- Improvement over current best CV: +0.000211
- Threshold gain is small (+0.000022), so the improvement mostly comes from the stack base addition rather than multiplier tuning.
- Decision: API submission is justified, but this is a borderline-confident candidate. Future submissions should prefer larger raw-CV gains.

## API Submission Attempt

Kaggle API submission failed with `400 Bad Request` after upload, and submission history did not register `exp024`. There are already 10 submissions on 2026-06-05, so the likely cause is the daily submission limit. Do not retry until the limit resets.

## Risks

- Leakage risk: medium。baseはOOFだが、meta hyperparameterとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。PB狙いではfold別の安定性とLB相関を確認する。
