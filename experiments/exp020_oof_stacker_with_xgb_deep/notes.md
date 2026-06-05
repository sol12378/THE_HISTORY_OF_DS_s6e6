# exp020_oof_stacker_with_xgb_deep

## Purpose

利用可能な自前モデルOOF probabilityを使い、LogisticRegressionのmeta-modelでOOF stackを試す。

## Result

- Base experiments: ['exp009_lgbm_color_redshift_small', 'exp010_catboost_gpu_basic', 'exp016_xgboost_gpu_basic', 'exp019_xgboost_gpu_deep']
- Best C: 0.01
- Raw CV balanced_accuracy: 0.965901
- Thresholded CV balanced_accuracy: 0.965901
- CV accuracy: 0.959896
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.0, 'STAR': 1.0}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

`exp017_lgbm_xgboost_blend` のCV 0.965235を超えるかどうかで採否を判断する。超えない場合も、stacker実装は次のモデル追加時に再利用できる。

## Submission Confidence

- Current self-model best before this experiment: `exp018_oof_logistic_stacker`, CV 0.965560, LB 0.96641
- Candidate CV: 0.965901
- Public LB: 0.96642
- Submission ref: 53391536
- Improvement over current self-model best CV: +0.000341
- Raw CV and thresholded CV are identical, so the gain is not caused by class multiplier overfitting.
- Submission distribution is close to recent self-model submissions.
- Decision: API submission is justified under the confident-CV-only rule.

## LB Check

Public LB improved only slightly from 0.96641 to 0.96642 despite the CV gain. Continue using CV as the trusted development metric, but require similarly strong raw-CV improvement for future API submissions.

## Risks

- Leakage risk: medium。baseはOOFだが、meta hyperparameterとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。PB狙いではfold別の安定性とLB相関を確認する。
