# exp022_oof_stacker_with_xgb_shallow

## Purpose

利用可能な自前モデルOOF probabilityを使い、LogisticRegressionのmeta-modelでOOF stackを試す。

## Result

- Base experiments: ['exp009_lgbm_color_redshift_small', 'exp010_catboost_gpu_basic', 'exp016_xgboost_gpu_basic', 'exp019_xgboost_gpu_deep', 'exp021_xgboost_gpu_shallow']
- Best C: 0.01
- Raw CV balanced_accuracy: 0.965856
- Thresholded CV balanced_accuracy: 0.965864
- CV accuracy: 0.959950
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.05, 'STAR': 1.0}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

`exp017_lgbm_xgboost_blend` のCV 0.965235を超えるかどうかで採否を判断する。超えない場合も、stacker実装は次のモデル追加時に再利用できる。

## Submission Decision

- Current self-model best: `exp020_oof_stacker_with_xgb_deep`, CV 0.965901, LB 0.96642
- Candidate CV: 0.965864
- Difference vs current best CV: -0.000037
- Decision: API提出しない。confident-CV-only ruleの基準未達。

## Risks

- Leakage risk: medium。baseはOOFだが、meta hyperparameterとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。PB狙いではfold別の安定性とLB相関を確認する。
