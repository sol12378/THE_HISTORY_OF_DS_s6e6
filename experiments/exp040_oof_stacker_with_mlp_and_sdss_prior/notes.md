# exp040_oof_stacker_with_mlp_and_sdss_prior

## Purpose

利用可能な自前モデルOOF probabilityを使い、LogisticRegressionのmeta-modelでOOF stackを試す。

## Result

- Base experiments: ['exp009_lgbm_color_redshift_small', 'exp010_catboost_gpu_basic', 'exp016_xgboost_gpu_basic', 'exp019_xgboost_gpu_deep', 'exp023_lgbm_basic_fast', 'exp029_catboost_v3_style', 'exp031_lgbm_sdss_prior_fast', 'exp035_torch_tabular_mlp']
- Best C: 0.3
- Raw CV balanced_accuracy: 0.966334
- Thresholded CV balanced_accuracy: 0.966407
- CV accuracy: 0.959925
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.1, 'STAR': 1.05}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

`exp017_lgbm_xgboost_blend` のCV 0.965235を超えるかどうかで採否を判断する。超えない場合も、stacker実装は次のモデル追加時に再利用できる。

## Risks

- Leakage risk: medium。baseはOOFだが、meta hyperparameterとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。PB狙いではfold別の安定性とLB相関を確認する。
