# exp016_xgboost_gpu_basic

## Purpose

GPU XGBoostでLightGBM/CatBoostと異なるOOF予測を作り、自前OOF stackの候補を増やす。

## Hypothesis

`hist` + `device=cuda` のXGBoostにclass-balanced sample weightsを使うことで、単体CVまたはblend/stackの多様性が改善する。

## Result

- CV balanced_accuracy: 0.964020
- CV accuracy: 0.963180
- Feature mode: `color_redshift`
- Fold count: 3
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Leakage Risk

Low。公式train/test特徴量のみを使用し、外部ラベルやtarget encodingは使っていない。

## Overfitting Risk

Medium。3-fold高速診断なのでfold varianceが残る。単体LBではなく、OOF blend/stackで価値を判断する。

## Next Actions

- `exp009_lgbm_color_redshift_small` とOOF probability blendを試す。
- 単体CVが弱くても、誤差相関が低ければstack候補として残す。
