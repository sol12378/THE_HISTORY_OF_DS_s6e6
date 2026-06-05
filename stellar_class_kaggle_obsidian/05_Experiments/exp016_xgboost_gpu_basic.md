# exp016_xgboost_gpu_basic

## Purpose

GPU XGBoostでLightGBM/CatBoostと異なるOOF予測を作り、自前OOF stack候補を増やす。

## Result

- CV balanced_accuracy: 0.964020
- CV accuracy: 0.963180
- Feature mode: `color_redshift`
- Fold count: 3
- GPU: `device=cuda`, `tree_method=hist`

## Interpretation

単体ではbest self-modelに届かないが、LightGBMとは異なる誤差を持つ可能性がある。blend/stack候補として残す。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold診断なのでfold varianceが残る。

## Next Actions

LightGBM OOF probabilityとblendし、CVが伸びるか確認する。
