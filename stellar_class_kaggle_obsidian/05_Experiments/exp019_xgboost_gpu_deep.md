# exp019_xgboost_gpu_deep

## Purpose

GPU XGBoostの深め・長め設定を追加し、OOF stacker用の多様なbase predictionを作る。

## Result

- CV balanced_accuracy: 0.962960
- CV accuracy: 0.965340
- Feature mode: `color_redshift`
- Fold count: 3
- Best iterations: 1668, 1792, 1669

## Interpretation

単体balanced_accuracyは`exp016`より弱い。accuracyは高めなので、class recallのバランスが悪化している。ただし誤差特性が違うため、stack baseとしては検証価値がある。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold診断で、深め設定のためfold依存がある。

## Next Actions

`exp020`でstackに入れて改善するか確認する。単体提出はしない。
