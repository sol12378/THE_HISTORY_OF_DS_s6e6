# exp025_lgbm_color_regularized

## Purpose

`color_redshift` featuresで強正則化・stochasticなLightGBMを作り、stack diversityを追加する。

## Result

- CV balanced_accuracy: 0.962930
- CV accuracy: 0.960073
- Feature mode: `color_redshift`
- Fold count: 3

## Interpretation

単体では弱い。`exp026`でstackに追加して有効性を判断する。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold診断。
