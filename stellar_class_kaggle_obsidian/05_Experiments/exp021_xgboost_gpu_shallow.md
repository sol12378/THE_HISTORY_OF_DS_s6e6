# exp021_xgboost_gpu_shallow

## Purpose

浅め・強正則化のGPU XGBoostを追加し、`exp019`とは異なる誤差を持つstack baseを作る。

## Result

- CV balanced_accuracy: 0.964128
- CV accuracy: 0.961361
- Feature mode: `color_redshift`
- Fold count: 3

## Interpretation

単体では`exp016`より少し強いが、best self-modelには届かない。提出せず、stack baseとして評価する。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold診断で、early stoppingは上限まで到達している。

## Next Actions

`exp022`でstackに追加し、`exp020`を超えるか確認する。
