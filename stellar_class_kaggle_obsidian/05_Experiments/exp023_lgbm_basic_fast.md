# exp023_lgbm_basic_fast

## Purpose

`exp001`に近いbasic featureのLightGBMを3-foldで再作成し、OOF/test_probaをstack用に保存する。

## Result

- CV balanced_accuracy: 0.963713
- CV accuracy: 0.964162
- Feature mode: `basic`
- Fold count: 3

## Interpretation

単体ではbestに届かないが、`exp009`とはfeature setとfold seedが違うためstack diversityとして有効か確認する。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold高速診断。

## Next Actions

`exp024`でstackに追加する。
