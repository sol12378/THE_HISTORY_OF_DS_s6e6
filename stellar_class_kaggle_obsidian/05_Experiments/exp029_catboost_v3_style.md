# exp029_catboost_v3_style

## Purpose

CatBoost v3風に、量子化bin・丸めbin・カテゴリinteraction・SDSS17 original低weight appendを使うCatBoost GPU baseを作る。

## Result

- CV balanced_accuracy: 0.964213
- CV accuracy: 0.958396
- Fold count: 3
- Iterations: 1600
- Original weight: 0.06
- Feature count: 103
- Categorical feature count: 47

## Interpretation

過去の`exp010_catboost_gpu_basic` CV 0.951213より大幅に改善。単体bestには届かないが、CatBoostらしい誤差を持つstack baseとして有望。

## Risks

- Leakage risk: medium。SDSS17 original labelを低weightでfold trainにappendするが、validation rowsは学習に入れない。direct label transferは使っていない。
- Overfitting risk: medium-high。カテゴリviewと外部データを使うため、stack改善とLB相関で判断する。

## Next Actions

`exp030`でstackへ追加してCV改善を確認する。
