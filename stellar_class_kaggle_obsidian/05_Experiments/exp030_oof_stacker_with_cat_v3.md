# exp030_oof_stacker_with_cat_v3

## Purpose

`exp024` stackに `exp029_catboost_v3_style` を追加し、CatBoost native categorical/original low-weight baseがCVを改善するか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp029`
- Raw/thresholded CV balanced_accuracy: 0.966222
- CV accuracy: 0.960381
- Multipliers: `GALAXY=1.0`, `QSO=1.0`, `STAR=1.0`

## Submission Decision

`exp024` CV 0.966112から +0.000110。改善はraw CVで、multiplier依存ではない。ただしAPI提出基準としては小幅のため提出しない。次のbase追加で0.9663以上を狙う。

## Interpretation

CatBoost v3-style baseはstackに有効。次はこの方向を強化するか、SDSS prior/prototype featuresをGBDTに追加するのがよい。

## Risks

- Leakage risk: medium。base predictionsはOOFだが、SDSS17 original labelsを低weightで使う。
- Overfitting risk: medium-high。CV改善幅は小さいため、追加検証が必要。
