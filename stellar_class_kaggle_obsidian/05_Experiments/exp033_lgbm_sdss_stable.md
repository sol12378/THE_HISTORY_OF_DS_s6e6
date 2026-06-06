# exp033_lgbm_sdss_stable

## Purpose

`exp031` で弱かった広いbin/count型SDSS特徴を削り、coarse prior と prototype距離だけをLightGBMに追加する。

## Result

- CV balanced_accuracy: 0.963558
- CV accuracy: 0.964081
- n_splits: 3
- Feature count: 52
- Categorical feature count: 3
- Submission checks: all passed

## Interpretation

`exp031` の 0.959431 から大きく回復したため、SDSS特徴は「広く入れる」より「粗く安定させる」方がよい。ただし既存の強いbaseには届かず、単体提出候補ではない。

## Risks

- Leakage risk: medium-low。SDSS17 labelsは粗いaggregate prior/prototypeのみに使用。
- Overfitting risk: medium。外部特徴のCV過適合は残る。
