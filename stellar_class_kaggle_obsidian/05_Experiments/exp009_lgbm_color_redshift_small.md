# exp009_lgbm_color_redshift_small

---
tags: [experiment]
status: completed
exp_id: exp009_lgbm_color_redshift_small
model: lightgbm_color_redshift_small
cv: 0.963628
lb:
---

## Hypothesis

feature importance 上位の color/redshift 周辺だけを追加すれば、runtime を抑えつつ `exp001_baseline` の CV balanced_accuracy 0.964030 を上回れる。

## Changes

- 全10色差
- `mag_mean`, `mag_std`, `mag_min`, `mag_max`, `mag_range`
- `redshift_abs`, `redshift_sq`, `redshift_signed_log1p`, `redshift_is_negative`
- selected `redshift_x_color`
- LightGBM 5-fold, `n_estimators=1200`, `learning_rate=0.06`, `early_stopping_rounds=80`

## Results

| Metric | Value |
|---|---:|
| CV balanced_accuracy | 0.963628 |
| CV accuracy | 0.964765 |
| LB | |

## Risks

- Leakage risk: target-derived feature は使わない。redshift/color は正当特徴。
- Overfitting risk: medium。exp002 より絞るが、feature additions と hyperparameters の影響を fold stability で確認する。

## Decision

completed

仮説は外れ。focused color/redshift/magnitude features は `exp001_baseline` の CV 0.964030 を下回った。ただし `test_proba.csv` を得られたため、threshold submission の土台として使う。
