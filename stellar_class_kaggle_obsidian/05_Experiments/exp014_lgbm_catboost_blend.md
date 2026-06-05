# exp014_lgbm_catboost_blend

---
tags: [experiment, ensemble]
status: submitted
exp_id: exp014_lgbm_catboost_blend
cv: 0.964652
lb: 0.96545
---

## Hypothesis

LightGBM と CatBoost GPU は誤差構造が異なるため、OOF probability の soft voting / class multiplier で CV balanced_accuracy が単体より改善する。

## Changes

- Blend `exp009_lgbm_color_redshift_small` and `exp010_catboost_gpu_basic`
- Search blend weight and class multipliers on OOF
- Create submission from blended test probabilities

## Results

| Metric | Value |
|---|---:|
| CV balanced_accuracy | 0.964652 |
| CV accuracy | 0.962783 |
| LB | 0.96545 |

## Risks

- Leakage risk: OOF label で blend weight / multipliers を選ぶため medium。
- Overfitting risk: medium-high。LB 確認が必要。

## Decision

submitted

Search result は `exp009` weight 1.0 / `exp010` CatBoost weight 0.0。CatBoost は今回の blend に寄与しなかったが、class multipliers `QSO=1.30`, `STAR=1.35` で `exp012` より CV/LB ともに改善。Kaggle API submission ref: `53390199`。
