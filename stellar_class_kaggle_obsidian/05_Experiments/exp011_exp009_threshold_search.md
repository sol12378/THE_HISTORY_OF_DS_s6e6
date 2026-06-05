# exp011_exp009_threshold_search

---
tags: [experiment]
status: diagnostic_success
exp_id: exp011_exp009_threshold_search
source_exp: exp009_lgbm_color_redshift_small
cv: 0.964604
lb:
---

## Hypothesis

`exp009` は raw argmax では balanced_accuracy が低いが、class multipliers を OOF で調整すれば official metric に合わせて改善できる。

## Results

| Metric | Value |
|---|---:|
| Baseline CV balanced_accuracy | 0.963628 |
| Adjusted CV balanced_accuracy | 0.964604 |
| Delta | +0.000976 |

Best multipliers: `GALAXY=1.0`, `QSO=1.28`, `STAR=1.30`。

## Risks

- Leakage risk: model training への leakage はないが OOF labels による post-processing selection。
- Overfitting risk: medium-high。

## Decision

diagnostic_success
