# exp003_oof_threshold_search

---
tags: [experiment]
status: completed
exp_id: exp003_oof_threshold_search
source_exp: exp001_baseline
cv: 0.964788
lb:
---

## Hypothesis

`exp001_baseline` は `STAR` recall が相対的に低いため、class multiplier / threshold を調整すれば macro recall が改善し、CV balanced_accuracy が上がる。

## Changes

- Training なし。
- `exp001_baseline/oof.csv` の probability に class multiplier を掛け、OOF balanced_accuracy を grid search で最大化する。

## Results

| Metric | Value |
|---|---:|
| Baseline CV balanced_accuracy | 0.964030 |
| Adjusted CV balanced_accuracy | 0.964788 |
| Delta | +0.000758 |
| Baseline accuracy | 0.965139 |
| Adjusted accuracy | 0.963414 |

Best multipliers: `GALAXY=1.0`, `QSO=1.22`, `STAR=1.30`。

## Risks

- Leakage risk: model training への leakage はないが、OOF label による post-processing selection。
- Overfitting risk: scalar tuning でも同じ OOF で探索・評価するため medium-high。

## Decision

diagnostic_success

balanced_accuracy は改善したが、同じ OOF labels で post-processing を選んでいるため過信しない。次の学習実験では test probabilities を保存し、CV/LB gap を確認する。
