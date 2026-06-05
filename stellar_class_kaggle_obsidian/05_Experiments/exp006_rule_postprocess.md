# exp006_rule_postprocess

---
tags: [experiment, submission]
status: submitted
exp_id: exp006_rule_postprocess
source_exp: exp001_baseline
cv: 0.964108
lb: 0.96463
---

## Hypothesis

`redshift` と categorical conditions を使って `exp001_baseline` の hard predictions を補正すれば、特に `STAR` / `GALAXY` 境界の macro recall が改善し、CV balanced_accuracy が上がる。

## Changes

- Training なし。
- OOF hard prediction と raw train features から simple rules を greedy search する。
- 同じ rules を `exp001_baseline/submission.csv` と raw test features に適用して submission を作成する。

## Results

| Metric | Value |
|---|---:|
| Baseline CV balanced_accuracy | 0.964030 |
| Adjusted CV balanced_accuracy | 0.964108 |
| Delta | +0.000078 |
| LB | 0.96463 |

Rules:

- `pred=GALAXY` and `redshift > 1.6` -> `QSO`
- `pred=STAR` and `spectral_type=M` and `redshift > 0.28` -> `GALAXY`
- `pred=GALAXY` and `spectral_type=O/B` and `redshift > 1.525` -> `QSO`

## Risks

- Leakage risk: target-derived training feature はないが、rules は OOF labels で選ぶ。
- Overfitting risk: medium-high。LB 確認が必要。

## Decision

submitted

LB は `exp001_baseline` の 0.96462 から 0.96463 に微増。改善幅は小さいが、CV と LB の方向は一致した。
