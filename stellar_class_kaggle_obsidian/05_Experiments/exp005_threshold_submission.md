# exp005_threshold_submission

---
tags: [experiment, submission]
status: planned
exp_id: exp005_threshold_submission
source_exp: exp004_lgbm_basic_proba
cv:
lb:
---

## Hypothesis

`exp003_oof_threshold_search` で CV balanced_accuracy が改善した multiplier を `exp004_lgbm_basic_proba` の test probabilities に適用すれば、Public LB でも `exp001_baseline` より改善する可能性がある。

## Changes

- Multipliers: `GALAXY=1.0`, `QSO=1.22`, `STAR=1.30`
- `exp004_lgbm_basic_proba/test_proba.csv` から `submission.csv` を作成する。

## Results

| Metric | Value |
|---|---:|
| Source CV balanced_accuracy | |
| Threshold diagnostic CV | 0.964788 |
| LB | |

## Risks

- Leakage risk: multiplier は OOF label で選んでいるため post-processing selection risk がある。
- Overfitting risk: medium-high。LB が改善しない場合は threshold を採用しない。

## Decision

planned
