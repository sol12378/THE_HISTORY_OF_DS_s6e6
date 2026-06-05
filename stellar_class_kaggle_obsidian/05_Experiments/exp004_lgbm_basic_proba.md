# exp004_lgbm_basic_proba

---
tags: [experiment]
status: timeout
exp_id: exp004_lgbm_basic_proba
model: lightgbm_basic_proba
cv:
lb:
---

## Hypothesis

同一CV設定と basic features で `exp001_baseline` と同等の CV を再現し、保存した `test_proba.csv` により `exp003` の class multiplier を LB で検証できる。

## Changes

- `exp001_baseline` 相当の LightGBM を再実行する。
- `test_proba.csv` を保存する。
- threshold / ensemble 用の基礎 artifact を作る。

## Results

| Metric | Value |
|---|---:|
| CV balanced_accuracy | N/A |
| CV accuracy | N/A |
| LB | |

## Risks

- Leakage risk: `exp001` と同じく target-derived encoding はなし。
- Overfitting risk: `exp003` threshold を適用する場合は OOF tuning の過適合を LB で確認する必要がある。

## Decision

timeout

20分 timeout。別 workspace の Python/Kaggle processes による CPU 競合が大きい状態だった。成果物は未生成。既存 `exp001` hard submission に適用可能な `exp006_rule_postprocess` へ切り替えた。
