# exp010_catboost_gpu_basic

---
tags: [experiment, gpu]
status: completed
exp_id: exp010_catboost_gpu_basic
model: catboost_gpu_basic
cv: 0.951213
lb:
---

## Hypothesis

CatBoost GPU は LightGBM と異なる model family であり、categorical features を自然に扱えるため、単体または ensemble 素材として CV balanced_accuracy を改善できる可能性がある。

## Changes

- Basic features + `spectral_type`, `galaxy_population`
- CatBoostClassifier GPU
- 3-fold diagnostic
- `iterations=600`, `depth=8`, `learning_rate=0.08`

## Results

| Metric | Value |
|---|---:|
| CV balanced_accuracy | 0.951213 |
| CV accuracy | 0.965011 |
| LB | |

## Risks

- Leakage risk: target-derived feature は使わない。
- Overfitting risk: 3-fold diagnostic なので 5-fold より不安定。
- Runtime/GPU risk: RTX 2080 SUPER 8GB で VRAM が不足すると失敗する可能性がある。

## Decision

completed

GPU CatBoost は約1分で3-fold diagnostic が完走したため、GPU 学習環境は有効。ただし balanced_accuracy は LightGBM baseline より大きく低く、この設定は単体採用しない。ensemble weight search でも CatBoost weight は 0 になった。
