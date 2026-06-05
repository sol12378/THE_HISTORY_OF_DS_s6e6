# exp013_catboost_threshold_search

---
tags: [experiment, gpu]
status: diagnostic_success
exp_id: exp013_catboost_threshold_search
source_exp: exp010_catboost_gpu_basic
cv: 0.954779
lb:
---

## Result

CatBoost GPU OOF に class multipliers を適用すると CV balanced_accuracy は 0.951213 から 0.954779 へ改善したが、LightGBM 系より大きく低い。

## Decision

Do not use as standalone. Keep only as GPU feasibility and model-diversity diagnostic.
