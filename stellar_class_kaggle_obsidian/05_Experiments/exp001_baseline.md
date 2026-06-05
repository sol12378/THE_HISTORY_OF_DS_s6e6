# exp001_baseline

---
tags: [experiment]
status: planned
exp_id: exp001_baseline
model: lightgbm_baseline
cv: 0.964030
lb: 0.96462
---

## Hypothesis

公式 metric の `balanced_accuracy` に合わせて、LightGBM + 色指数6個で再現可能な CV/OOF/submission pipeline を確立する。

## Changes

- Kaggle official metric は `balanced_accuracy`。
- 5-fold `StratifiedKFold(random_state=42)`。
- LightGBM multiclass + `class_weight="balanced"`。
- Features: raw tabular features, `spectral_type`, `galaxy_population`, and `u_g`, `g_r`, `r_i`, `i_z`, `u_r`, `g_i`。
- Saved `result.json`, `oof.csv`, `submission.csv`, `feature_importance.csv`, `notes.md` under `experiments/exp001_baseline/`。

## Results

| Metric | Value |
|---|---:|
| CV balanced_accuracy | 0.964030 |
| CV accuracy | 0.965139 |
| Public LB | 0.96462 |
| LB - CV gap | +0.000590 |

## Decision

submitted

CV/LB gap が小さく、初回 baseline として validation は機能している。次は ID-like columns の寄与と leakage/overfit リスクを確認する。

## Risks

- Leakage risk: ID-like columns は初回では残した。target-derived encoding は使っていないが、synthetic data の split に強く効く可能性はある。
- Overfitting risk: early stopping ありだが LightGBM 2000 estimators。次回以降も CV/LB gap を監視する。
