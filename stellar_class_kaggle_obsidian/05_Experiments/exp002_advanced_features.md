# exp002_advanced_features

---
tags: [experiment]
status: timeout
exp_id: exp002_advanced_features
model: lightgbm_advanced_features
cv:
lb:
---

## Hypothesis

baseline で重要だった `alpha`, `delta`, `redshift`, color 系を明示的に拡張することで、GALAXY/STAR/QSO 境界の局所的な取り違えを減らし、CV balanced_accuracy を 0.964030 から引き上げられる。

## Changes

- 全色指数、flux ratio、redshift x color、magnitude 統計、sky 座標変換を追加する。
- 初回実行は容量が大きく 40分で timeout したため、同じ 5-fold のまま `num_leaves=63`, `learning_rate=0.08`, `n_estimators=900`, `early_stopping_rounds=60`, `max_bin=127` の fast 設定に縮小する。

## Results

| Metric | Value |
|---|---:|
| CV balanced_accuracy | N/A |
| CV accuracy | N/A |
| LB | |

## Risks

- Leakage risk: target-derived encoding は使わない。`alpha`/`delta` の sky pattern は synthetic split に依存する可能性がある。
- Overfitting risk: feature count と model capacity を増やすため、CV/LB gap を exp001 より強く監視する。

## Decision

abandoned_for_now

40分 timeout 後に fast 設定へ縮小したが、さらに20分 timeout。全 advanced features を一度に5-fold LightGBMへ投入する設計は重すぎる。次回は feature group ごとに分割する。
