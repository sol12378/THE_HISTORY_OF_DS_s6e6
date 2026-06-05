# exp002_advanced_features

## Purpose

全色指数、flux ratio、redshift 交互作用、magnitude 統計、sky 座標変換を追加し、LightGBM baseline の CV balanced_accuracy を改善できるか検証する。

## Result

- Status: timeout
- CV balanced_accuracy: not available
- OOF/submission: not produced

## PDCA

- Plan: baseline で重要だった `alpha`, `delta`, `redshift`, color 系を拡張する。
- Do: high-capacity 設定を実行したが 40分 timeout。fast 設定に縮小したが 20分 timeout。
- Check: feature count 59 + 5-fold LightGBM は現在の運用方針（約20分以内）に対して重すぎる。
- Act: advanced feature は全投入ではなく、feature group ごとの小さな実験に分割する。先に OOF post-processing など軽い改善を検証する。

## Risks

- Leakage risk: target-derived feature は使っていないが、sky coordinate の強い寄与は synthetic split 依存の可能性がある。
- Overfitting risk: feature 数と model capacity が増えすぎるため高い。完走前に止めたのでスコア評価はしない。

## Decision

abandoned_for_now
