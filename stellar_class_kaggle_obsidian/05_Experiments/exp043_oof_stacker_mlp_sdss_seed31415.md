# exp043_oof_stacker_mlp_sdss_seed31415

## Purpose

`exp038` のbase setをmeta fold seed 31415で再評価し、別splitでの安定性を確認する。

## Result

- Base set: `exp038` と同一
- n_folds: 5
- fold_seed: 31415
- Raw CV balanced_accuracy: 0.966327
- Thresholded CV balanced_accuracy: 0.966354
- CV accuracy: 0.960033
- Best C: 0.01

## Interpretation

`exp038/042` より低いが、`exp030/032` は上回る。base set改善は再現している。

## Decision

no_submit。次はseed-averaged stack candidateを作る。
