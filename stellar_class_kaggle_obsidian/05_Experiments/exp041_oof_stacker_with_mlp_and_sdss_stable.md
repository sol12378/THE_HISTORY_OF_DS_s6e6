# exp041_oof_stacker_with_mlp_and_sdss_stable

## Purpose

`exp038` から `exp031` を外し、MLP + `exp033` の寄与を切り分ける。

## Result

- CV balanced_accuracy: 0.966379
- Raw CV balanced_accuracy: 0.966379
- CV accuracy: 0.960599
- Best C: 0.3
- Multipliers: all 1.0

## Interpretation

raw CVは強いが、最終CVは `exp038/040` 未満。`exp033` は安定だが、`exp031` ほどstack上限を押し上げない。

## Decision

no_submit。
