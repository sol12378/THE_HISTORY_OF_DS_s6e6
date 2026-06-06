# exp040_oof_stacker_with_mlp_and_sdss_prior

## Purpose

`exp038` から `exp033` を外し、MLP + `exp031` の寄与を切り分ける。

## Result

- CV balanced_accuracy: 0.966407
- Raw CV balanced_accuracy: 0.966334
- CV accuracy: 0.959925
- Best C: 0.3

## Interpretation

`exp038` よりわずかに低い。`exp031` はMLPと組むと寄与するが、`exp033` も同時に入れた方が最良。

## Decision

no_submit。
