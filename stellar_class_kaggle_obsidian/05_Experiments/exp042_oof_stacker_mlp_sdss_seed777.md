# exp042_oof_stacker_mlp_sdss_seed777

## Purpose

`exp038` のbase setをmeta fold seed 777で再評価し、seed 2026 固有の上振れではないか確認する。

## Result

- Base set: `exp038` と同一
- n_folds: 5
- fold_seed: 777
- Raw/thresholded CV balanced_accuracy: 0.966398
- CV accuracy: 0.960552
- Best C: 0.01
- Multipliers: all 1.0

## Interpretation

`exp038` より少し低いが、threshold-freeで強い。`exp038` のbase setは別splitでも有効。

## Decision

no_submit。seed単体submissionではなく、`exp038/042/043` のseed-averaged candidateを次に検討する。
