# exp039_torch_logit_stacker_with_mlp_and_sdss

## Purpose

`exp038` のbase setを純粋PyTorch logit stackerで確認し、改善がsklearn stacker固有ではないかを検証する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp029`, `exp031`, `exp033`, `exp035`
- Raw/thresholded CV balanced_accuracy: 0.966322
- CV accuracy: 0.960242
- Best C: 10.0
- Multipliers: all 1.0

## Interpretation

`exp030/032` は超えたが、`exp038` ほどの改善ではない。`exp038` の方向性は支持されるが、API提出に十分な確信までは届かない。

## Decision

no_submit。
