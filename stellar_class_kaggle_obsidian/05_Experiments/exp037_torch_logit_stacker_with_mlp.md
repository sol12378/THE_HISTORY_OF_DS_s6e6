# exp037_torch_logit_stacker_with_mlp

## Purpose

`exp036` の改善がstacker実装依存ではないか確認するため、同じbase群を純粋PyTorch logit stackerで再評価する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp029`, `exp035`
- Raw CV balanced_accuracy: 0.966135
- Thresholded CV balanced_accuracy: 0.966175
- CV accuracy: 0.959496
- Best C: 1.0
- Device: cuda

## Interpretation

`exp030/032` を下回るため、`exp036` の改善は提出に十分な確証とは見なさない。純粋PyTorch MLP自体は多様性を持つが、より強いNN baseまたはcalibrationが必要。

## Decision

no_submit。
