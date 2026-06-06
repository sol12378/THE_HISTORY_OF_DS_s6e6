# exp035_torch_tabular_mlp

## Purpose

純粋PyTorch (`nn.Module` + `DataLoader` + `AdamW`) のtabular MLP baseを作り、tree model stackへNN系の誤差多様性を追加できるか確認する。

## Result

- CV balanced_accuracy: 0.950964
- CV accuracy: 0.941623
- Device: cuda
- n_splits: 3
- epochs: 18
- hidden: 256
- dropout: 0.08
- Submission checks: all passed

## Interpretation

単体CVは弱い。初回実行では全NaN列により学習が壊れたため、全NaN feature drop と `nan_to_num` を追加して再実行した。単体提出候補ではないが、NN由来の異質な誤差としてstack診断へ回す。

## Risks

- Leakage risk: low。official train/test featuresのみ。
- Overfitting risk: medium。短epochのstochastic NNであり、fold varianceが残る。
