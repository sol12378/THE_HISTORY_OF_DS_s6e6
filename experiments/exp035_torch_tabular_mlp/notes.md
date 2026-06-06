# exp035_torch_tabular_mlp

## Purpose

純粋PyTorch (`nn.Module` + `DataLoader` + `AdamW`) のtabular MLP baseを作り、tree model stackへNN系の誤差多様性を追加できるか確認する。

## Result

- CV balanced_accuracy: 0.950964
- CV accuracy: 0.941623
- Device: `cuda`
- n_splits: 3
- epochs: 18
- hidden: 256
- dropout: 0.08
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

単体提出候補ではなく、次段のOOF stackで採否を判断する。現行best `exp032` CV 0.966223 を明確に超えない場合はAPI提出しない。

## Risks

- Leakage risk: low。official train/test featuresのみ。
- Overfitting risk: medium。NNはstochasticでfold varianceがあるため、stack改善幅を厳しく見る。
