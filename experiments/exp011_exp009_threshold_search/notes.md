# exp011_exp009_threshold_search

## Purpose

`exp001_baseline` の OOF probabilities に class multiplier を掛け、公式 metric の `balanced_accuracy` が改善するか確認する。

## Hypothesis

baseline は `STAR` recall が相対的に低いため、class prior / threshold を調整すれば macro recall が改善し、CV balanced_accuracy が上がる。

## Result

- Baseline balanced_accuracy: 0.963628
- Adjusted balanced_accuracy: 0.964604
- Delta balanced_accuracy: +0.000976
- Baseline accuracy: 0.964765
- Adjusted accuracy: 0.963078
- Delta accuracy: -0.001687
- Best multipliers: {'GALAXY': 1.0, 'QSO': 1.28, 'STAR': 1.3}

## Risks

- Leakage risk: model training 自体への target leakage はない。ただし OOF label を使った post-processing 最適化なので leaderboard へ過信しない。
- Overfitting risk: scalar 2個だけの tuning だが、同じ OOF で探索と評価をしているため medium-high。

## Decision

Use as a diagnostic. Test-time probabilities are not saved for exp001, so submission generation requires rerunning or future experiments saving test probabilities.
