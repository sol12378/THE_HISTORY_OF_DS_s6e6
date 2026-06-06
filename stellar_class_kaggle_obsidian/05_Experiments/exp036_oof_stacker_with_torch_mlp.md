# exp036_oof_stacker_with_torch_mlp

## Purpose

`exp035_torch_tabular_mlp` を `exp030` 相当のtree/CatBoost base群へ追加し、弱いNN baseでもstack diversityとして効くか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp029`, `exp035`
- Raw CV balanced_accuracy: 0.966279
- Thresholded CV balanced_accuracy: 0.966349
- CV accuracy: 0.959440
- Best C: 1.0
- Multipliers: GALAXY 1.0, QSO 1.1, STAR 1.1

## Interpretation

記録上のnew diagnostic best。ただし改善はsklearn logistic stackerとclass multiplier searchに依存する。confirmationとして実行した `exp037` が 0.966175 と低く、提出に足る自信はまだない。

## Decision

no_submit。API提出は、追加の確認またはより強いNN baseで改善が再現した後にする。
