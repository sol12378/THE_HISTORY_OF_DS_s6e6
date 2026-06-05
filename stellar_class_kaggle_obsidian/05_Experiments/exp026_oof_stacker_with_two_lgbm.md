# exp026_oof_stacker_with_two_lgbm

## Purpose

`exp024` stackに `exp025_lgbm_color_regularized` も追加し、2つのLightGBM追加baseがさらにCVを改善するか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp025`
- Raw/thresholded CV balanced_accuracy: 0.966092
- CV accuracy: 0.960140
- Multipliers: `GALAXY=1.0`, `QSO=1.0`, `STAR=1.0`

## Submission Decision

`exp024` CV 0.966112を下回ったため、API提出しない。なおKaggle submissionは日次上限らしき状態のため、このターンでは再試行しない。

## Interpretation

`exp025`の追加はstackに上積みしなかった。弱いLightGBM baseを増やすより、より強いLightGBM 5-fold/probaまたは別モデルが必要。

## Risks

- Leakage risk: medium。OOF baseを使っているが、meta CはOOF labelで選択している。
- Overfitting risk: medium-high。base数が増えており、OOF hyperparameter overfitに注意。
