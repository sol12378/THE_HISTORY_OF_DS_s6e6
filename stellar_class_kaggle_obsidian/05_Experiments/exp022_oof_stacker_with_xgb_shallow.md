# exp022_oof_stacker_with_xgb_shallow

## Purpose

`exp020` stackに `exp021_xgboost_gpu_shallow` を追加し、CVがさらに改善するか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp021`
- Best C: 0.01
- Raw CV balanced_accuracy: 0.965856
- Thresholded CV balanced_accuracy: 0.965864
- CV accuracy: 0.959950
- Multipliers: `GALAXY=1.0`, `QSO=1.05`, `STAR=1.0`

## Submission Decision

`exp020` のCV 0.965901を下回ったため、API提出しない。confident-CV-only ruleの基準未達。

## Interpretation

浅めXGBoostは単体では悪くないが、現在のstackには上積みしなかった。相関が高いか、meta-modelが過密になっている可能性がある。

## Risks

- Leakage risk: medium。baseはOOFだが、meta CとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。追加baseが増えるほどOOF hyperparameter overfitのリスクが増える。

## Next Actions

次はXGBoostを増やすより、LightGBM別seed/test_probaの作成を優先する。
