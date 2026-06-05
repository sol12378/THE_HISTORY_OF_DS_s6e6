# exp020_oof_stacker_with_xgb_deep

## Purpose

`exp018` のOOF stackerに `exp019_xgboost_gpu_deep` を追加し、CVをさらに改善できるか検証する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`
- Best C: 0.01
- Raw CV balanced_accuracy: 0.965901
- Thresholded CV balanced_accuracy: 0.965901
- CV accuracy: 0.959896
- Public LB: 0.96642
- Submission ref: 53391536
- Multipliers: `GALAXY=1.0`, `QSO=1.0`, `STAR=1.0`

## Submission Confidence

`exp018` のCV 0.965560から +0.000341。raw CVとthreshold CVが同一で、class multiplier依存ではないため、API提出基準を満たすと判断した。

## Interpretation

CVは明確に改善したが、Public LBは0.96641から0.96642への小幅改善に留まった。PB狙いでは有効な方向だが、今後の提出は同等以上のraw-CV改善が必要。

## Risks

- Leakage risk: medium。base predictionsはOOFだが、meta CをOOF labelで選んでいる。
- Overfitting risk: medium-high。CV改善幅に比べてPublic LB gainが小さいため、fold/seed安定性が必要。

## Next Actions

浅め・強正則化のXGBoostやLightGBM別seedを追加し、stackのraw CVをさらに上げる。
