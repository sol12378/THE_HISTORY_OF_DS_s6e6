# exp024_oof_stacker_with_lgbm_basic

## Purpose

`exp020` stackに `exp023_lgbm_basic_fast` を追加し、LightGBM basic baseがCVを改善するか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`
- Raw CV balanced_accuracy: 0.966090
- Thresholded CV balanced_accuracy: 0.966112
- CV accuracy: 0.959335
- Multipliers: `GALAXY=1.0`, `QSO=1.0`, `STAR=1.05`
- API submission: failed with `400 Bad Request`

## Submission Confidence

`exp020` CV 0.965901から +0.000211。threshold gainは+0.000022と小さく、主な改善はbase追加由来と判断した。

## API Failure

Kaggle APIはupload後のCreateSubmissionで `400 Bad Request`。submission historyには登録されていない。2026-06-05の提出数が10件に達しているため、日次提出上限が原因の可能性が高い。再試行しない。

## Interpretation

現時点のbest CV。ただしLB未確認なので、提出上限リセット後に再評価候補。

## Risks

- Leakage risk: medium。OOF baseを使っているが、meta CとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。改善幅が小さいため、次回提出時はraw CVとfold安定性を再確認する。
