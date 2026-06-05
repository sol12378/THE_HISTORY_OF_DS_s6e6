# exp004_lgbm_basic_proba

## Purpose

`exp001_baseline` 相当の LightGBM baseline を再実行し、今後の thresholding / ensemble 用に `test_proba.csv` を保存する。

## Result

- Status: timeout
- CV balanced_accuracy: not available
- `test_proba.csv`: not produced

## PDCA

- Plan: baseline を再実行して probability artifact を作る。
- Do: 20分 timeout。
- Check: 同時に別 workspace の Python/Kaggle processes が走っており、CPU 競合が大きい状態だった。
- Act: 再学習を一旦止め、既存 `exp001` hard submission に適用できる rule-based postprocess へ切り替えた。

## Decision

timeout
