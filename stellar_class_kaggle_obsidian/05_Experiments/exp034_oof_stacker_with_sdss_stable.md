# exp034_oof_stacker_with_sdss_stable

## Purpose

`exp033_lgbm_sdss_stable` を `exp030` 相当のbase群へ追加し、安定SDSS baseがstack diversityとして効くか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp029`, `exp033`
- Raw CV balanced_accuracy: 0.966130
- Thresholded CV balanced_accuracy: 0.966206
- CV accuracy: 0.959993
- Best C: 10.0
- Multipliers: GALAXY 1.0, QSO 1.0, STAR 1.05

## Interpretation

`exp030` 0.966222、`exp032` 0.966223 を下回る。`exp033` は `exp031` より健全だが、stack diversityとしては採用しない。

## Decision

no_submit。API提出基準を満たさない。

## Next

SDSS aggregate特徴はいったん優先度を下げ、次は純粋PyTorch tabular base、またはXGBoost/CatBoostの強化に進む。
