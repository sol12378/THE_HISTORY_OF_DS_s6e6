# exp032_oof_stacker_with_sdss_prior

## Purpose

`exp031_lgbm_sdss_prior_fast` を `exp030_oof_stacker_with_cat_v3` のbase群へ追加し、弱いが異質なSDSS prior baseがstack CVを押し上げるか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp029`, `exp031`
- Raw CV balanced_accuracy: 0.966160
- Thresholded CV balanced_accuracy: 0.966223
- CV accuracy: 0.959624
- Best C: 0.3
- Multipliers: GALAXY 1.0, QSO 1.1, STAR 1.05

## Interpretation

`exp030` の 0.966221799 から `exp032` の 0.966222902 へ、差分は約 +0.000001。これは実質同値で、しかも最良値はclass multiplier search込みである。API提出に必要な「自信のあるCV改善」には届かない。

## Decision

no_submit。`exp032` は記録上のbest diagnostic CVだが、提出候補としては `exp030` と同格以下に扱う。

## Risks

- Leakage risk: medium。baseはOOFだが、meta hyperparameterとclass multiplierはOOF labelsで選択している。
- Overfitting risk: medium-high。改善幅が極小のため、LB/PBに効く根拠としては弱い。
