# exp017_lgbm_xgboost_blend

## Purpose

LightGBM (`exp009_lgbm_color_redshift_small`) と GPU XGBoost (`exp016_xgboost_gpu_basic`) の probability をsoft votingし、model diversityでCV balanced_accuracyを改善する。

## Result

- CV balanced_accuracy: 0.965235
- CV accuracy: 0.961905
- Public LB: 0.96597
- Submission ref: 53391073
- Left weight: 0.450
- Right weight: 0.550
- Multipliers: `GALAXY=1.0`, `QSO=1.35`, `STAR=1.35`
- Submission checks: all passed

## Interpretation

XGBoost単体のCVは0.964020だったが、LightGBMとのblendでは自前最高CVと最高LBを更新した。GPU XGBoostはstack/blend候補として有効。

## Risks

- Leakage risk: medium。model trainingへの直接leakageはないが、blend weightsとclass multipliersをOOF labelsで選んでいる。
- Overfitting risk: medium-high。CV/LBは改善したが、PB狙いではstackerやholdout風の検証で過剰なmultiplier依存を減らす必要がある。

## Next Actions

- GPU XGBoostの追加seed/round/depthを探索する。
- OOF stackerを実装し、単純grid multiplierより汎化しやすいblendを試す。
