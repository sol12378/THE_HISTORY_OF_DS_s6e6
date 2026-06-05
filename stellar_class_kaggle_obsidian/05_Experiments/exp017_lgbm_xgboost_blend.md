# exp017_lgbm_xgboost_blend

## Purpose

`exp009_lgbm_color_redshift_small` と `exp016_xgboost_gpu_basic` のOOF/test probabilityをblendし、自前モデルのCVとLBを更新する。

## Result

- CV balanced_accuracy: 0.965235
- CV accuracy: 0.961905
- Public LB: 0.96597
- Submission ref: 53391073
- Best weights: LightGBM 0.45, XGBoost 0.55
- Multipliers: `GALAXY=1.0`, `QSO=1.35`, `STAR=1.35`

## Interpretation

XGBoost GPUは単体CVよりblend価値が高い。自前モデルとして現時点のbest CVとbest LBを更新した。

## Risks

- Leakage risk: medium。OOF labelでblend weightとclass multiplierを選んでいる。
- Overfitting risk: medium-high。threshold/multiplier探索の影響があるため、PB向けにはstacker/heldout検証が必要。

## Next Actions

より細かいOOF stackerを作り、単純grid multiplier依存を減らす。GPU XGBoostのround数・depth・正則化も追加探索する。
