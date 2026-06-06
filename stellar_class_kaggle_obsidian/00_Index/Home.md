# Predicting Stellar Class Home

## Competition

- [[Competition_Overview]]
- [[Data_Structure]]
- [[Evaluation]]
- [[Leakage_and_Risks]]

## Execution

- [[EXP_Index]]
- [[Decision_Log]]
- [[Hypothesis_Backlog]]
- [[LB_Tracking]]

## Current Focus

- Best submitted LB is `exp008_nina_weighted_vote` at 0.97074.
- Best diagnostic CV is `exp036_oof_stacker_with_torch_mlp` at 0.966349.
- Best self-model submitted LB is `exp020_oof_stacker_with_xgb_deep` at 0.96642.
- `exp036` is the best CV candidate, but not submitted because the gain was not reproduced by `exp037_torch_logit_stacker_with_mlp`.
- External SDSS17 direct label transfer is not supported by `exp015`; use it only as possible auxiliary training data after validation.
- Next fast experiments: improve pure PyTorch MLP calibration/architecture, stronger XGBoost/CatBoost variants, LightGBM 5-fold/proba rerun after resources allow, or safer LB blend search after submission limit resets.
- Research note: [[Winning_Strategy_Research_2026-06-06]]
