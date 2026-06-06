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
- Best diagnostic CV is `exp038_oof_stacker_with_mlp_and_sdss` at 0.966420.
- Best self-model submitted LB is `exp047_exp038_star105_candidate` at 0.96679.
- `exp047` is the best self-model CV/LB candidate; `exp042`/`exp043` confirm the base set across alternate meta split seeds.
- External SDSS17 direct label transfer is not supported by `exp015`; use it only as possible auxiliary training data after validation.
- Next fast experiments: improve pure PyTorch MLP calibration/architecture, stronger XGBoost/CatBoost variants, or create a calibrated `exp038` submission candidate after submission limit/confidence review.
- Research note: [[Winning_Strategy_Research_2026-06-06]]
