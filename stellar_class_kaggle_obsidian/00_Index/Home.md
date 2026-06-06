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
- Best diagnostic CV is `exp030_oof_stacker_with_cat_v3` at 0.966222.
- Best self-model submitted LB is `exp020_oof_stacker_with_xgb_deep` at 0.96642.
- `exp030` is the best CV candidate, but not submitted because the gain over `exp024` is only +0.000110.
- External SDSS17 direct label transfer is not supported by `exp015`; use it only as possible auxiliary training data after validation.
- Next fast experiments: LightGBM 5-fold/proba rerun after resources allow, external SDSS17 auxiliary-training test, or safer LB blend search after submission limit resets.
- Research note: [[Winning_Strategy_Research_2026-06-06]]
