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
- Best diagnostic CV is `exp032_oof_stacker_with_sdss_prior` at 0.966223, effectively tied with `exp030`.
- Best self-model submitted LB is `exp020_oof_stacker_with_xgb_deep` at 0.96642.
- `exp032` is the best CV candidate, but not submitted because the gain over `exp030` is only +0.000001 and depends on multiplier search.
- External SDSS17 direct label transfer is not supported by `exp015`; use it only as possible auxiliary training data after validation.
- Next fast experiments: fix SDSS prior feature weakness, LightGBM 5-fold/proba rerun after resources allow, pure PyTorch tabular base, or safer LB blend search after submission limit resets.
- Research note: [[Winning_Strategy_Research_2026-06-06]]
