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
- Best diagnostic CV is `exp020_oof_stacker_with_xgb_deep` at 0.965901.
- Best self-model submitted LB is `exp020_oof_stacker_with_xgb_deep` at 0.96642.
- External SDSS17 direct label transfer is not supported by `exp015`; use it only as possible auxiliary training data after validation.
- Next fast experiments: shallow/regularized GPU XGBoost diversity, LightGBM test_proba rerun, external SDSS17 auxiliary-training test, or safer LB blend search.
