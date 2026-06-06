# Experiment Index

| Exp | Date | Status | CV | LB | Notes |
|---|---|---|---:|---:|---|
| [[exp001_baseline]] | 2026-06-05 | submitted | 0.964030 | 0.96462 | LightGBM + color indices baseline |
| [[exp002_advanced_features]] | 2026-06-05 | timeout | | | Advanced features too slow as one 5-fold experiment |
| [[exp003_oof_threshold_search]] | 2026-06-05 | diagnostic_success | 0.964788 | | OOF class multiplier search; no submission yet |
| [[exp004_lgbm_basic_proba]] | 2026-06-05 | timeout | | | Baseline rerun for test_proba timed out |
| [[exp006_rule_postprocess]] | 2026-06-05 | submitted | 0.964108 | 0.96463 | Small rule-based postprocess gain |
| [[exp007_nina_simple_vote]] | 2026-06-05 | submitted | | 0.97072 | Reproduced specified public simple vote notebook |
| [[exp008_nina_weighted_vote]] | 2026-06-05 | submitted | | 0.97074 | Improved weighted vote over specified notebook |
| [[exp009_lgbm_color_redshift_small]] | 2026-06-05 | completed | 0.963628 | | Focused features hurt raw CV |
| [[exp010_catboost_gpu_basic]] | 2026-06-05 | completed | 0.951213 | | GPU works, CatBoost setup weak |
| [[exp011_exp009_threshold_search]] | 2026-06-05 | diagnostic_success | 0.964604 | | Threshold improved exp009 OOF |
| [[exp012_exp009_threshold_submission]] | 2026-06-05 | submitted | 0.964604 | 0.96523 | Best self-model LB before exp014 |
| [[exp013_catboost_threshold_search]] | 2026-06-05 | diagnostic_success | 0.954779 | | CatBoost still below LightGBM |
| [[exp014_lgbm_catboost_blend]] | 2026-06-05 | submitted | 0.964652 | 0.96545 | Best self-model LB; CatBoost weight 0 |
| [[exp015_sdss17_source_match]] | 2026-06-05 | diagnostic_completed | | | SDSS17 direct matching rejected for now |
| [[exp016_xgboost_gpu_basic]] | 2026-06-05 | completed | 0.964020 | | GPU XGBoost is useful as a diversity candidate |
| [[exp017_lgbm_xgboost_blend]] | 2026-06-05 | submitted | 0.965235 | 0.96597 | New best self-model CV and LB |
| [[exp018_oof_logistic_stacker]] | 2026-06-05 | submitted | 0.965560 | 0.96641 | New best self-model CV and LB |
| [[exp019_xgboost_gpu_deep]] | 2026-06-05 | completed | 0.962960 | | Weak standalone, useful stack diversity candidate |
| [[exp020_oof_stacker_with_xgb_deep]] | 2026-06-05 | submitted | 0.965901 | 0.96642 | New best self-model CV, tiny LB gain |
| [[exp021_xgboost_gpu_shallow]] | 2026-06-05 | completed | 0.964128 | | Shallow XGBoost diversity candidate |
| [[exp022_oof_stacker_with_xgb_shallow]] | 2026-06-05 | no_submit | 0.965864 | | Below exp020, not submitted |
| [[exp023_lgbm_basic_fast]] | 2026-06-05 | completed | 0.963713 | | Basic LightGBM stack base |
| [[exp024_oof_stacker_with_lgbm_basic]] | 2026-06-05 | api_failed | 0.966112 | | New best CV; submission failed likely due daily limit |
| [[exp025_lgbm_color_regularized]] | 2026-06-05 | completed | 0.962930 | | Regularized LightGBM diversity base |
| [[exp026_oof_stacker_with_two_lgbm]] | 2026-06-05 | no_submit | 0.966092 | | Below exp024, not submitted |
| [[exp028_torch_logit_stacker]] | 2026-06-06 | no_submit | 0.965935 | | Pure PyTorch logit stacker; below exp024 |
| [[exp029_catboost_v3_style]] | 2026-06-06 | completed | 0.964213 | | CatBoost v3-style base with SDSS original low-weight append |
| [[exp030_oof_stacker_with_cat_v3]] | 2026-06-06 | no_submit | 0.966222 | | New best CV, but gain too small for API submission |
| [[exp031_lgbm_sdss_prior_fast]] | 2026-06-06 | completed | 0.959431 | | SDSS17 aggregate prior/prototype/count LightGBM; weak standalone |
| [[exp032_oof_stacker_with_sdss_prior]] | 2026-06-06 | no_submit | 0.966223 | | Effectively tied with exp030; no confident API submission |
| [[exp033_lgbm_sdss_stable]] | 2026-06-06 | completed | 0.963558 | | Stable SDSS prior/prototype LightGBM; healthier but not strong |
| [[exp034_oof_stacker_with_sdss_stable]] | 2026-06-06 | no_submit | 0.966206 | | Below exp030/032, no API submission |
| [[exp035_torch_tabular_mlp]] | 2026-06-06 | completed | 0.950964 | | Pure PyTorch MLP; weak standalone but useful diversity diagnostic |
| [[exp036_oof_stacker_with_torch_mlp]] | 2026-06-06 | no_submit | 0.966349 | | New diagnostic best but not confirmed by pure PyTorch stacker |
| [[exp037_torch_logit_stacker_with_mlp]] | 2026-06-06 | no_submit | 0.966175 | | Pure PyTorch confirmation below exp030/032 |
| [[exp038_oof_stacker_with_mlp_and_sdss]] | 2026-06-06 | no_submit | 0.966420 | | New diagnostic best with MLP + SDSS weak bases |
| [[exp039_torch_logit_stacker_with_mlp_and_sdss]] | 2026-06-06 | no_submit | 0.966322 | | Pure PyTorch confirmation supports modest gain |
| [[exp040_oof_stacker_with_mlp_and_sdss_prior]] | 2026-06-06 | no_submit | 0.966407 | | MLP + exp031, below exp038 |
| [[exp041_oof_stacker_with_mlp_and_sdss_stable]] | 2026-06-06 | no_submit | 0.966379 | | MLP + exp033, below exp038 |
| [[exp042_oof_stacker_mlp_sdss_seed777]] | 2026-06-06 | no_submit | 0.966398 | | Alternate meta split confirms exp038 base set |
| [[exp043_oof_stacker_mlp_sdss_seed31415]] | 2026-06-06 | no_submit | 0.966354 | | Alternate meta split confirms exp038 base set |
| [[exp044_seed_average_stacker]] | 2026-06-06 | no_submit | 0.966383 | | Seed-average candidate below exp038/042 |
| [[exp045_exp038_raw_candidate]] | 2026-06-06 | no_submit | 0.966344 | | Raw exp038 candidate |
| [[exp046_exp038_star1025_candidate]] | 2026-06-06 | no_submit | 0.966404 | | Mild STAR multiplier candidate |
| [[exp047_exp038_star105_candidate]] | 2026-06-06 | no_submit | 0.966420 | | Fixed best STAR multiplier candidate |

| Exp | Status | CV | LB | Decision |
|---|---|---:|---:|---|
| exp001_baseline | planned | | | |
