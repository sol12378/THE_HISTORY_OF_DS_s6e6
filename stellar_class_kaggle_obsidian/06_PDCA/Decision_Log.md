# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | Use `balanced_accuracy` as primary CV metric | Kaggle official Evaluation page states submissions are evaluated on balanced accuracy | CV and experiment summaries now report balanced accuracy as primary metric |
| 2026-06-05 | Submit `exp001_baseline` as first baseline | OOF balanced_accuracy 0.964030 and submission validation passed | Established first LB 0.96462 and a reproducible pipeline |
| 2026-06-05 | Split advanced feature work into smaller groups | `exp002_advanced_features` timed out twice, even after capacity reduction | Avoid all-feature 5-fold experiments until feature groups are screened |
| 2026-06-05 | Treat OOF threshold search as diagnostic, not final proof | `exp003` improved CV but tuned multipliers on the same OOF labels | Future submissions need saved test probabilities and CV/LB gap confirmation |
| 2026-06-05 | Submit small rule-based postprocess as `exp006` | Re-training for test probabilities timed out, while hard-prediction rules could be applied immediately | New best LB 0.96463, but gain is only +0.00001 |
| 2026-06-05 | Use the requested `nina2025/ps-s6e6-simple-vote` notebook as external blend reference | User explicitly requested using that notebook; it provides high-LB public submission blend candidates | `exp007` LB 0.97072 and improved `exp008` weighted vote LB 0.97074, but no local CV and high overfit risk |
| 2026-06-05 | Keep GPU CatBoost as diagnostic, not current ensemble member | `exp010` GPU CatBoost was fast but low CV, and `exp014` optimized CatBoost blend weight to 0 | GPU is useful, but current CatBoost setup does not improve self-model CV |
| 2026-06-05 | Reject direct SDSS17 label transfer for now | `exp015` found essentially no rounded matches and only about 0.81 sampled nearest-neighbor label balanced_accuracy | External SDSS17 should be treated as auxiliary data, not a direct answer key |
| 2026-06-05 | Keep GPU XGBoost as an ensemble member | `exp016` standalone CV was 0.964020 and `exp017` blend improved CV to 0.965235 and LB to 0.96597 | XGBoost adds useful model diversity for self-model stack/blend |
| 2026-06-05 | Move self-model development toward OOF stacking | `exp018` improved CV to 0.965560 and LB to 0.96641 | Add more diverse base OOF/test probability models and restack |
| 2026-06-05 | Tighten API submission policy to confident CV candidates | User requested API submissions only for candidates with confident CV | Future submissions must record CV, comparison target, risk, and confidence rationale before submission |
| 2026-06-05 | Keep `exp019` only as a stack diversity base | Standalone CV was 0.962960, but adding it in `exp020` improved stack CV to 0.965901 | Do not submit weak standalone models; use them only if stack CV improves |
| 2026-06-05 | Do not submit `exp022` | CV 0.965864 was below current best `exp020` CV 0.965901 | Confident-CV-only API submission rule prevented a low-confidence submission |
| 2026-06-05 | Treat `exp024` as current best CV but not confirmed LB | CV improved to 0.966112, but Kaggle API returned 400 and submission history did not register it | Keep `exp020` as best confirmed self-model LB; retry `exp024` only after submission limit resets |
| 2026-06-05 | Do not submit `exp026` | CV 0.966092 was below `exp024` CV 0.966112 | Adding the weak regularized LightGBM did not improve the current best stack |
| 2026-06-06 | Do not submit `exp028` | Pure PyTorch logit stacker CV 0.965935 was below `exp024` CV 0.966112 | Keep pure PyTorch stacker as a reusable tool, but prioritize adding stronger base models |
| 2026-06-06 | Do not submit `exp030` yet | CV improved to 0.966222, but the gain over `exp024` was only +0.000110 | Keep as current best CV and require a larger confidence margin before API submission |
| 2026-06-06 | Do not submit `exp032` | CV improved from `exp030` 0.966221799 to 0.966222902, only about +0.000001 and with multiplier search | Treat as an effective tie; redesign SDSS prior/prototype features before using API submissions |
| 2026-06-06 | Require pure PyTorch for torch experiments | User explicitly requested pure PyTorch | Avoid high-level torch wrappers and third-party tabular NN packages unless explicitly approved |
| 2026-06-06 | Do not submit `exp034` | Stable SDSS stack CV 0.966206 was below `exp030`/`exp032` | Deprioritize SDSS aggregate features and move to pure PyTorch tabular or stronger GPU tree variants |
| 2026-06-06 | Do not submit `exp036` yet | CV 0.966349 is a new diagnostic best, but pure PyTorch confirmation `exp037` was only 0.966175 | Keep the MLP base as a diversity signal, but require stronger confirmation before API submission |
| 2026-06-06 | Do not submit `exp038` yet | CV 0.966420 is a new diagnostic best and `exp039` confirms a smaller gain, but the margin is still modest and stacker/multiplier dependent | Keep `exp038` as best candidate; verify with another meta-stack split or calibration before API submission |
| 2026-06-06 | Do not submit `exp042`/`exp043` individually | Alternate seeds confirm the base set, but individual seed submissions are arbitrary | Build a seed-averaged stack candidate before considering Kaggle API submission |
