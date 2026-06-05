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
