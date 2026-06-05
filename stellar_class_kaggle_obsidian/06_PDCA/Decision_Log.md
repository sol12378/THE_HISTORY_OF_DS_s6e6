# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | Use `balanced_accuracy` as primary CV metric | Kaggle official Evaluation page states submissions are evaluated on balanced accuracy | CV and experiment summaries now report balanced accuracy as primary metric |
| 2026-06-05 | Submit `exp001_baseline` as first baseline | OOF balanced_accuracy 0.964030 and submission validation passed | Established first LB 0.96462 and a reproducible pipeline |
| 2026-06-05 | Split advanced feature work into smaller groups | `exp002_advanced_features` timed out twice, even after capacity reduction | Avoid all-feature 5-fold experiments until feature groups are screened |
| 2026-06-05 | Treat OOF threshold search as diagnostic, not final proof | `exp003` improved CV but tuned multipliers on the same OOF labels | Future submissions need saved test probabilities and CV/LB gap confirmation |
| 2026-06-05 | Submit small rule-based postprocess as `exp006` | Re-training for test probabilities timed out, while hard-prediction rules could be applied immediately | New best LB 0.96463, but gain is only +0.00001 |
| 2026-06-05 | Use the requested `nina2025/ps-s6e6-simple-vote` notebook as external blend reference | User explicitly requested using that notebook; it provides high-LB public submission blend candidates | `exp007` LB 0.97072 and improved `exp008` weighted vote LB 0.97074, but no local CV and high overfit risk |
