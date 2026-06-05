# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | Use `balanced_accuracy` as primary CV metric | Kaggle official Evaluation page states submissions are evaluated on balanced accuracy | CV and experiment summaries now report balanced accuracy as primary metric |
| 2026-06-05 | Submit `exp001_baseline` as first baseline | OOF balanced_accuracy 0.964030 and submission validation passed | Established first LB 0.96462 and a reproducible pipeline |
| 2026-06-05 | Split advanced feature work into smaller groups | `exp002_advanced_features` timed out twice, even after capacity reduction | Avoid all-feature 5-fold experiments until feature groups are screened |
| 2026-06-05 | Treat OOF threshold search as diagnostic, not final proof | `exp003` improved CV but tuned multipliers on the same OOF labels | Future submissions need saved test probabilities and CV/LB gap confirmation |
