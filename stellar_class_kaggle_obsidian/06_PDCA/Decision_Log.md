# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | Use `balanced_accuracy` as primary CV metric | Kaggle official Evaluation page states submissions are evaluated on balanced accuracy | CV and experiment summaries now report balanced accuracy as primary metric |
| 2026-06-05 | Submit `exp001_baseline` as first baseline | OOF balanced_accuracy 0.964030 and submission validation passed | Established first LB 0.96462 and a reproducible pipeline |
