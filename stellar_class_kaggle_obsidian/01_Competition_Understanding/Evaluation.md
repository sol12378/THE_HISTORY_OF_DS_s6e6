# Evaluation

## Official Metric

- Metric: `balanced_accuracy`
- Direction: higher is better.
- Source checked: `kaggle competitions pages playground-series-s6e6 --content --page-name Evaluation` on 2026-06-05.

## Submission Format

- Required columns: `id,class`
- Labels: `GALAXY`, `STAR`, `QSO`
- `sample_submission.csv` shape: `(247435, 2)`

## CV/LB Relationship

| Date | Exp | CV balanced_accuracy | Public LB | Gap |
|---|---|---:|---:|---:|
| 2026-06-05 | exp001_baseline | 0.964030 | 0.96462 | +0.000590 |

Initial CV/LB gap is small, so 5-fold stratified CV is usable as the first validation baseline.
