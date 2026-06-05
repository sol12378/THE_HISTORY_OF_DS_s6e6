# Data Quality Checks

## Raw CSV Checks

Checked on 2026-06-05 for `exp001_baseline`.

| Check | Result |
|---|---|
| `train.csv` exists | PASS |
| `test.csv` exists | PASS |
| `sample_submission.csv` exists | PASS |
| Train rows | 577347 |
| Test rows | 247435 |
| Missing values in train/test | 0 / 0 |
| Target classes | `GALAXY`, `QSO`, `STAR` |

## Submission Checks

`experiments/exp001_baseline/submission.csv`

| Check | Result |
|---|---|
| Columns match sample submission | PASS |
| Row count matches test | PASS |
| IDs match sample submission order | PASS |
| Labels are valid strings | PASS |
| Missing values | PASS |
