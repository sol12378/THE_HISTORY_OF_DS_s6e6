# Data Structure

## Raw Files

Checked on 2026-06-05 after Kaggle download.

| File | Shape | Notes |
|---|---:|---|
| `train.csv` | `(577347, 12)` | Includes `class` target |
| `test.csv` | `(247435, 11)` | Same features excluding target |
| `sample_submission.csv` | `(247435, 2)` | `id,class` |

## Columns

- ID: `id`
- Numeric: `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`
- Categorical: `spectral_type`, `galaxy_population`
- Target: `class`

## Target Distribution

| Class | Count |
|---|---:|
| `GALAXY` | 377480 |
| `QSO` | 117143 |
| `STAR` | 82724 |

Missing values were not observed in train/test.

## Raw Inventory

- Fill this after `scripts/download_data.sh`.

## Expected Notes

- Train/test row counts
- Feature columns
- Target column
- Submission columns
