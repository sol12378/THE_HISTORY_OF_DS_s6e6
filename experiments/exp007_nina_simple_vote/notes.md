# exp007_nina_simple_vote

## Purpose

指定 notebook `nina2025/ps-s6e6-simple-vote` の simple vote をこの workspace で再現し、submission を作成する。

## Result

- Mode: `notebook_combo`
- Files: ['0.96933.csv', '0.97033.csv', '0.97047.csv', '0.97061.csv', '0.97079.csv']
- Agreement all: {'True': 244655, 'False': 2780}
- Submission distribution: {'GALAXY': 157215, 'QSO': 51375, 'STAR': 38845}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}
- Public LB: 0.97072
- Kaggle API submission ref: 53387851

## Risks

- Leakage risk: high。public submissions の blend であり、local CV 改善ではない。
- Overfitting risk: high。Public LB score 由来の選択に依存する。
