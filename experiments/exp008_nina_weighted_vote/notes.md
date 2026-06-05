# exp008_nina_weighted_vote

## Purpose

指定 notebook `nina2025/ps-s6e6-simple-vote` の simple vote をこの workspace で再現し、submission を作成する。

## Result

- Mode: `top_weighted`
- Files: ['0.97079.b.csv', '0.97079.csv', '0.97076.b.csv', '0.97076.csv', '0.97075.b.csv', '0.97075.csv', '0.97074.b.csv', '0.97074.csv', '0.97073.csv', '0.97071.csv', '0.97070.csv', '0.97065.csv']
- Agreement all: {'True': 246644, 'False': 791}
- Submission distribution: {'GALAXY': 157265, 'QSO': 51377, 'STAR': 38793}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}
- Public LB: 0.97074
- Kaggle API submission ref: 53387906

## Risks

- Leakage risk: high。public submissions の blend であり、local CV 改善ではない。
- Overfitting risk: high。Public LB score 由来の選択に依存する。
