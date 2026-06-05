# exp006_rule_postprocess

## Purpose

`exp001_baseline` の hard predictions に redshift/category based rules を適用し、公式 metric の `balanced_accuracy` が改善するか確認する。

## Result

- Baseline balanced_accuracy: 0.964030
- Adjusted balanced_accuracy: 0.964108
- Delta balanced_accuracy: +0.000078
- Baseline accuracy: 0.965139
- Adjusted accuracy: 0.965007
- Delta accuracy: -0.000132
- Public LB: 0.96463
- LB - CV gap: +0.000522
- Kaggle API submission ref: 53387393
- Rules: [{'from_class': 'GALAXY', 'to_class': 'QSO', 'direction': 'gt', 'threshold': 1.6, 'category_column': None, 'category_value': None, 'affected': 228}, {'from_class': 'STAR', 'to_class': 'GALAXY', 'direction': 'gt', 'threshold': 0.28, 'category_column': 'spectral_type', 'category_value': 'M', 'affected': 10}, {'from_class': 'GALAXY', 'to_class': 'QSO', 'direction': 'gt', 'threshold': 1.525, 'category_column': 'spectral_type', 'category_value': 'O/B', 'affected': 22}]
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Risks

- Leakage risk: target-derived training feature はないが、rule selection は OOF labels を使っている。
- Overfitting risk: medium-high。LB が改善しない場合は採用しない。
