# exp029_catboost_v3_style

## Purpose

CatBoost v3風に、量子化bin・丸めbin・カテゴリinteraction・SDSS17 original低weight appendを使うCatBoost GPU baseを作る。

## Result

- CV balanced_accuracy: 0.964213
- CV accuracy: 0.958396
- n_splits: 3
- iterations: 1600
- original_weight: 0.06
- Feature count: 103
- Categorical feature count: 47
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

単体提出ではなく、`exp024` stackに追加してCV改善するかで採否を判断する。

## Risks

- Leakage risk: medium。SDSS17 original labelを低weightでfold trainingにappendするが、validation rowsは学習に入らない。direct label transferは使わない。
- Overfitting risk: medium-high。カテゴリviewが多く、外部データを使うため、OOF stackとLBの相関を確認する必要がある。
