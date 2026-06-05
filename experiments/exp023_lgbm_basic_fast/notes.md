# exp023_lgbm_basic_fast

## Purpose

自前OOF stackerに追加するため、LightGBMの高速OOF/test_proba baseを作る。

## Hypothesis

`basic` featureのLightGBMを別fold seedで作ることで、既存の`exp009`とは異なるLightGBM系予測をstackへ追加できる。

## Result

- CV balanced_accuracy: 0.963713
- CV accuracy: 0.964162
- Feature mode: `basic`
- Fold count: 3
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

単体提出はしない。`exp020` stackに追加して、CV 0.965901を超えるかで採否を判断する。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold高速診断なのでfold varianceが残る。
