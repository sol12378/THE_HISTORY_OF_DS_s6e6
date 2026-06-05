# exp025_lgbm_color_regularized

## Purpose

自前OOF stackerに追加するため、LightGBMの高速OOF/test_proba baseを作る。

## Hypothesis

`basic` featureのLightGBMを別fold seedで作ることで、既存の`exp009`とは異なるLightGBM系予測をstackへ追加できる。

## Result

- CV balanced_accuracy: 0.962930
- CV accuracy: 0.960073
- Feature mode: `color_redshift`
- Fold count: 3
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

単体提出はしない。`exp020` stackに追加して、CV 0.965901を超えるかで採否を判断する。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold高速診断なのでfold varianceが残る。
