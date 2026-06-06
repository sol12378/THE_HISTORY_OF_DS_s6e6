# exp033_lgbm_sdss_stable

## Purpose

`exp031` で弱かった広いbin/count型SDSS特徴を削り、coarse prior と prototype距離だけをLightGBMに追加する。

## Result

- CV balanced_accuracy: 0.963558
- CV accuracy: 0.964081
- n_splits: 3
- Feature count: 52
- Categorical feature count: 3
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

単体CVとstack CVで採否を判断する。`exp032` の 0.966223 を明確に超えない場合、API提出しない。

## Risks

- Leakage risk: medium-low。SDSS17 labelsは粗いaggregate prior/prototypeのみに使い、直接照合やtest label transferはしていない。
- Overfitting risk: medium。外部由来特徴のCV過適合は残るため、改善幅を厳しく見る。
