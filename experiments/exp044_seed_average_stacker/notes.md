# exp044_seed_average_stacker

## Purpose

stacker seedを任意に1つ選ばず、`['exp038_oof_stacker_with_mlp_and_sdss', 'exp042_oof_stacker_mlp_sdss_seed777', 'exp043_oof_stacker_mlp_sdss_seed31415']` のOOF/test probabilitiesを等重み平均して、より安定したsubmission candidateを作る。

## Result

- Raw CV balanced_accuracy: 0.966366
- Thresholded CV balanced_accuracy: 0.966383
- CV accuracy: 0.960034
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.05, 'STAR': 1.05}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

`exp038` 単体より低い場合でも、seed平均によりmeta split任意性は下がる。API提出は、現行bestとの比較、raw CV、confirmation結果、submission limitを合わせて判断する。

## Risks

- Leakage risk: medium。sourceはOOFだが、multipliersはOOF labelsで選択。
- Overfitting risk: medium。seed averagingでsplit varianceは減るが、multiplier searchの過適合は残る。
