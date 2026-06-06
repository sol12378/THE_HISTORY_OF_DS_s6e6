# exp045_exp038_raw_candidate

## Purpose

`exp038_oof_stacker_with_mlp_and_sdss` のOOF/test probabilitiesに固定class multipliersを適用し、提出候補のthreshold依存度を確認する。

## Result

- CV balanced_accuracy: 0.966344
- CV accuracy: 0.960544
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.0, 'STAR': 1.0}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

倍率を強くしすぎずにCVが維持できるかを見る診断。API提出は、現行bestと安定性確認を合わせて判断する。
