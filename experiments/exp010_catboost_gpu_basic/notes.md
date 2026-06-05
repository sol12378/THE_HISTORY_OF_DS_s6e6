# exp010_catboost_gpu_basic

## Purpose

GPU CatBoost で LightGBM と異なる model family の OOF を作り、ensemble 素材として有効か確認する。

## Result

- CV balanced_accuracy: 0.951213
- CV accuracy: 0.965011
- n_splits: 3
- iterations: 600
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_test': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Risks

- Leakage risk: target-derived feature は使っていない。
- Overfitting risk: 3-fold diagnostic なので fold stability は弱い。良ければ 5-fold に拡張する。
