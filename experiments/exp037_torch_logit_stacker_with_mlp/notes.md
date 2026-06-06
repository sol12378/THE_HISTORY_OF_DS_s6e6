# exp037_torch_logit_stacker_with_mlp

## Purpose

純粋なPyTorchでCDeotte型logit-only multi-seed logistic stackerを検証する。

## Hypothesis

probabilityをlogit化し、PyTorchの`nn.Linear`だけでmulticlass logistic regressionを学習すれば、`exp024` CV 0.966112を超える、またはstacker実装の妥当性を検証できる。

## Result

- Base experiments: ['exp009_lgbm_color_redshift_small', 'exp010_catboost_gpu_basic', 'exp016_xgboost_gpu_basic', 'exp019_xgboost_gpu_deep', 'exp023_lgbm_basic_fast', 'exp029_catboost_v3_style', 'exp035_torch_tabular_mlp']
- Device: `cuda`
- Seeds: [42, 43, 44]
- Folds: 5
- Epochs: 1000
- Best C: 1.0
- Raw CV balanced_accuracy: 0.966135
- Thresholded CV balanced_accuracy: 0.966175
- CV accuracy: 0.959496
- Multipliers: {'GALAXY': 1.0, 'QSO': 1.1, 'STAR': 1.05}
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Submission Decision

Compare against `exp024_oof_stacker_with_lgbm_basic` CV 0.966112. Submit only if the improvement is confidence-worthy and the Kaggle submission limit allows it.

## Risks

- Leakage risk: medium。base predictionsはOOFだが、CとmultiplierをOOF labelで選んでいる。
- Overfitting risk: medium-high。multi-seedでsplit varianceは減るが、meta-parameter探索は過学習しうる。
