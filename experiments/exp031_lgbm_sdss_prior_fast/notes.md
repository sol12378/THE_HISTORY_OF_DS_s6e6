# exp031_lgbm_sdss_prior_fast

## Purpose

SDSS17外部元データを直接照合・ラベル転写せず、class prior、class prototype距離、train/test/original count特徴としてLightGBMに入れる。

## Hypothesis

外部ラベルを集約統計として使えば、PBに近い分布知識を取り込みつつ、直接リークより低リスクでself-model stackの多様性を増やせる。

## Result

- CV balanced_accuracy: 0.959431
- CV accuracy: 0.963480
- n_splits: 3
- Feature count: 146
- Categorical feature count: 23
- Original rows: 100000
- Submission checks: {'columns_match_sample_submission': True, 'row_count_matches_sample_submission': True, 'id_matches_sample_submission': True, 'labels_are_valid_strings': True, 'no_missing_values': True, 'all_passed': True}

## Interpretation

単体CVと次段stack CVで採否を決める。`exp030_oof_stacker_with_cat_v3` の CV 0.966222 を明確に超えない場合はAPI提出しない。

## Risks

- Leakage risk: medium-low。SDSS17 labelsはaggregate prior/prototypeのみに使用し、test rowへの直接ラベル転写はしていない。
- Overfitting risk: medium。bin/prior特徴がOOFに過適合する可能性があるため、stack改善幅とLB提出基準を分けて判断する。
