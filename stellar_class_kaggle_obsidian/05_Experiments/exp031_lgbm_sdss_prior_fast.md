# exp031_lgbm_sdss_prior_fast

## Purpose

SDSS17外部元データを直接照合・ラベル転写せず、aggregate class prior、class prototype距離、count特徴としてLightGBMへ追加する。

## Hypothesis

外部ラベルを集約統計として使えば、PBに近い分布知識を低めのleakage riskで取り込み、self-model stackの多様性を増やせる。

## Result

- CV balanced_accuracy: 0.959431
- CV accuracy: 0.963480
- n_splits: 3
- Feature count: 146
- Original rows: 100000
- Submission checks: all passed

## Interpretation

単体CVは既存LightGBM系より明確に弱い。SDSS prior/prototype特徴の設計が粗く、LightGBMが外部集約特徴を有効に使うよりもbin/ID/count特徴のノイズを拾った可能性がある。

## Risks

- Leakage risk: medium-low。SDSS17 labelsはaggregate prior/prototypeのみに使用し、test rowへの直接ラベル転写はしていない。
- Overfitting risk: medium。多数のbin/prior特徴がOOFへ過適合する可能性がある。

## Next

SDSS特徴は「全特徴に混ぜる」より、少数の安定prior/prototypeだけを既存強モデルへ足す形、またはfold-safe target encoding/feature selectionで再設計する。
