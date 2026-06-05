# exp015_sdss17_source_match

## Purpose

PB 1位狙いのため、公開SDSS17候補データと公式train/testの直接照合可能性を診断する。

## Result

- External source: `data/external/sdss17/star_classification.csv`
- Source URL: `https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17`
- Rounded match: decimals 6から2ではtrain/testとも0件、decimals 1でtest 1件のみ
- Sampled nearest-neighbor: threshold 0.1でtrain label balanced_accuracy 約0.8127

## Decision

直接label transferは採用しない。train側で高信頼にlabel再現できていないため、PB向けには危険。

## Risks

- Leakage risk: medium-high。直接写しはルール・出典確認が必要。
- Overfitting risk: medium。test match率で閾値を決めると危険。

## Next Actions

SDSS17は直接照合ではなく、補助学習データまたは分布比較に回す。
