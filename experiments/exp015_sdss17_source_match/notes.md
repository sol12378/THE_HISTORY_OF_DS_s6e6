# exp015_sdss17_source_match

## Purpose

PB 1位狙いのため、公開SDSS17候補データと公式train/testの照合可能性を診断した。

## Hypothesis

PlaygroundデータがSDSS17由来なら、`alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift` の丸め一致または近傍一致で一部行を高信頼に回収できる可能性がある。

## Result

- External source: `data\external\sdss17\star_classification.csv`
- External shape: (100000, 18)
- Best rounded train matches: decimals=6, train_matches=0, test_matches=0
- Nearest-neighbor safe threshold candidate: None

## Interpretation

`result.json` の `rounded_match_summary` と `nearest_neighbor_summary` を参照。採用する場合は、train側のlabel一致率が高い距離閾値だけを使い、testだけで都合よく閾値を決めない。

## Leakage Risk

Medium-high。外部元データのlabelをtestへ直接写す行為は、データ由来と競技ルール次第でleakage扱いになる可能性がある。提出前に、外部データが公開・利用可能であること、hidden targetの逆算ではないことを確認する。

## Overfitting Risk

Medium。丸め桁数や距離閾値をtest match率に合わせるとPBへ過適合する。train一致率と距離分布を主判断にする。

## Next Actions

- 0.99以上のtrain label balanced_accuracyを満たす近傍閾値が十分な行数を持つか確認する。
- 十分なら、external-match priorをOOF stackの特徴として追加する。
- 不十分なら、SDSS17は追加学習データとしてのみ扱い、直接label写しは採用しない。
