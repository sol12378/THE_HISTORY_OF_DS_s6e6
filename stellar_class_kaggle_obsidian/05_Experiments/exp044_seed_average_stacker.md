# exp044_seed_average_stacker

## Purpose

`exp038/042/043` のstacker seedを任意に1つ選ばず、OOF/test probabilitiesを等重み平均して、より安定したsubmission candidateを作る。

## Result

- Source experiments: `exp038`, `exp042`, `exp043`
- Raw CV balanced_accuracy: 0.966366
- Thresholded CV balanced_accuracy: 0.966383
- CV accuracy: 0.960034
- Multipliers: GALAXY 1.0, QSO 1.05, STAR 1.05
- Submission checks: all passed

## Interpretation

seed選択の任意性は下がるが、CVは `exp038` 0.966420 と `exp042` 0.966398 を下回った。提出候補としての安定性はあるが、best CV候補ではない。

## Decision

no_submit。API提出は、best CVである `exp038` の追加calibrationか、さらに強いbase/stack改善後に判断する。

## Risks

- Leakage risk: medium。sourceはOOFだが、multipliersはOOF labelsで選択。
- Overfitting risk: medium。seed averagingでsplit varianceは減るが、multiplier searchの過適合は残る。
