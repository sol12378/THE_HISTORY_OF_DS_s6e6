# exp046_exp038_star1025_candidate

## Purpose

`exp038` に mild な STAR multiplier 1.025 を適用し、過度なthreshold依存を避けた候補を作る。

## Result

- CV balanced_accuracy: 0.966404
- CV accuracy: 0.960393
- Multipliers: GALAXY 1.0, QSO 1.0, STAR 1.025
- Submission checks: all passed

## Interpretation

`exp047` のbestより少し低いが、より保守的。PB安定性を重視する場合の候補。

## Decision

no_submit。提出するなら `exp047` とどちらを優先するか、Kaggle提出枠とLB確認方針を明記する。
