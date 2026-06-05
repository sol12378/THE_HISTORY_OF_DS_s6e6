# CV Strategy

## Baseline

- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Primary local metric: `balanced_accuracy`, matching the official Kaggle metric.
- Also record plain `accuracy` as a secondary diagnostic only.

## exp001_baseline Check

- CV balanced_accuracy: 0.964030
- Public LB: 0.96462
- Gap: +0.000590

The first submission suggests the validation setup is directionally aligned with LB.

Start with stratified K-fold if the target is class-balanced enough. Revisit after schema and target distribution are known.
