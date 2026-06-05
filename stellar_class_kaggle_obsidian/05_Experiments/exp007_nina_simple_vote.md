# exp007_nina_simple_vote

---
tags: [experiment, submission, external]
status: submitted
exp_id: exp007_nina_simple_vote
source_notebook: nina2025/ps-s6e6-simple-vote
cv:
lb: 0.97072
---

## Hypothesis

指定 notebook の simple vote（public submissions 5本の多数決）を再現すれば、local model より高い LB を得られる。ただしこれは CV 改善ではなく public submission blend である。

## Changes

- Source notebook: https://www.kaggle.com/code/nina2025/ps-s6e6-simple-vote
- Source dataset: `nina2025/ps-s6e6`
- Notebook final combo: indices `[9, 19, 23, 24, 43]`

## Results

| Metric | Value |
|---|---:|
| CV | N/A |
| LB | 0.97072 |

## Risks

- Leakage risk: high。public submissions の blend で、local CV を証明しない。
- Overfitting risk: high。public LB 由来の submission selection。

## Decision

submitted

指定 notebook の final combo `[9, 19, 23, 24, 43]` を再現し、Public LB 0.97072。local CV はないため、model validation の証拠ではなく public submission blend として扱う。
