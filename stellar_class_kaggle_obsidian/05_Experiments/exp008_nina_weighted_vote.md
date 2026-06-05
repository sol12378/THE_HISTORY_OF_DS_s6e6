# exp008_nina_weighted_vote

---
tags: [experiment, submission, external]
status: submitted
exp_id: exp008_nina_weighted_vote
source_notebook: nina2025/ps-s6e6-simple-vote
cv:
lb: 0.97074
---

## Hypothesis

指定 notebook の majority vote を拡張し、filename の public score を重みにした top-N weighted vote にすると、単純5本 vote より安定した submission になる可能性がある。

## Changes

- Source dataset: `nina2025/ps-s6e6`
- Use top-N submissions by filename score.
- Weighted vote by parsed filename score.

## Results

| Metric | Value |
|---|---:|
| CV | N/A |
| LB | 0.97074 |

## Risks

- Leakage risk: high。public submissions blend。
- Overfitting risk: high。public score filenames に基づく選択。

## Decision

submitted

top-12 weighted vote は指定 notebook の simple vote をわずかに上回り、Public LB 0.97074。現状 best LB だが、外部 public submissions の blend であり leakage/overfitting risk は high。
