# Experiment Summary

## Current Best

| Type | Experiment | CV | LB | Notes |
|---|---|---:|---:|---|
| Baseline | exp001_baseline | 0.964030 | 0.96462 | LightGBM + color indices, official balanced accuracy |
| Best CV | exp001_baseline | 0.964030 | 0.96462 | First completed CV/submission pipeline |
| Best LB | exp001_baseline | 0.964030 | 0.96462 | CV/LB gap +0.000590 |

## Experiments

| Exp | Date | Hypothesis | CV | LB | Decision | Notes |
|---|---|---|---:|---:|---|---|
| exp001_baseline | 2026-06-05 | Establish a reproducible LightGBM baseline for official balanced accuracy. | 0.964030 | 0.96462 | submitted | CV/LB gap +0.000590; leakage risk low-medium due to ID-like columns retained |
