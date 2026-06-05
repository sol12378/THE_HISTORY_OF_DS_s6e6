# Reproducibility and Artifacts

## Experiment Artifacts

Each completed CV training experiment should save:

- `result.json`
- `notes.md`
- `oof.csv`
- `test_proba.csv`
- `submission.csv` only when intended for Kaggle
- `feature_importance.csv` when supported

`oof.csv`, `test_proba.csv`, and `submission.csv` are ignored by git because they are generated artifacts.

Keep raw data immutable and save experiment outputs under `experiments/expXXX_*`.
