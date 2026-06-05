# Predicting Stellar Class

Kaggle competition workspace for `playground-series-s6e6`.

Competition: https://www.kaggle.com/competitions/playground-series-s6e6

## Layout

- `data/raw/`: immutable Kaggle downloads
- `data/interim/`: temporary preprocessing outputs
- `data/processed/`: model-ready datasets
- `data/folds/`: saved CV split files
- `src/stellar_class/`: reusable package code
- `scripts/`: command line entrypoints
- `experiments/`: one directory per experiment
- `outputs/`: logs, models, figures, predictions, blends
- `reports/`: analysis notes and solution writeups
- `stellar_class_kaggle_obsidian/`: Obsidian vault for decisions and experiment notes

## First Steps

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh
```

Raw data should remain unchanged after download.
