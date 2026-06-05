from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_train_test(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = Path(raw_dir)
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    sample_submission = pd.read_csv(raw / "sample_submission.csv")
    return train, test, sample_submission
