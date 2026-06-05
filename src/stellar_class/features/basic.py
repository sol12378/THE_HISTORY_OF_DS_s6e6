from __future__ import annotations

import pandas as pd


def basic_features(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    drop_columns = drop_columns or []
    return df.drop(columns=[c for c in drop_columns if c in df.columns])
