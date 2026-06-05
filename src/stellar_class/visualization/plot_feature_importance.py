from __future__ import annotations

import pandas as pd
import seaborn as sns


def plot_feature_importance(importance: pd.DataFrame) -> None:
    sns.barplot(data=importance, x="importance", y="feature")
