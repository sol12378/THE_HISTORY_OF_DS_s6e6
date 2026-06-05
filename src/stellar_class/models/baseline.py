from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


def make_baseline_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=random_state)
