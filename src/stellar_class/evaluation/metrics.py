from __future__ import annotations

from sklearn.metrics import accuracy_score


def accuracy(y_true: object, y_pred: object) -> float:
    return float(accuracy_score(y_true, y_pred))
