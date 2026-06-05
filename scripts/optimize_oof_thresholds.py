from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


def score_predictions(y_idx: np.ndarray, pred_idx: np.ndarray) -> tuple[float, float, list[list[int]]]:
    n_classes = len(CLASSES)
    cm = np.bincount(y_idx * n_classes + pred_idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    recalls = np.diag(cm) / cm.sum(axis=1)
    balanced_accuracy = float(recalls.mean())
    accuracy = float(np.diag(cm).sum() / cm.sum())
    return balanced_accuracy, accuracy, cm.tolist()


def evaluate(y_idx: np.ndarray, proba: np.ndarray, multipliers: np.ndarray) -> dict:
    pred_idx = (proba * multipliers.reshape(1, -1)).argmax(axis=1)
    balanced_accuracy, accuracy, cm = score_predictions(y_idx, pred_idx)
    return {
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "multipliers": {cls: float(mult) for cls, mult in zip(CLASSES, multipliers)},
        "confusion_matrix": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize class multipliers on OOF probabilities.")
    parser.add_argument("--source-exp", default="exp001_baseline")
    parser.add_argument("--exp-id", default="exp003_oof_threshold_search")
    args = parser.parse_args()

    source_dir = Path("experiments") / args.source_exp
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(source_dir / "oof.csv")
    proba = oof[[f"proba_{cls}" for cls in CLASSES]].to_numpy(dtype=np.float64)
    y_idx = oof["true_class"].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)

    baseline = evaluate(y_idx, proba, np.ones(len(CLASSES), dtype=np.float64))
    best = baseline

    multipliers = np.ones(len(CLASSES), dtype=np.float64)
    search_ranges = [
        np.linspace(0.70, 1.30, 31),
        np.linspace(0.85, 1.15, 31),
        np.linspace(0.95, 1.05, 21),
    ]
    for values in search_ranges:
        for class_idx in [1, 2]:
            local_best = best
            local_multiplier = multipliers[class_idx]
            for value in values:
                trial = multipliers.copy()
                trial[class_idx] = value
                result = evaluate(y_idx, proba, trial)
                if result["balanced_accuracy"] > local_best["balanced_accuracy"]:
                    local_best = result
                    local_multiplier = value
            multipliers[class_idx] = local_multiplier
            best = local_best

    result = {
        "exp_id": args.exp_id,
        "source_exp": args.source_exp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Optimize class multipliers on exp001 OOF probabilities for official balanced accuracy.",
        "official_metric": "balanced_accuracy",
        "baseline": baseline,
        "best": best,
        "delta_balanced_accuracy": best["balanced_accuracy"] - baseline["balanced_accuracy"],
        "delta_accuracy": best["accuracy"] - baseline["accuracy"],
        "leakage_risk": "Medium: multipliers are optimized directly on OOF labels. This is post-processing risk, not target leakage into model training.",
        "overfitting_risk": "Medium-high: only two scalar multipliers are tuned, but the same OOF labels are used for selection and evaluation.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_idx = (proba * np.array([best["multipliers"][cls] for cls in CLASSES]).reshape(1, -1)).argmax(axis=1)
    adjusted = oof[["id", "true_class"]].copy()
    adjusted["pred_class"] = np.array(CLASSES, dtype=object)[pred_idx]
    for cls in CLASSES:
        adjusted[f"source_proba_{cls}"] = oof[f"proba_{cls}"]
    adjusted.to_csv(exp_dir / "oof.csv", index=False)

    notes = f"""# {args.exp_id}

## Purpose

`exp001_baseline` の OOF probabilities に class multiplier を掛け、公式 metric の `balanced_accuracy` が改善するか確認する。

## Hypothesis

baseline は `STAR` recall が相対的に低いため、class prior / threshold を調整すれば macro recall が改善し、CV balanced_accuracy が上がる。

## Result

- Baseline balanced_accuracy: {baseline["balanced_accuracy"]:.6f}
- Adjusted balanced_accuracy: {best["balanced_accuracy"]:.6f}
- Delta balanced_accuracy: {result["delta_balanced_accuracy"]:+.6f}
- Baseline accuracy: {baseline["accuracy"]:.6f}
- Adjusted accuracy: {best["accuracy"]:.6f}
- Delta accuracy: {result["delta_accuracy"]:+.6f}
- Best multipliers: {best["multipliers"]}

## Risks

- Leakage risk: model training 自体への target leakage はない。ただし OOF label を使った post-processing 最適化なので leaderboard へ過信しない。
- Overfitting risk: scalar 2個だけの tuning だが、同じ OOF で探索と評価をしているため medium-high。

## Decision

Use as a diagnostic. Test-time probabilities are not saved for exp001, so submission generation requires rerunning or future experiments saving test probabilities.
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
