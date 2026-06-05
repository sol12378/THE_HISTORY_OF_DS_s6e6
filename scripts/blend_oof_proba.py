from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
TARGET = "class"
ID_COL = "id"


def read_oof(exp_id: str) -> pd.DataFrame:
    return pd.read_csv(Path("experiments") / exp_id / "oof.csv")


def read_test_proba(exp_id: str) -> pd.DataFrame:
    return pd.read_csv(Path("experiments") / exp_id / "test_proba.csv")


def evaluate(y_idx: np.ndarray, proba: np.ndarray, multipliers: np.ndarray) -> dict:
    pred_idx = (proba * multipliers.reshape(1, -1)).argmax(axis=1)
    n_classes = len(CLASSES)
    cm = np.bincount(y_idx * n_classes + pred_idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    recalls = np.diag(cm) / cm.sum(axis=1)
    return {
        "balanced_accuracy": float(recalls.mean()),
        "accuracy": float(np.diag(cm).sum() / cm.sum()),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend OOF/test probabilities from two experiments.")
    parser.add_argument("--exp-id", default="exp014_lgbm_catboost_blend")
    parser.add_argument("--left-exp", required=True)
    parser.add_argument("--right-exp", required=True)
    args = parser.parse_args()

    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    left = read_oof(args.left_exp)
    right = read_oof(args.right_exp)
    merged = left[[ID_COL, "true_class", *[f"proba_{cls}" for cls in CLASSES]]].merge(
        right[[ID_COL, *[f"proba_{cls}" for cls in CLASSES]]],
        on=ID_COL,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    y_idx = merged["true_class"].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)
    left_proba = merged[[f"proba_{cls}_left" for cls in CLASSES]].to_numpy(dtype=np.float64)
    right_proba = merged[[f"proba_{cls}_right" for cls in CLASSES]].to_numpy(dtype=np.float64)

    best = None
    history = []
    for weight in np.linspace(0.0, 1.0, 21):
        proba = weight * left_proba + (1.0 - weight) * right_proba
        for qso_scale in np.linspace(1.0, 1.35, 8):
            for star_scale in np.linspace(1.0, 1.35, 8):
                multipliers = np.array([1.0, qso_scale, star_scale], dtype=np.float64)
                score = evaluate(y_idx, proba, multipliers)
                candidate = {
                    "left_weight": float(weight),
                    "right_weight": float(1.0 - weight),
                    "multipliers": {cls: float(mult) for cls, mult in zip(CLASSES, multipliers)},
                    **score,
                }
                history.append(candidate)
                if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
                    best = candidate

    assert best is not None

    left_test = read_test_proba(args.left_exp)
    right_test = read_test_proba(args.right_exp)
    test = left_test.merge(right_test, on=ID_COL, suffixes=("_left", "_right"), validate="one_to_one")
    left_test_proba = test[[f"proba_{cls}_left" for cls in CLASSES]].to_numpy(dtype=np.float64)
    right_test_proba = test[[f"proba_{cls}_right" for cls in CLASSES]].to_numpy(dtype=np.float64)
    test_proba = best["left_weight"] * left_test_proba + best["right_weight"] * right_test_proba
    multipliers = np.array([best["multipliers"][cls] for cls in CLASSES], dtype=np.float64)
    pred = np.array(CLASSES, dtype=object)[(test_proba * multipliers.reshape(1, -1)).argmax(axis=1)]

    sample = pd.read_csv("data/raw/sample_submission.csv")
    submission = sample.copy()
    submission[TARGET] = pred
    submission.to_csv(exp_dir / "submission.csv", index=False)

    oof_pred = np.array(CLASSES, dtype=object)[
        (
            (best["left_weight"] * left_proba + best["right_weight"] * right_proba)
            * multipliers.reshape(1, -1)
        ).argmax(axis=1)
    ]
    oof = merged[[ID_COL, "true_class"]].copy()
    oof["pred_class"] = oof_pred
    for idx, cls in enumerate(CLASSES):
        oof[f"proba_{cls}"] = (best["left_weight"] * left_proba + best["right_weight"] * right_proba)[:, idx]
    oof.to_csv(exp_dir / "oof.csv", index=False)

    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample),
        "id_matches_sample_submission": submission[ID_COL].equals(sample[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(CLASSES)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    result = {
        "exp_id": args.exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Blend LightGBM and CatBoost GPU probabilities for model diversity.",
        "official_metric": "balanced_accuracy",
        "left_exp": args.left_exp,
        "right_exp": args.right_exp,
        "best": best,
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Medium: blend weights and class multipliers are optimized on OOF labels.",
        "overfitting_risk": "Medium-high: OOF grid search may overfit and must be LB-checked.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {args.exp_id}

## Purpose

LightGBM (`{args.left_exp}`) と CatBoost GPU (`{args.right_exp}`) の probability を soft voting し、model diversity で CV balanced_accuracy が改善するか確認する。

## Result

- CV balanced_accuracy: {best["balanced_accuracy"]:.6f}
- CV accuracy: {best["accuracy"]:.6f}
- Left weight: {best["left_weight"]:.3f}
- Right weight: {best["right_weight"]:.3f}
- Multipliers: {best["multipliers"]}
- Submission checks: {checks}

## Risks

- Leakage risk: model training への leakage はないが、blend weights と multipliers を OOF labels で選んでいる。
- Overfitting risk: medium-high。LB確認が必要。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
