from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


ID_COL = "id"
TARGET = "class"
CLASSES = ["GALAXY", "QSO", "STAR"]
PROBA_COLS = [f"proba_{cls}" for cls in CLASSES]


def read_oof(exp_id: str) -> pd.DataFrame:
    path = Path("experiments") / exp_id / "oof.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_test_proba(exp_id: str) -> pd.DataFrame:
    path = Path("experiments") / exp_id / "test_proba.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def evaluate(y: np.ndarray, proba: np.ndarray, multipliers: np.ndarray) -> dict:
    pred = (proba * multipliers.reshape(1, -1)).argmax(axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "multipliers": {cls: float(mult) for cls, mult in zip(CLASSES, multipliers)},
    }


def search_multipliers(y: np.ndarray, proba: np.ndarray) -> dict:
    best = None
    for qso_scale in np.linspace(1.0, 1.25, 6):
        for star_scale in np.linspace(1.0, 1.25, 6):
            candidate = evaluate(y, proba, np.array([1.0, qso_scale, star_scale], dtype=np.float64))
            if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
                best = candidate
    assert best is not None
    return best


def run_average(exp_id: str, base_exp_ids: list[str]) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_oof = None
    base_test = None
    oof_sum = None
    test_sum = None
    for source_exp in base_exp_ids:
        oof = read_oof(source_exp)
        test = read_test_proba(source_exp)
        if base_oof is None:
            base_oof = oof[[ID_COL, "true_class"]].copy()
            base_test = test[[ID_COL]].copy()
            oof_sum = np.zeros((len(oof), len(CLASSES)), dtype=np.float64)
            test_sum = np.zeros((len(test), len(CLASSES)), dtype=np.float64)
        else:
            base_oof = base_oof.merge(oof[[ID_COL, "true_class"]], on=[ID_COL, "true_class"], validate="one_to_one")
            base_test = base_test.merge(test[[ID_COL]], on=ID_COL, validate="one_to_one")
        oof_sum += oof[PROBA_COLS].to_numpy(dtype=np.float64) / len(base_exp_ids)
        test_sum += test[PROBA_COLS].to_numpy(dtype=np.float64) / len(base_exp_ids)

    assert base_oof is not None and base_test is not None and oof_sum is not None and test_sum is not None
    y = base_oof["true_class"].map({cls: idx for idx, cls in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    raw = evaluate(y, oof_sum, np.ones(len(CLASSES), dtype=np.float64))
    threshold = search_multipliers(y, oof_sum)
    multipliers = np.array([threshold["multipliers"][cls] for cls in CLASSES], dtype=np.float64)
    pred_idx = (oof_sum * multipliers.reshape(1, -1)).argmax(axis=1)
    test_pred = np.array(CLASSES, dtype=object)[(test_sum * multipliers.reshape(1, -1)).argmax(axis=1)]

    oof_out = base_oof.copy()
    oof_out["pred_class"] = np.array(CLASSES, dtype=object)[pred_idx]
    for idx, cls in enumerate(CLASSES):
        oof_out[f"proba_{cls}"] = oof_sum[:, idx].astype(np.float32)
    oof_out.to_csv(exp_dir / "oof.csv", index=False)

    test_proba_out = base_test.copy()
    for idx, cls in enumerate(CLASSES):
        test_proba_out[f"proba_{cls}"] = test_sum[:, idx].astype(np.float32)
    test_proba_out.to_csv(exp_dir / "test_proba.csv", index=False)

    sample = pd.read_csv("data/raw/sample_submission.csv")
    submission = sample.copy()
    submission[TARGET] = test_pred
    submission.to_csv(exp_dir / "submission.csv", index=False)
    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample),
        "id_matches_sample_submission": submission[ID_COL].equals(sample[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(CLASSES)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())

    result = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Average OOF/test probabilities from several stacker seeds.",
        "hypothesis": "Averaging stacker seeds can reduce meta-split variance and create a more robust submission candidate than choosing one seed arbitrarily.",
        "official_metric": "balanced_accuracy",
        "source_exp_ids": base_exp_ids,
        "raw_cv_balanced_accuracy": raw["balanced_accuracy"],
        "raw_cv_accuracy": raw["accuracy"],
        "cv_balanced_accuracy": threshold["balanced_accuracy"],
        "cv_accuracy": threshold["accuracy"],
        "multipliers": threshold["multipliers"],
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Medium: source predictions are OOF, but class multipliers are selected on OOF labels.",
        "overfitting_risk": "Medium: seed averaging reduces split variance, but multiplier search can still overfit.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

stacker seedを任意に1つ選ばず、`{base_exp_ids}` のOOF/test probabilitiesを等重み平均して、より安定したsubmission candidateを作る。

## Result

- Raw CV balanced_accuracy: {raw["balanced_accuracy"]:.6f}
- Thresholded CV balanced_accuracy: {threshold["balanced_accuracy"]:.6f}
- CV accuracy: {threshold["accuracy"]:.6f}
- Multipliers: {threshold["multipliers"]}
- Submission checks: {checks}

## Interpretation

`exp038` 単体より低い場合でも、seed平均によりmeta split任意性は下がる。API提出は、現行bestとの比較、raw CV、confirmation結果、submission limitを合わせて判断する。

## Risks

- Leakage risk: medium。sourceはOOFだが、multipliersはOOF labelsで選択。
- Overfitting risk: medium。seed averagingでsplit varianceは減るが、multiplier searchの過適合は残る。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Average OOF/test probabilities from multiple experiments.")
    parser.add_argument("--exp-id", default="exp044_seed_average_stacker")
    parser.add_argument("--source-exp", action="append", required=True)
    args = parser.parse_args()
    run_average(args.exp_id, args.source_exp)


if __name__ == "__main__":
    main()
