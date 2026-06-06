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


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> dict:
    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample),
        "id_matches_sample_submission": submission[ID_COL].equals(sample[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(CLASSES)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def run(exp_id: str, source_exp: str, multipliers: dict[str, float]) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path("experiments") / source_exp

    oof = pd.read_csv(source_dir / "oof.csv")
    test_proba = pd.read_csv(source_dir / "test_proba.csv")
    sample = pd.read_csv("data/raw/sample_submission.csv")
    mult = np.array([multipliers[cls] for cls in CLASSES], dtype=np.float64)
    y = oof["true_class"].map({cls: idx for idx, cls in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    oof_proba = oof[PROBA_COLS].to_numpy(dtype=np.float64)
    pred_idx = (oof_proba * mult.reshape(1, -1)).argmax(axis=1)

    out_oof = oof[[ID_COL, "true_class"]].copy()
    out_oof["pred_class"] = np.array(CLASSES, dtype=object)[pred_idx]
    for col in PROBA_COLS:
        out_oof[col] = oof[col].to_numpy(dtype=np.float32)
    out_oof.to_csv(exp_dir / "oof.csv", index=False)

    out_test = test_proba[[ID_COL, *PROBA_COLS]].copy()
    out_test.to_csv(exp_dir / "test_proba.csv", index=False)
    test_pred = np.array(CLASSES, dtype=object)[
        (test_proba[PROBA_COLS].to_numpy(dtype=np.float64) * mult.reshape(1, -1)).argmax(axis=1)
    ]
    submission = sample.copy()
    submission[TARGET] = test_pred
    submission.to_csv(exp_dir / "submission.csv", index=False)
    checks = validate_submission(submission, sample)

    result = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Apply fixed class multipliers to an existing OOF/test probability experiment.",
        "source_exp": source_exp,
        "official_metric": "balanced_accuracy",
        "multipliers": multipliers,
        "cv_balanced_accuracy": float(balanced_accuracy_score(y, pred_idx)),
        "cv_accuracy": float(accuracy_score(y, pred_idx)),
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Medium-low: no model refit; fixed multipliers are evaluated on OOF labels.",
        "overfitting_risk": "Medium: class multipliers can overfit OOF label distribution.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

`{source_exp}` のOOF/test probabilitiesに固定class multipliersを適用し、提出候補のthreshold依存度を確認する。

## Result

- CV balanced_accuracy: {result["cv_balanced_accuracy"]:.6f}
- CV accuracy: {result["cv_accuracy"]:.6f}
- Multipliers: {multipliers}
- Submission checks: {checks}

## Interpretation

倍率を強くしすぎずにCVが維持できるかを見る診断。API提出は、現行bestと安定性確認を合わせて判断する。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply fixed class multipliers to an experiment.")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--source-exp", required=True)
    parser.add_argument("--galaxy", type=float, default=1.0)
    parser.add_argument("--qso", type=float, default=1.0)
    parser.add_argument("--star", type=float, default=1.0)
    args = parser.parse_args()
    run(
        args.exp_id,
        args.source_exp,
        {"GALAXY": args.galaxy, "QSO": args.qso, "STAR": args.star},
    )


if __name__ == "__main__":
    main()
