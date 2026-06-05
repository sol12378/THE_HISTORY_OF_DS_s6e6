from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET = "class"
ID_COL = "id"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a threshold-adjusted submission from test probabilities.")
    parser.add_argument("--source-exp", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--galaxy", type=float, default=1.0)
    parser.add_argument("--qso", type=float, default=1.22)
    parser.add_argument("--star", type=float, default=1.30)
    args = parser.parse_args()

    source_dir = Path("experiments") / args.source_exp
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    test_proba = pd.read_csv(source_dir / "test_proba.csv")
    sample_submission = pd.read_csv("data/raw/sample_submission.csv")
    multipliers = np.array([args.galaxy, args.qso, args.star], dtype=np.float64)
    proba = test_proba[[f"proba_{cls}" for cls in CLASSES]].to_numpy(dtype=np.float64)
    pred = np.array(CLASSES, dtype=object)[(proba * multipliers.reshape(1, -1)).argmax(axis=1)]

    submission = sample_submission.copy()
    submission[TARGET] = pred
    submission_path = exp_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample_submission.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample_submission),
        "id_matches_sample_submission": submission[ID_COL].equals(sample_submission[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(CLASSES)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")

    result = {
        "exp_id": args.exp_id,
        "source_exp": args.source_exp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Apply OOF-optimized class multipliers to saved test probabilities and create a Kaggle submission.",
        "official_metric": "balanced_accuracy",
        "multipliers": {cls: float(mult) for cls, mult in zip(CLASSES, multipliers)},
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Medium: multipliers were selected on exp001 OOF labels, then applied to test probabilities.",
        "overfitting_risk": "Medium-high: expected LB benefit may be smaller than OOF gain because threshold tuning used OOF labels.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {args.exp_id}

## Purpose

`{args.source_exp}` の `test_proba.csv` に OOF-optimized class multipliers を適用し、Kaggle submission を作成する。

## Hypothesis

`exp003_oof_threshold_search` で CV balanced_accuracy が改善した multiplier を test probabilities に適用すれば、Public LB でも `exp001_baseline` より改善する可能性がある。

## Result

- Multipliers: {result["multipliers"]}
- Submission distribution: {result["submission_distribution"]}
- Submission checks: {checks}
- LB: pending

## Risks

- Leakage risk: target-derived training feature はないが、multiplier は OOF label で選んでいる。
- Overfitting risk: medium-high。LB で CV/LB gap を確認する。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
