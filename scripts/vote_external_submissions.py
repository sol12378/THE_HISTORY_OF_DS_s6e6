from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


TARGET = "class"
ID_COL = "id"
CLASSES = ["GALAXY", "QSO", "STAR"]

NOTEBOOK_DDF = [
    "0.96652.csv",
    "0.96707.csv",
    "0.96722.csv",
    "0.96751.csv",
    "0.96771.csv",
    "0.96807.csv",
    "0.96889.csv",
    "0.96918.csv",
    "0.96920.csv",
    "0.96933.csv",
    "0.96973.csv",
    "0.96983.csv",
    "0.97007.csv",
    "0.96936.csv",
    "0.96755.csv",
    "0.96977.csv",
    "0.96986.csv",
    "0.97015.csv",
    "0.97021.csv",
    "0.97033.csv",
    "0.97034.csv",
    "0.97038.csv",
    "0.97044.csv",
    "0.97047.csv",
    "0.97061.csv",
    "0.97070.csv",
    "0.97076.csv",
    "0.96771.b.csv",
    "0.97037.csv",
    "0.97009.csv",
    "0.97062.csv",
    "0.97071.csv",
    "0.97075.csv",
    "0.97075.b.csv",
    "0.97055.csv",
    "0.97058.csv",
    "0.97064.csv",
    "0.97074.csv",
    "0.97051.csv",
    "0.97076.b.csv",
    "0.97073.csv",
    "0.97065.csv",
    "0.97074.b.csv",
    "0.97079.csv",
]


def parse_score(path: Path) -> float:
    match = re.match(r"(\d+\.\d+)", path.name)
    if not match:
        raise ValueError(f"cannot parse score from {path.name}")
    return float(match.group(1))


def load_predictions(files: list[Path]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for path in files:
        col = path.stem.replace(".", "_")
        sub = pd.read_csv(path).rename(columns={TARGET: col})
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on=ID_COL, how="inner")
    if merged is None:
        raise ValueError("no files to load")
    return merged


def majority_vote(labels: list[str], fallback: str) -> str:
    counts = Counter(labels)
    max_count = max(counts.values())
    winners = {label for label, count in counts.items() if count == max_count}
    if fallback in winners:
        return fallback
    for cls in CLASSES:
        if cls in winners:
            return cls
    raise RuntimeError(f"unexpected labels: {labels}")


def weighted_vote(labels: list[str], weights: list[float], fallback: str) -> str:
    scores = {cls: 0.0 for cls in CLASSES}
    for label, weight in zip(labels, weights):
        scores[label] += weight
    max_score = max(scores.values())
    winners = {label for label, score in scores.items() if score == max_score}
    if fallback in winners:
        return fallback
    return max(CLASSES, key=lambda cls: (scores[cls], cls))


def create_submission(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "notebook_combo":
        indices = [int(x) for x in args.indices.split(",")]
        names = [NOTEBOOK_DDF[idx] for idx in indices]
        files = [input_dir / name for name in names]
        mode_description = f"notebook indices {indices}"
    elif args.mode == "top_weighted":
        all_files = sorted(input_dir.glob("*.csv"), key=parse_score, reverse=True)
        files = all_files[: args.top_n]
        names = [path.name for path in files]
        mode_description = f"top {args.top_n} weighted by filename LB score"
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing files: {missing}")

    preds = load_predictions(files)
    pred_cols = [path.stem.replace(".", "_") for path in files]
    fallback_col = pred_cols[-1]
    weights = [parse_score(path) for path in files]

    if args.mode == "notebook_combo":
        preds["vote"] = [
            majority_vote(list(row), fallback=row[-1])
            for row in preds[pred_cols].itertuples(index=False, name=None)
        ]
    else:
        preds["vote"] = [
            weighted_vote(list(row), weights=weights, fallback=row[-1])
            for row in preds[pred_cols].itertuples(index=False, name=None)
        ]

    sample = pd.read_csv("data/raw/sample_submission.csv")
    submission = sample.copy()
    submission[TARGET] = preds["vote"].to_numpy()
    submission_path = exp_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample),
        "id_matches_sample_submission": submission[ID_COL].equals(sample[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(CLASSES)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")

    agreement_all = (preds[pred_cols].nunique(axis=1) == 1).value_counts().to_dict()
    result = {
        "exp_id": args.exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Use nina2025/ps-s6e6-simple-vote external public submissions and improve voting robustness.",
        "mode": args.mode,
        "mode_description": mode_description,
        "source_notebook": "https://www.kaggle.com/code/nina2025/ps-s6e6-simple-vote",
        "source_dataset": "nina2025/ps-s6e6",
        "files": names,
        "weights": {path.name: parse_score(path) for path in files},
        "agreement_all": {str(key): int(value) for key, value in agreement_all.items()},
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "High: this blends public submissions and is not a model CV result. It can improve LB but does not prove local validation quality.",
        "overfitting_risk": "High: source files are selected by public LB-style filenames and public notebook experiments.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {args.exp_id}

## Purpose

指定 notebook `nina2025/ps-s6e6-simple-vote` の simple vote をこの workspace で再現し、submission を作成する。

## Result

- Mode: `{args.mode}`
- Files: {names}
- Agreement all: {result["agreement_all"]}
- Submission distribution: {result["submission_distribution"]}
- Submission checks: {checks}
- LB: pending

## Risks

- Leakage risk: high。public submissions の blend であり、local CV 改善ではない。
- Overfitting risk: high。Public LB score 由来の選択に依存する。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Vote external public submissions.")
    parser.add_argument("--input-dir", default="data/external/nina2025_ps_s6e6")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--mode", choices=["notebook_combo", "top_weighted"], required=True)
    parser.add_argument("--indices", default="9,19,23,24,43")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    print(json.dumps(create_submission(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
