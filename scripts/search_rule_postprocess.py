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


def score(y_idx: np.ndarray, pred_idx: np.ndarray) -> tuple[float, float, list[list[int]]]:
    n_classes = len(CLASSES)
    cm = np.bincount(y_idx * n_classes + pred_idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    recalls = np.diag(cm) / cm.sum(axis=1)
    return float(recalls.mean()), float(np.diag(cm).sum() / cm.sum()), cm.tolist()


def make_base_table(source_exp: str) -> pd.DataFrame:
    train = pd.read_csv("data/raw/train.csv")
    oof = pd.read_csv(Path("experiments") / source_exp / "oof.csv")
    return train.merge(oof[[ID_COL, "pred_class"]], on=ID_COL)


def candidate_masks(df: pd.DataFrame) -> list[dict]:
    thresholds_low = np.round(np.linspace(-0.01, 0.35, 73), 6)
    thresholds_high = np.round(np.linspace(0.0, 2.5, 101), 6)
    category_filters: list[tuple[str | None, str | None]] = [(None, None)]
    category_filters += [("spectral_type", value) for value in sorted(df["spectral_type"].dropna().unique())]
    category_filters += [("galaxy_population", value) for value in sorted(df["galaxy_population"].dropna().unique())]

    rules = []
    transforms = [
        ("GALAXY", "STAR", "lt", thresholds_low),
        ("GALAXY", "QSO", "gt", thresholds_high),
        ("STAR", "GALAXY", "gt", thresholds_low),
        ("STAR", "QSO", "gt", thresholds_high),
        ("QSO", "GALAXY", "lt", thresholds_high),
        ("QSO", "STAR", "lt", thresholds_low),
    ]
    for from_class, to_class, direction, thresholds in transforms:
        from_mask = df["pred_class"].eq(from_class).to_numpy()
        for cat_col, cat_val in category_filters:
            if cat_col is None:
                cat_mask = np.ones(len(df), dtype=bool)
            else:
                cat_mask = df[cat_col].eq(cat_val).to_numpy()
            base_mask = from_mask & cat_mask
            if not base_mask.any():
                continue
            for threshold in thresholds:
                if direction == "lt":
                    mask = base_mask & (df["redshift"].to_numpy() < threshold)
                else:
                    mask = base_mask & (df["redshift"].to_numpy() > threshold)
                if mask.sum() == 0:
                    continue
                rules.append(
                    {
                        "from_class": from_class,
                        "to_class": to_class,
                        "direction": direction,
                        "threshold": float(threshold),
                        "category_column": cat_col,
                        "category_value": cat_val,
                        "mask": mask,
                        "affected": int(mask.sum()),
                    }
                )
    return rules


def serializable_rule(rule: dict) -> dict:
    return {key: value for key, value in rule.items() if key != "mask"}


def apply_rules(df: pd.DataFrame, pred_idx: np.ndarray, rules: list[dict]) -> np.ndarray:
    adjusted = pred_idx.copy()
    redshift = df["redshift"].to_numpy()
    for rule in rules:
        mask = df["pred_class"].eq(rule["from_class"]).to_numpy().copy()
        if rule["category_column"] is not None:
            mask &= df[rule["category_column"]].eq(rule["category_value"]).to_numpy()
        if rule["direction"] == "lt":
            mask &= redshift < rule["threshold"]
        else:
            mask &= redshift > rule["threshold"]
        adjusted[mask] = CLASS_TO_IDX[rule["to_class"]]
    return adjusted


def create_submission(source_exp: str, exp_dir: Path, rules: list[dict]) -> dict:
    test = pd.read_csv("data/raw/test.csv")
    base_submission = pd.read_csv(Path("experiments") / source_exp / "submission.csv")
    sample_submission = pd.read_csv("data/raw/sample_submission.csv")
    df = test.merge(base_submission, on=ID_COL).rename(columns={TARGET: "pred_class"})
    pred_idx = df["pred_class"].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)
    adjusted_idx = apply_rules(df, pred_idx, rules)
    submission = sample_submission.copy()
    submission[TARGET] = np.array(CLASSES, dtype=object)[adjusted_idx]
    submission.to_csv(exp_dir / "submission.csv", index=False)
    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample_submission.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample_submission),
        "id_matches_sample_submission": submission[ID_COL].equals(sample_submission[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(CLASSES)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    return {
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search simple rule-based postprocessing on OOF hard predictions.")
    parser.add_argument("--source-exp", default="exp001_baseline")
    parser.add_argument("--exp-id", default="exp006_rule_postprocess")
    parser.add_argument("--max-rules", type=int, default=3)
    args = parser.parse_args()

    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    df = make_base_table(args.source_exp)
    y_idx = df[TARGET].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)
    base_pred_idx = df["pred_class"].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)
    baseline_bal, baseline_acc, baseline_cm = score(y_idx, base_pred_idx)

    selected_rules: list[dict] = []
    current_pred_idx = base_pred_idx.copy()
    current_bal = baseline_bal
    current_acc = baseline_acc
    available_rules = candidate_masks(df)
    history = []

    for step in range(1, args.max_rules + 1):
        best_rule = None
        best_pred_idx = current_pred_idx
        best_bal = current_bal
        best_acc = current_acc
        best_cm = None
        for rule in available_rules:
            trial_pred_idx = current_pred_idx.copy()
            trial_pred_idx[rule["mask"]] = CLASS_TO_IDX[rule["to_class"]]
            bal, acc, cm = score(y_idx, trial_pred_idx)
            if bal > best_bal:
                best_rule = rule
                best_pred_idx = trial_pred_idx
                best_bal = bal
                best_acc = acc
                best_cm = cm
        if best_rule is None:
            break
        selected_rules.append(serializable_rule(best_rule))
        current_pred_idx = best_pred_idx
        current_bal = best_bal
        current_acc = best_acc
        history.append(
            {
                "step": step,
                "rule": serializable_rule(best_rule),
                "balanced_accuracy": best_bal,
                "accuracy": best_acc,
                "confusion_matrix": best_cm,
            }
        )

    final_bal, final_acc, final_cm = score(y_idx, current_pred_idx)
    adjusted = df[[ID_COL, TARGET]].copy()
    adjusted["pred_class"] = np.array(CLASSES, dtype=object)[current_pred_idx]
    adjusted.to_csv(exp_dir / "oof.csv", index=False)
    submission_info = create_submission(args.source_exp, exp_dir, selected_rules)

    result = {
        "exp_id": args.exp_id,
        "source_exp": args.source_exp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Search simple redshift/category rules on exp001 hard predictions to improve balanced accuracy.",
        "official_metric": "balanced_accuracy",
        "baseline": {
            "balanced_accuracy": baseline_bal,
            "accuracy": baseline_acc,
            "confusion_matrix": baseline_cm,
        },
        "best": {
            "balanced_accuracy": final_bal,
            "accuracy": final_acc,
            "confusion_matrix": final_cm,
            "rules": selected_rules,
        },
        "history": history,
        "delta_balanced_accuracy": final_bal - baseline_bal,
        "delta_accuracy": final_acc - baseline_acc,
        **submission_info,
        "leakage_risk": "Medium: rules are selected on OOF labels and raw features, then applied to test.",
        "overfitting_risk": "Medium-high: greedy rules may overfit OOF idiosyncrasies; LB confirmation is required.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {args.exp_id}

## Purpose

`exp001_baseline` の hard predictions に redshift/category based rules を適用し、公式 metric の `balanced_accuracy` が改善するか確認する。

## Result

- Baseline balanced_accuracy: {baseline_bal:.6f}
- Adjusted balanced_accuracy: {final_bal:.6f}
- Delta balanced_accuracy: {final_bal - baseline_bal:+.6f}
- Baseline accuracy: {baseline_acc:.6f}
- Adjusted accuracy: {final_acc:.6f}
- Delta accuracy: {final_acc - baseline_acc:+.6f}
- Rules: {selected_rules}
- Submission checks: {submission_info["submission_checks"]}

## Risks

- Leakage risk: target-derived training feature はないが、rule selection は OOF labels を使っている。
- Overfitting risk: medium-high。LB が改善しない場合は採用しない。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
