from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


def make_stack_features(exp_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    base = None
    test_base = None
    for exp_id in exp_ids:
        oof = read_oof(exp_id)
        test = read_test_proba(exp_id)
        if base is None:
            base = oof[[ID_COL, "true_class"]].copy()
            test_base = test[[ID_COL]].copy()
        else:
            base = base.merge(oof[[ID_COL, "true_class"]], on=[ID_COL, "true_class"], validate="one_to_one")
            test_base = test_base.merge(test[[ID_COL]], on=ID_COL, validate="one_to_one")

        train_proba = oof[PROBA_COLS].clip(1e-6, 1 - 1e-6)
        test_proba = test[PROBA_COLS].clip(1e-6, 1 - 1e-6)
        train_part = train_proba.copy()
        test_part = test_proba.copy()
        train_part.columns = [f"{exp_id}_{col}" for col in PROBA_COLS]
        test_part.columns = [f"{exp_id}_{col}" for col in PROBA_COLS]
        train_logit = np.log(train_proba.to_numpy() / (1.0 - train_proba.to_numpy()))
        test_logit = np.log(test_proba.to_numpy() / (1.0 - test_proba.to_numpy()))
        for idx, cls in enumerate(CLASSES):
            train_part[f"{exp_id}_logit_{cls}"] = train_logit[:, idx]
            test_part[f"{exp_id}_logit_{cls}"] = test_logit[:, idx]
        train_parts.append(train_part.reset_index(drop=True))
        test_parts.append(test_part.reset_index(drop=True))

    assert base is not None and test_base is not None
    train_x = pd.concat([base.reset_index(drop=True), *train_parts], axis=1)
    test_x = pd.concat([test_base.reset_index(drop=True), *test_parts], axis=1)
    return train_x, test_x


def evaluate_with_multipliers(y_idx: np.ndarray, proba: np.ndarray, multipliers: np.ndarray) -> dict:
    pred_idx = (proba * multipliers.reshape(1, -1)).argmax(axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_idx, pred_idx)),
        "accuracy": float(accuracy_score(y_idx, pred_idx)),
        "multipliers": {cls: float(mult) for cls, mult in zip(CLASSES, multipliers)},
    }


def search_multipliers(y_idx: np.ndarray, proba: np.ndarray) -> dict:
    best = None
    for qso_scale in np.linspace(1.0, 1.45, 10):
        for star_scale in np.linspace(1.0, 1.45, 10):
            candidate = evaluate_with_multipliers(y_idx, proba, np.array([1.0, qso_scale, star_scale]))
            if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
                best = candidate
    assert best is not None
    return best


def run_stack(exp_id: str, base_exp_ids: list[str]) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    train_x, test_x = make_stack_features(base_exp_ids)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train_x["true_class"])
    feature_cols = [col for col in train_x.columns if col not in [ID_COL, "true_class"]]
    x = train_x[feature_cols].to_numpy(dtype=np.float64)
    x_test = test_x[feature_cols].to_numpy(dtype=np.float64)

    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    candidates = []
    for c_value in c_grid:
        oof_proba = np.zeros((len(train_x), len(CLASSES)), dtype=np.float32)
        fold_results = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x[tr_idx])
            x_va = scaler.transform(x[va_idx])
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=100 + fold,
            )
            model.fit(x_tr, y[tr_idx])
            va_proba = model.predict_proba(x_va)
            oof_proba[va_idx] = va_proba.astype(np.float32)
            fold_results.append(
                {
                    "fold": fold,
                    "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], va_proba.argmax(axis=1))),
                    "accuracy": float(accuracy_score(y[va_idx], va_proba.argmax(axis=1))),
                }
            )
        raw = evaluate_with_multipliers(y, oof_proba, np.ones(len(CLASSES)))
        best_threshold = search_multipliers(y, oof_proba)
        candidates.append(
            {
                "C": c_value,
                "raw": raw,
                "best_threshold": best_threshold,
                "fold_results": fold_results,
                "oof_proba": oof_proba,
            }
        )

    best = max(candidates, key=lambda row: row["best_threshold"]["balanced_accuracy"])
    best_c = best["C"]
    best_oof_proba = best["oof_proba"]
    multipliers = np.array([best["best_threshold"]["multipliers"][cls] for cls in CLASSES], dtype=np.float64)
    oof_pred_idx = (best_oof_proba * multipliers.reshape(1, -1)).argmax(axis=1)

    final_scaler = StandardScaler()
    x_scaled = final_scaler.fit_transform(x)
    x_test_scaled = final_scaler.transform(x_test)
    final_model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=777,
    )
    final_model.fit(x_scaled, y)
    test_proba = final_model.predict_proba(x_test_scaled)
    test_pred = np.array(CLASSES, dtype=object)[(test_proba * multipliers.reshape(1, -1)).argmax(axis=1)]

    oof = train_x[[ID_COL, "true_class"]].copy()
    oof["pred_class"] = np.array(CLASSES, dtype=object)[oof_pred_idx]
    for idx, cls in enumerate(CLASSES):
        oof[f"proba_{cls}"] = best_oof_proba[:, idx]
    oof.to_csv(exp_dir / "oof.csv", index=False)

    test_proba_df = test_x[[ID_COL]].copy()
    for idx, cls in enumerate(CLASSES):
        test_proba_df[f"proba_{cls}"] = test_proba[:, idx]
    test_proba_df.to_csv(exp_dir / "test_proba.csv", index=False)

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
        "purpose": "Train a nested-CV logistic OOF stacker over available self-model probabilities.",
        "hypothesis": "A meta-model over LightGBM, CatBoost, and GPU XGBoost OOF probabilities can learn class-specific blending better than a fixed two-model grid.",
        "official_metric": "balanced_accuracy",
        "base_exp_ids": base_exp_ids,
        "feature_count": len(feature_cols),
        "best_C": best_c,
        "raw_cv_balanced_accuracy": best["raw"]["balanced_accuracy"],
        "cv_balanced_accuracy": best["best_threshold"]["balanced_accuracy"],
        "cv_accuracy": best["best_threshold"]["accuracy"],
        "multipliers": best["best_threshold"]["multipliers"],
        "candidate_summary": [
            {
                "C": row["C"],
                "raw_cv_balanced_accuracy": row["raw"]["balanced_accuracy"],
                "threshold_cv_balanced_accuracy": row["best_threshold"]["balanced_accuracy"],
                "threshold_cv_accuracy": row["best_threshold"]["accuracy"],
                "multipliers": row["best_threshold"]["multipliers"],
            }
            for row in candidates
        ],
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Medium: base predictions are OOF, but meta-model hyperparameters and class multipliers are selected on OOF labels.",
        "overfitting_risk": "Medium-high: nested CV reduces direct stacker leakage, but multiplier search can still overfit.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

利用可能な自前モデルOOF probabilityを使い、LogisticRegressionのmeta-modelでOOF stackを試す。

## Result

- Base experiments: {base_exp_ids}
- Best C: {best_c}
- Raw CV balanced_accuracy: {best["raw"]["balanced_accuracy"]:.6f}
- Thresholded CV balanced_accuracy: {best["best_threshold"]["balanced_accuracy"]:.6f}
- CV accuracy: {best["best_threshold"]["accuracy"]:.6f}
- Multipliers: {best["best_threshold"]["multipliers"]}
- Submission checks: {checks}

## Interpretation

`exp017_lgbm_xgboost_blend` のCV 0.965235を超えるかどうかで採否を判断する。超えない場合も、stacker実装は次のモデル追加時に再利用できる。

## Risks

- Leakage risk: medium。baseはOOFだが、meta hyperparameterとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。PB狙いではfold別の安定性とLB相関を確認する。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a logistic OOF stacker.")
    parser.add_argument("--exp-id", default="exp018_oof_logistic_stacker")
    parser.add_argument(
        "--base-exp",
        action="append",
        required=True,
        help="Base experiment id with both oof.csv and test_proba.csv. Repeat this option.",
    )
    args = parser.parse_args()
    run_stack(args.exp_id, args.base_exp)


if __name__ == "__main__":
    main()
