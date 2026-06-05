from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp import ID_COL, RAW_DIR, TARGET, add_features, validate_submission


def run_catboost_gpu(exp_id: str, n_splits: int = 3, iterations: int = 600) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train = add_features(pd.read_csv(RAW_DIR / "train.csv"))
    test = add_features(pd.read_csv(RAW_DIR / "test.csv"))
    sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET])
    classes = encoder.classes_
    features = [col for col in train.columns if col not in [ID_COL, TARGET]]
    cat_features = [col for col in features if train[col].dtype == "object" or pd.api.types.is_string_dtype(train[col])]

    x_train = train[features].copy()
    x_test = test[features].copy()
    for col in cat_features:
        x_train[col] = x_train[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(train), len(classes)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float32)
    fold_results: list[dict] = []
    feature_importance = pd.DataFrame({"feature": features})

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": iterations,
        "learning_rate": 0.08,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "task_type": "GPU",
        "devices": "0",
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "od_type": "Iter",
        "od_wait": 50,
        "verbose": 100,
        "allow_writing_files": False,
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        model = CatBoostClassifier(**{**params, "random_seed": 42 + fold})
        train_pool = Pool(x_train.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        valid_pool = Pool(x_train.iloc[va_idx], y[va_idx], cat_features=cat_features)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_proba = model.predict_proba(valid_pool)
        test_pool = Pool(x_test, cat_features=cat_features)
        fold_test_proba = model.predict_proba(test_pool)
        oof_proba[va_idx] = va_proba
        test_proba += fold_test_proba.astype(np.float32) / n_splits

        va_pred = va_proba.argmax(axis=1)
        fold_result = {
            "fold": fold,
            "best_iteration": int(model.get_best_iteration() or iterations),
            "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], va_pred)),
            "accuracy": float(accuracy_score(y[va_idx], va_pred)),
        }
        fold_results.append(fold_result)
        feature_importance[f"fold_{fold}"] = model.get_feature_importance()
        print(
            f"fold={fold} "
            f"balanced_accuracy={fold_result['balanced_accuracy']:.6f} "
            f"accuracy={fold_result['accuracy']:.6f} "
            f"best_iteration={fold_result['best_iteration']}"
        )

    oof_pred = oof_proba.argmax(axis=1)
    cv_balanced_accuracy = float(balanced_accuracy_score(y, oof_pred))
    cv_accuracy = float(accuracy_score(y, oof_pred))

    oof = pd.DataFrame(
        {
            ID_COL: train[ID_COL].to_numpy(),
            "fold": 0,
            "true_class": train[TARGET].to_numpy(),
            "pred_class": encoder.inverse_transform(oof_pred),
        }
    )
    for fold, (_, va_idx) in enumerate(skf.split(x_train, y), start=1):
        oof.loc[va_idx, "fold"] = fold
    for idx, cls in enumerate(classes):
        oof[f"proba_{cls}"] = oof_proba[:, idx]
    oof.to_csv(exp_dir / "oof.csv", index=False)

    test_proba_df = pd.DataFrame({ID_COL: test[ID_COL].to_numpy()})
    for idx, cls in enumerate(classes):
        test_proba_df[f"proba_{cls}"] = test_proba[:, idx]
    test_proba_df.to_csv(exp_dir / "test_proba.csv", index=False)

    submission = sample_submission.copy()
    submission[TARGET] = encoder.inverse_transform(test_proba.argmax(axis=1))
    submission.to_csv(exp_dir / "submission.csv", index=False)
    feature_importance["mean"] = feature_importance[[c for c in feature_importance.columns if c.startswith("fold_")]].mean(axis=1)
    feature_importance.sort_values("mean", ascending=False).to_csv(exp_dir / "feature_importance.csv", index=False)

    submission_checks = validate_submission(submission, sample_submission, test, classes)
    result = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Run a CatBoost GPU diagnostic model for model diversity and future ensembling.",
        "official_metric": "balanced_accuracy",
        "model": "CatBoostClassifier",
        "task_type": "GPU",
        "n_splits": n_splits,
        "params": params,
        "cv_balanced_accuracy": cv_balanced_accuracy,
        "cv_accuracy": cv_accuracy,
        "fold_results": fold_results,
        "classes": classes.tolist(),
        "features": features,
        "categorical_features": cat_features,
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": submission_checks,
        "leakage_risk": "Low-medium: no target-derived features; same validation split style as baseline.",
        "overfitting_risk": "Medium: 3-fold diagnostic is less stable than 5-fold and should be expanded if promising.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

GPU CatBoost で LightGBM と異なる model family の OOF を作り、ensemble 素材として有効か確認する。

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- n_splits: {n_splits}
- iterations: {iterations}
- Submission checks: {submission_checks}

## Risks

- Leakage risk: target-derived feature は使っていない。
- Overfitting risk: 3-fold diagnostic なので fold stability は弱い。良ければ 5-fold に拡張する。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")

    if not submission_checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {submission_checks}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CatBoost GPU diagnostic experiment.")
    parser.add_argument("--exp-id", default="exp010_catboost_gpu_basic")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()
    run_catboost_gpu(args.exp_id, n_splits=args.n_splits, iterations=args.iterations)


if __name__ == "__main__":
    main()
