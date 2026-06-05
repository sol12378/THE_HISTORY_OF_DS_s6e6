from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


TARGET = "class"
ID_COL = "id"
RAW_DIR = Path("data/raw")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    return out


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_features(df)
    bands = ["u", "g", "r", "i", "z"]

    for left, right in itertools.combinations(bands, 2):
        color = f"{left}_{right}"
        out[color] = out[left] - out[right]
        out[f"flux_ratio_{left}_{right}"] = np.power(10.0, -0.4 * out[color])
        out[f"redshift_x_{color}"] = out["redshift"] * out[color]

    mag = out[bands]
    out["mag_mean"] = mag.mean(axis=1)
    out["mag_std"] = mag.std(axis=1)
    out["mag_min"] = mag.min(axis=1)
    out["mag_max"] = mag.max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["mag_sum"] = mag.sum(axis=1)

    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_sq"] = out["redshift"] ** 2
    out["redshift_signed_log1p"] = np.sign(out["redshift"]) * np.log1p(out["redshift"].abs())
    out["redshift_x_mag_mean"] = out["redshift"] * out["mag_mean"]
    out["redshift_x_mag_range"] = out["redshift"] * out["mag_range"]

    alpha_rad = np.deg2rad(out["alpha"])
    delta_rad = np.deg2rad(out["delta"])
    out["alpha_sin"] = np.sin(alpha_rad)
    out["alpha_cos"] = np.cos(alpha_rad)
    out["delta_sin"] = np.sin(delta_rad)
    out["delta_cos"] = np.cos(delta_rad)
    out["sky_x"] = np.cos(delta_rad) * np.cos(alpha_rad)
    out["sky_y"] = np.cos(delta_rad) * np.sin(alpha_rad)
    out["sky_z"] = np.sin(delta_rad)

    if "spectral_type" in out.columns and "galaxy_population" in out.columns:
        out["spectral_population"] = out["spectral_type"].astype(str) + "_" + out["galaxy_population"].astype(str)

    return out


def align_categories(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    categorical_features: list[str] = []
    for col in features:
        train_is_text = pd.api.types.is_object_dtype(train[col]) or pd.api.types.is_string_dtype(train[col])
        test_is_text = pd.api.types.is_object_dtype(test[col]) or pd.api.types.is_string_dtype(test[col])
        if train_is_text or test_is_text:
            categories = pd.Index(pd.concat([train[col], test[col]], axis=0).dropna().unique())
            train[col] = pd.Categorical(train[col], categories=categories)
            test[col] = pd.Categorical(test[col], categories=categories)
            categorical_features.append(col)
    return train, test, categorical_features


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame, test: pd.DataFrame, classes: np.ndarray) -> dict:
    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample_submission.columns),
        "row_count_matches_test": len(submission) == len(test),
        "id_matches_sample_submission": submission[ID_COL].equals(sample_submission[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(classes)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def run_lgbm_experiment(
    exp_id: str,
    feature_mode: str,
    purpose: str,
    hypothesis: str,
    model_params: dict | None = None,
    early_stopping_rounds: int = 100,
) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")

    if feature_mode == "basic":
        train = add_features(train)
        test = add_features(test)
    elif feature_mode == "advanced":
        train = add_advanced_features(train)
        test = add_advanced_features(test)
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET])
    classes = encoder.classes_

    drop_cols = [ID_COL, TARGET]
    features = [col for col in train.columns if col not in drop_cols]
    train, test, categorical_features = align_categories(train, test, features)
    x_train = train[features]
    x_test = test[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(train), len(classes)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float32)
    fold_results: list[dict] = []
    feature_importance = pd.DataFrame({"feature": features})

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        x_tr = x_train.iloc[tr_idx]
        x_va = x_train.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        params = {
            "objective": "multiclass",
            "num_class": len(classes),
            "metric": "multi_logloss",
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }
        if model_params:
            params.update(model_params)
        model = lgb.LGBMClassifier(
            **params,
        )
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            eval_metric="multi_logloss",
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)],
        )

        va_proba = model.predict_proba(x_va, num_iteration=model.best_iteration_)
        fold_test_proba = model.predict_proba(x_test, num_iteration=model.best_iteration_)
        oof_proba[va_idx] = va_proba
        test_proba += fold_test_proba.astype(np.float32) / skf.n_splits

        va_pred = va_proba.argmax(axis=1)
        fold_result = {
            "fold": fold,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "balanced_accuracy": float(balanced_accuracy_score(y_va, va_pred)),
            "accuracy": float(accuracy_score(y_va, va_pred)),
        }
        fold_results.append(fold_result)
        feature_importance[f"fold_{fold}"] = model.feature_importances_
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
    submission_distribution = submission[TARGET].value_counts().to_dict()
    result = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": purpose,
        "hypothesis": hypothesis,
        "official_metric": "balanced_accuracy",
        "feature_mode": feature_mode,
        "model_params": params,
        "early_stopping_rounds": early_stopping_rounds,
        "cv_balanced_accuracy": cv_balanced_accuracy,
        "cv_accuracy": cv_accuracy,
        "fold_results": fold_results,
        "classes": classes.tolist(),
        "features": features,
        "categorical_features": categorical_features,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_submission_shape": list(sample_submission.shape),
        "target_distribution": train[TARGET].value_counts().to_dict(),
        "missing_train_total": int(train.isna().sum().sum()),
        "missing_test_total": int(test.isna().sum().sum()),
        "submission_distribution": submission_distribution,
        "submission_checks": submission_checks,
        "leakage_risk": "Low to medium: ID-like columns are included as baseline inputs; no target-derived encoding or test-label leakage is used.",
        "overfitting_risk": "Medium: LightGBM with 2000 estimators uses early stopping; CV/LB gap must be monitored after submission.",
    }

    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

{purpose}

## Hypothesis

{hypothesis}

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- Feature mode: {feature_mode}
- Feature count: {len(features)}
- Train shape: {train.shape}
- Test shape: {test.shape}
- Target distribution: {train[TARGET].value_counts().to_dict()}
- Submission distribution: {submission_distribution}
- Submission checks: {submission_checks}

## Changes

- 5-fold `StratifiedKFold(random_state=42)`。
- LightGBM multiclass + `class_weight=\"balanced\"`。
- Feature mode: `{feature_mode}`。
- `spectral_type` と `galaxy_population` は categorical feature として使用。
- `oof.csv`, `test_proba.csv`, `submission.csv`, `feature_importance.csv`, `result.json` を保存。

## Risks

- Leakage risk: ID-like columns は初回 baseline では残している。target encoding や test label 由来の処理は使っていないため直接 leakage はないが、ID 系が synthetic split に強く効く可能性は次回検証する。
- Overfitting risk: LightGBM 2000 estimators は early stopping 付き。提出後に CV/LB gap を確認し、validation 連動性を検証する。

## Next Actions

- Kaggle に初回提出し、LB と CV gap を記録する。
- 次フェーズでは ID 系 drop 比較、全色指数、redshift interaction を検証する。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")

    if not submission_checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {submission_checks}")

    print(f"OOF balanced_accuracy: {cv_balanced_accuracy:.6f}")
    print(f"OOF accuracy: {cv_accuracy:.6f}")
    print(f"Saved outputs to {exp_dir}")
    return result


def run_exp001(exp_id: str) -> dict:
    return run_lgbm_experiment(
        exp_id=exp_id,
        feature_mode="basic",
        purpose="初回提出用に、公式 metric の `balanced_accuracy` に合わせた LightGBM baseline と再現可能な CV/OOF/submission pipeline を確立する。",
        hypothesis="`redshift`、測光特徴量、`spectral_type` / `galaxy_population`、色指数6個を LightGBM に入れることで、単体 baseline でも `balanced_accuracy` 0.95 前後に到達し、初回提出に十分な形式検証済み `submission.csv` を作れる。",
    )


def run_exp002(exp_id: str) -> dict:
    return run_lgbm_experiment(
        exp_id=exp_id,
        feature_mode="advanced",
        purpose="全色指数、flux ratio、redshift 交互作用、magnitude 統計、sky 座標変換を追加し、LightGBM baseline の CV balanced_accuracy を改善する。",
        hypothesis="baseline で重要だった `alpha`, `delta`, `redshift`, color 系を明示的に拡張することで、GALAXY/STAR/QSO 境界の局所的な取り違えを減らし、CV balanced_accuracy を 0.964030 から引き上げられる。",
        model_params={
            "n_estimators": 900,
            "learning_rate": 0.08,
            "num_leaves": 63,
            "min_child_samples": 30,
            "reg_alpha": 0.05,
            "reg_lambda": 0.5,
            "max_bin": 127,
            "force_col_wise": True,
        },
        early_stopping_rounds=60,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Stellar Class experiment.")
    parser.add_argument("exp_id", help="Experiment id, e.g. exp001_baseline")
    args = parser.parse_args()

    if args.exp_id == "exp001_baseline":
        run_exp001(args.exp_id)
    elif args.exp_id == "exp002_advanced_features":
        run_exp002(args.exp_id)
    else:
        raise ValueError("Implemented experiments: exp001_baseline, exp002_advanced_features")


if __name__ == "__main__":
    main()
