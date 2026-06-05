from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


TARGET = "class"
ID_COL = "id"
RAW_DIR = Path("data/raw")
CONFIG_DIR = Path("src/stellar_class/config")


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    return out


def add_color_redshift_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_basic_features(df)
    bands = ["u", "g", "r", "i", "z"]
    for left, right in itertools.combinations(bands, 2):
        out[f"{left}_{right}"] = out[left] - out[right]

    mag = out[bands]
    out["mag_mean"] = mag.mean(axis=1)
    out["mag_std"] = mag.std(axis=1)
    out["mag_min"] = mag.min(axis=1)
    out["mag_max"] = mag.max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_sq"] = out["redshift"] ** 2
    out["redshift_signed_log1p"] = np.sign(out["redshift"]) * np.log1p(out["redshift"].abs())
    out["redshift_is_negative"] = (out["redshift"] < 0).astype(np.int8)
    for color in ["u_g", "g_r", "r_i", "i_z", "u_z", "g_z", "r_z"]:
        if color in out.columns:
            out[f"redshift_x_{color}"] = out["redshift"] * out[color]
    return out


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, feature_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if feature_mode == "basic":
        train = add_basic_features(train)
        test = add_basic_features(test)
    elif feature_mode == "color_redshift":
        train = add_color_redshift_features(train)
        test = add_color_redshift_features(test)
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")

    features = [col for col in train.columns if col not in [ID_COL, TARGET]]
    categorical_features: list[str] = []
    for col in features:
        train_is_text = pd.api.types.is_object_dtype(train[col]) or pd.api.types.is_string_dtype(train[col])
        test_is_text = pd.api.types.is_object_dtype(test[col]) or pd.api.types.is_string_dtype(test[col])
        if train_is_text or test_is_text:
            categories = pd.Index(pd.concat([train[col], test[col]], axis=0).dropna().unique())
            train[col] = pd.Categorical(train[col], categories=categories)
            test[col] = pd.Categorical(test[col], categories=categories)
            categorical_features.append(col)
    return train, test, features, categorical_features


def load_config(exp_id: str) -> dict:
    config_path = CONFIG_DIR / f"{exp_id}.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame, classes: np.ndarray) -> dict:
    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample_submission.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample_submission),
        "id_matches_sample_submission": submission[ID_COL].equals(sample_submission[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(classes)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def run_experiment(exp_id: str) -> dict:
    config = load_config(exp_id)
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")

    feature_mode = config.get("feature_mode", "basic")
    train, test, features, categorical_features = prepare_features(train, test, feature_mode)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET])
    classes = encoder.classes_
    x_train = train[features]
    x_test = test[features]

    n_splits = int(config.get("n_splits", 3))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config.get("fold_seed", 42)))
    oof_proba = np.zeros((len(train), len(classes)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float32)
    fold_results = []

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "n_estimators": 900,
        "learning_rate": 0.06,
        "num_leaves": 63,
        "min_child_samples": 35,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "class_weight": "balanced",
        "reg_alpha": 0.03,
        "reg_lambda": 0.3,
        "max_bin": 255,
        "force_col_wise": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(config.get("lightgbm_params", {}))
    early_stopping_rounds = int(config.get("early_stopping_rounds", 80))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        fold_params = dict(params)
        fold_params["random_state"] = int(params.get("random_state", 42)) + fold
        model = lgb.LGBMClassifier(**fold_params)
        model.fit(
            x_train.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)],
        )
        va_proba = model.predict_proba(x_train.iloc[va_idx], num_iteration=model.best_iteration_)
        fold_test_proba = model.predict_proba(x_test, num_iteration=model.best_iteration_)
        oof_proba[va_idx] = va_proba.astype(np.float32)
        test_proba += fold_test_proba.astype(np.float32) / n_splits

        va_pred = va_proba.argmax(axis=1)
        fold_result = {
            "fold": fold,
            "best_iteration": int(model.best_iteration_ or fold_params["n_estimators"]),
            "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], va_pred)),
            "accuracy": float(accuracy_score(y[va_idx], va_pred)),
        }
        fold_results.append(fold_result)
        print(
            f"fold={fold} balanced_accuracy={fold_result['balanced_accuracy']:.6f} "
            f"accuracy={fold_result['accuracy']:.6f} best_iteration={fold_result['best_iteration']}"
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
    checks = validate_submission(submission, sample_submission, classes)

    result = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Create a fast LightGBM OOF/test_proba base for the self-model stacker.",
        "hypothesis": "A basic-feature LightGBM base with a different fold seed can add diversity to the current OOF stacker and improve trusted CV.",
        "official_metric": "balanced_accuracy",
        "feature_mode": feature_mode,
        "n_splits": n_splits,
        "lightgbm_params": params,
        "early_stopping_rounds": early_stopping_rounds,
        "cv_balanced_accuracy": cv_balanced_accuracy,
        "cv_accuracy": cv_accuracy,
        "fold_results": fold_results,
        "classes": classes.tolist(),
        "features": features,
        "categorical_features": categorical_features,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Low: official train/test features only; no external labels or target-derived encodings.",
        "overfitting_risk": "Medium: fast 3-fold LightGBM diagnostic; stack value should be judged by OOF stack improvement.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {exp_id}

## Purpose

自前OOF stackerに追加するため、LightGBMの高速OOF/test_proba baseを作る。

## Hypothesis

`basic` featureのLightGBMを別fold seedで作ることで、既存の`exp009`とは異なるLightGBM系予測をstackへ追加できる。

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- Feature mode: `{feature_mode}`
- Fold count: {n_splits}
- Submission checks: {checks}

## Interpretation

単体提出はしない。`exp020` stackに追加して、CV 0.965901を超えるかで採否を判断する。

## Risks

- Leakage risk: low。公式特徴量のみ。
- Overfitting risk: medium。3-fold高速診断なのでfold varianceが残る。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fast LightGBM OOF/test_proba experiment.")
    parser.add_argument("exp_id")
    args = parser.parse_args()
    run_experiment(args.exp_id)


if __name__ == "__main__":
    main()
