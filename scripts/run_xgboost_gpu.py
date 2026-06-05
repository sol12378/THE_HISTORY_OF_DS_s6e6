from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


TARGET = "class"
ID_COL = "id"
RAW_DIR = Path("data/raw")
CONFIG_DIR = Path("src/stellar_class/config")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    return out


def add_color_redshift_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_features(df)
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


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, feature_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if feature_mode == "basic":
        train = add_features(train)
        test = add_features(test)
    elif feature_mode == "color_redshift":
        train = add_color_redshift_features(train)
        test = add_color_redshift_features(test)
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")

    features = [col for col in train.columns if col not in [ID_COL, TARGET]]
    for col in features:
        if pd.api.types.is_object_dtype(train[col]) or pd.api.types.is_string_dtype(train[col]):
            categories = pd.Index(pd.concat([train[col], test[col]], axis=0).dropna().unique())
            mapping = {value: idx for idx, value in enumerate(categories)}
            train[col] = train[col].map(mapping).fillna(-1).astype(np.int16)
            test[col] = test[col].map(mapping).fillna(-1).astype(np.int16)
    return train, test, features


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    weights = len(y) / (len(counts) * counts)
    return weights[y].astype(np.float32)


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


def load_config(exp_id: str) -> dict:
    config_path = CONFIG_DIR / f"{exp_id}.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def run_experiment(exp_id: str) -> dict:
    config = load_config(exp_id)
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")

    feature_mode = config.get("feature_mode", "color_redshift")
    train, test, features = prepare_features(train, test, feature_mode)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET])
    classes = encoder.classes_
    x_train = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)

    n_splits = int(config.get("n_splits", 3))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config.get("fold_seed", 42)))
    oof_proba = np.zeros((len(train), len(classes)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float32)
    fold_results = []

    params = {
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "eta": 0.04,
        "max_depth": 8,
        "min_child_weight": 8.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 2.0,
        "alpha": 0.05,
        "max_bin": 256,
        "seed": 42,
    }
    params.update(config.get("xgboost_params", {}))
    num_boost_round = int(config.get("num_boost_round", 1200))
    early_stopping_rounds = int(config.get("early_stopping_rounds", 80))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        fold_params = dict(params)
        fold_params["seed"] = int(params.get("seed", 42)) + fold
        dtrain = xgb.DMatrix(x_train[tr_idx], label=y[tr_idx], weight=class_balanced_weights(y[tr_idx]))
        dvalid = xgb.DMatrix(x_train[va_idx], label=y[va_idx])
        dtest = xgb.DMatrix(x_test)

        booster = xgb.train(
            fold_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
        )
        best_iteration = int(booster.best_iteration if booster.best_iteration is not None else num_boost_round - 1)
        iteration_range = (0, best_iteration + 1)
        va_proba = booster.predict(dvalid, iteration_range=iteration_range)
        fold_test_proba = booster.predict(dtest, iteration_range=iteration_range)
        oof_proba[va_idx] = va_proba.astype(np.float32)
        test_proba += fold_test_proba.astype(np.float32) / n_splits

        va_pred = va_proba.argmax(axis=1)
        fold_result = {
            "fold": fold,
            "best_iteration": best_iteration,
            "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], va_pred)),
            "accuracy": float(accuracy_score(y[va_idx], va_pred)),
        }
        fold_results.append(fold_result)
        print(
            f"fold={fold} balanced_accuracy={fold_result['balanced_accuracy']:.6f} "
            f"accuracy={fold_result['accuracy']:.6f} best_iteration={best_iteration}"
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

    submission_checks = validate_submission(submission, sample_submission, classes)
    result = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "GPU XGBoostでLightGBM/CatBoostと異なるOOF予測を作り、自前OOF stack候補を増やす。",
        "hypothesis": "hist GPU XGBoostにclass-balanced sample weightsを使うと、LightGBMと異なる誤差を持つモデルとしてstack/blendでCVを改善できる。",
        "official_metric": "balanced_accuracy",
        "feature_mode": feature_mode,
        "n_splits": n_splits,
        "xgboost_params": params,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "cv_balanced_accuracy": cv_balanced_accuracy,
        "cv_accuracy": cv_accuracy,
        "fold_results": fold_results,
        "classes": classes.tolist(),
        "features": features,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": submission_checks,
        "leakage_risk": "Low: official train/test features only; no external labels or target-derived encodings.",
        "overfitting_risk": "Medium: 3-fold diagnostic with GPU XGBoost and class-balanced weights; stack value must be judged by OOF blend, not standalone LB.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {exp_id}

## Purpose

GPU XGBoostでLightGBM/CatBoostと異なるOOF予測を作り、自前OOF stackの候補を増やす。

## Hypothesis

`hist` + `device=cuda` のXGBoostにclass-balanced sample weightsを使うことで、単体CVまたはblend/stackの多様性が改善する。

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- Feature mode: `{feature_mode}`
- Fold count: {n_splits}
- Submission checks: {submission_checks}

## Leakage Risk

Low。公式train/test特徴量のみを使用し、外部ラベルやtarget encodingは使っていない。

## Overfitting Risk

Medium。3-fold高速診断なのでfold varianceが残る。単体LBではなく、OOF blend/stackで価値を判断する。

## Next Actions

- `exp009_lgbm_color_redshift_small` とOOF probability blendを試す。
- 単体CVが弱くても、誤差相関が低ければstack候補として残す。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not submission_checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {submission_checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GPU XGBoost Stellar Class experiment.")
    parser.add_argument("exp_id", help="Experiment id, e.g. exp016_xgboost_gpu_basic")
    args = parser.parse_args()
    run_experiment(args.exp_id)


if __name__ == "__main__":
    main()
