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
SDSS_PATH = Path("data/external/sdss17/star_classification.csv")
CLASSES = ["GALAXY", "QSO", "STAR"]
BANDS = ["u", "g", "r", "i", "z"]
RAW_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]


def reconstruct_original_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type"] = pd.cut(
        out["r"] - out["g"],
        [-np.inf, -1.0, -0.5, 0.0, np.inf],
        labels=["M", "G/K", "A/F", "O/B"],
    ).astype(str)
    out["galaxy_population"] = pd.cut(
        out["u"] - out["r"],
        [-np.inf, 2.2, np.inf],
        labels=["Blue_Cloud", "Red_Sequence"],
    ).astype(str)
    out["spectral_x_population"] = out["spectral_type"] + "_" + out["galaxy_population"]
    return out


def add_color_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RAW_NUM_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    for left, right in itertools.combinations(BANDS, 2):
        out[f"{left}_{right}"] = (out[left] - out[right]).astype("float32")
    mag = out[BANDS].to_numpy(dtype=np.float32)
    out["mag_mean"] = np.nanmean(mag, axis=1).astype("float32")
    out["mag_std"] = np.nanstd(mag, axis=1).astype("float32")
    out["mag_min"] = np.nanmin(mag, axis=1).astype("float32")
    out["mag_max"] = np.nanmax(mag, axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")
    out["redshift_abs"] = out["redshift"].abs().astype("float32")
    out["redshift_sq"] = (out["redshift"] ** 2).astype("float32")
    out["redshift_signed_log1p"] = np.sign(out["redshift"]) * np.log1p(out["redshift_abs"])
    for color in ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z"]:
        if color in out.columns:
            out[f"redshift_x_{color}"] = (out["redshift"] * out[color]).astype("float32")
    return out


def add_prior_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    original_x: pd.DataFrame,
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_out = train_x.copy()
    test_out = test_x.copy()
    features: list[str] = []
    global_prior = original_x[TARGET].value_counts(normalize=True).reindex(CLASSES, fill_value=0.0)
    for col in group_cols:
        grouped = (
            original_x.groupby(col, dropna=False)[TARGET]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reindex(columns=CLASSES, fill_value=0.0)
        )
        support = original_x.groupby(col, dropna=False)[TARGET].size()
        for cls in CLASSES:
            name = f"sdss_prior_{col}_{cls}"
            train_out[name] = train_out[col].map(grouped[cls]).fillna(global_prior[cls]).astype("float32")
            test_out[name] = test_out[col].map(grouped[cls]).fillna(global_prior[cls]).astype("float32")
            features.append(name)
        support_name = f"sdss_support_{col}_log1p"
        train_out[support_name] = np.log1p(train_out[col].map(support).fillna(0)).astype("float32")
        test_out[support_name] = np.log1p(test_out[col].map(support).fillna(0)).astype("float32")
        features.append(support_name)
    return train_out, test_out, features


def add_prototype_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    original_x: pd.DataFrame,
    proto_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_out = train_x.copy()
    test_out = test_x.copy()
    original_values = original_x[proto_cols].astype("float64")
    means = original_values.mean(axis=0)
    stds = original_values.std(axis=0).replace(0, 1.0).fillna(1.0)
    original_scaled = (original_values - means) / stds
    centroids = {
        cls: original_scaled.loc[original_x[TARGET].to_numpy() == cls].mean(axis=0).to_numpy(dtype=np.float64)
        for cls in CLASSES
    }
    names: list[str] = []
    for frame, out in [(train_x, train_out), (test_x, test_out)]:
        values = ((frame[proto_cols].astype("float64") - means) / stds).to_numpy(dtype=np.float64)
        distances = []
        for cls in CLASSES:
            name = f"sdss_proto_dist_{cls}"
            dist = np.sqrt(np.nanmean((values - centroids[cls].reshape(1, -1)) ** 2, axis=1))
            out[name] = dist.astype("float32")
            distances.append(dist)
            if name not in names:
                names.append(name)
        dist_arr = np.vstack(distances).T
        out["sdss_proto_margin_dist"] = (
            np.partition(dist_arr, kth=1, axis=1)[:, 1] - np.partition(dist_arr, kth=0, axis=1)[:, 0]
        ).astype("float32")
    names.append("sdss_proto_margin_dist")
    return train_out, test_out, names


def build_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    original_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train = add_color_features(reconstruct_original_categories(train_raw))
    test = add_color_features(reconstruct_original_categories(test_raw))
    original = add_color_features(reconstruct_original_categories(original_raw))
    original[TARGET] = original_raw[TARGET].to_numpy()

    group_cols = ["spectral_type", "galaxy_population", "spectral_x_population"]
    train, test, prior_features = add_prior_features(train, test, original, group_cols)
    proto_cols = ["u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z", "u_r", "mag_mean", "mag_std"]
    train, test, proto_features = add_prototype_features(train, test, original, proto_cols)

    categorical_features = group_cols
    for col in categorical_features:
        categories = pd.Index(pd.concat([train[col], test[col]], axis=0).dropna().astype(str).unique())
        train[col] = pd.Categorical(train[col].astype(str), categories=categories)
        test[col] = pd.Categorical(test[col].astype(str), categories=categories)

    drop_cols = [ID_COL, TARGET]
    features = [col for col in train.columns if col not in drop_cols]
    features = list(dict.fromkeys(features + prior_features + proto_features))
    return train, test, features, categorical_features


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> dict:
    checks = {
        "columns_match_sample_submission": list(submission.columns) == list(sample_submission.columns),
        "row_count_matches_sample_submission": len(submission) == len(sample_submission),
        "id_matches_sample_submission": submission[ID_COL].equals(sample_submission[ID_COL]),
        "labels_are_valid_strings": set(submission[TARGET].unique()).issubset(set(CLASSES)),
        "no_missing_values": int(submission.isna().sum().sum()) == 0,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def run_experiment(exp_id: str, n_splits: int) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(RAW_DIR / "train.csv")
    test_raw = pd.read_csv(RAW_DIR / "test.csv")
    sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")
    original_raw = pd.read_csv(SDSS_PATH)
    train, test, features, categorical_features = build_features(train_raw, test_raw, original_raw)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET])
    x_train = train[features]
    x_test = test[features]
    params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "multi_logloss",
        "n_estimators": 950,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_child_samples": 34,
        "subsample": 0.86,
        "colsample_bytree": 0.84,
        "class_weight": "balanced",
        "reg_alpha": 0.04,
        "reg_lambda": 0.45,
        "max_bin": 255,
        "force_col_wise": True,
        "random_state": 314,
        "n_jobs": -1,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=314)
    oof_proba = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASSES)), dtype=np.float32)
    fold_results: list[dict] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        print(f"===== fold {fold}/{n_splits} =====")
        model = lgb.LGBMClassifier(**{**params, "random_state": 314 + fold})
        model.fit(
            x_train.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(90), lgb.log_evaluation(100)],
        )
        va_proba = model.predict_proba(x_train.iloc[va_idx], num_iteration=model.best_iteration_)
        fold_test_proba = model.predict_proba(x_test, num_iteration=model.best_iteration_)
        oof_proba[va_idx] = va_proba.astype(np.float32)
        test_proba += fold_test_proba.astype(np.float32) / n_splits
        va_pred = va_proba.argmax(axis=1)
        fold_result = {
            "fold": fold,
            "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
            "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], va_pred)),
            "accuracy": float(accuracy_score(y[va_idx], va_pred)),
        }
        fold_results.append(fold_result)
        print(json.dumps(fold_result, ensure_ascii=False))

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
    for idx, cls in enumerate(CLASSES):
        oof[f"proba_{cls}"] = oof_proba[:, idx]
    oof.to_csv(exp_dir / "oof.csv", index=False)

    test_proba_df = pd.DataFrame({ID_COL: test[ID_COL].to_numpy()})
    for idx, cls in enumerate(CLASSES):
        test_proba_df[f"proba_{cls}"] = test_proba[:, idx]
    test_proba_df.to_csv(exp_dir / "test_proba.csv", index=False)

    submission = sample_submission.copy()
    submission[TARGET] = encoder.inverse_transform(test_proba.argmax(axis=1))
    submission.to_csv(exp_dir / "submission.csv", index=False)
    checks = validate_submission(submission, sample_submission)

    result = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Train a LightGBM base with only stable SDSS17 prior/prototype features.",
        "hypothesis": "Keeping SDSS17 signal to coarse original priors and prototype distances can avoid the noisy bin/count overfit seen in exp031 while preserving PB-oriented external distribution signal.",
        "official_metric": "balanced_accuracy",
        "n_splits": n_splits,
        "params": params,
        "cv_balanced_accuracy": cv_balanced_accuracy,
        "cv_accuracy": cv_accuracy,
        "fold_results": fold_results,
        "features": features,
        "categorical_features": categorical_features,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "original_rows": int(len(original_raw)),
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Medium-low: SDSS17 labels are used only as coarse aggregate priors and class prototypes; no direct row matching, no test label transfer.",
        "overfitting_risk": "Medium: external aggregate features can still overfit CV, so stack improvement and margin decide submission.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

`exp031` で弱かった広いbin/count型SDSS特徴を削り、coarse prior と prototype距離だけをLightGBMに追加する。

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- n_splits: {n_splits}
- Feature count: {len(features)}
- Categorical feature count: {len(categorical_features)}
- Submission checks: {checks}

## Interpretation

単体CVとstack CVで採否を判断する。`exp032` の 0.966223 を明確に超えない場合、API提出しない。

## Risks

- Leakage risk: medium-low。SDSS17 labelsは粗いaggregate prior/prototypeのみに使い、直接照合やtest label transferはしていない。
- Overfitting risk: medium。外部由来特徴のCV過適合は残るため、改善幅を厳しく見る。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stable SDSS prior/prototype LightGBM experiment.")
    parser.add_argument("--exp-id", default="exp033_lgbm_sdss_stable")
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()
    run_experiment(args.exp_id, args.n_splits)


if __name__ == "__main__":
    main()
