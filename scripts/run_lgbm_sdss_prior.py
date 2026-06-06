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
RAW_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BANDS = ["u", "g", "r", "i", "z"]


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


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RAW_NUM_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    for left, right in itertools.combinations(BANDS, 2):
        color = out[left] - out[right]
        out[f"{left}_{right}"] = color.astype("float32")
        out[f"{left}_{right}_abs"] = color.abs().astype("float32")
        out[f"redshift_x_{left}_{right}"] = (out["redshift"] * color).astype("float32")

    band = out[BANDS].to_numpy(dtype=np.float32)
    out["mag_mean"] = np.nanmean(band, axis=1).astype("float32")
    out["mag_std"] = np.nanstd(band, axis=1).astype("float32")
    out["mag_min"] = np.nanmin(band, axis=1).astype("float32")
    out["mag_max"] = np.nanmax(band, axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")
    out["curv_ugr"] = (out["u"] - 2.0 * out["g"] + out["r"]).astype("float32")
    out["curv_gri"] = (out["g"] - 2.0 * out["r"] + out["i"]).astype("float32")
    out["curv_riz"] = (out["r"] - 2.0 * out["i"] + out["z"]).astype("float32")
    out["redshift_abs"] = out["redshift"].abs().astype("float32")
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"]).astype("float32")
    out["redshift_sq"] = (out["redshift"] ** 2).astype("float32")
    out["redshift_neg"] = (out["redshift"] < 0).astype("int8")
    out["redshift_low"] = (out["redshift"] < 0.02).astype("int8")
    alpha_rad = np.deg2rad(out["alpha"].astype("float64"))
    delta_rad = np.deg2rad(out["delta"].astype("float64"))
    out["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    out["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    out["delta_sin"] = np.sin(delta_rad).astype("float32")
    out["delta_cos"] = np.cos(delta_rad).astype("float32")
    return out


def add_quantile_bins(all_df: pd.DataFrame, cols: list[str], bins_list: list[int]) -> tuple[pd.DataFrame, list[str]]:
    out = all_df.copy()
    bin_cols: list[str] = []
    for col in cols:
        values = out[col].astype("float64")
        for bins in bins_list:
            name = f"{col}_q{bins}"
            try:
                out[name] = pd.qcut(values, q=bins, labels=False, duplicates="drop").fillna(-1).astype("int16")
            except ValueError:
                out[name] = np.int16(-1)
            bin_cols.append(name)
    return out, bin_cols


def add_count_features(all_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = all_df.copy()
    for col in group_cols:
        counts = out[col].astype(str).value_counts(dropna=False)
        out[f"{col}_count_all"] = out[col].astype(str).map(counts).fillna(0).astype("float32")
        out[f"{col}_log_count_all"] = np.log1p(out[f"{col}_count_all"]).astype("float32")
    return out


def add_sdss_prior_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    original_x: pd.DataFrame,
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_out = train_x.copy()
    test_out = test_x.copy()
    prior_features: list[str] = []
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
            mapping = grouped[cls]
            train_out[name] = train_out[col].map(mapping).fillna(global_prior[cls]).astype("float32")
            test_out[name] = test_out[col].map(mapping).fillna(global_prior[cls]).astype("float32")
            prior_features.append(name)
        support_name = f"sdss_prior_{col}_support_log1p"
        train_out[support_name] = np.log1p(train_out[col].map(support).fillna(0)).astype("float32")
        test_out[support_name] = np.log1p(test_out[col].map(support).fillna(0)).astype("float32")
        prior_features.append(support_name)
    return train_out, test_out, prior_features


def add_sdss_prototype_features(
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
    feature_names: list[str] = []
    for frame, out in [(train_x, train_out), (test_x, test_out)]:
        values = ((frame[proto_cols].astype("float64") - means) / stds).to_numpy(dtype=np.float64)
        distances = []
        for cls in CLASSES:
            dist = np.sqrt(np.nanmean((values - centroids[cls].reshape(1, -1)) ** 2, axis=1))
            out[f"sdss_proto_dist_{cls}"] = dist.astype("float32")
            if f"sdss_proto_dist_{cls}" not in feature_names:
                feature_names.append(f"sdss_proto_dist_{cls}")
            distances.append(dist)
        dist_arr = np.vstack(distances).T
        out["sdss_proto_min_dist"] = np.nanmin(dist_arr, axis=1).astype("float32")
        out["sdss_proto_margin_dist"] = (
            np.partition(dist_arr, kth=1, axis=1)[:, 1] - np.partition(dist_arr, kth=0, axis=1)[:, 0]
        ).astype("float32")
    feature_names.extend(["sdss_proto_min_dist", "sdss_proto_margin_dist"])
    return train_out, test_out, feature_names


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, original_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_core = reconstruct_original_categories(train_raw)
    test_core = reconstruct_original_categories(test_raw)
    original_core = reconstruct_original_categories(original_raw)
    original_core[ID_COL] = -1 - np.arange(len(original_core), dtype=np.int32)

    train_core["_source"] = "train"
    test_core["_source"] = "test"
    original_core["_source"] = "original"
    all_df = pd.concat(
        [train_core.drop(columns=[TARGET]), test_core, original_core.drop(columns=[TARGET])],
        axis=0,
        ignore_index=True,
    )
    all_df = add_numeric_features(all_df)

    q_sources = ["alpha", "delta", "redshift", "u_g", "g_r", "r_i", "i_z", "u_r", "mag_mean", "mag_std"]
    all_df, q_cols = add_quantile_bins(all_df, q_sources, [16, 64])
    group_cols = [
        "spectral_type",
        "galaxy_population",
        "spectral_x_population",
        "redshift_q16",
        "redshift_q64",
        "u_g_q64",
        "g_r_q64",
        "r_i_q64",
        "i_z_q64",
    ]
    for col in [*group_cols, *q_cols]:
        all_df[col] = all_df[col].astype(str).fillna("__NA__")
    all_df = add_count_features(all_df, group_cols)

    n_train = len(train_raw)
    n_test = len(test_raw)
    train_x = all_df.iloc[:n_train].reset_index(drop=True)
    test_x = all_df.iloc[n_train : n_train + n_test].reset_index(drop=True)
    original_x = all_df.iloc[n_train + n_test :].reset_index(drop=True)
    train_x[TARGET] = train_raw[TARGET].to_numpy()
    original_x[TARGET] = original_core[TARGET].to_numpy()

    train_x, test_x, prior_features = add_sdss_prior_features(train_x, test_x, original_x, group_cols)
    proto_cols = [
        "alpha",
        "delta",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
        "u_g",
        "g_r",
        "r_i",
        "i_z",
        "u_r",
        "mag_mean",
        "mag_std",
        "mag_range",
        "redshift_abs",
        "redshift_sq",
    ]
    train_x, test_x, proto_features = add_sdss_prototype_features(train_x, test_x, original_x, proto_cols)

    categorical_features = ["spectral_type", "galaxy_population", "spectral_x_population", *q_cols]
    for col in categorical_features:
        categories = pd.Index(pd.concat([train_x[col], test_x[col]], axis=0).dropna().astype(str).unique())
        train_x[col] = pd.Categorical(train_x[col].astype(str), categories=categories)
        test_x[col] = pd.Categorical(test_x[col].astype(str), categories=categories)

    drop_cols = [ID_COL, TARGET, "_source"]
    features = [col for col in train_x.columns if col not in drop_cols]
    features = list(dict.fromkeys(features + prior_features + proto_features))
    categorical_features = [col for col in categorical_features if col in features]
    return train_x, test_x, features, categorical_features


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
        "n_estimators": 1200,
        "learning_rate": 0.045,
        "num_leaves": 95,
        "min_child_samples": 28,
        "subsample": 0.88,
        "colsample_bytree": 0.82,
        "class_weight": "balanced",
        "reg_alpha": 0.05,
        "reg_lambda": 0.6,
        "max_bin": 255,
        "force_col_wise": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASSES)), dtype=np.float32)
    fold_results: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        print(f"===== fold {fold}/{n_splits} =====")
        model = lgb.LGBMClassifier(**{**params, "random_state": 42 + fold})
        model.fit(
            x_train.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
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
        "purpose": "Train a fast LightGBM base with SDSS17 class prior, prototype distance, and count features.",
        "hypothesis": "External SDSS17 labels can improve PB-oriented generalization when used as aggregate priors and class prototypes, without direct row matching or label transfer.",
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
        "leakage_risk": "Medium-low: SDSS17 labels are used only for aggregate class priors and class prototypes; no direct train/test row label transfer and no validation labels are encoded.",
        "overfitting_risk": "Medium: many prior/bin features can overfit OOF; accept only if standalone or stack CV improves clearly.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {exp_id}

## Purpose

SDSS17外部元データを直接照合・ラベル転写せず、class prior、class prototype距離、train/test/original count特徴としてLightGBMに入れる。

## Hypothesis

外部ラベルを集約統計として使えば、PBに近い分布知識を取り込みつつ、直接リークより低リスクでself-model stackの多様性を増やせる。

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- n_splits: {n_splits}
- Feature count: {len(features)}
- Categorical feature count: {len(categorical_features)}
- Original rows: {len(original_raw)}
- Submission checks: {checks}

## Interpretation

単体CVと次段stack CVで採否を決める。`exp030_oof_stacker_with_cat_v3` の CV 0.966222 を明確に超えない場合はAPI提出しない。

## Risks

- Leakage risk: medium-low。SDSS17 labelsはaggregate prior/prototypeのみに使用し、test rowへの直接ラベル転写はしていない。
- Overfitting risk: medium。bin/prior特徴がOOFに過適合する可能性があるため、stack改善幅とLB提出基準を分けて判断する。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")

    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM with SDSS17 prior/prototype features.")
    parser.add_argument("--exp-id", default="exp031_lgbm_sdss_prior_fast")
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()
    run_experiment(args.exp_id, args.n_splits)


if __name__ == "__main__":
    main()
