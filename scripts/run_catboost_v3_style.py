from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
BASE_CATS = ["spectral_type", "galaxy_population"]
ORIGINAL_WEIGHT = 0.06


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
    out["blue_slope"] = ((out["g"] - out["u"]) + (out["r"] - out["g"])).astype("float32")
    out["red_slope"] = ((out["i"] - out["r"]) + (out["z"] - out["i"])).astype("float32")
    out["curv_ugr"] = (out["u"] - 2.0 * out["g"] + out["r"]).astype("float32")
    out["curv_gri"] = (out["g"] - 2.0 * out["r"] + out["i"]).astype("float32")
    out["curv_riz"] = (out["r"] - 2.0 * out["i"] + out["z"]).astype("float32")
    out["redshift_abs"] = out["redshift"].abs().astype("float32")
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"]).astype("float32")
    out["redshift_sq"] = (out["redshift"] ** 2).astype("float32")
    out["redshift_neg"] = (out["redshift"] < 0).astype("int8")
    out["redshift_low"] = (out["redshift"] < 0.02).astype("int8")
    out["sky_radius"] = np.sqrt(out["alpha"] ** 2 + out["delta"] ** 2).astype("float32")

    alpha_rad = np.deg2rad(out["alpha"].astype("float64"))
    delta_rad = np.deg2rad(out["delta"].astype("float64"))
    out["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    out["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    out["delta_sin"] = np.sin(delta_rad).astype("float32")
    out["delta_cos"] = np.cos(delta_rad).astype("float32")
    return out


def add_quantile_bins(all_df: pd.DataFrame, source_cols: list[str], bins_list: list[int]) -> tuple[pd.DataFrame, list[str]]:
    out = all_df.copy()
    cat_cols: list[str] = []
    for col in source_cols:
        values = out[col].astype("float64")
        for bins in bins_list:
            name = f"{col}_q{bins}"
            try:
                out[name] = pd.qcut(values, q=bins, labels=False, duplicates="drop").fillna(-1).astype("int32")
            except ValueError:
                out[name] = -1
            cat_cols.append(name)
    return out, cat_cols


def add_rounded_bins(df: pd.DataFrame, source_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cat_cols: list[str] = []
    for col in source_cols:
        for decimals in [1, 2]:
            name = f"{col}_r{decimals}"
            out[name] = np.rint(out[col].astype("float64") * (10**decimals)).fillna(-999999).astype("int32")
            cat_cols.append(name)
    return out, cat_cols


def add_interaction_cats(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cat_cols: list[str] = []
    for left, right in pairs:
        name = f"{left}_x_{right}"
        out[name] = out[left].astype(str) + "_" + out[right].astype(str)
        cat_cols.append(name)
    return out, cat_cols


def build_features(train: pd.DataFrame, test: pd.DataFrame, original: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    original = reconstruct_original_categories(original)
    original[ID_COL] = -1 - np.arange(len(original), dtype=np.int32)

    train_core = train[[ID_COL, *RAW_NUM_COLS, *BASE_CATS, TARGET]].copy()
    test_core = test[[ID_COL, *RAW_NUM_COLS, *BASE_CATS]].copy()
    original_core = original[[ID_COL, *RAW_NUM_COLS, *BASE_CATS, TARGET]].copy()

    train_core["_source"] = "train"
    test_core["_source"] = "test"
    original_core["_source"] = "original"
    all_df = pd.concat(
        [train_core.drop(columns=[TARGET]), test_core, original_core.drop(columns=[TARGET])],
        axis=0,
        ignore_index=True,
    )
    all_df = add_numeric_features(all_df)

    q_sources = [
        "alpha",
        "delta",
        "redshift",
        "u_g",
        "g_r",
        "r_i",
        "i_z",
        "u_r",
        "mag_mean",
        "mag_std",
        "mag_range",
    ]
    all_df, q_cat_cols = add_quantile_bins(all_df, q_sources, [16, 64])
    all_df, r_cat_cols = add_rounded_bins(all_df, ["redshift", "u_g", "g_r", "r_i", "i_z", "u_r", "mag_mean"])

    for col in BASE_CATS:
        all_df[col] = all_df[col].astype(str).fillna("__NA__")
    base_cat_cols = BASE_CATS + ["redshift_neg", "redshift_low"]
    interaction_pairs = [
        ("spectral_type", "galaxy_population"),
        ("redshift_q64", "spectral_type"),
        ("redshift_q64", "galaxy_population"),
        ("u_g_q64", "g_r_q64"),
        ("g_r_q64", "r_i_q64"),
        ("r_i_q64", "i_z_q64"),
        ("alpha_q16", "delta_q16"),
    ]
    all_df, int_cat_cols = add_interaction_cats(all_df, interaction_pairs)
    cat_cols = list(dict.fromkeys(base_cat_cols + q_cat_cols + r_cat_cols + int_cat_cols))
    for col in cat_cols:
        all_df[col] = all_df[col].astype(str).fillna("__NA__")

    n_train = len(train)
    n_test = len(test)
    train_x = all_df.iloc[:n_train].reset_index(drop=True)
    test_x = all_df.iloc[n_train : n_train + n_test].reset_index(drop=True)
    original_x = all_df.iloc[n_train + n_test :].reset_index(drop=True)
    train_x[TARGET] = train[TARGET].to_numpy()
    original_x[TARGET] = original_core[TARGET].to_numpy()

    drop_cols = [ID_COL, "_source", TARGET]
    features = [col for col in train_x.columns if col not in drop_cols]
    cat_cols = [col for col in cat_cols if col in features]
    return train_x, test_x, original_x, features, cat_cols


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


def run_experiment(exp_id: str, n_splits: int, iterations: int, original_weight: float) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(RAW_DIR / "train.csv")
    test_raw = pd.read_csv(RAW_DIR / "test.csv")
    sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")
    original_raw = pd.read_csv(SDSS_PATH)
    train, test, original, features, cat_features = build_features(train_raw, test_raw, original_raw)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET])
    y_original = encoder.transform(original[TARGET])

    x_train = train[features].copy()
    x_test = test[features].copy()
    x_original = original[features].copy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASSES)), dtype=np.float32)
    fold_results: list[dict] = []

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "iterations": iterations,
        "learning_rate": 0.045,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.2,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.2,
        "one_hot_max_size": 16,
        "max_ctr_complexity": 3,
        "class_weights": [1.0, 3.25, 5.0],
        "border_count": 254,
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.85,
        "od_type": "Iter",
        "od_wait": 180,
        "allow_writing_files": False,
        "verbose": 200,
        "thread_count": 4,
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        print(f"===== fold {fold}/{n_splits} =====")
        x_fit = pd.concat([x_train.iloc[tr_idx], x_original], axis=0, ignore_index=True)
        y_fit = np.concatenate([y[tr_idx], y_original]).astype(np.int8)
        weights = np.ones(len(y_fit), dtype=np.float32)
        weights[len(tr_idx) :] = np.float32(original_weight)

        train_pool = Pool(x_fit, y_fit, cat_features=cat_features, weight=weights)
        valid_pool = Pool(x_train.iloc[va_idx], y[va_idx], cat_features=cat_features)
        test_pool = Pool(x_test, cat_features=cat_features)

        model = CatBoostClassifier(**{**params, "random_seed": 42 + fold})
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_proba = model.predict_proba(valid_pool).astype(np.float32)
        fold_test_proba = model.predict_proba(test_pool).astype(np.float32)
        oof_proba[va_idx] = va_proba
        test_proba += fold_test_proba / n_splits

        va_pred = va_proba.argmax(axis=1)
        fold_result = {
            "fold": fold,
            "best_iteration": int(model.get_best_iteration() or iterations),
            "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], va_pred)),
            "accuracy": float(accuracy_score(y[va_idx], va_pred)),
            "fit_rows": int(len(y_fit)),
        }
        fold_results.append(fold_result)
        print(json.dumps(fold_result, ensure_ascii=False))

        del model, train_pool, valid_pool, test_pool, x_fit, y_fit, weights, va_proba, fold_test_proba

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
        "purpose": "CatBoost v3-style native categorical diagnostic with low-weight SDSS17 original rows.",
        "hypothesis": "Binned/rounded/interaction categorical views plus low-weight original rows can create CatBoost diversity and improve the OOF stack.",
        "official_metric": "balanced_accuracy",
        "n_splits": n_splits,
        "iterations": iterations,
        "original_weight": original_weight,
        "params": params,
        "cv_balanced_accuracy": cv_balanced_accuracy,
        "cv_accuracy": cv_accuracy,
        "fold_results": fold_results,
        "features": features,
        "categorical_features": cat_features,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "original_shape": list(original.shape),
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Medium: SDSS17 original labels are appended with low sample weight, but validation rows are never appended to their own fold. Direct test label transfer is not used.",
        "overfitting_risk": "Medium-high: many categorical views and external rows can overfit; judge by OOF stack improvement and LB only if confident.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

CatBoost v3風に、量子化bin・丸めbin・カテゴリinteraction・SDSS17 original低weight appendを使うCatBoost GPU baseを作る。

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- n_splits: {n_splits}
- iterations: {iterations}
- original_weight: {original_weight}
- Feature count: {len(features)}
- Categorical feature count: {len(cat_features)}
- Submission checks: {checks}

## Interpretation

単体提出ではなく、`exp024` stackに追加してCV改善するかで採否を判断する。

## Risks

- Leakage risk: medium。SDSS17 original labelを低weightでfold trainingにappendするが、validation rowsは学習に入らない。direct label transferは使わない。
- Overfitting risk: medium-high。カテゴリviewが多く、外部データを使うため、OOF stackとLBの相関を確認する必要がある。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")

    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CatBoost v3-style diagnostic experiment.")
    parser.add_argument("--exp-id", default="exp029_catboost_v3_style")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--original-weight", type=float, default=ORIGINAL_WEIGHT)
    args = parser.parse_args()
    run_experiment(args.exp_id, args.n_splits, args.iterations, args.original_weight)


if __name__ == "__main__":
    main()
