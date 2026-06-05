from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ID_COL = "id"
TARGET = "class"
RAW_DIR = Path("data/raw")
SDSS_PATH = Path("data/external/sdss17/star_classification.csv")
MATCH_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CLASSES = ["GALAXY", "QSO", "STAR"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    external = pd.read_csv(SDSS_PATH)
    missing = [col for col in MATCH_COLS + [TARGET] if col not in external.columns]
    if missing:
        raise ValueError(f"external SDSS17 data is missing required columns: {missing}")
    return train, test, external


def rounded_match(left: pd.DataFrame, external: pd.DataFrame, decimals: int) -> pd.DataFrame:
    key_cols = [f"{col}_round" for col in MATCH_COLS]
    left_keys = left[MATCH_COLS].round(decimals).copy()
    left_keys.columns = key_cols
    ext = external[MATCH_COLS + [TARGET]].round({col: decimals for col in MATCH_COLS}).copy()
    ext = ext.rename(columns={col: f"{col}_round" for col in MATCH_COLS})
    ext_counts = ext.groupby(key_cols, observed=True).agg(
        external_match_count=(TARGET, "size"),
        external_class=(TARGET, "first"),
        external_class_unique=(TARGET, "nunique"),
    ).reset_index()
    matched = pd.concat([left[[ID_COL]].reset_index(drop=True), left_keys.reset_index(drop=True)], axis=1)
    matched = matched.merge(
        ext_counts,
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    return matched[[ID_COL, "external_match_count", "external_class", "external_class_unique"]]


def summarize_rounded(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    decimals_grid: list[int],
) -> list[dict]:
    rows = []
    for decimals in decimals_grid:
        train_match = rounded_match(train, external, decimals)
        test_match = rounded_match(test, external, decimals)

        train_hit = train_match["external_match_count"].notna()
        test_hit = test_match["external_match_count"].notna()
        if train_hit.any():
            train_pred = train_match.loc[train_hit, "external_class"].to_numpy()
            train_true = train.loc[train_hit, TARGET].to_numpy()
            label_accuracy = float((train_pred == train_true).mean())
            label_balanced_accuracy = float(balanced_accuracy_score(train_true, train_pred))
        else:
            label_accuracy = None
            label_balanced_accuracy = None

        rows.append(
            {
                "decimals": decimals,
                "train_matches": int(train_hit.sum()),
                "train_match_rate": float(train_hit.mean()),
                "test_matches": int(test_hit.sum()),
                "test_match_rate": float(test_hit.mean()),
                "train_label_accuracy_on_matches": label_accuracy,
                "train_label_balanced_accuracy_on_matches": label_balanced_accuracy,
                "train_ambiguous_matches": int(
                    train_match.loc[train_hit, "external_class_unique"].fillna(0).gt(1).sum()
                ),
                "test_ambiguous_matches": int(
                    test_match.loc[test_hit, "external_class_unique"].fillna(0).gt(1).sum()
                ),
            }
        )
    return rows


def sample_frame(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df
    if TARGET in df.columns:
        return df.groupby(TARGET, group_keys=False).sample(frac=sample_size / len(df), random_state=seed)
    return df.sample(n=sample_size, random_state=seed)


def nearest_neighbor_diagnostic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    sample_size: int,
) -> dict:
    train_eval = sample_frame(train, sample_size, seed=42).reset_index(drop=True)
    test_eval = sample_frame(test, sample_size, seed=43).reset_index(drop=True)

    scaler = StandardScaler()
    ext_x = external[MATCH_COLS].to_numpy(dtype=np.float64)
    train_x = train_eval[MATCH_COLS].to_numpy(dtype=np.float64)
    test_x = test_eval[MATCH_COLS].to_numpy(dtype=np.float64)
    scaler.fit(np.vstack([ext_x, train_x, test_x]))

    ext_scaled = scaler.transform(ext_x)
    train_scaled = scaler.transform(train_x)
    test_scaled = scaler.transform(test_x)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(ext_scaled)
    train_dist, train_idx = nn.kneighbors(train_scaled, return_distance=True)
    test_dist, test_idx = nn.kneighbors(test_scaled, return_distance=True)
    train_dist = train_dist[:, 0]
    test_dist = test_dist[:, 0]
    train_idx = train_idx[:, 0]
    test_idx = test_idx[:, 0]

    external_class = external[TARGET].to_numpy()
    train_pred = external_class[train_idx]
    train_true = train_eval[TARGET].to_numpy()

    thresholds = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    threshold_rows = []
    for threshold in thresholds:
        train_hit = train_dist <= threshold
        test_hit = test_dist <= threshold
        if train_hit.any():
            acc = float((train_pred[train_hit] == train_true[train_hit]).mean())
            bal_acc = float(balanced_accuracy_score(train_true[train_hit], train_pred[train_hit]))
        else:
            acc = None
            bal_acc = None
        threshold_rows.append(
            {
                "scaled_distance_threshold": threshold,
                "train_matches": int(train_hit.sum()),
                "train_match_rate": float(train_hit.mean()),
                "test_matches": int(test_hit.sum()),
                "test_match_rate": float(test_hit.mean()),
                "train_label_accuracy_on_matches": acc,
                "train_label_balanced_accuracy_on_matches": bal_acc,
            }
        )

    return {
        "train_distance_quantiles": {
            str(q): float(np.quantile(train_dist, q)) for q in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9]
        },
        "test_distance_quantiles": {
            str(q): float(np.quantile(test_dist, q)) for q in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9]
        },
        "thresholds": threshold_rows,
        "train_sample_size": int(len(train_eval)),
        "test_sample_size": int(len(test_eval)),
        "external_size": int(len(external)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose source-data matches against public SDSS17 data.")
    parser.add_argument("--exp-id", default="exp015_sdss17_source_match")
    parser.add_argument("--nn-sample-size", type=int, default=50000)
    parser.add_argument("--skip-nn", action="store_true")
    args = parser.parse_args()

    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train, test, external = load_data()
    rounded_rows = summarize_rounded(train, test, external, decimals_grid=[6, 5, 4, 3, 2, 1])
    nn_result = None if args.skip_nn else nearest_neighbor_diagnostic(train, test, external, args.nn_sample_size)

    result = {
        "exp_id": args.exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "PB狙いに向けて、公式train/testと公開SDSS17元データ候補の一致可能性を診断する。",
        "hypothesis": "PlaygroundデータがSDSS17由来なら、丸め一致または近傍一致で一部行を高信頼に照合でき、PBで強い補助信号になる可能性がある。",
        "official_metric": "balanced_accuracy",
        "external_source": str(SDSS_PATH),
        "external_source_url": "https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17",
        "match_columns": MATCH_COLS,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "external_shape": list(external.shape),
        "train_target_distribution": train[TARGET].value_counts().to_dict(),
        "external_target_distribution": external[TARGET].value_counts().to_dict(),
        "rounded_match_summary": rounded_rows,
        "nearest_neighbor_summary": nn_result,
        "leakage_risk": "Medium-high: external source labels may reveal source objects if exact/near matches are accepted. Use only thresholds validated by train match label agreement and document rule compatibility before submission.",
        "overfitting_risk": "Medium: thresholds chosen after observing train/test match rates can overfit. Prefer predeclared thresholds based on train label agreement and distance distribution.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    rounded_df = pd.DataFrame(rounded_rows)
    rounded_df.to_csv(exp_dir / "rounded_match_summary.csv", index=False)
    if nn_result is not None:
        nn_df = pd.DataFrame(nn_result["thresholds"])
        nn_df.to_csv(exp_dir / "nearest_neighbor_thresholds.csv", index=False)

    best_rounded = max(rounded_rows, key=lambda row: row["train_matches"])
    if nn_result is not None:
        safe_nn = [
            row
            for row in nn_result["thresholds"]
            if row["train_label_balanced_accuracy_on_matches"] is not None
            and row["train_label_balanced_accuracy_on_matches"] >= 0.99
        ]
        best_safe_nn = max(safe_nn, key=lambda row: row["train_matches"], default=None)
    else:
        best_safe_nn = None

    notes = f"""# {args.exp_id}

## Purpose

PB 1位狙いのため、公開SDSS17候補データと公式train/testの照合可能性を診断した。

## Hypothesis

PlaygroundデータがSDSS17由来なら、`alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift` の丸め一致または近傍一致で一部行を高信頼に回収できる可能性がある。

## Result

- External source: `{SDSS_PATH}`
- External shape: {external.shape}
- Best rounded train matches: decimals={best_rounded["decimals"]}, train_matches={best_rounded["train_matches"]}, test_matches={best_rounded["test_matches"]}
- Nearest-neighbor safe threshold candidate: {best_safe_nn}

## Interpretation

`result.json` の `rounded_match_summary` と `nearest_neighbor_summary` を参照。採用する場合は、train側のlabel一致率が高い距離閾値だけを使い、testだけで都合よく閾値を決めない。

## Leakage Risk

Medium-high。外部元データのlabelをtestへ直接写す行為は、データ由来と競技ルール次第でleakage扱いになる可能性がある。提出前に、外部データが公開・利用可能であること、hidden targetの逆算ではないことを確認する。

## Overfitting Risk

Medium。丸め桁数や距離閾値をtest match率に合わせるとPBへ過適合する。train一致率と距離分布を主判断にする。

## Next Actions

- 0.99以上のtrain label balanced_accuracyを満たす近傍閾値が十分な行数を持つか確認する。
- 十分なら、external-match priorをOOF stackの特徴として追加する。
- 不十分なら、SDSS17は追加学習データとしてのみ扱い、直接label写しは採用しない。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
