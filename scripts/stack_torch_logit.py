from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


ID_COL = "id"
TARGET = "class"
CLASSES = ["GALAXY", "QSO", "STAR"]
PROBA_COLS = [f"proba_{cls}" for cls in CLASSES]
EPS = 1e-15
LOGIT_CLIP = 30.0


class TorchMultiLogReg(nn.Module):
    def __init__(self, in_features: int, out_features: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def prob_to_logit(values: np.ndarray) -> np.ndarray:
    p = np.clip(values.astype(np.float64), EPS, 1.0 - EPS)
    return np.clip(np.log(p / (1.0 - p)), -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)


def make_stack_features(exp_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_base = None
    test_base = None
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for exp_id in exp_ids:
        oof = read_oof(exp_id)
        test = read_test_proba(exp_id)
        if train_base is None:
            train_base = oof[[ID_COL, "true_class"]].copy()
            test_base = test[[ID_COL]].copy()
        else:
            train_base = train_base.merge(oof[[ID_COL, "true_class"]], on=[ID_COL, "true_class"], validate="one_to_one")
            test_base = test_base.merge(test[[ID_COL]], on=ID_COL, validate="one_to_one")

        train_logits = prob_to_logit(oof[PROBA_COLS].to_numpy())
        test_logits = prob_to_logit(test[PROBA_COLS].to_numpy())
        train_parts.append(pd.DataFrame(train_logits, columns=[f"{exp_id}_logit_{cls}" for cls in CLASSES]))
        test_parts.append(pd.DataFrame(test_logits, columns=[f"{exp_id}_logit_{cls}" for cls in CLASSES]))

    assert train_base is not None and test_base is not None
    train_x = pd.concat([train_base.reset_index(drop=True), *train_parts], axis=1)
    test_x = pd.concat([test_base.reset_index(drop=True), *test_parts], axis=1)
    feature_cols = [col for col in train_x.columns if col not in [ID_COL, "true_class"]]
    return train_x, test_x, feature_cols


def evaluate_with_multipliers(y_idx: np.ndarray, proba: np.ndarray, multipliers: np.ndarray) -> dict:
    pred_idx = (proba * multipliers.reshape(1, -1)).argmax(axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_idx, pred_idx)),
        "accuracy": float(accuracy_score(y_idx, pred_idx)),
        "multipliers": {cls: float(mult) for cls, mult in zip(CLASSES, multipliers)},
    }


def search_multipliers(y_idx: np.ndarray, proba: np.ndarray) -> dict:
    best = None
    for qso_scale in np.linspace(1.0, 1.25, 6):
        for star_scale in np.linspace(1.0, 1.25, 6):
            candidate = evaluate_with_multipliers(y_idx, proba, np.array([1.0, qso_scale, star_scale]))
            if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
                best = candidate
    assert best is not None
    return best


def train_one_fold(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
    seed: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)

    class_weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y_tr).astype(np.float32)
    sample_weights = torch.tensor(class_weights[y_tr], dtype=torch.float32, device=device)

    model = TorchMultiLogReg(x_tr.shape[1], len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    weight_decay = 1.0 / (c_value * len(y_tr))
    optimizer = torch.optim.Adam(
        [
            {"params": model.linear.weight, "weight_decay": weight_decay},
            {"params": model.linear.bias, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tr_t)
        loss = (criterion(logits, y_tr_t) * sample_weights).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_proba = torch.softmax(model(x_val_t), dim=1).detach().cpu().numpy().astype(np.float32)
        test_proba = torch.softmax(model(x_test_t), dim=1).detach().cpu().numpy().astype(np.float32)

    del model, x_tr_t, y_tr_t, x_val_t, x_test_t, sample_weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_proba, test_proba


def fit_predict_multiseed(
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
    seeds: list[int],
    n_folds: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof_sum = np.zeros((len(x), len(CLASSES)), dtype=np.float64)
    test_sum = np.zeros((len(x_test), len(CLASSES)), dtype=np.float64)
    fold_rows: list[dict] = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_seed = np.zeros((len(x), len(CLASSES)), dtype=np.float32)
        test_seed = np.zeros((len(x_test), len(CLASSES)), dtype=np.float64)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
            val_proba, test_proba = train_one_fold(
                x[tr_idx],
                y[tr_idx],
                x[va_idx],
                x_test,
                c_value=c_value,
                seed=seed + fold,
                epochs=epochs,
                lr=lr,
                device=device,
            )
            oof_seed[va_idx] = val_proba
            test_seed += test_proba / n_folds
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], val_proba.argmax(axis=1))),
                    "accuracy": float(accuracy_score(y[va_idx], val_proba.argmax(axis=1))),
                }
            )
        oof_sum += oof_seed / len(seeds)
        test_sum += test_seed / len(seeds)
    return oof_sum.astype(np.float32), test_sum.astype(np.float32), fold_rows


def run_stack(
    exp_id: str,
    base_exp_ids: list[str],
    seeds: list[int],
    n_folds: int,
    c_grid: list[float],
    epochs: int,
    lr: float,
) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, test_x, feature_cols = make_stack_features(base_exp_ids)
    encoder = LabelEncoder()
    y = encoder.fit_transform(train_x["true_class"])
    x = train_x[feature_cols].to_numpy(dtype=np.float32)
    x_test = test_x[feature_cols].to_numpy(dtype=np.float32)

    candidates = []
    for c_value in c_grid:
        print(f"running pure PyTorch C={c_value} seeds={seeds} folds={n_folds} epochs={epochs} device={device}")
        oof_proba, test_proba, fold_rows = fit_predict_multiseed(
            x=x,
            y=y,
            x_test=x_test,
            c_value=c_value,
            seeds=seeds,
            n_folds=n_folds,
            epochs=epochs,
            lr=lr,
            device=device,
        )
        raw = evaluate_with_multipliers(y, oof_proba, np.ones(len(CLASSES)))
        threshold = search_multipliers(y, oof_proba)
        candidates.append(
            {
                "C": c_value,
                "raw": raw,
                "best_threshold": threshold,
                "fold_results": fold_rows,
                "oof_proba": oof_proba,
                "test_proba": test_proba,
            }
        )
        print(f"C={c_value} raw={raw['balanced_accuracy']:.6f} threshold={threshold['balanced_accuracy']:.6f}")

    best = max(candidates, key=lambda row: row["best_threshold"]["balanced_accuracy"])
    best_oof = best["oof_proba"]
    best_test = best["test_proba"]
    multipliers = np.array([best["best_threshold"]["multipliers"][cls] for cls in CLASSES], dtype=np.float64)
    pred_idx = (best_oof * multipliers.reshape(1, -1)).argmax(axis=1)
    test_pred = np.array(CLASSES, dtype=object)[(best_test * multipliers.reshape(1, -1)).argmax(axis=1)]

    oof = train_x[[ID_COL, "true_class"]].copy()
    oof["pred_class"] = np.array(CLASSES, dtype=object)[pred_idx]
    for idx, cls in enumerate(CLASSES):
        oof[f"proba_{cls}"] = best_oof[:, idx]
    oof.to_csv(exp_dir / "oof.csv", index=False)

    test_proba_df = test_x[[ID_COL]].copy()
    for idx, cls in enumerate(CLASSES):
        test_proba_df[f"proba_{cls}"] = best_test[:, idx]
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
        "purpose": "Pure PyTorch CDeotte-style logit multi-seed logistic stacker over self-model OOF/test predictions.",
        "hypothesis": "A pure PyTorch logit stacker with repeated stratified folds can improve or validate the current logistic stacker CV.",
        "official_metric": "balanced_accuracy",
        "base_exp_ids": base_exp_ids,
        "feature_count": len(feature_cols),
        "n_folds": n_folds,
        "seeds": seeds,
        "c_grid": c_grid,
        "epochs": epochs,
        "learning_rate": lr,
        "device": str(device),
        "best_C": best["C"],
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
        "leakage_risk": "Medium: base predictions are OOF, but C and class multipliers are selected on OOF labels.",
        "overfitting_risk": "Medium-high: repeated-fold stacker reduces split variance, but meta-parameter search can still overfit.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

純粋なPyTorchでCDeotte型logit-only multi-seed logistic stackerを検証する。

## Hypothesis

probabilityをlogit化し、PyTorchの`nn.Linear`だけでmulticlass logistic regressionを学習すれば、`exp024` CV 0.966112を超える、またはstacker実装の妥当性を検証できる。

## Result

- Base experiments: {base_exp_ids}
- Device: `{device}`
- Seeds: {seeds}
- Folds: {n_folds}
- Epochs: {epochs}
- Best C: {best["C"]}
- Raw CV balanced_accuracy: {best["raw"]["balanced_accuracy"]:.6f}
- Thresholded CV balanced_accuracy: {best["best_threshold"]["balanced_accuracy"]:.6f}
- CV accuracy: {best["best_threshold"]["accuracy"]:.6f}
- Multipliers: {best["best_threshold"]["multipliers"]}
- Submission checks: {checks}

## Submission Decision

Compare against `exp024_oof_stacker_with_lgbm_basic` CV 0.966112. Submit only if the improvement is confidence-worthy and the Kaggle submission limit allows it.

## Risks

- Leakage risk: medium。base predictionsはOOFだが、CとmultiplierをOOF labelで選んでいる。
- Overfitting risk: medium-high。multi-seedでsplit varianceは減るが、meta-parameter探索は過学習しうる。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pure PyTorch CDeotte-style logit stacker.")
    parser.add_argument("--exp-id", default="exp028_torch_logit_stacker")
    parser.add_argument("--base-exp", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--c", action="append", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    seeds = args.seed if args.seed else [42, 43, 44]
    c_grid = args.c if args.c else [0.1, 1.0, 10.0]
    run_stack(args.exp_id, args.base_exp, seeds, args.n_folds, c_grid, args.epochs, args.lr)


if __name__ == "__main__":
    main()
