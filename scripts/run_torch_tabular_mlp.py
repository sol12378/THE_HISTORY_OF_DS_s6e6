from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET = "class"
ID_COL = "id"
RAW_DIR = Path("data/raw")
CLASSES = ["GALAXY", "QSO", "STAR"]
BANDS = ["u", "g", "r", "i", "z"]


class TabularMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in [TARGET]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for left, right in itertools.combinations(BANDS, 2):
        out[f"{left}_{right}"] = out[left] - out[right]
    mag = out[BANDS]
    out["mag_mean"] = mag.mean(axis=1)
    out["mag_std"] = mag.std(axis=1)
    out["mag_min"] = mag.min(axis=1)
    out["mag_max"] = mag.max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_sq"] = out["redshift"] ** 2
    out["redshift_signed_log1p"] = np.sign(out["redshift"]) * np.log1p(out["redshift_abs"])
    out["redshift_is_negative"] = (out["redshift"] < 0).astype(np.float32)
    for color in ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z"]:
        if color in out.columns:
            out[f"redshift_x_{color}"] = out["redshift"] * out[color]
    return out


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = add_features(train)
    test = add_features(test)
    features = [col for col in train.columns if col not in [ID_COL, TARGET]]
    for col in features:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
    train[features] = train[features].replace([np.inf, -np.inf], np.nan)
    test[features] = test[features].replace([np.inf, -np.inf], np.nan)
    features = [col for col in features if not train[col].isna().all()]
    medians = train[features].median(axis=0)
    train[features] = train[features].fillna(medians).fillna(0.0)
    test[features] = test[features].fillna(medians).fillna(0.0)
    return train, test, features


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def train_one_fold(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_test: np.ndarray,
    class_weights: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    model = TabularMLP(x_tr.shape[1], hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    loader = make_loader(x_tr, y_tr, batch_size=batch_size, shuffle=True)
    x_va_t = torch.tensor(x_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")
    patience = 5
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(yb)
            total_rows += len(yb)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_va_t), y_va_t).detach().cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"epoch={epoch} train_loss={total_loss / max(total_rows, 1):.6f} val_loss={val_loss:.6f}")
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_proba = torch.softmax(model(x_va_t), dim=1).detach().cpu().numpy().astype(np.float32)
        test_chunks = []
        for start in range(0, len(x_test), batch_size * 4):
            test_chunks.append(torch.softmax(model(x_test_t[start : start + batch_size * 4]), dim=1).detach().cpu().numpy())
        test_proba = np.concatenate(test_chunks, axis=0).astype(np.float32)

    info = {"best_val_loss": best_loss, "epochs_ran": epoch}
    del model, x_va_t, y_va_t, x_test_t, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return va_proba, test_proba, info


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


def run_experiment(
    exp_id: str,
    n_splits: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
) -> dict:
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(RAW_DIR / "train.csv")
    test_raw = pd.read_csv(RAW_DIR / "test.csv")
    sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")
    train, test, features = prepare_features(train_raw, test_raw)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET])
    classes = encoder.classes_
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=202606)
    oof_proba = np.zeros((len(train), len(classes)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float32)
    fold_results: list[dict] = []

    x_all = train[features].to_numpy(dtype=np.float32)
    x_test_all = test[features].to_numpy(dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_all, y), start=1):
        print(f"===== fold {fold}/{n_splits} device={device} =====")
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_all[tr_idx]).astype(np.float32)
        x_va = scaler.transform(x_all[va_idx]).astype(np.float32)
        x_test = scaler.transform(x_test_all).astype(np.float32)
        x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0)
        x_va = np.nan_to_num(x_va, nan=0.0, posinf=0.0, neginf=0.0)
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
        fold_counts = np.bincount(y[tr_idx], minlength=len(classes)).astype(np.float32)
        class_weights = (len(tr_idx) / (len(classes) * fold_counts)).astype(np.float32)
        va_proba, fold_test_proba, info = train_one_fold(
            x_tr=x_tr,
            y_tr=y[tr_idx],
            x_va=x_va,
            y_va=y[va_idx],
            x_test=x_test,
            class_weights=class_weights,
            seed=202606 + fold,
            epochs=epochs,
            batch_size=batch_size,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        oof_proba[va_idx] = va_proba
        test_proba += fold_test_proba / n_splits
        va_pred = va_proba.argmax(axis=1)
        fold_result = {
            "fold": fold,
            "balanced_accuracy": float(balanced_accuracy_score(y[va_idx], va_pred)),
            "accuracy": float(accuracy_score(y[va_idx], va_pred)),
            **info,
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
    for fold, (_, va_idx) in enumerate(skf.split(x_all, y), start=1):
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
        "purpose": "Train a pure PyTorch tabular MLP base model for OOF stack diversity.",
        "hypothesis": "A nonlinear pure PyTorch MLP over standardized numeric/color/redshift features can add neural-network error diversity to the tree-model stack.",
        "official_metric": "balanced_accuracy",
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden": hidden,
        "dropout": dropout,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "device": str(device),
        "cv_balanced_accuracy": cv_balanced_accuracy,
        "cv_accuracy": cv_accuracy,
        "fold_results": fold_results,
        "features": features,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "submission_distribution": submission[TARGET].value_counts().to_dict(),
        "submission_checks": checks,
        "leakage_risk": "Low: official train/test features only; no external labels or target encoding.",
        "overfitting_risk": "Medium: neural net diagnostic is short and stochastic; accept only by OOF stack gain.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {exp_id}

## Purpose

純粋PyTorch (`nn.Module` + `DataLoader` + `AdamW`) のtabular MLP baseを作り、tree model stackへNN系の誤差多様性を追加できるか確認する。

## Result

- CV balanced_accuracy: {cv_balanced_accuracy:.6f}
- CV accuracy: {cv_accuracy:.6f}
- Device: `{device}`
- n_splits: {n_splits}
- epochs: {epochs}
- hidden: {hidden}
- dropout: {dropout}
- Submission checks: {checks}

## Interpretation

単体提出候補ではなく、次段のOOF stackで採否を判断する。現行best `exp032` CV 0.966223 を明確に超えない場合はAPI提出しない。

## Risks

- Leakage risk: low。official train/test featuresのみ。
- Overfitting risk: medium。NNはstochasticでfold varianceがあるため、stack改善幅を厳しく見る。
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    if not checks["all_passed"]:
        raise RuntimeError(f"submission validation failed: {checks}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pure PyTorch tabular MLP base experiment.")
    parser.add_argument("--exp-id", default="exp035_torch_tabular_mlp")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    args = parser.parse_args()
    run_experiment(
        exp_id=args.exp_id,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
