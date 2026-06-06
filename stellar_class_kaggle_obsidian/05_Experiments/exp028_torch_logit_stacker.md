# exp028_torch_logit_stacker

## Purpose

CDeotte型のlogit-only multi-seed logistic stackerを、純粋なPyTorch (`nn.Linear`) で実装して検証する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`
- Device: `cuda`
- Seeds: `42`, `43`, `44`
- Folds: 5
- Epochs: 1000
- C grid: `0.1`, `1.0`, `10.0`
- Raw/thresholded CV balanced_accuracy: 0.965935
- CV accuracy: 0.959924
- Multipliers: `GALAXY=1.0`, `QSO=1.0`, `STAR=1.0`

## Submission Decision

`exp024_oof_stacker_with_lgbm_basic` のCV 0.966112を下回ったため、API提出しない。

## Interpretation

純PyTorch logit stackerは実装でき、GPUで高速に動作した。ただし現base構成では`exp024`を超えない。Cに対する結果がほぼ同一なので、現在のlogit入力ではlinear meta-modelの表現力よりbase model不足が主因と考える。

## Risks

- Leakage risk: medium。base predictionsはOOFだが、CとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。multi-seedでsplit varianceは減るが、meta-parameter探索は過学習しうる。

## Next Actions

- PyTorch stacker自体より、RealMLP/TabM/CatBoost v3/SDSS prior featuresなどbase model追加を優先する。
