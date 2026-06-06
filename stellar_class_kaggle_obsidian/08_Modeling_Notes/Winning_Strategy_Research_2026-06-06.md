# Winning Strategy Research 2026-06-06

## 現状

- Current public #1: `0.97127` (`nybbler`, Kaggle leaderboard via API, 2026-06-06 JST)
- Our best confirmed LB: `exp008_nina_weighted_vote`, `0.97074`, external public submission blend, high risk
- Our best self-model confirmed LB: `exp020_oof_stacker_with_xgb_deep`, `0.96642`
- Our best self-model CV: `exp024_oof_stacker_with_lgbm_basic`, `0.966112`, LB未確認

## 重要な公開notebookからの示唆

### CDeotte GPU Logistic Regression Stacker

Source: `https://www.kaggle.com/code/cdeotte/gpu-logistic-regression-stacker`

- OOF/test probabilityをlogit変換して、GPU PyTorchのmulticlass logistic regressionでstackする。
- `N_FOLDS=5`, `N_SEEDS=5` のseed averaging。
- base modelはGBDTだけでなく、RealMLP, TabM, TabICL, NN, Logistic Regressionまで含む。
- stacker入力はモデルごとの3-class probability/logit。probability calibrationはstackerに任せる思想。
- 1位級に必要なのは「強い単体モデル」より「多様なOOF/test predsの大量確保」。

主なbase:

- `xgb-0`, `xgb-1`, `xgb-3`, `xgb-5`
- `lgbm-3`
- `cat-0`, `cat-3`
- `realmlp-0`, `realmlp-1`, `realmlp-2`, `realmlp-5`
- `tabm-0`
- `tabicl-2`
- `logreg-1`
- `nn-1`, `nn-2`

### TabPFN-3 Stacker

Source: `https://www.kaggle.com/code/philippsinger/tabpfn-3-stacker`

- Chris Deotteのstackerを土台に、meta-modelをTabPFN-3へ置き換える。
- TabPFN-3は1M行級まで対応するtabular foundation modelとして紹介されている。
- `balance_probabilities=True` を使うため、balanced_accuracy向けのmeta-model候補。

### RealMLP PyTorch

Source: `https://www.kaggle.com/code/yekenot/ps-s6-e6-realmlp-pytorch`

- RealMLP_TD系のNNを5-foldで学習し、OOF/test predsを保存する。
- 数値前処理は `median_center`, `robust_scale`, `smooth_clip`, `l2_normalize`。
- categoricalはone-hotまたはembedding。
- PBLD/periodic-like embedding、label smoothing、内部ensembleを使う。
- GBDTとは異なる誤差が期待でき、stackerのbaseとして重要。

### Single LightGBM LB 0.96728

Source: `https://www.kaggle.com/code/evgendvorkin/single-lightgbm-lb-0-96728`

- 外部SDSS17 original dataを直接appendではなく、target prior/prototype系featureに利用。
- `spectral_type` と `galaxy_population` をoriginal側で再構築。
- original class prior features:
  - `orig_spec_target`
  - `orig_gal_target`
  - `orig_combo_target`
- color/redshift/sky featuresを追加。
- 10-fold LightGBM + differential evolution class weight calibration。
- 直接match/label transferではなく、original dataを「prior feature」として使う方向が安全で有望。

### CDeotte LGBM v3

Source: `https://www.kaggle.com/code/cdeotte/lgbm-v3-for-s6e6`

- SDSS17 original dataから以下を作る。
  - denoised categorical bins
  - train/test count features
  - original-dataset class priors
  - class-prototype distance features
- original rowsは学習にappendしない。
- 5-fold LightGBMでOOF/test predsを出す。
- local OOF BAは低めでも、stack baseとして使われている。

### CDeotte XGB v5

Source: `https://www.kaggle.com/code/cdeotte/xgb-v5-for-s6e6`

- RAPIDS/cuDF/cuMLでGPU feature engineering。
- cuML TargetEncoderをouter fold内でfold-safeに実行。
- original labelsはprior featuresにのみ使い、original rowsはappendしない。
- selected 370 features、XGBoost CUDA hist。
- custom balanced_error metricでearly stopping。

### CDeotte CatBoost v3

Source: `https://www.kaggle.com/code/cdeotte/cat-v3-for-s6e6`

- local 5-fold CV `0.96887081` と記載。
- original SDSS rowsを各fold training poolへ `ORIGINAL_WEIGHT=0.06` でappend。
- 多数のnative categorical views:
  - quantile bins
  - rounded numeric artifacts
  - color/magnitude bins
  - sky-position bins
  - compact hashed category interactions
- CatBoost GPUでnative categoricalを強く使う。

## 論文・一般手法からの示唆

### SHEEP: photometric redshift-aided ensemble

Source: `https://arxiv.org/abs/2204.02080`

- XGBoost, LightGBM, CatBoostのoutputsを組み合わせる天体分類ensemble。
- photometric redshift推定を分類前に行い、その推定値を追加特徴量として使う。
- one-vs-all + meta-learnerの構成も有効。

### TabPFN-3

Source: `https://arxiv.org/abs/2605.13986`

- 1M training rows規模まで対応するtabular foundation model。
- tuned/ensembled GBDT baselineを上回るケースがある。
- 今回の577k train、3 class、30から数百featureのmeta-stackに特に適合しそう。

### TabM

Source: `https://arxiv.org/abs/2410.24210`

- parameter-efficient ensemble of MLPs。
- 1つのモデルでMLP ensembleのように複数予測を作る。
- RealMLP/TabMはGBDTと誤差相関が低く、stack baseとして価値が高い。

### PLAsTiCC / astronomical Kaggle

Sources:

- `https://arxiv.org/abs/2012.12392`
- `https://higepon.hatenablog.com/entry/2018/12/21/130853`

天体分類系Kaggleでは、boosted trees, neural networks, RNN/CNN/MLPなどのhybrid ensembleが上位で使われる。今回も時系列ではないが、天体ドメインではモデル多様性と外部天文知識が効く。

## 1位を狙うために必要な情報

### 必須

1. 公開上位stackerに含まれる各base modelのOOF/test preds
   - CDeotte stackerの `STACKING_FILES` にある全モデル。
   - 取得できるなら、local stackに取り込み、our CV/LBとの対応を見る。

2. 各base modelのCV/LB/OOF相関
   - 単体CVだけでなく、stacker係数・重要度を確認する。
   - 低CVでも相関が低いbaseは残す。

3. original SDSS17 feature利用の安全線
   - direct label transferは`exp015`で不採用。
   - prior/prototype/low-weight appendは候補。
   - leakage riskを明確化する。

4. submission limit reset後の提出計画
   - `exp024`を最初に確認。
   - 次はCV `0.9663+`以上の候補のみ提出する。

### あると強い

- RAPIDS/cuDF/cuML環境
- TabPFN-3 model cache/API利用可否
- RealMLP/TabM/TabICLをローカルGPUで動かせる環境
- 各公開notebookのoutput artifactsをKaggleから入力datasetとして取得できるか

## 実験優先順位

1. `exp024`をsubmission limit reset後に提出確認
   - 現best CV `0.966112`
   - LB未確認

2. CDeotte型logit stackerへ更新
   - 現在の`stack_oof_meta.py`はlogit+probabilityだが、seed averagingが弱い。
   - PyTorch logistic regression with class weights, 5 seeds, 5 foldsへ変更。

3. External SDSS priors/prototypes実装
   - `orig_spec_target`, `orig_gal_target`, `orig_combo_target`
   - class prototype distances
   - count features
   - direct label transferではない。

4. CatBoost v3風base
   - original rowsをweight 0.06でappend。
   - quantile/rounded/color/sky hashed categorical views。
   - CatBoost GPU native categorical。

5. RealMLPまたはTabM base
   - RealMLP_TD/PBLD/label smoothing。
   - 5-fold OOF/test predsを作成。

6. TabPFN-3 meta-stacker
   - 既存base predsを入力にする。
   - `balance_probabilities=True`。

## 判断

LB 1位だけなら公開submission vote/searchで届く可能性があるが、PB 1位には危険。PB 1位を狙う本線は、CDeotte stackerと同系統の「多様なOOF/test preds + logit/meta stack + external SDSS prior/prototype feature」である。
