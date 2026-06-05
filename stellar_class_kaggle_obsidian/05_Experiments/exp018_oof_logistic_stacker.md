# exp018_oof_logistic_stacker

## Purpose

利用可能な自前モデルOOF probabilityを使い、LogisticRegressionのmeta-modelでOOF stackを試す。

## Result

- Base experiments: `exp009_lgbm_color_redshift_small`, `exp010_catboost_gpu_basic`, `exp016_xgboost_gpu_basic`
- Best C: 1.0
- Raw CV balanced_accuracy: 0.965475
- Thresholded CV balanced_accuracy: 0.965560
- CV accuracy: 0.958716
- Public LB: 0.96641
- Submission ref: 53391289
- Multipliers: `GALAXY=1.0`, `QSO=1.15`, `STAR=1.0`

## Interpretation

`exp017_lgbm_xgboost_blend` をCV/LBともに上回った。固定weightのsoft votingより、OOF stackerの方が自前モデル開発では有望。

## Risks

- Leakage risk: medium。base predictionsはOOFだが、meta hyperparameterとmultiplierはOOF labelで選んでいる。
- Overfitting risk: medium-high。PB狙いではfold安定性、追加seed、より保守的なmultiplierの比較が必要。

## Next Actions

- GPU XGBoostの追加設定・seedを作り、stackerのbaseモデルを増やす。
- LightGBMの別seed/別featureセットのtest_probaを追加する。
- Public LB用blendとは分けて管理する。
