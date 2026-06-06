# exp038_oof_stacker_with_mlp_and_sdss

## Purpose

`exp036` のMLP diversityに、SDSS系の弱base `exp031` と `exp033` を同時追加し、弱いが異質なbase群がstackで相補的に効くか確認する。

## Result

- Base experiments: `exp009`, `exp010`, `exp016`, `exp019`, `exp023`, `exp029`, `exp031`, `exp033`, `exp035`
- Raw CV balanced_accuracy: 0.966344
- Thresholded CV balanced_accuracy: 0.966420
- CV accuracy: 0.960173
- Best C: 0.03
- Multipliers: GALAXY 1.0, QSO 1.0, STAR 1.05

## Interpretation

記録上のnew diagnostic best。`exp036` よりさらに改善し、weak/diverse baseの組み合わせが有効だった。confirmation `exp039` も 0.966322 で `exp030/032` を超えたため、改善シグナルは完全な偶然ではない。

## Decision

no_submit。改善幅はまだ小さく、最良値はsklearn stackerのC選択と軽いmultiplier searchに依存する。もう一段の確認またはCV改善が出たらAPI提出候補に戻す。
