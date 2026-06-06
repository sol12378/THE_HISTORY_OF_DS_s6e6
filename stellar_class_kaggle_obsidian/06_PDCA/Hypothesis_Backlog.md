# Hypothesis Backlog

| Priority | Hypothesis | Suggested Exp | Risk |
|---|---|---|---|
| High | Class multiplier tuning can improve official balanced_accuracy, but needs test probabilities and LB confirmation | rerun LightGBM with `test_proba.csv`, then thresholded submission | Medium-high overfitting |
| High | Advanced features should be split into groups instead of all at once | color-only, redshift-interaction-only, sky-only experiments | Runtime / overfitting |
| Medium | Model diversity can improve OOF ensemble | CatBoost / XGBoost OOF, then soft voting or stacking | Runtime |
| Medium | Public submission weighted voting can improve LB, but not local CV | exp008-style vote variants | High leakage / public LB overfit |
| High | CDeotte型logit stacker with 5 seeds can improve current logistic stack CV | PyTorch weighted logistic stacker over existing OOF/test preds | Medium overfit; needs nested CV |
| High | SDSS17 original priors/prototypes can improve LightGBM/XGB/CatBoost without direct label transfer | orig prior + prototype distance features | Medium leakage if documented poorly |
| High | CatBoost v3-style native categorical views plus low-weight original rows can add strong stack diversity | CatBoost GPU with binned/hashed categories and original_weight=0.06 | Medium-high leakage/overfit; needs honest fold split |
| High | RealMLP/TabM/TabICL/NN base predictions are necessary for top public/PB stack | Build one neural tabular base and add to stack | Runtime/GPU complexity |
| Medium | TabPFN-3 can be a stronger meta-stacker than logistic regression on base logits | TabPFN-3 stacker on existing OOF/test logits | Dependency/model access risk |

| Hypothesis | Priority | Status | Notes |
|---|---:|---|---|
