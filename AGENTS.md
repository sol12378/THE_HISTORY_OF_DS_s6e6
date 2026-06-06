# AGENTS.md

You are working in the Predicting Stellar Class Kaggle competition workspace.

## Language Rules

- Write reasoning notes, hypotheses, decisions, experiment results, interpretation, and next actions in Japanese by default.
- Keep code, file names, config keys, Kaggle official terms, and log output in English where natural.
- Obsidian notes should primarily be written in Japanese.

## Always Follow

- Give each experiment one clear purpose.
- Never modify `data/raw`.
- Do not commit secrets, raw Kaggle data, processed parquet files, model artifacts, OOF files, or submissions.
- Save reproducible outputs under `experiments/expXXX_*`.
- Update `EXP_SUMMARY.md` when an experiment changes the project state.
- Update `stellar_class_kaggle_obsidian/` when competition understanding, validation, features, submissions, or decisions change.
- Prefer fast experiments that can finish in about 20 minutes before scaling up.
- Do not optimize only for public LB.
- Explicitly document leakage risk and overfitting risk.

## Before Coding

- Read `EXP_SUMMARY.md`.
- Read `stellar_class_kaggle_obsidian/00_Index/Home.md`.
- Check recent best CV, best LB, and leak-risk experiments.
- State the experiment hypothesis clearly.

## After Coding

- Save `result.json`.
- Save `oof.csv` when training with CV.
- Save `submission.csv` only when the output is intended for Kaggle.
- Write `notes.md`.
- Update the matching note in `stellar_class_kaggle_obsidian/05_Experiments/`.
- Update `stellar_class_kaggle_obsidian/06_PDCA/Daily_Log/YYYY-MM-DD.md`.
- Record major decisions in `stellar_class_kaggle_obsidian/06_PDCA/Decision_Log.md`.
- After a Kaggle submission, update `stellar_class_kaggle_obsidian/09_Submissions/LB_Tracking.md`.

## Orchestration and Worker PDCA

- Make the main agent act as the orchestrator for implementation, experiments, and project decisions.
- Use workers for bounded implementation, inspection, verification, or remediation tasks when useful.
- Run workers with low reasoning effort unless the user explicitly requests otherwise.
- Review worker deliverables against clear acceptance criteria; if a deliverable is below the criteria, send it back to the worker for recursive correction.
- Keep running PDCA cycles until the stated objective is reached or a real external blocker prevents further progress.
- Submit through the Kaggle API only when the candidate has a confident validation basis: it clearly improves the trusted CV/balanced_accuracy versus the current relevant best, or it has a predeclared validation rationale strong enough to justify the submission.
- Before every Kaggle API submission, record the candidate CV, comparison target, expected risk, and reason for confidence in the experiment notes or submission tracking.

## Torch Policy

- When using torch models or torch-based stackers, implement them in pure PyTorch.
- Do not depend on high-level torch training wrappers or third-party tabular neural-network packages unless the user explicitly approves them.
