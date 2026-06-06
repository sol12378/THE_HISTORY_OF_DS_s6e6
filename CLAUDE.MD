# CLAUDE.MD

Follow `AGENTS.md` as the canonical project instruction file.

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
