# CLAUDE.MD

Follow `AGENTS.md` as the canonical project instruction file.

## Orchestration and Worker PDCA

- Make the main agent act as the orchestrator for implementation, experiments, and project decisions.
- Use workers for bounded implementation, inspection, verification, or remediation tasks when useful.
- Run workers with low reasoning effort unless the user explicitly requests otherwise.
- Review worker deliverables against clear acceptance criteria; if a deliverable is below the criteria, send it back to the worker for recursive correction.
- Keep running PDCA cycles until the stated objective is reached or a real external blocker prevents further progress.
