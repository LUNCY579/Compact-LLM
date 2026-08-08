# File responsibilities

| File | Responsibility |
|---|---|
| `student_playground.py` | Loads a full model or PEFT adapter, reuses official Phase 9/9B prompts/generation, scores cases, measures latency/throughput/VRAM, and writes run artifacts. |
| `run_playground.ps1` | Windows-friendly wrapper for common recording runs. |
| `datasets/showcase_suite.jsonl` | Combined 34-case novel demo suite. |
| `datasets/logic_games.jsonl` | Compact closed-context logic cases. |
| `datasets/evidence_reasoning.jsonl` | Evidence-grounded answer/abstain/conflict cases. |
| `datasets/memory_reconstruction.jsonl` | Current-state + source-state memory applicability cases. |
| `schemas/playground_case.schema.json` | Authoring contract for custom cases. |
| `configs/default.json` | Version/provenance defaults and the explicit non-gating policy. |
| `README.md` | Commands, workflow placement, interpretation, and contamination rules. |
