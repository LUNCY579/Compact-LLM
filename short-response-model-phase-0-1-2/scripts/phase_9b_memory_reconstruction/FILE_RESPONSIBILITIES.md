# Phase 9B file responsibilities

| File | Responsibility |
|---|---|
| `phase9b_common.py` | Memory schema, source-state prompts, JSON parsing, conservative metadata baseline, metrics, and training-row construction. |
| `build_reconstruction_candidates.py` | Convert Phase 9 prompts/evidence into reviewable Phase 9B candidates without silently approving labels. |
| `generate_teacher_reconstructions.py` | Run the authoritative teacher to add a second pending reconstruction candidate and raw generation evidence. |
| `build_reconstruction_dataset.py` | Require accepted rows, preserve connected-group split isolation, and write reconstruction/answer/direct/preference corpora. |
| `evaluate_memory_reconstruction.py` | Compare memory-off, raw, prompted, trained two-pass, and direct strategies with quality, negative-transfer, latency, token, and VRAM metrics. |
| `check_phase_9b_gate.py` | Compare proportional reconstruction benefit against proportional runtime/token cost and write the human stop gate. |
| `finalize_phase_9b_selection.py` | Merge/materialize the exact human-approved checkpoint and write the Phase 9B handoff/rollback manifest. |
| `run_phase_9b.ps1` | Executable Phase 9B orchestration. |
