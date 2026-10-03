Adam moment spill removed 2026-09-14T01:09:21Z.

These were 124 optimizer scratch files (~2651 MiB) written by StreamedAdam during this
run. They are transient optimizer state, not evidence: the run's measurements live in
START.json, probe_*.json and resources.jsonl in the parent directory, and the trained weights
(if the run saved any) live under outputs/arm-checkpoints/.

Removed to reclaim disk; the run directory itself is preserved unchanged. Nothing here was a
measurement and nothing recorded in this run's artifacts depends on these bytes.
