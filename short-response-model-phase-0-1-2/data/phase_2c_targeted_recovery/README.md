# Phase 2C Dataset

`train.jsonl` and `dev.jsonl` retain the approved Phase 1 SFT data and add deterministic concise-reasoning records covering implication, ordering, comparison, arithmetic, contradiction, set membership, temporal reasoning, and clarification.

Added answers are derived from explicit rules rather than a language model. Train/dev names and objects are disjoint, grammar is generated with article/plural handling, IDs and prompt-target pairs are unique, exact train/dev prompt-target overlap is rejected, and sealed data is never read.

The added rows are deterministically shuffled and interleaved with original records. This ensures limited-prefix calibration and evaluation include both specialized reasoning cases and original conversational controls.

Regenerate reproducibly:

```powershell
python .\scripts\phase_2c_targeted_recovery\build_phase_2c_dataset.py --force
```

`dataset_manifest.json` records counts, ordering, provenance, and SHA-256 hashes for source and generated files.
