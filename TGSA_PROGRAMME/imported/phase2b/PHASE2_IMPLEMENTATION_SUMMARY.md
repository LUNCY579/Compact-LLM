# Phase-2 TGSA + MTP Implementation Summary

## Added execution layers

- `tgsa/mtp.py`: zero-start offset heads, future/duplicate/shuffled targets,
  normalized auxiliary loss, explicit SFT target masking.
- `tgsa/phase2_model.py`: exact paired 2 x 2 model construction and initialization
  audit, with optional duplicate-next controls.
- `tgsa/phase2_decode.py`: correctness-first verified block decoding with cached
  accepted verification states.
- `tgsa/phase2_data.py`: deterministic, hash-materialized recurrence families.
- `tgsa/phase2_runner.py`: fixed-step paired training, evaluation, checkpointing,
  competence construction, and factorial analysis.
- `tgsa/phase2_analysis.py`: paired effects, interaction, bootstrap intervals,
  and complementarity gates.
- `tgsa/phase2_verify.py`: closed artifact-chain verification and final gate.

## Reused pretrained-retrofit safeguards

The new branch imports the validated Qwen3/Rineya retrofit implementation rather
than reimplementing GQA/RoPE/cache semantics inside the synthetic lab. The
addition `rineya_tgsa/mtp_adapter.py` composes that converter with output-neutral
MTP heads, adapter-only state, real-logit neutrality auditing, and a combined
warm-start schedule.

## Validation performed while developing this branch

- New focused unit suite: target alignment, no-future-input semantics, gradients,
  deterministic data, paired initialization, factorial estimands, exact decoding,
  pretrained-wrapper neutrality, adapter filtering, and stage freezing.
- Local result: **16 focused tests passed**.
- Full non-evidentiary rehearsal: **12/12 cells completed** (four primary plus
  two auxiliary controls across two seeds).
- Exact verified decode: passed for every primary cell.
- Closed evidence verifier: passed.
- Competence: correctly `NO_GO` under the intentionally tiny smoke budget.
- Final smoke decision: `SMOKE_ONLY`, not a quality result.

The real Supa/Rineya checkpoint smoke and adapter training require local model
weights and a Transformers environment; they are executable but were not run in
the CPU-only implementation workspace.
