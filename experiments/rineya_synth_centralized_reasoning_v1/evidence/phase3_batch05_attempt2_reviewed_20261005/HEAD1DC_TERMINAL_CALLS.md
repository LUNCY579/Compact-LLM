# Exact HEAD1DC05A2/06A1 terminal calls

New wrapper `run_head1dc_terminal.py` changes only the frozen call frame:
HEAD `1dc787533d4e57257c5ae20b8e5324fb49894b70`, exact reviewed05A2/06A1
inventory paths/hashes, config anchors and new attempt basenames. It reuses the
preserved29790d body through the previously reviewed sole ordinal4→explicit5/6
selector adaptation. Original source131 and all six auditor dependencies remain
unchanged. Eleven source/inert-call controls pass with unchanged source bytes.
No live raw/evidence, model/runtime, probe, worker or actual copy/audit was read
or invoked during preparation. The original no-worker/failed preservation and
complete current root/descendant death/cleanup/stable-copy gates remain unchanged.

Batch05 attempt2 still uses the exact unchanged
`configs/SYNTH_SHORT_M0_ATOMIC_BATCH05_PHASE3_A1.json` file, SHA
`b8e2efe1e832ca865bc8a5de8f48b9d601246cd14335a2e6e09ecc4a1362ba00`.
Its new inventory is `SYNTH_SHORT_PHASE3_BATCH05_A2_HEAD1DC_AUDIT_INVENTORY_001.json`,
file SHA `f2fad910e7d3dd9f82793db0864d633f4bbf1cf656af7e082fcd7ae83a6aab96`.
Raw/preserved basename is `short_m0_atomic_batch05_phase3_attempt2_20261005`.

Batch06 config SHA is
`06677a414195afb137daba2f434edd8f4ba0a4f82b029c52e17582d8fb6e095c`;
inventory `SYNTH_SHORT_PHASE3_BATCH06_A1_HEAD1DC_AUDIT_INVENTORY_001.json`,
file SHA `9dba239f5ccce2ab604f2f80e1e8c17e14610382a36eb0c9c9d5ad1a6a01f847`.
Raw/preserved basename is `short_m0_atomic_batch06_phase3_attempt1_20261005`.
Both retain exact profile body
`2d5547910aa57b0509a532972d899ceb5c913e41274bfa773d25536e6d8c9676`.

Root invokes preservation only after terminal, with separately selected exact
supervisor file and current full-worker/all-observed-birth death proof anchors.
The proof stays outside the immutable attempt leaf:

```powershell
python -I -B experiments/rineya_synth_centralized_reasoning_v1/artifacts/phase3_remaining_batches_work/run_head1dc_terminal.py preserve --ordinal 5 --root-terminal-frozen --supervisor-file-sha256 ROOT_FROZEN_SUPERVISOR_SHA --current-death-evidence ROOT_CURRENT_DEATH_PROOF_PATH --current-death-file-sha256 ROOT_CURRENT_DEATH_PROOF_FILE_SHA
```

For06 use `--ordinal 6`; all paths/anchors remain fixed by that ordinal. A
no-worker admission denial can omit both death arguments and stays failure-only.
Preservation prints the exact new body hash. After Root independently selects
that anchor, successful evidence may use the unchanged full native auditor:

```powershell
python -I -B experiments/rineya_synth_centralized_reasoning_v1/artifacts/phase3_remaining_batches_work/run_head1dc_terminal.py audit --ordinal 5 --root-terminal-and-preserved --preservation-body-sha256 ROOT_SELECTED_PRESERVATION_BODY_SHA
```

For06 again use `--ordinal 6`. Failed, no-worker and CPU-only receipts refuse a
successful native audit. No dynamic dispatch, retry, policy substitution or
future authority is added by this wrapper.
