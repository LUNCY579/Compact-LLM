# Phase 2B verified execution manual

“Verified” means the scripts contain explicit provenance, reload checks, common evaluation, and stop conditions. It does not claim a GPU run has already succeeded.

## Why this is inside Phase 2

Phase 2A found no selective one-, two-, or four-layer adapter that recovered the all-layer SFT result. Phase 2B changes the question from **where to train** to **what can be removed and then recovered**. Running it before DPO gives Phase 3 a chance to align the reduced architecture instead of deleting layers only after all behavior training is complete.

## Preflight

1. Keep `artifacts/dense_sft/adapter` and `artifacts/dense_sft_evaluation` unchanged.
2. Confirm `reports/PHASE_1_STOP_GATE.md` contains an explicit human PASS.
3. Confirm adequate disk space. One Qwen3-0.6B full checkpoint can be roughly gigabyte-scale depending on precision.
4. Activate the CUDA environment used by Phase 1.
5. Close other GPU workloads before comparing latency or memory.

## Recommended first run

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -MaxDrops 4 `
  -RecoveryTopK 3 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

The wrapper uses Quality Evaluator v4.5 and exposes search, provenance, output, and quality-profile controls. It does not expose absolute promotion thresholds. `ProbeLimit=50` remains a workload setting, not a quality standard. The default `probe_50_v4_5.jsonl` and the thorough `probe_150_thorough_v4_5.jsonl` contain explicit case-level response contracts; legacy uncontracted probes are rejected for official Phase 2B selection.

For the established thorough probe, use an isolated run:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2_progressive_reduction_v1_3_relative `
  -Report .\reports\PHASE_2B_V1_2_THOROUGH_REPORT.md `
  -GateReport .\reports\PHASE_2B_V1_2_THOROUGH_STOP_GATE.md `
  -MaxDrops 10 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Increase `MaxDrops` only after reviewing the first frontier. Use `-Resume` only when code, evaluator, opportunity policy/profile, probe, model, and data have not changed. `ProbeLimit` controls probe creation and provenance metadata; supplying `150` avoids a 150-row probe being reported as a 50-case configuration. Custom output roots derive unique report names automatically, but explicit report paths are preferred for important experiments.

## Review order

1. Read `PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md`.
2. Inspect every accepted stage's `OUTPUT_DEVIATION.md`.
3. Inspect the lowest-score rejected neighbors to confirm the stopping point is meaningful.
4. Check critical and major concrete defect/contract labels manually.
5. Confirm target NLL did not sharply worsen even when generated score remained 1.0.
6. Compare checkpoint weight bytes and parameters, not only adapter size.
7. Choose approved profiles in `PHASE_2B_STOP_GATE.md`, for example:

```text
- Human decision: `PASS`
- Approved profiles: `full_quality,balanced`
```

Do not approve `min_memory` without manually reviewing its raw generations, evidence provenance, and opportunity components.

## Continuing Phase 3

Full quality, preserving the existing behavior:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2Profile full_quality
```

Balanced reduced branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2Profile balanced
```

Minimum-memory branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2Profile min_memory
```

Profile-specific Phase 3 outputs are written side by side so the original accepted Phase 3 artifacts are not overwritten.

## Failure and rollback

A failed structural sweep is still useful evidence. Keep the reports, leave reduced profiles unapproved, and continue with `full_quality`. The original base model, SFT adapter, full materialized reference, and every accepted frontier model are never modified in place.

## Continuation behavior after a completed catalog

The continuation runner treats the accepted frontier as durable experiment state. With `-Resume`, it does not replay completed stages merely to reconstruct their parent. It reads `model_catalog.json`, verifies provenance, restores the final accepted `inference_model`, restores the original-layer lineage, and begins at `max(frontier.stage) + 1`.

For the existing Stage 10 run, the expected console message is similar to:

```text
[resume] continuing from accepted stage 10; next stage is 11; parent=...stage_10...\model
```

A candidate is promotable only when both comparisons pass:

```text
cumulative memory gain from full reference > cumulative quality opportunity loss
AND
incremental memory gain from immediate parent > incremental quality opportunity loss
```

This prevents a poor late deletion from consuming margin earned by earlier successful deletions.

### Recovery-on-stall

Each stage still recovery-trains only the configured top `RecoveryTopK` raw candidates first. If neither a raw candidate nor one of those focused recovery candidates is eligible, the runner recovery-trains every remaining raw candidate once. Only after this exhaustive fallback also fails does it record:

```text
no_marginally_worthwhile_deletion_after_exhaustive_recovery
```

If recovery is disabled, the distinct terminal reason is:

```text
no_marginally_worthwhile_deletion_recovery_disabled
```

### Resume safety

A post-hotfix catalog stores file and adapter fingerprints plus recovery, evaluator, data, and quality-policy settings. A later `-Resume` refuses to mix changed evidence in the same output root. Increasing only `MaxDrops` is allowed. Use a new `OutputRoot` when changing the probe, training data, recovery settings, evaluator, policy, adapter, or model.

The first resume from a pre-hotfix catalog cannot retroactively verify settings the old catalog never stored. Those fields are listed transparently in:

```text
<OutputRoot>\resume_validation.json
```

The runner also audits every previously accepted frontier transition with the incremental rule before beginning the next stage. If a historical accepted transition fails that audit, continuation stops rather than silently building on a subsidized deletion.

## Re-score a pre-v4.5 run

Use `run_phase_2b_rescore_v4_5.ps1` to re-score stored generations without loading a model. Official migration mode backs up old result files, rebuilds full-reference deviations, archives stale parent-relative decisions, and prepares a Stage-1 decision replay. Run the original Phase 2B command with `-Resume` afterward. Evidence is reused only when its parent-model lineage, retained layer IDs, benchmark, evaluator configuration, and recovery settings match.

A changed evaluator can select a different parent. Downstream checkpoints produced from the old parent cannot be made valid through scoring alone, so the runner regenerates only that divergent suffix of the search tree.

## Incremental evidence, interruption, and report recovery

Phase 2B treats every completed example and candidate decision as durable experiment state. The generation evaluator appends one JSON object to `generation/results.jsonl` immediately after the case finishes, flushes it, and calls `fsync`. It is therefore safe to inspect partial evidence or interrupt the sweep without losing all completed cases.

### Live files

The run root is refreshed throughout the experiment:

| File | Update cadence | Meaning |
|---|---|---|
| `LIVE_PROGRESS.json` | Every completed generation example | Atomic machine-readable current state and provisional metrics. |
| `LIVE_PROGRESS.md` | Every `ProgressUpdateInterval` examples, default 5 | Human-readable view. Incomplete candidates display an explicit `PROVISIONAL` warning. |
| `RUN_STATE.json` | Candidate/stage/orchestration transitions | Durable current frontier and last accepted checkpoint. |
| `PARTIAL_RUN_REPORT.md` | After each gated candidate and completed stage | Cumulative completed/incomplete stages, candidates, gate reasons, progression, missing artifacts, and pending work. |

Watch the Markdown safely with path polling because the file is atomically replaced:

```powershell
while ($true) {
  Clear-Host
  Get-Content .\artifacts\phase_2_progressive_reduction\LIVE_PROGRESS.md -Raw
  Start-Sleep -Seconds 5
}
```

### Candidate and stage durability

Each candidate directory contains `candidate_status.json`. The lifecycle states are:

- `pending`, `preparing`, `evaluating`
- `evaluation_complete`
- `recovering`, `recovery_complete`
- `gating`
- `accepted`, `rejected`
- `interrupted`, `failed`

A completed candidate also receives `candidate_row.json` and `CANDIDATE_REPORT.md`. The containing stage receives `stage_status.json` and `STAGE_XX_REPORT.md`. A stage report is generated independently of later stages.

`accepted` means the existing relative quality-versus-resource gate found the candidate worthwhile. It does not itself prove the candidate was selected. `promoted=true` and `promoted_checkpoint` record that the candidate actually became the next parent.

The orchestrator verifies both of these persisted conditions immediately before materialization:

```text
candidate_status.status == "accepted"
candidate_status.promotion_allowed == true
```

A rejected candidate therefore cannot become the parent of the next pruning stage, even if an in-memory ranking row is malformed or stale.

### Resume after interruption

Use the exact original settings and add `-Resume`:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

Resume validates the evaluation manifest and Phase 2B run provenance, repairs a malformed final JSONL line without deleting earlier valid rows, skips stable example IDs already completed, reuses fully gated candidates, and restores the last accepted frontier parent. A changed model, benchmark, evaluator policy, recovery setting, or data fingerprint requires a new output root.

Ctrl+C and Python/subprocess failures mark the in-flight candidate and run, refresh both live files, rebuild the partial report when possible, and print a complete Python resume command. A report-generation failure after a candidate gate does not erase the already persisted accepted/rejected decision.

### Rebuild reports without inference

Use the normal wrapper:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -OutputRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -RebuildReport `
  -AllowPartialReport
```

Or invoke the builder directly:

```powershell
python .\scripts\phase_2_layer_selection\build_phase_2b_report.py `
  --run-root .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  --allow-partial `
  --rebuild
```

The builder never runs model inference. It scans saved JSON/JSONL/status artifacts, repairs recoverable final JSONL tails, reconstructs missing generation aggregates where possible, writes candidate and stage reports, and regenerates `PARTIAL_RUN_REPORT.md`. It creates the canonical final report only when the run is durably marked complete and its final-report inputs exist.

Older artifact folders are supported on a best-effort basis. When no saved gate assessment proves acceptance, the builder labels the status as inferred and leaves `promotion_allowed=false`. It does not invent a historical winner.

### Limitations

- Incremental per-example persistence applies to generation evaluation. Teacher-forcing NLL and footprint measurement remain candidate-level subprocess artifacts.
- A hard machine power loss can leave an orphaned atomic temp sibling. JSON/JSONL readers recover a complete validated orphan only when the destination is absent and the original writer process is no longer alive.
- Live quality and defect counts remain provisional until all required benchmark examples complete; they must not be used for promotion.
- Final release approval still requires the existing human stop gate.
