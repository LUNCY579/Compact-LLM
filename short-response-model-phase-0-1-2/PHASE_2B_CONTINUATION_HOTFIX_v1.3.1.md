# Phase 2B Continuation Hotfix v1.3.1

## Purpose

This hotfix lets an already completed Phase 2B catalog continue from its last accepted frontier stage without replaying the earlier stages. It also prevents accumulated early-stage savings from subsidizing a damaging later deletion and guarantees an exhaustive recovery check before the frontier is declared terminal.

The package contains source, documentation, and tests only. It does not contain model checkpoints, adapters, generated evidence, reports, or other artifacts.

## Changed files

| File | Change |
|---|---|
| `scripts/phase_2_layer_selection/progressive_reduction.py` | Direct frontier resume, provenance validation, legacy-catalog migration, incremental opportunity checks, exhaustive recovery-on-stall, explicit termination records, model-keyed profile validation, and safer existing-output handling. |
| `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1` | Describes and prints the new cumulative/incremental decision and continuation behavior. |
| `scripts/phase_2_layer_selection/README.md` | Adds the exact Stage 10 continuation command and expected behavior. |
| `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md` | Documents resume restoration, marginal acceptance, exhaustive recovery, and safety rules. |
| `scripts/phase_2_layer_selection/FILE_RESPONSIBILITIES.md` | Updates the Phase 2B runner responsibility boundary. |
| `tests/test_phase_2_progressive_reduction.py` | Adds regression tests for direct Stage 11 restoration, settings validation, incremental rejection, and stale profile-cache prevention. |

## Core behavior changes

### 1. Real continuation from the accepted frontier

With `-Resume`, the runner now reads the existing catalog and registry, restores the final accepted frontier checkpoint, and begins at:

```text
max(accepted frontier stage) + 1
```

For the current ten-stage run, the next attempted stage is Stage 11. Stages 1–10 are not reconstructed.

### 2. Cumulative and incremental acceptance

A new deletion is accepted only when both conditions are true:

```text
cumulative memory gain from the full reference
    > cumulative combined quality opportunity loss

incremental memory gain from the immediate accepted parent
    > incremental combined quality opportunity loss
```

The parent-relative comparison prevents a late destructive deletion from consuming the positive margin earned by earlier successful removals.

### 3. Exhaustive recovery before stopping

Each stage first recovery-trains only the configured `RecoveryTopK` raw branches. If no raw or focused-recovery branch is eligible, all remaining raw branches are recovery-trained and evaluated once. The run stops only if this exhaustive fallback also fails.

### 4. Explicit terminal state

The catalog and reports now record a termination reason, including:

- `max_drops_reached`
- `minimum_layer_guard_reached`
- `no_marginally_worthwhile_deletion_after_exhaustive_recovery`
- `no_marginally_worthwhile_deletion_recovery_disabled`

A later `-Resume` does not replay a catalog that already records a true terminal no-progress condition.

### 5. Provenance-safe resume

Post-hotfix catalogs persist:

- model and adapter identity;
- adapter fingerprint;
- train, development, probe, and benchmark hashes;
- evaluator and policy hashes;
- recovery settings and seeds;
- QLoRA and gradient-checkpointing settings;
- response token limit and quality profile.

A later resume refuses to mix changed evidence in the same output root. Increasing `MaxDrops` is explicitly allowed.

The existing pre-hotfix catalog cannot retroactively prove fields it never stored. On its first continuation, all available fields are verified and unavailable fields are written transparently to:

```text
artifacts\phase_2b_quality_v3_thorough_20260801\resume_validation.json
```

### 6. Stale profile-evaluation prevention

Profile validation output directories are keyed by the selected model path. A newly selected Stage 11+ profile cannot accidentally reuse a `summary.json` generated for the old Stage 10 profile.

## Installation

1. Back up the six files listed above.
2. Extract this ZIP into the repository root and allow those files to be replaced.
3. From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_phase_2_progressive_reduction -v
```

The expected result is 12 passing Phase 2B tests. The complete repository test suite used for this hotfix contains 46 passing tests.

## Continue the current thorough run

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Report .\reports\PHASE_2B_phase_2b_quality_v3_thorough_20260801_REPORT.md `
  -GateReport .\reports\PHASE_2B_phase_2b_quality_v3_thorough_20260801_STOP_GATE.md `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

Expected initial continuation message:

```text
[resume] continuing from accepted stage 10; next stage is 11; parent=...stage_10...\model
```

`MaxDrops 27` is an upper bound, not a requirement to delete 27 layers. The runner stops earlier when no layer is worthwhile. Because the source has 28 layers and the safety guard preserves at least two, no more than 26 layers can be accepted for deletion.

## Review after completion

The continuation rewrites the Phase 2B report, catalog, registry, and stop gate with the extended frontier. The stop gate remains `PENDING_HUMAN_REVIEW`; manually review the accepted stages and rejected terminal neighbors before approving a reduced profile for Phase 2C.
