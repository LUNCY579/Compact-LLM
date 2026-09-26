# Cross-Scale Transport Failure Localization — Plan-Index Report (v3)

Protocol index: `00_protocol/amendment_v3_transport_diagnostics.md` `3.0.0-diagnostics-index`

Executed measurement parents (immutable):

- `13_v3_transport_diagnosis/` — E1–E4, E5–E7 (donor side)
- Reports: `13_v3_transport_diagnosis/report/v3_report.md`,
  `13_v3_transport_diagnosis/report/v3_1_donor_side_report.md`

This tree (`13_v3_transport_diagnostics/`) is the **plan deliverable index**. It
does not rewrite frozen measurements. It maps plan phases onto executed evidence,
records Gate decisions, and states escalation status.

Sealed holdout: **CLOSED** (never read). DEV-FULL: **not justified**. Candidate
checkpoints created by this index: **0**.

## What was run (this index session)

| Plan phase | Action | Result |
|---|---|---|
| Phase 0 | Protocol index + evidence tree | complete |
| Phase 1 | `sequential_runner.py`, `resource_trace.py` | complete |
| Phase 2 | Curate atlas from E4+E5 | complete (`atlas/`) |
| Exp A | Transport sanity from E1/E1c/E2 | **FAIL** |
| Exp B | Localization from E3 | **FAIL** (no promising region) |
| Exp C | Consensus vs Δ_Supa from E2 | **FAIL** (margin ≪ 0.02) |
| Exp D | Translator curation from E6 | **FAIL** (donor redundant; low-rank weak) |
| Phase 7 | Micro-injection | **NO_GO** (`candidates/NO_GO.json`) |
| Phase 8 | This report + registry | complete |

Heavy donor reloads were **not** repeated (hardware-safe). Artifact hashes were
re-verified via `verify_cached_artifacts()` → status `ok`.

## What was skipped, and why

- **New full-model merges / TIES/DARE sweeps:** forbidden by plan; already near-null in v2.
- **Micro-injection / α sweep:** upstream Gates FAIL; creating candidates would not be evidence-backed.
- **MERGEDEV-fast / DEV-FULL / sealed:** no survivor; sealed remains closed.
- **Re-running E4/E5/E6 captures:** inputs unchanged; would only pressure RAM/disk.

## Resource consumption (index assembly)

| Quantity | Value |
|---|---:|
| GPU hours | 0.0 |
| Model loads | 0 |
| Peak concurrent models | 0 |
| Wall time | ~16 s (assemble script) |
| See also | `report/resource_assemble.json`; full E-series accounting in diagnosis reports |

## Strongest result

Functional cross-scale correspondence is real: E5 linear CKA at matched relative
depth is ~0.82–0.96 against prompt-shuffled controls ~0.03–0.21. Uniform
relative-depth mapping is functionally diagonal. P4's closed form shows it
transmits ~275 effective scalars and **zero donor directions**.

## Strongest negative result

1. The v1 headline 0.932 cross-donor cosine does **not** survive shape-matched
   Gaussian nulls (null mean ~0.9988 > real 0.9682) — operator/shape induced.
2. Transported consensus vs Δ_Supa cosine ≈ **+0.0002**, indistinguishable from
   random/sign/layer-shuffle controls.
3. Full ridge basis maps work (held-out cosine up to ~0.98), but rank-4/8/16 are
   weak and **adding donor features does not improve** predicting Supa residuals
   over Qwen3-0.6B alone.

## Decision policy outcome

| Outcome | Status |
|---|---|
| A — Parameter transfer works | **Not reached** |
| B — Functional transfer exists | **Partial** (representations align; not a tiny portable update) |
| C — Localized transfer | **Rejected** (E3 all buckets ≈ 0) |
| D — Non-portable for tested T | **Supported for P4/P5** |

Plan-index code: `B_partial_then_D_for_tested_T`.

## Did the cross-scale hypothesis get stronger or weaker?

**Weaker for spectral/statistic transport (P4/P5).** Those operators cannot carry
directions into a useful target subspace.

**Not settled against all cross-scale transfer.** Representation alignment exists;
the remaining open path is an **aligned** transport that rebases donor *weights*
through a fitted basis map (see frozen `14_v4_aligned_transport` amendment), not
another hand-designed merge on P4 deltas.

## Exact next experiment justified by the evidence

Proceed only under amendment v4 (aligned transport): fit per-layer orthogonal
Procrustes maps and test whether a rebased donor delta becomes distinguishable
from controls in target space **before** any Supa injection. Do not reopen
sealed data. Do not run DEV-FULL until a Gate-3 two-control survivor exists.

## Escalation

| Gate | Justified? |
|---|---|
| DEV-FULL | **No** |
| Sealed | **No — remains CLOSED** |

## Machine-readable pointers

- `atlas/similarity_matrix.json`, `atlas/heatmap.md`
- `transport_sanity/ranked_table.json`
- `localization/promising_regions.json` (empty)
- `consensus_analysis/target_basis_report.json`
- `functional_translator/translator_summary.json`
- `candidates/NO_GO.json`
- `candidates/registry.jsonl`
- `report/decision_summary.json`
