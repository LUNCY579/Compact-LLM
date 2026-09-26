# Amendment v3 — Transport Failure Localization (Plan Index)

Protocol version: `3.0.0-diagnostics-index`

Parents (immutable):

- v1 `preregistration.md` `1.0.0`
- `amendment_v2_exploratory.md` `2.0.0-exploratory`
- Executed diagnostic amendment: `13_v3_transport_diagnosis/protocol/amendment_v3_transport_diagnosis.md` `3.0.0-diagnostic`
- Executed donor-side amendment: `13_v3_transport_diagnosis/protocol/amendment_v3_1_donor_side.md` `3.1.0-diagnostic`

Research branch: `research/rineya-cross-scale-qwen4b-8b`

## Purpose of this file

This amendment is the **plan-index companion** to the already-executed
`13_v3_transport_diagnosis` evidence tree. It does not reopen sealed evaluation,
does not rewrite frozen v1/v2/v3 measurement files, and does not authorise new
merge recipes.

It exists so the plan's named phases (atlas → Exp A–D → conditional
micro-injection → report) map one-to-one onto the measured E-series without
duplicating compute or pressure on limited host memory.

## Mapping: plan phases → executed experiments

| Plan phase | Plan name | Executed evidence |
|---|---|---|
| Phase 2 | Functional alignment atlas | E4 (0.6B), E4b (mean-pool), E5 (cross-scale) |
| Exp A | Transport sanity | E1, E1b, E1c (channel capacity + nulls) |
| Exp B | Functional localization | E3 (depth/module); E5 depth surface |
| Exp C | Consensus vs target basis | E2 |
| Exp D | Low-rank functional translator | E6, E6b |
| Phase 7 | Micro-injection | **NO_GO** — E2/E3 found no target-relevant region |
| Phase 8 | Research report | `13_v3_transport_diagnosis/report/` + this tree's `report/` |

## Frozen rules (inherited)

- Sealed holdout remains **CLOSED**.
- DEV-FULL is not authorised without a Gate-3 two-control survivor.
- Two-control rule unchanged: beat ungated Supa **and** v1 `CONS_MEAN_A1500`.
- Candidate creation cap for this index tree: **0** unless upstream PASS is newly
  demonstrated (E2/E3 currently fail that bar).
- CPU-first; one model at a time; never load 4B+8B+0.6B concurrently.
- Do not reopen v1/v2 TIES/DARE/sign-consensus grids without new scientific reason.

## Artifact hashes (parents)

Verified at index freeze via `verify_cached_artifacts()`:

- Δ4 digest `8edf2d4f…`
- Δ8 digest `2951457e…`
- projected4 weight SHA `43d2d2e7…`
- v1 consensus weight SHA `5758b7af…`

## Outcome policy (plan decision table)

| Outcome | Condition | Status after executed E-series |
|---|---|---|
| A — Parameter transfer works | Micro-injection beats both controls | **Not reached** |
| B — Functional transfer exists | Atlas shows structure; weight transport fails | **Partial**: E5 shows structure; E6 shows donor is linearly redundant for predicting Supa residuals |
| C — Localized transfer | Signal in 1–2 buckets | **Rejected** by E3 (all buckets ≈ 0 vs Δ_Supa) |
| D — Non-portable (for tested T) | Strong param consensus, no useful target signal | **Supported for P4/P5** |

## What this index tree may contain

- Curated summaries and pointers into `13_v3_transport_diagnosis/`
- Reusable orchestration modules (`sequential_runner`, `resource_trace`)
- Explicit micro-injection NO_GO record
- Plan-aligned research report

It may **not** silently discard negative evidence or reopen sealed data.
