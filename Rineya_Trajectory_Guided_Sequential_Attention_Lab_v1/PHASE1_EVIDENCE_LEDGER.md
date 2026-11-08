# TGSA Phase 1 evidence ledger

This ledger distinguishes implemented evidence machinery from empirical outcomes. It must not be edited to imply a capability result before the registered full matrix has run.

## Lineage

- Parent branch: `research/trajectory-guided-sequential-attention`
- Pinned parent commit: `aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e`
- Working branch: `research/tgsa-phase1-model-identifiability`
- Phase 0 claim retained: engineering viability only
- Phase 1 claim state at implementation time: **not yet evaluated**

## Implemented evidence contracts

| Contract | Status | Evidence produced by |
|---|---|---|
| Materialize all examples before training | Implemented | `prepare_phase1_data.py`, dataset manifest |
| Hash train/dev/test files and instance IDs | Implemented | `dataset_manifest.json` |
| Reject latent-signature overlap across splits | Implemented | generator and manifest validator |
| Pair examples, order, updates, optimizer, and seeds across arms | Implemented | deterministic batch stream and run manifests |
| Exact history/delta parameter parity | Implemented | preflight state-structure and parameter audit |
| Exact history/delta initialization parity | Implemented | full state-dictionary SHA-256 equality |
| Exact shared cascade/delta initialization | Implemented | canonical learned-delta supernet copy plus shared-state audit |
| Fixed final checkpoint; no dev selection | Implemented | registered config and run manifest |
| Preserve per-instance predictions | Implemented | condition-specific prediction JSONL |
| Preserve gradients, batch IDs, and batch signatures | Implemented | `train_log.jsonl` |
| Preserve pre/post mechanism telemetry | Implemented | run manifest telemetry records |
| Preserve crashes rather than silently retry | Implemented | `failure_bundle.json` |
| Reject stale complete runs after source/config/data drift | Implemented | matrix CLI reuse guard |
| Refuse confirmatory decision from partial matrix | Implemented | frozen analyzer |
| Keep CI smoke non-confirmatory | Implemented | smoke config plus analysis scope enforcement |

## Registered causal contrasts

| Contrast | Role | Required interpretation |
|---|---|---|
| `learned_delta - history_control` | Primary | Tests finite-difference coordinate against a same-information, same-parameter learned history pathway |
| `learned_delta - cascade` | Supporting | Tests added capability beyond sequential recurrence alone |
| aligned learned delta - batch-rolled learned delta | Falsifier | Tests whether example-aligned control content matters at inference |
| learned delta - history on direct lookup | Guardrail | Detects degradation of short-horizon retrieval |

A delta-versus-cascade win without a delta-versus-history win is **not** evidence that finite differences are the cause. Damage from zeroing or negating the control path is diagnostic only; the distribution-preserving batch-roll contrast is required for the alignment claim.

## Registered advancement gates

The full study advances only when all fixed gates in `configs/phase1_model_identifiability.json` pass:

- delta minus history observed macro accuracy at least `+0.02`, lower 95% bound above zero;
- delta minus cascade observed macro accuracy at least `+0.02`, lower 95% bound above zero;
- aligned minus batch-rolled delta at least `+0.015`, lower 95% bound above zero;
- direct-lookup lower bound versus history at least `-0.01`;
- all five paired seeds, all three arms, hashes, initialization proofs, and source-lineage checks valid.

## Outcome entries

### CI smoke

Purpose: execution, integrity, intervention, resume, and analyzer wiring only.  
Claim value: none.  
Permitted decision: `smoke_only`.

Record the workflow run, branch commit, artifact hash, and any failure bundle here only after the remote workflow completes.

### Registered full Phase 1 matrix

Status: **not run by this implementation commit**.

When run, record:

- exact branch commit;
- dataset-manifest hash;
- matrix-manifest hash;
- analyzer-source hash;
- all arm/seed run-manifest hashes;
- final decision and confidence intervals;
- failed or replaced runs with preserved failure bundles;
- whether any registered gate failed.

## Claim ceiling after a positive result

At most:

> In a controlled tiny decoder trained on the registered synthetic iterative tasks, an aligned finite-difference control representation improved held-out capability beyond an information-matched history representation and the sequential cascade alone.

This ledger does not authorize claims about Qwen/GQA, natural language, long context, efficiency, scaling, novelty, or paper-primary evidence.
