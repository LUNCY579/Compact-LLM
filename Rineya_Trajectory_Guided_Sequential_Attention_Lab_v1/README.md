# Rineya Trajectory-Guided Sequential Attention Lab v1

**Working hypothesis:** a causal attention block can trade head-level parallelism for iterative refinement by treating sequential heads as steps along a differentiable state trajectory. Later heads observe not only the current state but also its finite-difference velocity, and that velocity controls subsequent computation.

The working name is **Trajectory-Guided Sequential Attention (TGSA)**. This name is deliberately narrower than “differential attention”: the central quantity is a finite difference across sequential-head depth, not a subtraction between simultaneous attention maps.

## Current status

**Phase 1 v3 is complete and frozen. Phase 1 v4 rejected its registered regime family. This branch is the canonical Phase 2A trajectory-mechanism branch; all current TGSA × MTP runs remain development-only.** 396 tests.

### Canonical branch role

The TGSA × MTP programme uses the existing branches as a staged chain rather than merging two different MTP interventions into one ambiguous implementation:

1. `research/tgsa-phase1-v4-competence-first` is the frozen scientific boundary. Its registered task/regime family is rejected and must not receive Phase-2 implementation commits.
2. `research/tgsa-phase2a-ci-and-competence` (this branch) is **Phase 2A: trajectory-mechanism identification**. Its MTP semantics are `sequential_teacher_forced_recursive_v1`. Its archived predecessor `research/tgsa-phase2-mtp-complementarity` held this role until 2026-08-31 and is fully contained here; read it through `archive/tgsa-phase2a-superseded-2026-08-31`.
3. `research/tgsa-mtp-rineya-complementarity-v1` is **Phase 2B: system and decoding complementarity**. Its MTP semantics are `parallel_independent_offsets_v1`.

The Phase 2A and Phase 2B branches are related stages, not replications. Their checkpoints, evidence roots, estimands, and schema identities must remain distinct. See `TGSA_MTP_CANONICAL_PATH.md`.

### What has been demonstrated

- A typed PyTorch reference kernel, a conventional parallel-MHA control, seven mechanism-preserving TGSA ablations, a shared carrier state, learned trajectory encoding, bounded predictor/corrector gates, Q/K/V conditioning, and soft token-importance biasing.
- A completed Phase 1 v3 confirmatory study (`tgsa-phase1-identifiability-v3-2026-08-29`) whose decision was `inconclusive` because no arm was competent.
- A completed Phase 1 v4 competence programme whose registered regime family was rejected before a confirmatory mechanism matrix was dispatched. The apparent transition was state-update arithmetic crossing from floor to saturation; the pointer families remained inert. This is an apparatus result, not evidence for or against TGSA.
- A parameter-matched sequential future-prediction reference module that directly reuses TGSA's existing differentiable control coordinate, plus an eight-cell development rehearsal, full-ladder mechanical tests, evidence schemas, and a fail-closed competence precondition.
- A Phase-2 integrity layer that binds MTP semantics, data diversity, sample presentations, effective epochs, device identity, and regime/study/cell provenance into each development bundle.

> **Neither v3 nor v4 is evidence against TGSA.** A contrast over non-learning, inert, or saturated task families cannot identify the mechanism in either direction.

### What has NOT been demonstrated

- Anything about the TGSA mechanism. No hypothesis in the H1a/H1b/H1c ladder has been confirmed or refuted.
- Improved model quality, at any scale.
- That trajectory-conditioned sequential future prediction improves TGSA.
- That this Phase 2A predictor provides useful speculative-decoding acceleration; it is a mechanism/training reference, not the production verifier path.
- Any Qwen/GQA, natural-language, long-context, or wall-clock efficiency claim.

### Phase 1 v4 — frozen result boundary

The v4 gate prevented a 9–26 GPU-hour confirmatory matrix from being run on an uninformative band. At equal 1.28 million sample presentations, 25.6 effective epochs made the arithmetic transition common but produced excessive memorization, while 6.4 effective epochs reduced memorization but made transition completion unreliable. More importantly, `direct_lookup` and `pointer_chase` remained inert and `state_update` saturated.

Status: `V4_REGIME_FAMILY_REJECTED`. The old task/regime family is permitted only for regression, integrity, and smoke testing. Any new mechanism study requires a new competence preregistration with data diversity and sample presentations varied separately, and with primary families that are both learnable and non-saturated.

### Phase 2A — trajectory mechanism bridge

This branch adds one identical sequential future-prediction module to every cell and separates three conditions:

- `dormant`: module present, auxiliary objective excluded;
- `backbone`: future-token supervision from the backbone state with zero control;
- `trajectory`: the same module receives TGSA's registered finite-difference or information-matched history coordinate.

The predictor is sequential across future horizons. During teacher-forced training, the previous true future-token embedding can enter the next horizon transition; recursive inference instead consumes the previous prediction. This is intentionally a different intervention from Phase 2B's independent offset heads.

The permitted Phase 2A claim, after a new competence-qualified regime exists, is narrow:

> Whether the registered TGSA trajectory coordinate contributes incremental future-token predictive information beyond parameter-matched backbone and history controls.

The supplied configuration is explicitly `development_only` and `obsolete_regime`; it may exercise mechanics and evidence closure, but it cannot authorize a mechanism claim or a confirmatory launch.

## Core recurrence

For a shared carrier state `c[j]`:

```math
v_{j-1}=c_{j-1}-c_{j-2}
```

```math
m_j=\Phi_j(\operatorname{RMSNorm}(v_{j-1}))
```

```math
\hat c_j=c_{j-1}+\alpha_j\odot P_jm_j
```

Head `j` uses `m[j]` to control its feature injection and optionally its Q/K/V projections and token-importance prior. Attention then supplies a correction:

```math
c_j=\operatorname{Norm}(\hat c_j+\beta_j\odot C_jh_j).
```

The trajectory is therefore a **control signal**, not merely another linear residual. See `ARCHITECTURE.md` for the full operator and the linear-collapse warning.

## Quick start

From this directory:

```powershell
python -m pip install -e ".[test]"
pytest -q
python scripts/run_smoke.py
python scripts/validate_manifest.py artifacts/phase0_smoke.json
python scripts/run_ablation_matrix.py
```

Phase-2A rehearsal:

```powershell
python scripts/phase2_tgsa_mtp_rehearsal.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir artifacts/phase2a_smoke/dataset `
  --output-dir artifacts/phase2a_smoke/study `
  --repo-root ..
python scripts/verify_phase2_integrity.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir artifacts/phase2a_smoke/dataset `
  --output-dir artifacts/phase2a_smoke/study
```

The generated `artifacts/` directory is intentionally ignored. Promote an artifact into a paper evidence bundle only after attaching immutable model/data/config hashes and the exact commit.

## Repository map

| Path | Purpose |
|---|---|
| `TGSA_MTP_CANONICAL_PATH.md` | Branch/stage map, semantic split, and promotion gates |
| `tgsa/` | Reference attention, configuration, sequential future predictor, integrity, telemetry, and statistics |
| `tests/` | Causality, gradient, mask, shape, ablation, MTP, integrity, and statistical tests |
| `scripts/` | Smoke execution, ablation sanity matrix, manifest validation, Phase-2A rehearsal |
| `configs/` | Frozen/rehearsal configurations and preregistration templates |
| `evidence/` | Run-manifest, integrity, and study-report schemas |
| `ARCHITECTURE.md` | Equations, invariants, complexity, and causal semantics |
| `EXPERIMENT_PLAN.md` | Staged scientific and engineering pipeline |
| `PREREGISTRATION.md` | Primary hypotheses, endpoints, controls, and analysis lock |
| `QWEN_INTEGRATION_PLAN.md` | RoPE/GQA/KV-cache-compatible implementation plan |
| `RELATED_WORK_SCOPE.md` | Prior-art boundaries and literature-audit checklist |
| `CLAIMS_AND_FALSIFIERS.md` | Claim ladder and explicit disconfirmation criteria |
| `DESIGN_DECISIONS.md` | Architectural decisions, rationale, and deferred questions |
| `EVIDENCE_LEDGER.md` | Demonstrated evidence versus still-missing claims |
| `PAPER_EVIDENCE_CHECKLIST.md` | Reproducibility and arXiv release gate |
| `PHASE1_V3_PREREGISTRATION.md` | The frozen v3 confirmatory design |
| `PHASE1_V3_RESULT.md` | The v3 result and what it does and does not support |
| `PHASE1_V3_COMPETENCE_POSTMORTEM.md` | Why v3 returned `inconclusive` |
| `PHASE1_V4_PREREGISTRATION.md` | v4 design, competence gate, escalation ladder |
| `PHASE1_V4_COMPETENCE_RESULT.md` | Frozen rejection of the registered v4 regime family |
| `PHASE1_V4_PILOT_REPORT.md` | Outcome-blind pilot and frozen thresholds |
| `PHASE1_V4_RUNBOOK.md` | Validated v4 execution sequence and budget |
| `PHASE2_TGSA_MTP_DESIGN.md` | Phase 2A architecture, mechanism estimands, claims, and gates |
| `PHASE2_TGSA_MTP_RUNBOOK.md` | Phase 2A rehearsal and future-regime roadmap |
| `PHASE2_V4_REGIME_INTEGRATION.md` | How the v4 result constrains all Phase-2 work |

## Non-negotiable research rules

1. A raw linear `delta` term is not sufficient evidence of a new mechanism; it can often be algebraically absorbed into weights on the two preceding states.
2. Sequential depth must be separated from trajectory guidance through matched cascade controls.
3. Parameter, training-compute, inference-compute, and wall-clock views must all be reported; no single fairness view is sufficient.
4. The existing long-context demo may be diagnostic, but a paper claim requires an independently authored sealed bank.
5. Telemetry is explanatory evidence only unless its predictive relationship is preregistered and validated out of sample.
6. No “first” or novelty claim is permitted until the final related-work search and citation audit are complete.
7. Future-token accuracy or serial-round reduction is not speculative-decoding speed; speed requires an exact cache-aware verifier and end-to-end timing.
8. A development competence report cannot certify a confirmatory matrix, even when its numeric verdict is `GO`.
9. `study_kind` and `mtp_semantics` are part of scientific identity; evidence from Phase 2A and Phase 2B may never be pooled or relabeled as replications.
10. The rejected v4 regime may not be revived by adding MTP, changing the objective, or increasing steps under the same registration.
