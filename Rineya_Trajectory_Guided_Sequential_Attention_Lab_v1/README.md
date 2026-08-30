# Rineya Trajectory-Guided Sequential Attention Lab v1

**Working hypothesis:** a causal attention block can trade head-level parallelism for iterative refinement by treating sequential heads as steps along a differentiable state trajectory. Later heads observe not only the current state but also its finite-difference velocity, and that velocity controls subsequent computation.

The working name is **Trajectory-Guided Sequential Attention (TGSA)**. This name is deliberately narrower than “differential attention”: the central quantity is a finite difference across sequential-head depth, not a subtraction between simultaneous attention maps.

## Current status

**Phase 1 v3 is complete and frozen. Phase 1 v4 is in development. Phase 2 TGSA × MTP is implemented as a competence-gated rehearsal.** 336 tests.

### What has been demonstrated

- A typed PyTorch reference kernel, a conventional parallel-MHA control, seven
  mechanism-preserving TGSA ablations, a shared carrier state, learned trajectory
  encoding, bounded predictor/corrector gates, Q/K/V conditioning, and soft
  token-importance biasing.
- A **completed confirmatory study** (`tgsa-phase1-identifiability-v3-2026-08-29`):
  60/60 cells, a clean train → barrier → single-evaluate sequence, 2001 passing
  evidence-chain checks. Tagged `tgsa-phase1-v3-result-2026-08-29` and immutable.
- The v3 `decision` was **`inconclusive`**, and the reason is now understood: no
  arm was competent. All three memorized the 5,000-example training set and
  scored at the majority baseline on held-out data — an in-band held-out test
  (same difficulty, novel instances, zero latent-signature overlap) gave 0.236
  seen against 0.045 held-out, versus a 0.0407 majority baseline.
- A parameter-matched sequential multi-token-prediction reference module, direct
  reuse of TGSA's existing differentiable control coordinate, an eight-cell
  development rehearsal, full-ladder mechanical tests, evidence schemas, and a
  fail-closed Phase-1 competence precondition for any later confirmatory run.
- A Phase-2 integrity layer that binds data diversity, sample presentations,
  effective epochs, device identity, and regime/study/cell provenance into each
  development bundle.

> **v3 is not evidence against TGSA.** A contrast between arms that cannot
> perform the task is uninformative in both directions. It constrains the
> experimental setup, not the mechanism.

### What has NOT been demonstrated

- Anything about the TGSA mechanism. No hypothesis in the ladder has been
  confirmed or refuted.
- Improved model quality, at any scale.
- That MTP improves TGSA or that a TGSA × MTP interaction is positive.
- Speculative-decoding speedup. Future-token accuracy and serial target-round
  reduction are not replacements for a cache-aware verifier and timed
  end-to-end generation.
- Any Qwen/GQA, natural-language, long-context, or efficiency claim. The Rineya
  retrofit track is explicitly **non-evidentiary** until the mechanism has been
  fairly tested here.

### Phase 1 v4 — competence-first (in development)

v4 changes only the regime in which the contrast is run, plus a **competence
gate** the confirmatory runner consults before it creates an output root. Eight
arm-blind checks, every threshold frozen from an outcome-blind pilot. The purpose
is not to make TGSA win; it is to build a system in which TGSA can win or lose
for a scientifically meaningful reason.

Status: `V4_COMPETENCE_NOT_YET_ESTABLISHED`. Development work shows that training
exposure and data diversity can trade reliable optimization transitions against
memorization, so they are now explicit regime variables rather than hidden
properties of a step count. No development arm margin is interpreted as a TGSA
result. The registered escalation ladder remains authoritative.

### Phase 2 — TGSA × multi-token prediction

Phase 2 keeps the Phase-1 kernel and control definitions authoritative. It adds
one identical sequential MTP module to every cell and separates three conditions:

- `dormant`: module present, auxiliary objective excluded;
- `backbone`: future-token supervision from the backbone state with zero control;
- `trajectory`: the same module receives TGSA's registered finite-difference or
  information-matched history coordinate.

The current eight-cell CPU configuration is a non-evidentiary rehearsal. The
sealed test split is materialized but never read, all initial states and output
artifacts are hashed, and `confirmatory=true` fails before dispatch unless an
external, frozen, non-development Phase-1 v4 competence report returns `GO`.
Development reports are transitively marked and cannot be laundered into that
prerequisite. See `PHASE2_TGSA_MTP_DESIGN.md`, `PHASE2_TGSA_MTP_RUNBOOK.md`, and
`PHASE2_V4_REGIME_INTEGRATION.md`.

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

Phase-2 rehearsal:

```powershell
python scripts/phase2_tgsa_mtp_rehearsal.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir artifacts/phase2_smoke/dataset `
  --output-dir artifacts/phase2_smoke/study `
  --repo-root ..
python scripts/verify_phase2_integrity.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir artifacts/phase2_smoke/dataset `
  --output-dir artifacts/phase2_smoke/study
```

The generated `artifacts/` directory is intentionally ignored. Promote an artifact into a paper evidence bundle only after attaching immutable model/data/config hashes and the exact commit.

## Repository map

| Path | Purpose |
|---|---|
| `tgsa/` | Reference attention, configuration, MTP extension, integrity, telemetry, and statistics |
| `tests/` | Causality, gradient, mask, shape, ablation, MTP, integrity, and statistical tests |
| `scripts/` | Smoke execution, ablation sanity matrix, manifest validation, Phase-2 rehearsal |
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
| `PHASE1_V4_PILOT_REPORT.md` | Outcome-blind pilot and the frozen thresholds |
| `PHASE1_V4_RUNBOOK.md` | Validated v4 execution sequence and budget |
| `PHASE2_TGSA_MTP_DESIGN.md` | MTP architecture, factorial design, claims, and gates |
| `PHASE2_TGSA_MTP_RUNBOOK.md` | Validated rehearsal commands and family-model roadmap |
| `PHASE2_V4_REGIME_INTEGRATION.md` | How v4's dose/diversity trend constrains Phase-2 design |

## Non-negotiable research rules

1. A raw linear `delta` term is not sufficient evidence of a new mechanism; it can often be algebraically absorbed into weights on the two preceding states.
2. Sequential depth must be separated from trajectory guidance through matched cascade controls.
3. Parameter, training-compute, inference-compute, and wall-clock views must all be reported; no single fairness view is sufficient.
4. The existing long-context demo may be diagnostic, but a paper claim requires an independently authored sealed bank.
5. Telemetry is explanatory evidence only unless its predictive relationship is preregistered and validated out of sample.
6. No “first” or novelty claim is permitted until the final related-work search and citation audit are complete.
7. Future-token accuracy or serial-round reduction is not speculative-decoding speed; speed requires an exact cache-aware verifier and end-to-end timing.
8. A development competence report cannot certify a confirmatory matrix, even when its numeric verdict is `GO`.
