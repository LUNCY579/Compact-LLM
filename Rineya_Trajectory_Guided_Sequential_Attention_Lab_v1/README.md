# Rineya Trajectory-Guided Sequential Attention Lab v1

**Working hypothesis:** a causal attention block can trade head-level parallelism for iterative refinement by treating sequential heads as steps along a differentiable state trajectory. Later heads observe not only the current state but also its finite-difference velocity, and that velocity controls subsequent computation.

The working name is **Trajectory-Guided Sequential Attention (TGSA)**. This name is deliberately narrower than “differential attention”: the central quantity is a finite difference across sequential-head depth, not a subtraction between simultaneous attention maps.

## Current status

**Phase 1 v3 is complete and frozen. Phase 1 v4 is in development. Phase 2 TGSA + MTP is executable and non-evidentiary.** 311 tests.

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
- An executable TGSA × multi-token-prediction (MTP) pipeline with paired
  initialization, prompt-masked future-token objectives, exact verified block
  decoding, factorial analysis, arm-blind competence gating, and a closed
  evidence verifier. Its smoke configuration is intentionally `SMOKE_ONLY`.

> **v3 is not evidence against TGSA.** A contrast between arms that cannot
> perform the task is uninformative in both directions. It constrains the
> experimental setup, not the mechanism.

### What has NOT been demonstrated

- Anything about the TGSA mechanism. No hypothesis in the ladder has been
  confirmed or refuted.
- Improved model quality, at any scale.
- A real Rineya TGSA + MTP quality result or wall-clock speedup. The combined
  retrofit track is explicitly **non-evidentiary** until a real checkpoint,
  held-out evaluation, and deployment-path timing are completed.

### Phase 1 v4 — competence-first (in development)

v4 changes only the regime in which the contrast is run, plus a **competence
gate** the confirmatory runner consults before it creates an output root. Eight
arm-blind checks, every threshold frozen from an outcome-blind pilot. The purpose
is not to make TGSA win; it is to build a system in which TGSA can win or lose
for a scientifically meaningful reason.

Status: `V4_COMPETENCE_NOT_YET_ESTABLISHED`. A development pilot found a regime
reaching ~6.6× the strongest trivial baseline, but its seeds are bimodal — a
training phase transition that some seeds complete and others do not — so it
fails the reliability check. The registered escalation ladder is running.

### Phase 2 — TGSA + multi-token prediction complementarity

Phase 2 keeps mechanism identification separate from product-level
complementarity. Its primary paired design is parallel/TGSA × AR/MTP. Every
cell carries the same MTP parameter structure; AR cells keep it dormant, and
within-family cells are byte-identical at initialization. Auxiliary duplicate-
next controls distinguish future-horizon supervision from extra parameters or
gradient signal.

The reference decoder verifies every MTP block against the main LM head and
must reproduce ordinary greedy decoding exactly. It reports serial-round
speedup and fractional round reduction, not an unmeasured wall-clock claim. See
`TGSA_MTP_COMPLEMENTARITY_PROTOCOL.md` and `PHASE2_RUNBOOK.md`.

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

Phase-2 smoke:

```powershell
python scripts/run_phase2_mtp_study.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir artifacts/phase2_smoke/dataset `
  --output-dir artifacts/phase2_smoke/study
python scripts/verify_phase2_mtp_evidence.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir artifacts/phase2_smoke/dataset `
  --output-dir artifacts/phase2_smoke/study
```

The generated `artifacts/` directory is intentionally ignored. Promote an artifact into a paper evidence bundle only after attaching immutable model/data/config hashes and the exact commit.

## Repository map

| Path | Purpose |
|---|---|
| `tgsa/` | Reference attention, configuration, telemetry, MTP, and statistics |
| `tests/` | Causality, gradient, mask, shape, ablation, decoding, and statistical tests |
| `scripts/` | Smoke execution, ablation matrices, Phase-2 execution, and verification |
| `configs/` | Frozen Phase-0/1 configs and Phase-2 smoke/development configs |
| `evidence/` | Run-manifest schemas and artifact contracts |
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
| `TGSA_MTP_COMPLEMENTARITY_PROTOCOL.md` | Combined mechanism, controls, estimands, gates, and claim ceiling |
| `PHASE2_RUNBOOK.md` | Synthetic and real-Rineya execution sequence |
| `PHASE2_IMPLEMENTATION_SUMMARY.md` | Added modules, reused safeguards, and validation record |

## Non-negotiable research rules

1. A raw linear `delta` term is not sufficient evidence of a new mechanism; it can often be algebraically absorbed into weights on the two preceding states.
2. Sequential depth must be separated from trajectory guidance through matched cascade controls.
3. Parameter, training-compute, inference-compute, and wall-clock views must all be reported; no single fairness view is sufficient.
4. The existing long-context demo may be diagnostic, but a paper claim requires an independently authored sealed bank.
5. Telemetry is explanatory evidence only unless its predictive relationship is preregistered and validated out of sample.
6. No “first” or novelty claim is permitted until the final related-work search and citation audit are complete.
