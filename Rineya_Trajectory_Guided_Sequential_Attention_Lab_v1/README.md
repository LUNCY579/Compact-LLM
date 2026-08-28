# Rineya Trajectory-Guided Sequential Attention Lab v1

**Working hypothesis:** a causal attention block can trade head-level parallelism for iterative refinement by treating sequential heads as steps along a differentiable state trajectory. Later heads observe not only the current state but also its finite-difference velocity, and that velocity controls subsequent computation.

The working name is **Trajectory-Guided Sequential Attention (TGSA)**. This name is deliberately narrower than “differential attention”: the central quantity is a finite difference across sequential-head depth, not a subtraction between simultaneous attention maps.

## Current status

This branch establishes an auditable Phase-0 foundation:

- a typed PyTorch reference kernel;
- a conventional parallel-MHA control;
- seven mechanism-preserving TGSA ablations plus a conventional parallel control;
- a shared carrier state that makes inter-head differences comparable;
- learned trajectory encoding, bounded predictor/corrector gates, Q/K/V conditioning, and soft token-importance biasing;
- causal masking, padding masks, additive bias support, and summary/full telemetry;
- deterministic smoke and evidence-manifest scripts;
- 22 unit tests covering shapes, causality, gradients, masks, telemetry, all ablations, and paired statistics;
- preregistration, falsification criteria, Qwen/GQA integration requirements, and an arXiv evidence checklist.

This is a **reference implementation**, not a fused production kernel and not yet evidence of improved model quality. The first local validation recorded:

- 22/22 tests passing;
- all reference ablations finite and differentiable;
- zero observed causal-prefix perturbation error in the configured synthetic check;
- an expected large latency penalty for the unfused sequential CPU implementation, which remains a first-class optimization target.

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

The generated `artifacts/` directory is intentionally ignored. Promote an artifact into a paper evidence bundle only after attaching immutable model/data/config hashes and the exact commit.

## Repository map

| Path | Purpose |
|---|---|
| `tgsa/` | Reference attention, configuration, telemetry, and statistics |
| `tests/` | Causality, gradient, mask, shape, ablation, and statistical tests |
| `scripts/` | Smoke execution, ablation sanity matrix, manifest validation |
| `configs/` | Frozen Phase-0 config and Phase-1 preregistration template |
| `evidence/` | Run-manifest schema and artifact contract |
| `ARCHITECTURE.md` | Equations, invariants, complexity, and causal semantics |
| `EXPERIMENT_PLAN.md` | Staged scientific and engineering pipeline |
| `PREREGISTRATION.md` | Primary hypotheses, endpoints, controls, and analysis lock |
| `QWEN_INTEGRATION_PLAN.md` | RoPE/GQA/KV-cache-compatible implementation plan |
| `RELATED_WORK_SCOPE.md` | Prior-art boundaries and literature-audit checklist |
| `CLAIMS_AND_FALSIFIERS.md` | Claim ladder and explicit disconfirmation criteria |
| `DESIGN_DECISIONS.md` | Architectural decisions, rationale, and deferred questions |
| `EVIDENCE_LEDGER.md` | Demonstrated evidence versus still-missing claims |
| `PAPER_EVIDENCE_CHECKLIST.md` | Reproducibility and arXiv release gate |

## Non-negotiable research rules

1. A raw linear `delta` term is not sufficient evidence of a new mechanism; it can often be algebraically absorbed into weights on the two preceding states.
2. Sequential depth must be separated from trajectory guidance through matched cascade controls.
3. Parameter, training-compute, inference-compute, and wall-clock views must all be reported; no single fairness view is sufficient.
4. The existing long-context demo may be diagnostic, but a paper claim requires an independently authored sealed bank.
5. Telemetry is explanatory evidence only unless its predictive relationship is preregistered and validated out of sample.
6. No “first” or novelty claim is permitted until the final related-work search and citation audit are complete.
