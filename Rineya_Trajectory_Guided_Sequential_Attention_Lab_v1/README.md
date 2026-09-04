# Rineya Trajectory-Guided Sequential Attention Lab v1

Trajectory-Guided Sequential Attention (TGSA) treats attention heads as ordered
refinement steps. Later heads receive a differentiable control signal derived
from finite differences in a shared carrier state rather than operating only as
independent parallel projections.

## Current status

**TGSA v5.1 is prospectively frozen as a staged research programme on
`research/tgsa-v5-1-unified-programme`. The laboratory contains 457 tests.**

The branch starts from
`research/tgsa-v5-competence-and-mechanism` at
`b09149002a7892a87a461af0b3c852ed14842905`. That commit already contains
`research/tgsa-v5-competence-execution` at
`de3bd632b491df294f9b8736079bdcbbdd64e4d3`, so v5.1 is not a blind merge of two
divergent codebases. Both source branches remain unchanged.

V5.1 combines their strongest ideas through a machine-checked programme:

- null-anchored apparatus qualification before any mechanism contrast;
- depth-before-width capacity selection and a complete data-diversity/exposure
  grid;
- paired cascade, history, and aligned-trajectory controls;
- resumable, fingerprint-bound evidence production and closed verification;
- linear-memory `control_only` trajectory telemetry from one registered block;
- isolated `frozen_probe`, `stop_gradient`, and `joint` MTP gradient semantics;
- separate Phase 2A trajectory-information and Phase 2B exact-decoding system
  lanes;
- real-checkpoint initialization-drift, exact-decoding, quality, and
  serial-round gates;
- a separate wall-clock benchmark after checkpoint qualification.

### What has been demonstrated

- The typed TGSA reference kernel, controls, telemetry, causal masking, gradient
  connectivity, and deterministic evidence machinery are implemented.
- Phase 1 v3 completed but was inconclusive because no primary arm was competent.
- Phase 1 v4 correctly rejected its registered task-regime family before a
  confirmatory matrix was dispatched.
- The v5 apparatus implementation supports family-specific nulls, one-hop
  prerequisite selection, capacity/exposure ladders, resume integrity, and
  verified reports.
- Phase 2A provides sequential trajectory-conditioned future-prediction
  machinery with isolated gradient modes.
- Phase 2B provides a paired parallel/TGSA × AR/MTP factorial, exact verified
  decoding, and algorithmic serial-round analysis.
- The Qwen3/Rineya retrofit preserves GQA, RoPE, Q/K normalization, cache
  semantics, deterministic conversion manifests, and real-path drift audits.

### What has not been demonstrated

- **Anything about the TGSA mechanism on a competent confirmatory regime.** No
  hypothesis in the H1a/H1b/H1c ladder has yet been confirmed or refuted.
- Improved Rineya quality, long-context reasoning, or broad capability.
- A real-checkpoint TGSA+MTP serial-round benefit.
- End-to-end wall-clock acceleration.

Neither v3 nor v4 is evidence against TGSA. Their failures constrain the
experimental apparatus and task regime, not the mechanism. Current synthetic
MTP work is non-evidentiary for Rineya, and every smoke run remains
`development_only`; it cannot authorize a mechanism claim.

## Why Phase 2A and Phase 2B remain separate

| Stage | MTP semantic identity | Scientific question |
|---|---|---|
| Phase 2A | `sequential_teacher_forced_recursive_v1` | Does the registered TGSA trajectory coordinate contain incremental future-token information? |
| Phase 2B | `parallel_independent_offsets_v1` | Can exact verified MTP preserve registered quality while reducing serial target-model rounds? |

These are complementary interventions, not replications. Their checkpoints,
evidence roots, estimands, and claims may not be pooled.

## V5.1 stage graph

```text
v5 apparatus qualification
    |
    +--> MTP-free mechanism study --> Phase 2A trajectory-MTP bridge
    |                                      |
    |                                      +--> mechanism-interpretable Rineya lane
    |
    +--> within-scope pretrained competence --> Phase 2B system factorial
                                                   |
                                                   +--> Rineya checkpoint qualification
                                                            |
                                                            +--> wall-clock benchmark
```

A Phase 2B system result may be tested after a mechanism `NO_GO`, but it may not
be relabeled as evidence that the trajectory mechanism caused the result.

## First commands

From this directory:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests -q

python scripts/verify_v5_1_programme_contract.py `
  --contract configs/v5_1_programme_contract.json `
  --repo-root ..
```

For development smoke and the nondevelopment capacity/exposure sequence, follow
`PHASE2_TGSA_MTP_RUNBOOK.md`. Keep evidence outside the checkout so the clean
worktree gate remains meaningful.

## Repository map

| Path | Purpose |
|---|---|
| `V5_1_UNIFIED_PROGRAMME.md` | Audit of both v5 branches and the final staged v5.1 design |
| `V5_1_DOCUMENTATION_STATUS.md` | Current versus historical document authority |
| `TGSA_MTP_CANONICAL_PATH.md` | Canonical lineage, semantic identities, and promotion graph |
| `PHASE2_TGSA_MTP_RUNBOOK.md` | End-to-end operator commands and stop rules |
| `PHASE1_V5_PREREGISTRATION.md` | Inherited v5 apparatus rationale and frozen decisions |
| `PHASE2_TGSA_MTP_DESIGN.md` | Phase 2A mechanism-bridge design and controls |
| `ARCHITECTURE.md` | Core TGSA equations and causal architecture |
| `CLAIMS_AND_FALSIFIERS.md` | Claim ladder and disconfirmation rules |
| `EVIDENCE_LEDGER.md` | Demonstrated versus missing evidence |
| `PAPER_EVIDENCE_CHECKLIST.md` | Reproducibility and publication gate |

Machine-facing files:

| Path | Purpose |
|---|---|
| `configs/v5_1_programme_contract.json` | Frozen lineage, stage, and claim contract |
| `tgsa/v5_1_programme.py` | Fail-closed contract and Git-lineage verifier |
| `scripts/verify_v5_1_programme_contract.py` | Pre-output verification command |
| `scripts/phase1_v5_1_qualify.py` | Canonical locked apparatus runner |
| `scripts/select_phase1_v5_1.py` | Capacity/exposure selection from registered reports |
| `tests/test_v5_1_programme_contract.py` | Anti-pooling and anti-laundering regressions |

## Non-negotiable research rules

1. A raw linear delta is not enough to establish a trajectory mechanism; it may
   collapse into weights on earlier states.
2. Serial depth, information-matched history, aligned change, replayed pathway,
   and deranged alignment require separate controls.
3. Apparatus competence precedes mechanism interpretation.
4. Phase 2A and Phase 2B semantic identities are immutable within an evidence
   bundle.
5. Development evidence cannot certify confirmatory execution.
6. A complete paired four-cell block is required for the Phase 2B system claim.
7. Exact decoding means identical token IDs, termination, accepted-prefix order,
   rollback behavior, and final greedy sequence.
8. Serial-round reduction is an algorithmic proxy, not tokens per second.
9. A single Rineya checkpoint cannot support a broad family-quality claim.
10. Any scientific amendment requires a new commit, contract digest, and fresh
    evidence root.
