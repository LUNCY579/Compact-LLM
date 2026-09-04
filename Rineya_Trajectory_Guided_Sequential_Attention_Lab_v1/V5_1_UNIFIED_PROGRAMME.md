# TGSA v5.1 Unified Programme

**Branch:** `research/tgsa-v5-1-unified-programme`
**Base:** `b09149002a7892a87a461af0b3c852ed14842905`
**Status:** prospective coordination contract frozen; nondevelopment v5 apparatus evidence and real Rineya evidence remain unrun.

## 1. Decision

The two v5 branches are not divergent implementations that should be merged
line by line.

- `research/tgsa-v5-competence-execution` at `de3bd632...` establishes the
  null-anchored v5 apparatus design, scope identity, competence rules, and the
  Phase 2A/Phase 2B semantic boundary.
- `research/tgsa-v5-competence-and-mechanism` at `b091490...` is a strict
  descendant of that work. It adds the executable/resumable apparatus runner,
  closed-chain verification, one-block `control_only` trajectory telemetry, and
  the `frozen_probe -> stop_gradient -> joint` MTP gradient ladder.

Therefore v5.1 uses `b091490...` as its code base. It does **not** alter either
source branch. The advantages of Phase 2A, Phase 2B, and the pretrained retrofit
are combined through a machine-checked stage graph, not by mixing their causal
interventions into a single ambiguous MTP module.

## 2. What each v5 branch gets right

### 2.1 Competence-execution branch

The earlier branch correctly repairs the central failure of v3/v4: an arm
contrast is uninterpretable if the task family is inert, saturated, or judged
against the wrong null. Its strongest contributions are:

- a one-hop `pointer_chase` prerequisite before a full multi-hop qualification;
- a depth-before-width capacity ladder (`C1`, `C2`, `C3`);
- a complete diversity-by-presentation grid (`E1`-`E4`), so data diversity is
  not silently confounded with optimizer exposure;
- registered family-specific nulls rather than one global chance number;
- `pointer_chase` as primary, `direct_lookup` as guardrail, and `state_update`
  only as a non-gating canary;
- paired `cascade`, `history_control`, and `learned_delta` arms;
- scope hashes that bind task, model, objective, competence rule, and execution
  environment.

These are foundational and are carried forward unchanged.

### 2.2 Competence-and-mechanism branch

The later branch contains the earlier branch and makes it operational. Its most
important additions are:

- fingerprint-bound resume/reuse of each arm/seed cell;
- periodic optimizer checkpoints and fixed-final-checkpoint semantics;
- closed report verification before capacity or exposure selection;
- linear-memory `control_only` trajectory telemetry from exactly one registered
  decoder block;
- three explicit auxiliary-gradient regimes:
  - `frozen_probe`: the backbone and shared projections are frozen/detached;
  - `stop_gradient`: the AR objective may adapt the backbone, but auxiliary
    gradients cannot enter it;
  - `joint`: auxiliary and AR objectives may co-adapt the system;
- detachment of hidden states, trajectory controls, tied embeddings, and the
  shared LM projection in isolated modes, closing a subtle gradient leakage path.

These additions make the later branch the correct technical base for v5.1.

## 3. Audit findings that v5.1 corrects

### 3.1 Registration timing

The inherited nondevelopment plan says
`registration_status=implementation_complete_not_registered`, yet the runner
can execute it on a clean descendant checkout. That leaves room for a costly
probe to occur before a prospective registration is frozen.

V5.1 does not edit the inherited plan. Instead it pins its exact Git blob
`0538faf4...` inside `configs/v5_1_programme_contract.json`, freezes the
programme before any nondevelopment probe, and requires an explicit operator
attestation that no nondevelopment outcome was inspected before this lock.

### 3.2 Documentation drift

The source README identified an older Phase 2A branch as “this branch” and
claimed 396 tests while the source suite collected 436. The source CI therefore
reported 435 passing tests plus one stale-documentation failure. V5.1 updates
branch identity and records 451 tests: 444 after adding four programme-contract
regressions, four more after registering `every_seed_beats_its_family_null`
(see PHASE1_V5_PREREGISTRATION.md §7), and two covering interrupt-safe resume.

### 3.3 Obsolete v4 promotion wording

The historical v4 branch has since been retired; its immutable result boundary is now fetched through `archive/tgsa-phase1-v4-regime-rejected`. The commit identity remains `8b5737e260b5478427b8bb033cffa394a5f3fc5d`.

An inherited runbook still said Phase 2 could follow a v4 competence `GO`.
That is impossible: v4 froze `V4_REGIME_FAMILY_REJECTED` at `8b5737e...`.
V5.1 makes the v5 apparatus the only current competence path and keeps v4 for
regression/integrity use only.

### 3.4 Rineya promotion was too coarse

A blanket “no Rineya until a positive mechanism result” mixes two different
questions:

1. **mechanism interpretation:** does aligned trajectory information itself
   explain a benefit?; and
2. **system utility:** does a complete TGSA+MTP system preserve registered
   quality while reducing exact-decoding target rounds?

V5.1 separates those lanes. Phase 2B and a real-checkpoint system qualification
may proceed after a mechanism `NO_GO`, but the result must remain a system-level
result and may never be relabeled as mechanism evidence.

### 3.5 Two MTP interventions must remain distinct

Phase 2A and Phase 2B use different causal graphs:

| Stage | Semantic identity | Purpose |
|---|---|---|
| Phase 2A | `sequential_teacher_forced_recursive_v1` | Test incremental future-token information in the trajectory coordinate |
| Phase 2B | `parallel_independent_offsets_v1` | Test exact-decoding quality/serial-round system complementarity |

They are complementary, not interchangeable. Their checkpoints, schemas,
evidence roots, estimands, and claims remain separate.

## 4. Final v5.1 stage graph

### Stage 0 — v5 apparatus qualification

No MTP objective is present. Run the one-hop prerequisite through the registered
capacity ladder, freeze the first passing rung, then complete all four exposure
cells. Select by the frozen resource-order rule, not by the most favorable arm
contrast.

A `GO` means only that the scratch apparatus is competent, non-saturated, and
stable enough for a separately preregistered mechanism experiment.

### Stage 1 — MTP-free TGSA mechanism study

The primary mechanism experiment remains MTP-free. It must include:

- `cascade` for parallel-versus-serial depth;
- `history_control` for information-matched sequential history;
- `learned_delta` for the aligned trajectory coordinate;
- `replay_control` for pathway/capacity without current aligned change;
- `deranged_control` for alignment falsification.

This stage prevents future-token supervision from creating the very trajectory
signal it is later asked to read.

### Stage 2A — trajectory-conditioned sequential MTP bridge

Use identical MTP capacity across `dormant`, `backbone`, and `trajectory` modes.
Run the gradient ladder in this order:

1. `frozen_probe` — primary information-identification result;
2. `stop_gradient` — secondary readout utility under AR-only backbone adaptation;
3. `joint` — tertiary co-training utility, never interpreted as a pure mechanism
   result.

Read only `control_only` telemetry from one registered block. A cascade arm may
not receive the TGSA trajectory tensor through a hidden side channel.

### Stage 2B — independent-offset exact system factorial

Preserve the paired four-cell design:

| Cell | Attention | Objective | Decoder |
|---|---|---|---|
| `parallel_ar` | parallel | AR | canonical greedy |
| `parallel_mtp` | parallel | AR + independent future offsets | exact verified MTP |
| `tgsa_ar` | TGSA | AR | canonical greedy |
| `tgsa_mtp` | TGSA | AR + independent future offsets | exact verified MTP |

Every cell has the same MTP parameter structure; AR cells keep it dormant and
frozen. A valid combined result requires the registered initialization checks,
exact token/termination/rollback equivalence, quality non-inferiority, and the
minimum serial-round reduction on TGSA.

Serial rounds are an algorithmic proxy. They are not wall-clock speed.

### Stage 3 — first real Rineya checkpoint qualification

The first real checkpoint run must preserve all four Phase 2B cells and bind:

- the complete source-checkpoint tree hash;
- data, tokenizer, renderer, seed, dtype, attention implementation, and code;
- within-family exact state and cross-family compatible-tensor identity;
- main-logit initialization drift in the actual deployment arithmetic path;
- exact verified decoding;
- within-scope competence;
- registered quality non-inferiority and serial-round criteria;
- a closed evidence verifier.

Start with adapters and a small distributed layer set. Expand to native
attention, converted blocks, or full tuning only after the previous stage
preserves the registered baseline.

### Stage 4 — deployment wall-clock benchmark

Only a separate, prospectively frozen hardware study may support speed claims.
It must specify warm-up/compilation, synchronization, cache policy, batch size,
context and generation lengths, target/proposal cost, TTFT, TPOT, decode
throughput, memory, utilization, repeated trials, and uncertainty.

## 5. Promotion lanes

### Mechanism-interpretable lane

`v5 apparatus GO -> MTP-free mechanism GO -> Phase 2A GO -> real-checkpoint GO`

Only this lane can eventually support a checkpoint-scoped mechanism
interpretation.

### System-utility lane

`within-scope pretrained competence GO -> Phase 2B GO -> real-checkpoint GO`

This lane does not require a positive mechanism result. It can support a narrow
system-level quality/serial-round result, but it cannot explain *why* the system
worked.

### Wall-clock lane

`real-checkpoint GO -> separately preregistered deployment benchmark GO`

No serial-round proxy can substitute for the final benchmark.

## 6. Machine enforcement

The programme is enforced by:

- `configs/v5_1_programme_contract.json` — frozen stage, lineage, and claim
  identity;
- `tgsa/v5_1_programme.py` — fail-closed contract and Git-lineage verifier;
- `scripts/verify_v5_1_programme_contract.py` — pre-output verification;
- `scripts/phase1_v5_1_qualify.py` — canonical locked apparatus runner;
- `scripts/select_phase1_v5_1.py` — selection from reports bound to the same
  prospective contract;
- `tests/test_v5_1_programme_contract.py` — anti-laundering regression tests.

The wrapper verifies a clean checkout, full pinned lineage, exact plan blob, and
prospective registration before creating an evidence root. A scientific
`NO_GO` remains a successful execution and exits normally; it is never treated
as a software failure.

## 7. Present claim ceiling

At branch creation, v5.1 establishes protocol and implementation readiness only.
It does not establish TGSA mechanism value, MTP value, Rineya quality,
long-context improvement, serial-round benefit on a real checkpoint, or
wall-clock acceleration.
