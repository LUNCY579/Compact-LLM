# TGSA Phase 1B — architecture amplification (frozen design, NOT for execution)

**Status:** frozen design only. **Do not execute.**
**Prerequisite:** Phase 1 (study `tgsa-phase1-identifiability-v3-*`) must first return a
positive primary result under its registered hierarchy — H1a, H1b, and H1c all
satisfied. If Phase 1 is inconclusive or negative, this design does not run; the
correct follow-on is a revised identification study, not an amplification study.

## Question

Phase 1 asks *whether* a learned finite-difference trajectory signal adds
capability beyond sequential depth and an information-matched history pathway.
Phase 1B asks the next question only if that one is answered:

> Once trajectory guidance itself is identified, how much additional accuracy can
> we extract by letting that signal control more of the attention computation?

Accuracy is the primary endpoint. Efficiency is descriptive.

## Arms

Four amplification levels, plus the two Phase-1 anchor controls retained
unchanged so the new study is directly comparable to the old one.

| Arm | Control injection | Q/K/V conditioning | Predictor | Importance bias | Role |
|---|---|---|---|---|---|
| `cascade` | no | no | no | no | anchor control (no trajectory signal) |
| `history_control` | yes | no | no | no | anchor control (information-matched history) |
| `learned_delta` | yes | no | no | no | Phase-1 reference point |
| `qkv_conditioned` | yes | **yes** | no | no | amplification level 1 |
| `predictor_corrector` | yes | yes | **yes** | no | amplification level 2 |
| `full` | yes | yes | yes | **yes** | amplification level 3 |

These map exactly onto `TGSAConfig.for_ablation`, which already defines every
arm. No new mechanism is introduced — Phase 1B only widens the surface the
already-identified signal is allowed to control.

## Primary endpoint and estimand

The estimand is unchanged from Phase 1: the paired per-example accuracy
difference, averaged within registered stratum, macro-averaged across strata,
then averaged across paired training seeds.

The primary contrasts are the **ordered amplification ladder against
`learned_delta`**:

- A1: `qkv_conditioned` − `learned_delta`
- A2: `predictor_corrector` − `qkv_conditioned`
- A3: `full` − `predictor_corrector`

plus the **cumulative** contrast `full` − `learned_delta`, and a preregistered
monotone trend contrast across the four ordered amplification levels.

Anchor contrasts (`learned_delta` − `history_control`, `learned_delta` −
`cascade`) are re-estimated so Phase 1B independently reproduces the Phase-1
result on new seeds. A Phase 1B that fails to reproduce the Phase-1 anchor
contrast is a replication failure and invalidates its own amplification claims.

## Inference

Identical to the registered Phase-1 v3 procedure, and for the same reasons:

- estimator: `crossed_cluster_t` (two-way cluster-robust, `t_{m-1}` reference);
- primary gate: lower 95% confidence bound above zero;
- the practical effect the study is *powered* for is reported descriptively and
  does not gate;
- `seed_cluster` and the pigeonhole variants reported as sensitivity analyses.

**Multiplicity.** Phase 1B has four primary contrasts rather than one, so it must
control the family-wise error rate across the ladder. The registered procedure is
a **fixed-sequence (hierarchical) test** in the order A1 → A2 → A3, stopping at
the first failure, with the cumulative contrast tested only if A1 passes. A
fixed-sequence procedure needs no alpha adjustment while the ordering is
preregistered, and the ordering here is forced by the architecture: each level
strictly contains the previous one.

**Power.** The seed count must be re-derived by the same outcome-blind audit
before registration. It will not simply inherit Phase 1's number: the
amplification increments A1–A3 are plausibly *smaller* than the Phase-1
identification effect, and a smaller target effect needs more replication. Run
`scripts/phase1_power_simulation.py` against the Phase-1B effect ladder and
report the answer, including "no candidate is adequate" if that is the answer.

## What is deliberately excluded at this stage

Per the research priority — determine whether sequential trajectory-guided
computation increases accuracy, even when the price is slower inference — the
following are **out of scope for Phase 1B** and belong to a later efficiency study:

- adaptive stopping or early exit;
- head skipping or dynamic head selection;
- trajectory-rank compression or low-rank distillation;
- latency-matched or FLOP-matched comparison constraints;
- any gate, threshold, or arm selection that references wall-clock, memory, or
  throughput.

Efficiency metrics are still recorded for completeness — training wall clock,
prefill latency, decode-proxy latency, throughput, peak memory, parameter count —
under a `gating: false` key, exactly as in Phase 1. A slower arm that is reliably
more accurate is an acceptable and potentially important result. Do not optimize
away sequential computation before establishing whether it helps.

## Inherited invariants

Everything Phase 1 v2/v3 established carries over unchanged and must be
re-verified by the same tooling before launch:

- canonical per-seed supernet initialization with byte-identical shared tensors
  (the canonical arm becomes `full`, the widest graph, so every narrower arm is a
  byte-exact restriction of it);
- residual-around-identity trajectory encoder with zero-initialized residual;
- generation-time rejection of trivial primary examples;
- the `direct_lookup` non-inferiority guardrail;
- the replay-derangement alignment falsifier with registered derangement indices;
- train → full-matrix barrier → confirmatory-evaluate orchestration with no
  progressive test exposure;
- immutable failure evidence and complete hash-bound evidence-chain verification;
- smoke configurations permanently forced to `smoke_only`.

## Claim ceiling

Even with every gate satisfied, Phase 1B supports at most:

> In a controlled tiny decoder on the registered synthetic iterative tasks,
> allowing an identified trajectory signal to additionally condition Q/K/V,
> prediction, and key importance increased held-out generated-task accuracy
> beyond conditioning the head input alone.

It does **not** establish natural-language generalization, long-context
behaviour, Qwen/GQA compatibility, efficiency, scaling, novelty, or broad
architectural superiority.

## Registration checklist before Phase 1B may run

1. Phase 1 v3 returns a positive primary result under its full hierarchy.
2. New study ID and new output root.
3. Outcome-blind power audit re-run against the amplification effect ladder.
4. Fixed-sequence multiplicity procedure registered with its ordering.
5. Canonical supernet re-verified with `full` as the canonical arm.
6. Complete test suite plus end-to-end smoke matrix green.
7. CI green on the exact pinned launch commit.
8. Evidence-chain verifier green on the smoke bundle.
