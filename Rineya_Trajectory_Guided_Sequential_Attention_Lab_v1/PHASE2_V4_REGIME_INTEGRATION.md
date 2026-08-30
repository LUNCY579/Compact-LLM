# Phase 2 integration of the Phase-1 v4 regime trend

**Status:** engineering and preregistration guidance. This document does not
report a TGSA mechanism result.

## What v4 currently changes for Phase 2

Phase 1 v3 produced no interpretable TGSA contrast because no arm learned the
task. The in-flight v4 work has instead exposed a regime-selection problem:
training exposure and data diversity can move competence reliability and
memorization in opposite directions.

The reported controlled comparison held sample presentations fixed at
1,280,000:

| regime | unique train examples | optimizer steps × batch | effective epochs |
|---|---:|---:|---:|
| 50k × 20k | 50,000 | 20,000 × 64 | 25.6 |
| 200k × 20k | 200,000 | 20,000 × 64 | 6.4 |
| registered remaining rung | 200,000 | 30,000 × 64 | 9.6 |

The partial v4 reports associate the higher-exposure regime with more reliable
optimization transitions but a larger train–holdout gap, and the lower-exposure
regime with less memorization but unreliable transitions. Those observations
are about the experimental regime. They are not evidence for, against, or about
the size of the TGSA effect.

## Structural response in the TGSA × MTP pipeline

Every Phase-2 evidence bundle now records, separately:

1. unique training examples;
2. sample presentations;
3. optimizer steps and microbatch size;
4. effective epochs (presentations / unique examples);
5. split-generation order and unbiased split;
6. resolved device identity and a device fingerprint;
7. a regime fingerprint, study fingerprint, and per-cell fingerprint.

This prevents equal-compute but unequal-diversity studies from being described
as the same regime, and prevents CPU/CUDA or runtime changes from being spliced
into one evidence set.

## Competence and marker rules

A Phase-2 development run must set `development_only=true`. Any competence
report produced by that run is marked development-only and
`certifies_confirmatory=false`.

A Phase-2 confirmatory run must set `development_only=false`, carry a frozen
registration, and reference an external current-schema Phase-1 competence `GO`
that is itself not development-only. The marker is transitive; editing one end
of the chain cannot launder development evidence.

`mechanism_interpretation_allowed` remains false unless those prerequisites are
met. Even when true, it is only permission to apply the frozen estimator and
decision rule; it is not a positive result.

## MTP calibration after a real competence GO

MTP depth, auxiliary-loss weight, transition rank, and target-verifier policy
must be calibrated arm-blind. Selection is based on the weakest required
factorial cell's absolute competence and guardrails, never on a favorable
`learned_delta` margin.

The calibration surface must vary data diversity and sample presentations as
separate axes. Effective epochs are reported, not treated as a universal causal
law. A TGSA+MTP model may change the optimization landscape, so v4's TinyLM
transition range cannot simply be copied as an MTP optimum.

The existing development configuration is deliberately non-certifying. In the
paired-factorial branch it presents 128,000 samples over 3,000 unique training
examples, approximately 42.67 effective epochs. That is useful for an
engineering stress run, but it may sit in a memorization-prone region and cannot
promote itself into confirmatory evidence.

## If the v4 ladder exhausts

Exhaustion rejects that registered regime family; it does not authorize an
unregistered fifth rung. A subsequent study may preregister a change such as
model depth or capacity if the question is whether an easier-to-find
composition circuit reduces the exposure needed for reliable competence.

Phase 2 should then be rebased on the new competence-certified regime before
claiming complementarity. MTP must not be used to disguise an incompetent TGSA
baseline: all factorial cells must first demonstrate stable held-out competence.

## Speed claim boundary

Future-token accuracy and fewer serial target-model rounds are feasibility
signals. They are not wall-clock speed. A speed result requires a cache-aware
verifier and end-to-end measurements of target and draft cost, accepted tokens,
rollback, TTFT, TPOT, tokens per second, and memory on the intended Rineya
deployment path.
