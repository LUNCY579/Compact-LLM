# Final interpretation

## Decision: the P2 NO-GO is WITHDRAWN as unsupported

The previous version of this file recorded **NO-GO**. That decision is withdrawn.
It was not wrong about the numbers; it was wrong about what the numbers could
mean. The experiment could not have produced any other result, whether or not
fast-weight memory works on Supa-A1.1.

This is not a reinterpretation of the same evidence. It rests on measurements
that P2 never took, in `reports/MECHANISM_DIAGNOSTIC.md`.

### Why the NO-GO cannot stand

P2 concluded NO-GO from "arms B, C and D emit identical predictions". That
inference needs two things to be true. Neither is.

**1. It needs the pathway to have been contributing something visible.** It was
contributing, and in the predicted order — the evaluation simply could not see
it. Measured at layer 14 on real eval items:

| Arm | share of the attention output | vs. arm A: max logit delta | mean KL |
|---|---:|---:|---:|
| B (static) | 0.12 % | 0.63 | 0.016 |
| C (recurrent) | 1.41 % | 1.44 | 0.071 |
| D (sig-gated) | 3.22 % | 3.37 | 0.251 |

The recurrent carry made C's contribution **11x** B's. B vs C differed by up to
0.80 logits. Those are far above fp16 noise. Identical argmax came from a
saturated evaluation, not a dead pathway.

**2. It needs the memory to have been able to reach the query.** It could not.
The trained retention half-life was **87.5 tokens** against ~990-token contexts,
and `beta = 1.0` sits exactly on the NLMS deadbeat point, where a write
*completely replaces* the stored value along its key direction (retention
`decay - eta*||x||^2 = -2.2e-5`). A fact at relative position 0.05 is ~940 tokens
from the query; `0.99211^940 = 6e-4` of it survives. `lambda` never moved from
its initialisation, so this horizon was a hyperparameter, not a finding.

**And the ruler could not have measured a difference anyway.** The filler was one
sentence repeated 21–42 times, so the needle was the only non-duplicated
sentence; the answer space was 10 colour words; and there was one item per
(length, position) cell, so a paired exact test needed 6 of 14 items to flip
(42.9 pp) to reach p < 0.05. The interference battery needed 6 of 7.

Three further defects mean the recorded configs do not describe the recorded run:
`run_experiment.py` hardcoded `lengths = [256, 512]` over the config's
`[512, 1024, 2048, 4096]`; `train.py` hardcoded `init_alpha = 0.05` over the
config's `0.0`; and every run used `data/synthetic/` (10 distinct answers) while
the official CC0 unsealed shard sitting in `data/unsealed/` (207 distinct
answers, hashed, committed) was never read.

## Established

Claims backed by direct measurement, independent of whether FWA helps.

- **A prefix-aligned FALCON-2 pathway can be attached to a frozen Supa-A1.1 as
  an auxiliary branch.** `alpha = 0` reproduces unmodified logits on GPU.
- **The recurrent state is bounded and does not grow with context.** 8192 bytes
  at rank 16, constant from 1K to 8K, against a KV cache that would grow to
  896 MiB at 8K.
- **That bound buys no memory saving in this design.** Peak VRAM is identical for
  arms A and C to within 0.01 GiB at every length, because full attention still
  runs. Only Arm E (replacement) could convert the bound into a saving.
- **Inference is strictly slower with FWA**: 1.5x–4.2x lower prefill at 1K–4K.
  Generation is worse still, because the wrapper refuses KV-cache mixing, making
  decoding O(new_tokens x T).
- **Hardware envelope**: 1K/2K/4K native (3.64 GiB at 4K); 8K only via
  host-memory spill at ~6x slowdown; **16K OOMs and is not measurable here**.
- **The retention horizon is set by `(beta, lambda, ||x||^2)`**, and `||x||^2` is
  pinned to `rank` by the RMSNorm on the key features — so the horizon is
  effectively an initialisation choice, not something training discovers in a
  short run.
- **Arm B and arm C are parameter-matched** at 524,593 trainable tensors each.

## Failed — but as experiments, not as hypotheses

The P2 report listed H2, H3 and H4 as *failed hypotheses*. That was the central
error. They were **untested**, and are now reclassified:

| Hypothesis | P2 said | Actual status |
|---|---|---|
| H2 (Q2) recurrence beats extra parameters | failed | **untested** — C and B differed by 11x in contribution; the eval was saturated |
| H3 (Q3) middle-vs-edge robustness | not supported | **untested** — an 87.5-token horizon cannot produce a position effect at 990 tokens |
| H4 (Q4) significance gating helps | failed | **untested** — and the gate was confounded: it replaced `beta`/`lambda` with absolute ranges, so arm D had a different retention horizon from arm C, and `beta_raw`/`lambda_raw` were orphaned from the graph (bit-identical to init in the saved adapters) |

## Untested

- **H1–H4 on a suite that can discriminate.** P2b, preregistered in
  `PREREGISTRATION.md` Amendment 1, is calibrated so the frozen backbone scores
  **0.778** at 2048 tokens with **0.333** at relative position 0.05.
- **H6** (new): whether the retention horizon is the binding constraint. Arm
  C-short holds the original short horizon; predicted to match arm B.
- **The layer placement itself.** The brief (section 6) asked for a
  layer-location pilot over early / middle / late / distributed placements before
  settling on one. That pilot was never run: P1 fixed layer 14 (the 0-based middle
  of 28) and every result since, including P2b, inherits it unexamined. So every
  number in this lab is conditional on one arbitrary placement, and a null result
  is a null result *at layer 14*, not for auxiliary fast-weight memory on
  Supa-A1.1 in general. Adding placements now would confound the preregistered
  contrasts, so it belongs in a follow-up rather than in P2b.
- Contexts 8K and 16K — not measurable on this GPU.
- Arm E (attention replacement), the only variant that could trade KV cache for
  bounded state.
- Chunk-parallel FALCON-2 / FALCON-3, and the inner-product family.
- Seed 33.
- Whether any of this transfers off the synthetic vault/unit/code task.

## Strongest evidence against the working hypothesis

Stated plainly, because withdrawing a negative result is not the same as
obtaining a positive one:

1. **Arm C is a 1.41 % perturbation of one layer's attention output.** Nothing
   guarantees a bounded rank-16 state adds information that 28 layers of full
   attention over an intact context do not already have. The auxiliary design
   gives the memory no privileged access to anything.
2. **The measured cost is unambiguous and the benefit is not yet measured at
   all.** 1.5x–4.2x slower prefill, no VRAM saving, and quadratic generation.
   A benefit would have to be large to be worth that.
3. **The static control was not far behind on contribution magnitude** once the
   horizon was short — which is consistent with a story where most of what the
   pathway does is a per-token feature transform, not memory.
4. **P2b changes the retention horizon by fiat.** If arm C now wins, part of that
   win is attributable to choosing a horizon that suits the task. Arm C-short
   exists to quantify that, and the result must be reported with it.

## Status of P2b

Apparatus committed, calibrated and preregistered. **Not yet run**: the single
8 GB GPU is being used by an unrelated experiment in
`/c/Users/clclc/BusinessProject/SupaExperiment/`, which holds ~5.7 GB. Training
needs more headroom than is currently free, and pre-empting another session's
run is not an acceptable way to get it.

Nothing in this file claims a P2b result. When P2b runs, `RESULTS.md` and this
file are updated with the paired McNemar outcomes and the decision label from
Amendment 1's rule.

## Terminology

Throughout: **fast-weight state**, **recurrent memory**, **inference-time state
update**. The state changes during a forward pass; the model's parameters do not.
This is not fine-tuning at inference and not lifelong learning.
