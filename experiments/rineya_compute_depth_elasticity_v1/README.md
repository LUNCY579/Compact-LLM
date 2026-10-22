# CDE-001 — Rineya compute-depth elasticity

**Status: offline methods implemented; native integration BLOCKED.**
This is an isolated research package inspired by TRM4CO and FlexLoop, not a
replication of either paper, a trained Rineya model, or evidence of an accuracy
or speed improvement. It does not replace the repository's model-quality status.

## Why this branch exists

TRM4CO motivates asking where fixed-parameter inference compute belongs: one
longer recurrent trajectory, more independent trajectories, or both. FlexLoop
motivates training useful intermediate exits instead of assuming a model trained
at maximum recurrent depth is reliable at every smaller depth. Their authors'
abstracts support these mechanisms in combinatorial optimization and deep RL,
respectively; transfer to Rineya language modeling remains a hypothesis. [P1, P2]

The Rineya question is **usefulness first, allocation second**. A controller cannot
recover a good answer that the recurrent host never produces. Stable predictions
can also be confidently wrong. Test both failures before promoting an adaptive
policy or architecture change.

## What is runnable now

| Component | Implemented boundary |
|---|---|
| `core.py` | Depth/width matrix; nested seeds; strict same-prefix traces; parent-balanced metrics; separate final, selected, and oracle quality; causal agreement policy; DEV-only threshold calibration; TEST-parent exclusion. |
| `losses.py` | Five callable PyTorch objectives: full depth, sampled depth, all exits, adjacent-depth KL, maximum-depth-to-shallow KL. Masked response positions and detached teachers. |
| `relabel.py` | Append-only label-change proposals; replay old and candidate through the same verifier; reject protected parents, invalid solutions, ties and decoder-contract mismatches. |
| `fixture.py` | Scripted immediate success, late rescue, reversion, stable-wrong, width opportunity, and permanent-failure traces. No learned model. |
| CLI | `plan`, `smoke`, and an always-blocked `native-preflight`. No network, checkpoint download or training entry point. |
| Tests | CPU method/gradient/guard tests, not native checkpoint qualification. See `VALIDATION.md`. |

There is no native trace collector, optimizer/training loop, dataset relabel
writer, external code-execution sandbox, online stopping controller, or deployed
verifier integration in this delivery. `Trace.checkpoint_id` and `prefix_id` are
consistency labels, **not independently checked artifact hashes or authorization
receipts**. Existing guards are not bypassed by supplying these fields.

## Run from the repository root

The plan/smoke/preflight commands use only Python's standard library. The loss
module and test suite additionally need PyTorch and pytest. This package was
tested with Python 3.13.5, CPU PyTorch 2.10.0 and pytest 9.0.2.

```bash
python -m experiments.rineya_compute_depth_elasticity_v1 plan
python -m experiments.rineya_compute_depth_elasticity_v1 smoke
python -m pytest -q experiments/rineya_compute_depth_elasticity_v1/tests
python -m experiments.rineya_compute_depth_elasticity_v1 native-preflight
```

The last command must print a BLOCKED receipt and exit **2**. Commands accept
`--out path/to/new-receipt.json` and refuse to overwrite an existing file.
The smoke run calibrates on 12 scripted DEV parents and evaluates 12 distinct
scripted TEST parents across 16 depth/width cells. These are apparatus fixtures,
not a measured Rineya benchmark, training run, or statistically meaningful result.

### Tensor-level training objective

```python
from experiments.rineya_compute_depth_elasticity_v1.losses import objective

# native_logits: [R, B, T, V]; targets/mask: [B, T].
# The caller aligns next-token targets ONCE; all exits use identical prefixes.
terms = objective(native_logits, targets, response_mask,
                  arm="adjacent", coefficient=0.1, temperature=1.0)
# Integrate terms["loss"] into an already-qualified native training loop.
```

The snippet is an integration contract, not a runnable native training command.
The coefficient is an illustrative development value, not a validated setting.

## Priorities and expected benefit to the whole Rineya program

1. **Fixed-checkpoint depth/width screen.** Decide whether extra recurrent work
   repairs errors, whether correct answers revert, and whether ordinary sampling
   is a better use of the same measured budget.
2. **Intermediate-exit training.** Only on a qualified recurrent host, determine
   whether shallow exits improve without sacrificing full-depth reasoning or
   existing retention capabilities.
3. **Calibrated stopping.** Freeze a DEV-only policy, measure missed rescues on
   untouched parents, then measure real batch-1 latency in the host. Replay work
   reductions are not measured speedups.
4. **Verified self-relabeling, separately.** Improve TRAIN solutions using a frozen
   executable verifier and evaluate subsequent low-budget model capability. This
   can use ordinary sampling before recurrence is proven useful.

The potential benefits are better reasoning per parameter, more reliable latency
budgets, auditable selection/stopping, and useful verified training solutions.
None is an established Rineya gain. Long-context, abstention, general language,
composable skill retention and cache efficiency remain independent endpoints.
See [PROTOCOL.md](PROTOCOL.md) for controls, stop conditions and the host handoff.

## Provenance and integration map

Created on `research/rineya-compute-depth-elasticity-v1` from the integrated
`post-training-Mimo` commit **`03db218ceebc61a2bc04f2482cf7e094045e01c1`**.
Only the new package and its scoped CPU workflow are added. No model weights,
existing experiments, shipping defaults, data banks or sealed evaluation sets
are changed.

The live repository inspection on 2026-09-29 established these boundaries:

| Existing work | Relationship / next integration, not an implemented bridge |
|---|---|
| `experiments/loopspec_v1/core.py` and its README at the pinned parent | Reuse the actual recurrent `prefill/start/advance/readout/cache_bytes` contract and independent prefix/cache qualification. LoopSpec scheduling must remain a separate systems variable. |
| `experiments/ced_native_loader_issue4_v1` at the parent | Native loading/identity qualification remains upstream; this package does not declare it passed. |
| `research/rineya-verified-resampling-v1` at `ac9b79ac07ad650d8815c76eb4194a180f23e8d4` | Confirmed branch tip. Future width/verifier integration; no code was imported or runtime bridge qualified here. |
| `research/rineya-recursive-self-distillation-v1` at `07703ce5cfc788f4e98f6dbe0f644c700f66f288` | Confirmed branch tip. Future comparison with existing self-distillation work; adjacent-depth LM losses here are a separate adaptation. |
| `research/post-training-mimo-centralized-v1` at `ba683ac0fcadbf04e4e179200d035b6a980e388f` | Confirmed separate branch; not the base of this delivery and not modified. |

The sibling branch **tips** were checked; their entire code or current efficacy
was not re-audited. Nothing from them is silently merged into this experiment.

## Sources and source-verification limit

- **[P1]** Zhengxi Zhang and Paul Swoboda. *Compute Time Scaling with Recursive
  Models for Combinatorial Optimization.* arXiv:2609.34585, submitted 2026-09-28.
  https://arxiv.org/abs/2609.34585
- **[P2]** Xun Wang, Ruishuo Chen, Yu Chen, Zhuoran Li, Longbo Huang. *FlexLoop:
  Depth-Elastic Looped Policies for Adaptive Test-Time Computation in Deep RL.*
  arXiv:2609.34488, submitted 2026-09-28.
  https://arxiv.org/abs/2609.34488

Both official arXiv abstracts were retrieved in this implementation session.
Direct full-HTML retrieval failed, so exact paper algorithms, table values,
training schedules and code equivalence were **not freshly verified** here.
The implementation is original, hypothesis-testing apparatus based on the
verified abstract-level mechanisms and the preceding research discussion. No
paper code was copied and no exact-reproduction claim is made.
