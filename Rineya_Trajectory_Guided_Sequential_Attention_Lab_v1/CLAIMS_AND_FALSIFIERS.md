# Claim ladder and falsification criteria

The project should become easier to disprove as it becomes more ambitious. This file prevents implementation complexity from being mistaken for scientific evidence.

## Level 0 — engineering existence

**Claim:** the operator is differentiable, causal, numerically finite, and executable.

Required evidence:

- unit tests;
- suffix-perturbation causal check;
- nonzero trajectory-path gradients;
- all ablations execute;
- reproducible run manifest.

Current status: supported only for the small PyTorch reference kernel and tested shapes.

Falsifiers:

- future-token leakage;
- disconnected trajectory gradients;
- NaN/Inf under the registered range;
- an ablation flag that leaves a supposedly removed mechanism active.

## Level 1 — mechanism identifiability

**Claim:** learned trajectory control does something beyond a two-state linear recurrence and beyond ordinary sequential depth.

Required comparisons:

- cascade;
- cascade plus two-state history;
- raw finite difference;
- learned finite difference;
- parameter-matched nonlinear two-state controller without explicit subtraction;
- recurrent-depth control with comparable sequential steps.

Falsifiers:

- raw delta and learned trajectory perform equivalently within uncertainty;
- a generic two-state MLP matches TGSA;
- all gains disappear under parameter or training-compute matching.

## Level 2 — optimization or representation benefit

**Claim:** trajectory guidance improves training dynamics or representational efficiency.

Possible evidence:

- lower held-out NLL at matched tokens/FLOPs;
- faster convergence to the same NLL;
- improved synthetic retrieval at matched capacity;
- stable, noncollapsed trajectory diagnostics.

Falsifiers:

- gains are seed-specific;
- trajectory gates saturate to constants;
- the model ignores the trajectory signal (`m` or control gates approach zero);
- benefits are fully reproduced by adding ordinary layers.

## Level 3 — task-quality benefit

**Primary candidate claim:** TGSA improves sealed long-context macro accuracy over cascaded sequential attention while satisfying short-context and NLL guardrails.

Required evidence:

- frozen five-seed protocol;
- stratified paired bootstrap confidence interval;
- independently authored sealed evaluation;
- per-domain and per-context-position breakdown;
- exact outputs and scoring records;
- parameter/FLOP/latency views.

Falsifiers:

- primary confidence interval includes zero;
- effect is below the registered practical threshold;
- a short-context guardrail fails;
- improvement is confined to a previously inspected diagnostic set;
- performance gain comes only from extra trainable capacity.

## Level 4 — compute-quality frontier benefit

**Claim:** TGSA creates a useful quality/compute trade-off, not merely higher quality at arbitrary latency.

Required evidence:

- prefill and decode measured separately;
- optimized kernel or honest reference-kernel result;
- equal-latency or equal-FLOP comparisons;
- adaptive/head-skipping variant if fixed sequential execution is too slow;
- peak memory and KV-cache footprint.

Falsifiers:

- all TGSA points are Pareto-dominated by deeper standard Transformers, recurrent-depth controls, or cascaded heads;
- latency variance is larger than the claimed difference;
- compression savings disappear on real hardware.

## Level 5 — trajectory explains behavior

**Claim:** trajectory quantities predict correctness, uncertainty, or computational difficulty.

This is strictly secondary and requires out-of-sample validation.

Required evidence:

- prediction rule trained on development data only;
- calibration and discrimination on sealed data;
- comparison to simpler baselines such as entropy, margin, or layerwise logit change;
- intervention showing that changing the trajectory controller changes behavior consistently.

Falsifiers:

- telemetry correlations vanish out of distribution;
- ordinary confidence signals explain the same variance;
- probes succeed but interventions fail.

## Level 6 — general architectural contribution

**Claim:** the mechanism generalizes across scales, model families, and task types.

Required evidence:

- at least one from-scratch model and one pretrained retrofit;
- at least two scales;
- at least two task families;
- no dependence on one tokenizer, one data ordering, or one head count.

Falsifiers:

- benefit exists only in one small model or one benchmark;
- scaling reverses the effect;
- GQA/KV-cache constraints eliminate the gain.

## Statements prohibited before evidence

- “TGSA is the first sequential-head architecture.”
- “The finite difference is a true continuous derivative.”
- “Trajectory convergence proves reasoning.”
- “Token importance scores are explanations.”
- “TGSA is more efficient” without specifying parameter, FLOP, latency, and memory views.
- “The architecture beats Qwen” without a capacity/training/data-matched protocol.
