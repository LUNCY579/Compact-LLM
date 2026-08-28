# Design decisions and open questions

This is an append-only architectural decision record. A later implementation may overturn a decision, but it must preserve the original rationale and add a superseding entry.

## D001 — shared carrier before differencing

**Status:** accepted for all TGSA experiments.

Raw head outputs may inhabit different learned coordinate systems, so subtracting head `j-1` from head `j-2` can mix representation change with basis change. TGSA instead maps the evolving computation into one shared carrier space and defines velocity there.

**Consequence:** every implementation needs an explicit carrier update and a test that all differenced states have identical shape/semantics.

## D002 — call the quantity a finite-difference velocity

**Status:** accepted.

Head index is discrete. The project may use a dynamical-systems analogy, but must not represent `c[j]-c[j-1]` as an exact continuous derivative. “Finite-difference trajectory/velocity across head depth” is the default language.

## D003 — trajectory must control computation nonlinearly

**Status:** accepted.

A linearly added delta is often algebraically absorbable into weights on the two prior states. The load-bearing mechanism therefore uses nonlinear encoding and multiplicative gates, and can condition Q/K/V, prediction, significance, and correction.

**Required control:** raw delta.

## D004 — parameter-identical history-mean control

**Status:** accepted.

The strongest no-delta control uses the mean of the current and previous carrier states, passes it through the same encoder and downstream paths, and has exactly the same parameter count as learned delta. Given the current carrier, mean and delta each permit recovery of the preceding carrier; their comparison isolates directional inductive bias rather than information access.

## D005 — predictor–corrector rather than momentum-only update

**Status:** accepted provisionally.

The trajectory first predicts a candidate next carrier, and attention supplies a correction. This makes the role of the trajectory inspectable and permits separate predictor/correction ablations.

**Risk:** the predictor may add instability or be redundant with ordinary residual depth.

## D006 — soft significance before hard compression

**Status:** accepted.

Phase 0 uses a bounded differentiable key prior and never removes tokens. Hard pruning would entangle mechanism quality with discrete routing and hardware effects. It is deferred until the soft significance contribution is established.

## D007 — do not modify final vocabulary probabilities initially

**Status:** accepted.

The first architecture modifies internal representations, attention probabilities, and token significance. Intermediate LM-head probabilities may be measured as a convergence signal later, but final vocabulary logits are not directly rewritten by the attention module in Phase 0/1.

## D008 — clear independent-KV reference before GQA optimization

**Status:** accepted.

The reference kernel uses independent head-specific K/V semantics because it is easiest to audit. Qwen integration must then compare independent, grouped sequential, and native-anchored K/V variants rather than silently changing the mechanism for cache convenience.

## D009 — telemetry is opt-in during performance measurement

**Status:** accepted.

Summary telemetry is useful for mechanism analysis; full maps are expensive. Timing helpers explicitly request no telemetry. Any benchmark that allocates full telemetry is invalid unless the same cost is imposed on every arm and the purpose is telemetry performance itself.

## D010 — fixed head order first

**Status:** accepted for Phase 0/1; adaptive order deferred.

A fixed order permits causal attribution and simpler kernels. Learned head routing/order, early exit, and head skipping are later efficiency/expressivity experiments and must not be bundled into the first quality test.

## D011 — first-order before acceleration

**Status:** accepted.

Second-order acceleration `a[j]=v[j]-v[j-1]` is mathematically natural but deferred. It introduces another state and another source of gain/instability. The first paper-quality mechanism experiment must establish the first-order contribution first.

## Open questions

| ID | Question | Required evidence |
|---|---|---|
| O1 | Is carrier width best set to one head width, a larger shared width, or model width? | matched-capacity sweep |
| O2 | Should trajectory encoders be per-head, shared, or group-shared? | quality/latency/parameter Pareto study |
| O3 | Should Q, K, and V all be conditioned? | separate ablations and cache cost |
| O4 | Does carrier normalization erase useful magnitude information? | norm/no-norm stability and quality comparison |
| O5 | Is fixed head order meaningful after pretrained conversion? | head-order permutation and learned-order study |
| O6 | Does the trajectory converge, oscillate, or collapse as scale grows? | held-out telemetry and intervention |
| O7 | Can soft significance support progressive KV/token reduction without quality loss? | soft-to-hard compression study |
| O8 | Can adaptive stopping recover serial latency? | calibrated stopping with equal-quality controls |
| O9 | Does a generic recurrent-depth block dominate TGSA? | matched recurrent-depth baseline |
| O10 | Does the benefit survive GQA and real KV caching? | Qwen/Supa integration stages Q1–Q6 |
