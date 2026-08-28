# Qwen/GQA integration plan

The Phase-0 kernel establishes semantics, not drop-in production compatibility. A credible Qwen-family implementation must preserve native behavior before testing the new mechanism.

## 1. Compatibility requirements

A production module must preserve:

- the host model’s exact pre-norm/post-norm placement;
- rotary position embeddings and position-id semantics;
- grouped-query attention head mapping;
- causal and padding mask conventions;
- KV-cache format, cache positions, and generation APIs;
- tensor parallel / sharded projection conventions where present;
- attention dropout and dtype behavior;
- eager, SDPA, and FlashAttention fallback expectations;
- state-dict naming or a fully documented conversion map;
- generation equivalence when all TGSA controls are disabled.

No quality experiment should begin until the disabled-control path is numerically matched against the native implementation within a registered tolerance.

## 2. Three integration variants

### Variant A — independent sequential K/V

Each sequential query head produces its own trajectory-conditioned K/V stream.

Advantages:

- closest to the Phase-0 operator;
- maximal head-specific expressivity.

Costs:

- MHA-like KV cache rather than native GQA savings;
- highest migration and latency cost.

Use for a tiny from-scratch mechanistic model before Qwen retrofit.

### Variant B — grouped sequential K/V

Retain Qwen’s K/V groups. Query heads within each group run sequentially, while group K/V are generated from either:

- the group’s initial carrier anchor; or
- a group-level trajectory state updated after each query head.

This is the primary Qwen-compatible candidate because it retains most GQA cache savings while allowing head-order refinement.

### Variant C — native anchored K/V

Copy and freeze or lightly tune native K/V. Sequential trajectory primarily conditions Q, the carrier, and an additive key prior.

Advantages:

- smallest checkpoint disruption;
- easiest near-identity initialization;
- lowest extra cache cost.

Risk:

- insufficient expressivity if trajectory-dependent memory reconstruction is essential.

## 3. Weight conversion

For a native projection matrix, slice each query head’s rows into the corresponding sequential-head projection. For K/V, follow the selected variant’s GQA mapping exactly.

New pathways must start near identity/no-op:

- small trajectory-encoder output projection;
- zero or near-zero trajectory injection projections;
- bounded gates with documented initial values;
- importance-logit scale initialized so it does not sharply suppress tokens;
- carrier initialization chosen from a deterministic native projection or calibrated linear map.

Conversion output must include:

- base repository and revision;
- input and output tensor hashes;
- parameter mapping table;
- list of copied, transformed, newly initialized, frozen, and trainable tensors;
- numerical equivalence test with controls disabled.

## 4. Cache algorithm for one decode token

For token `t`:

1. compute native hidden state and initialize `c[0,t]`;
2. for each sequential head `j`:
   - compute `v[j-1,t]` from the two latest carrier states for this token;
   - encode trajectory and predict `c_hat[j,t]`;
   - construct Q and, depending on the integration variant, K/V;
   - apply RoPE using the native cache position;
   - append current token’s K/V to the correct head/group cache;
   - attend to cached positions `≤t`;
   - correct the carrier;
3. concatenate head outputs and apply the native output projection;
4. return the host model’s expected cache object unchanged in public shape.

Cache updates must not cause a later head at token `t` to observe K/V from a future head as though it were a future token. Head index and token index are distinct axes.

## 5. Implementation sequence

### Gate Q0 — native attention characterization

Capture exact tensor shapes, RoPE order, GQA mapping, masks, cache classes, dtypes, and output contracts for the selected Qwen/Supa revision.

### Gate Q1 — wrapper equivalence

Implement a wrapper that reproduces native attention with no sequential trajectory. Require:

- forward maximum absolute error below the registered dtype-specific threshold;
- matching generation tokens under deterministic decoding;
- matching cache shapes and lengths;
- no regression in existing repository tests.

### Gate Q2 — cascade only

Enable ordered head execution and shared carrier without finite-difference control. This becomes the direct architectural baseline.

### Gate Q3 — learned trajectory control

Add the low-rank encoder, velocity injection, and telemetry. Keep Q/K/V conditioning, importance, and adaptive computation disabled.

### Gate Q4 — mechanism ladder

Enable Q, K, V conditioning separately; then predictor–corrector; then soft token significance. Never introduce two untested mechanisms in one comparison.

### Gate Q5 — cache and generation validation

Test prefill/decode parity, beam/sample generation, long-cache growth, batched unequal lengths, padding, and cache reuse.

### Gate Q6 — optimized execution

Profile before optimizing. Candidate strategies:

- fuse small carrier/gate projections;
- batch matrix operations across heads where dependencies permit;
- use CUDA graphs for fixed head order;
- share trajectory encoders across head groups;
- progressive soft-to-hard token reduction;
- early exit or routed head subsets;
- Triton/custom kernel only after semantics are frozen.

## 6. Training curriculum

A safe retrofit curriculum:

1. **calibration:** freeze native weights; train only carrier and trajectory controls to reconstruct native attention outputs;
2. **controlled adaptation:** unfreeze output projection and selected target-layer attention weights;
3. **task training:** train all registered arms on identical data/order/tokens;
4. **long-context specialization:** introduce position-stratified retrieval and reasoning only after general NLL stability;
5. **optional full unfreeze:** only if adapter/control-only training plateaus, with a matched native-attention control receiving the same trainable capacity.

The reconstruction phase is an initialization aid, not evidence of the research hypothesis.

## 7. First target scale

Use a tiny from-scratch decoder to validate mechanism causality and learning behavior, then a single-layer or small-layer replacement in the Qwen3-0.6B/Supa-compatible family. This minimizes hardware risk and lets the experiment distinguish architecture effects before committing to full-model conversion.

On constrained hardware, preserve scientific fairness through gradient accumulation, activation checkpointing, mixed precision, and identical effective token counts. Do not compensate one arm with additional updates or data.

## 8. Mandatory integration tests

- disabled-path numerical equivalence;
- full-sequence versus cached-token equivalence;
- suffix perturbation causal invariance;
- left/right padding behavior;
- RoPE position reset and offset cases;
- GQA head-to-group mapping;
- cache reorder for beam search where supported;
- deterministic greedy generation;
- gradient flow into every enabled control path;
- finite bf16/fp16 forward/backward;
- state-dict save/load round trip;
- peak-memory and cache-byte accounting;
- no telemetry allocation when disabled.
