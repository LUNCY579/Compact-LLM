# TWM-001 research protocol

## 1. Claim and scope

The target is a frozen 28-layer, width-1024, dense Qwen3-derived Rineya 0.6B checkpoint. It retains its native 16-query/8-KV-head, head-dimension-128 attention. The memory heads have their own independent geometry; do not infer native head dimensions from hidden width divided by query-head count.

Primary hypothesis: the triadic sidecar improves association binding, targeted factual updating, and compositional use of facts over both ordinary gated-delta memory and a similarly sized stateless adapter. The experiment does not assume that associative-state interference and persistent parameter/skill interference are the same phenomenon.

This is an augmentation of a softmax-attention backbone, not the paper's GDN-to-triadic upcycling experiment. Factorized Memory Attention's token-indexed value construction and historical trajectory-guided sequential attention are distinct. Do not merge them into this first campaign or retire their branches based on this implementation.

## 2. Exact reference mechanism

State layout is `[B,H,E,K,V]`. A token produces normalized `k,q` and, for E>1, positive softplus/L2-normalized `k2,q2`. E=1 uses constant second addresses equal to one. A causal depthwise convolution with width four is applied before address/value nonlinearities; its history is part of every live cache.

For each slice e:

```text
D_e = alpha_e * S_e
residual = v - sum_e k2_e * (D_e.T @ k)       # ONE joint residual
S_e_new = D_e + beta * k2_e * outer(k, residual)
o = sum_e q2_e * (S_e_new.T @ q) / sqrt(K)
```

The independent matrix reference flattens the state to `[B,H,E*K,V]` and uses the joint address `k2 outer k`. Both implementations must agree in outputs and gradients. A bank of independent residuals is not equivalent.

The sidecar uses `alpha=exp(-softplus(decay_projection(x)))`, `beta=sigmoid(write_projection(x))`, input/output RMS normalization, and a tanh-bounded scalar residual gate. This is a Rineya-oriented adaptation: it does not copy the paper's complete A-log parameterization or its internal output-gate architecture. Do not describe the package as a full numerical reproduction of the published model.

The gate starts at zero; the output projection does not. At initialization, only the gate may receive a useful nonzero gradient. Later steps open gradients to the memory parameters. A disabled module bypasses the augmentation exactly, but this does not guarantee retained behavior while the module is enabled.

No in-place state update is used. State detach is explicit. Padding neither writes, decays, nor advances convolution history. A document reset clears both recurrent memory and history before the new token. The native wrapper intentionally supports one document per batch row, not packed-document attention.

## 3. Registered first campaign

`configs/pilot.json` fixes the insertion after zero-based layer 13, E=4, memory heads=4, K=V=64, three training seeds (17,43,91), 100 optimizer steps, batch size one, learning rate 0.001, gradient clipping 1.0, teacher-KL weight 0.02, maximum 512 total input/output tokens, and greedy decoding with 24 generated tokens.

This is a bounded initial dose, not a claim that 100 steps are sufficient. Development-only revisions require a new config/source identity and new qualification before holdout evaluation. Do not tune a published candidate after opening its prospective holdout. Every native run records the complete config, checkpoint lock, source hash, data identity, and environment; adapters cannot silently load into a different configuration or backbone.

Teacher KL uses the last up-to-16 prompt positions against the disabled frozen model on the same training input. It is an auxiliary preservation signal, not a broad-retention benchmark. No full-backbone gradient update, LoRA, CSS merge, optimizer-state carryover, or other skill intervention is permitted in this first factorial.

Compare five arms:

1. Frozen unchanged baseline (no training).
2. Stateless low-rank residual adapter with active parameter count within 1% of the triadic arm.
3. Ordinary GDN sidecar, E=1, same first-key/value dimensions.
4. Triadic GDN sidecar, E=4.
5. Single-matrix GDN sidecar with key width multiplied by four, matching recurrent-state elements.

Arms 2–4 answer the primary augmentation question. Arm 5 separates expanded state from efficient factorized addressing, but has more projection parameters and different total buffers/decay geometry. Report both these differences and measured time; do not shrink the backbone MLP merely to hide an extra-state cost.

## 4. Data and leakage controls

Generate new train, development, and prospective holdout files with disjoint entity namespaces and surface templates. The generator covers binding aliases, chronological updates, preservation of unrelated attributes, scopes, two/three-hop chains, absent facts, unordered conflicts, private role state, and format constraints. Existing sealed/TEST2 resources are explicitly excluded and never auto-discovered.

`--retrieval-probes` adds direct-edge retrieval queries to hop problems. They share their parent's group ID; bootstrap by group, not by row, to avoid treating repeated source facts as independent samples. These probes are a separate diagnostic, not extra evidence for successful composition.

Oracle prompts retain relevant supporting records, not an asserted final result. Compare ordinary-context and oracle-facts development performance to distinguish retrieval failures from composition failures. Oracle-versus-ordinary comparisons intentionally have different comparison identities and are not falsely reported as identical-input paired ablations.

The primary acquisition family set is `binding update unrelated_after_update scope two_hop three_hop`. Report `hop_edge_retrieval`, `missing`, `conflict`, `role_state`, and `format_constraint` separately. Preserve exact output tokens/stop reasons; normalize whitespace only for answer scoring, never strip explanations or extract a convenient embedded answer.

`retention_demo` is merely a generated loader/smoke fixture. Actual capability retention requires an independently chosen, disjoint, established Rineya retention bank supplied explicitly with split `retention`. The package does not search for such a bank. Do not call a contextual association-retention result permanent skill retention.

Data files and output directories are never silently overwritten. Training rejects any non-training split. Evaluation rejects training rows and requires `--allow-holdout` for the fresh prospective holdout; this flag is not authorization to open any pre-existing sealed project set.

## 5. Gates and interpretation

Mechanical reference gate: tensor/flattened equivalence, gradients, E=1 reduction, joint correction, causal/reset/padding behavior, state budgets, streaming history, live gate gradients, wrapper lifecycle, and split/provenance guards.

Native mechanical gate: execute the optional tiny Qwen3 tests in the pinned environment, then `qualify` on the exact 0.6B checkpoint. Require exact neutral and disabled logits, active full-versus-incremental agreement, and exact greedy token equality. The runner records FP32 or BF16 tolerances and binds the native receipt to the code/config/checkpoint/device/runtime. Training/evaluation reject missing, incomplete, or stale native qualification.

Quality gate: after a frozen development choice, require paired acquisition improvement over BOTH stateless and E=1 controls, not just over baseline. Use at least three training seeds. A single-seed cluster interval is descriptive; do not pool seed/example pairs as independent questions. Positive evidence should be consistent across seeds and have a positive lower 95% paired interval on the registered primary acquisition metric.

Retention gate: compare the enabled candidate against the unchanged baseline on a disjoint, sufficiently powered retention bank. The proposed non-inferiority margin is one percentage point on the registered aggregate, with separate checks for instruction adherence, unsupported-answer rate, and regressions in established task categories. A small pilot cannot tightly establish a one-point margin; increase the evaluation sample before making that claim.

Cost gate: record added parameters, recurrent and convolution bytes, retained native KV memory, actual peak CUDA allocation, prefill latency, decode latency, output token count, and input count. Warm-up and repeat timing runs on the same device and compare matched batches/lengths before making a speed claim. The runner's per-example timings are raw observations, not a calibrated throughput benchmark. The first pilot has a 6.5 GiB free-memory admission guard and a 7.0 GiB allocated-peak guard, not a promise that training fits. Allocated peak excludes other processes and is not the same as reserved/total device memory.

A fixed rate slowdown threshold is not adopted from the paper's H100 results. Set an acceptable laptop budget using development-only resource profiling before an expensive native sweep. Do not rent cloud compute or start remote jobs implicitly.

The comparison helper deliberately emits `NOT_EVALUATED_REQUIRES_MULTISEED_CONTROLS_RETENTION_AND_COST`. It does not automatically promote a model.

## 6. Mechanistic interventions

`erase_reads` removes the sidecar's read contribution while still updating memory. `constant_second_keys` replaces both second addresses with normalized constants; heterogeneous learned decay can still distinguish slices, so this is an address-collapse intervention, not an exact reduction to E=1. `read_only` freezes recurrent evidence writes and decay but advances query-convolution history.

Offline `shuffle_second_keys` permutes second addresses within each supplied evidence document. It must not include the final answer/query tokens, mix documents, or be presented as causal streaming evaluation. The native CLI exposes only causal interventions. Compare matched fresh-prefill runs; do not switch enabled mode or ablations mid-cache.

Use effective address rank, state norm, and intervention effects to test whether the larger state is used. Diagnostics alone do not prove semantic relation disentanglement. A gain that survives erasing memory could reflect a surrounding adaptation pathway or an evaluation error rather than better associative storage.

## 7. Later work, conditional on native evidence

Only after a meaningful acquisition/retention/cost result should the branch grow to longer contexts, more insertion points, E=8, or feedback-conditioned repeated reads. A future repeated-read factorial must match read count/compute and initially keep source evidence read-only during deliberation. Generated hypotheses must not overwrite evidence by default.

Optimized RTX-compatible kernels, full gradient checkpointing, packed native documents, quantized/offloaded backbones, beam/speculative decoding, multi-device execution, and production generation integration are unqualified. Current guards fail closed rather than silently approximate them. Do not replace native attention or claim KV-cache savings based on this augmentation alone.
