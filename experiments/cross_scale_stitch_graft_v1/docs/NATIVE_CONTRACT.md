# Native integration boundary and unresolved prerequisites

## What this delivery does NOT certify

Transformers and official pretrained weights were not present in the implementation environment. The CPU qualification uses small random causal models, not downloaded Qwen or Rineya weights. Source-shaped arithmetic and toy parity cannot certify native behavior.

`native_qwen.py` is a source-reviewed adapter candidate for an already-loaded **exact stock** `Qwen3ForCausalLM` under Transformers **4.51.3**, CPU FP32, eager attention, zero dropout, no sliding window and no scaled RoPE. It accepts no model path and invokes no loader. It does not download, cast, quantize, merge or modify source weights. The caller must already own materialized, frozen, eval-mode modules, including nonpersistent rotary buffers.

The helper preserves native blocks and native rotary embeddings. Its FFN capture/graft interface splits the block at raw post-attention residuals. Its composed forwards are not `GenerationMixin` implementations and deliberately refuse caches, batch sizes above one and requests beyond the configured token bound.

## Historical Supa-A1.1 is not stock Qwen3

The inspected parent audit records `Qwen3AbstainForCausalLM`, FP16 storage and a non-weight forward logit adjustment. The stock helper rejects custom subclasses. `Decoder` also refuses `custom_a11_unqualified` identities rather than silently turning them into dense baselines.

The next implementation gate must examine the exact custom class and reproduce its original logits, policy application order, tokenizer/template and generation behavior. An explicitly authorized raw-dense arm could be an additional control, but cannot replace A1.1 without relabeling the experiment. Switching to official Supa Core 1.7 is also a distinct subject decision.

No native subject is implicitly selected by filling a plausible model name into an `Identity`. Those hashes are data contracts, not proof that the assets were inspected or authorization was granted.

## Required receipts before collecting native evidence

1. Exact source, weights and complete asset inventory; immutable revisions; custom code and model-output policy; storage and execution precision; runtime environment and resolved attention implementation.
2. Reviewed fresh open task banks and split registry; named scorer with accepted-answer semantics; exact input token IDs, thinking flags, EOS IDs and decoding bounds.
3. Measured available RAM and device/cgroup limits; load/capture/output-byte budgets; one-donor-at-a-time plan; watchdog and explicit no-bypass policy.
4. Unchanged native forward versus wrapper/segment parity; raw capture-site parity; causal-prefix invariance; finite behavior and deterministic bounded generation. Identity reconnection and bypass must be tested at native shapes/precision.
5. Source integrity before and after use; candidate composition identity including both model sources, cut/site, fitted maps, gain and output policy. Do not label a candidate using only its host's `Identity`.
6. Actual complete headroom panels and controls; then active graft/stitch measurements. Artifact hashes and small mechanics tests do not count as native samples.

Any failure in loading, resource preflight or collection is recorded as engineering failure/blocked, not as an incorrect answer. A runtime with a different Transformers version needs its own adapter review and receipt. Stock source validation alone does not qualify custom A1.1.

## Work deliberately deferred

A native weight loader and asset verifier; custom A1.1 output-policy integration; native activation and task-bank collectors; native random/shuffled control orchestration; bounded generation and scoring against approved banks; native bridge-only training; KV caching; quantized execution; GPU execution; production serialization/deployment; full-model recovery.

This is not a TODO-only project: bridge fitting, module composition, controls, loss masking, optimizer ownership, safe artifacts, scoring and deterministic fixture qualification are executable. The deferred work is the source-specific transition from these mechanics to an actual model experiment.
