# BP-FUT-001 — Block-parallel training

**Status: DEFERRED_DOCS_ONLY. No implementation, experiment, resource reservation, or execution approval.**

This is the first of two method notes from the SAME diffusion-training paper. It is not an additional paper beyond CSBP-FUT-001. See [the index](README.md) for source counting and the activation policy.

## What the idea is about

Suresh et al. assign each corrupted target block's computation to one distributed rank, exploiting the additive block-diffusion loss. Block-local corrupted states need not be exchanged with other block owners. This is a training-execution idea, not a claim of independent learned skills. [P1, Sections 3–4.2]

[P1]: https://arxiv.org/html/2609.19242v1

## Proposed Rineya research question

For a separately qualified Rineya block-diffusion workload, can changing execution ownership reduce training cost while preserving the reference objective and gradients?

This is NOT the same question as whether converting an autoregressive Rineya checkpoint to block diffusion improves its usefulness. Test that modeling question separately before investing in distributed optimization. A faster bad model is not a successful Rineya integration.

## Possible future integration

The inspected centralized [method inventory at 5e5fe36](https://github.com/LUNCY579/ChatBubble/blob/5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80/experiments/post_training_mimo_centralized_v1/METHOD_INVENTORY.csv) lists Discrete Diffusion as PROPOSED_04, not as a qualified BP implementation. Reuse that direction's future autoregressive comparator rather than creating a second, inconsistently defined conversion benchmark. The inventory is evidence of a proposal; this note does not establish repository-wide absence of diffusion code.

Candidate external implementation: [ScalingIntelligence/Turbo-dLLM](https://github.com/ScalingIntelligence/Turbo-dLLM). The project README was inspected, but code was not imported, executed, or certified compatible with Rineya. Before reuse, pin a commit; inspect applicable licenses and notices, supported objectives, model adapters, dependencies, numerical assumptions, and hardware requirements. A generic Qwen-family example does not qualify an exact Titan checkpoint.

A future implementation should live in an isolated experiment package and environment. Do not modify production generation, the RSD/CLARE treatments, centralized qualification thresholds, or existing dependencies merely to explore this note. Reuse existing provenance/data-contract conventions only after reviewing their concrete APIs on the then-current execution branch.

## Proposed staged experiment — not an authorized runbook

**Stage A: establish the model workload.** Register an exact checkpoint/tokenizer/template, corruption process, block masks, loss weighting, position handling, and sampler. Compare a small conversion pilot with the original autoregressive model and an ordinary continued-training control using parent-disjoint data. Measure unassisted quality, retention, termination, generation work, latency, and memory. Changing the objective and changing parallelism must be separate interventions.

**Stage B: mechanics before speed.** On a tiny deterministic model, compare a conventional reference with block-owned execution using identical samples and corruptions. Register tolerances before inspecting the comparison. Check loss, relevant logits, every trainable parameter gradient, and one optimizer update. Include padding, unequal last blocks, variable supervised-token counts, different block assignments, and fixed random streams. Verify global loss normalization, shared-prefix gradient contributions, and shared-parameter gradient reduction; local target ownership is not permission to detach shared context or omit reductions.

**Stage C: resource-approved distributed pilot.** Use at least two approved devices and profile the actual workload. Record useful supervised tokens, total processed tokens, wall time, communication, maximum per-device memory, and synchronization overhead. Establish parity before comparing performance. A CPU simulation or multiple processes on one GPU can check bookkeeping but cannot establish a multi-GPU speedup.

## Revisit only when

There is a useful block-diffusion or otherwise demonstrated block-separable training workload, an explicit owner and resource budget, a viable reference implementation, and evidence that distributed execution is worth optimizing. Do not reserve or rent devices under this note's authority.

If no such workload exists, leave this record deferred. If numerical parity fails, stop and debug before benchmarking. If parity passes but total cost is not improved, retain the negative result rather than expanding the workload solely to obtain a headline.

## Evidence and boundary

Everything after the paper summary is a proposed Rineya integration design, not a result from the paper or this repository. No weights, datasets, implementations, native tests, GPU work, or paid services were used for this record. [CSBP-FUT-001](CONTEXT_SHARDED_BP.md) records the related long-context extension.
