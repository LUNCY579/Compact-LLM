# CSBP-FUT-001 — Context-sharded block parallelism

**Status: DEFERRED_DOCS_ONLY. No implementation, experiment, resource reservation, or execution approval.**

This is an extension of [BP-FUT-001](BLOCK_PARALLELISM.md) from the SAME paper, not a third distinct remaining paper.

## What the idea is about

Pure BP can replicate overlapping clean prefixes. CSBP additionally shards the clean context while keeping corrupted-block computation local. The method changes distributed execution, preserving the intended block-diffusion computation rather than introducing a new reasoning objective. [Paper, Sections 4.3–4.4](https://arxiv.org/html/2609.19242v1).

## Proposed Rineya research question

Once a useful Rineya block-diffusion workload exists, does context sharding remove a measured memory/communication bottleneck at useful total cost, without changing the quality of the trained model?

This note records a systems hypothesis. It does not promise a speedup, require million-token training, or claim that a longer accepted input improves reasoning. On a single-device run, there is no inter-device communication to eliminate; that is a reason to defer this optimization, not to imitate a distributed benchmark locally.

## Possible future integration

Treat this as a candidate execution backend for the workload qualified in BP-FUT-001. First inspect the [official Turbo-dLLM repository](https://github.com/ScalingIntelligence/Turbo-dLLM) at a pinned revision instead of immediately writing a second kernel stack. Current dependency compatibility, exact checkpoint support, distributed topology, and licensing need a fresh review at activation. Nothing from that repository has been installed or copied here.

The centralized [inventory at 5e5fe36](https://github.com/LUNCY579/ChatBubble/blob/5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80/experiments/post_training_mimo_centralized_v1/METHOD_INVENTORY.csv) also names `experiments/factorized_memory_attention_v1` and `experiments/native_sparse_attention_v1` as architecture research with unresolved efficacy/native boundaries. Possible future reuse is their evaluation questions and provenance conventions, not an assumption that their attention computations can be interchanged with CSBP. Those paths refer to the pinned centralized snapshot, not an assertion that their contents were merged into this main-based notes branch.

A later joint experiment could ask whether a separately qualified attention approximation and a distributed execution scheme have complementary benefits. Do not combine them in the first test: doing so would change both the computation and its placement, defeating a clean parity comparison.

## Proposed qualification and benchmark design

**Correctness first.** Compare the same checkpoint and examples under a conventional context-parallel reference, pure BP where it fits, and CSBP. Preserve corruption draws, masks, global positions, precision, loss normalization, and effective global batch. Test forward outputs, all relevant gradients, and a bounded optimizer step against registered tolerances. Include uneven block counts and assignment changes; inspect shared-context gradient accumulation and global parameter reductions.

**Two separate performance questions.** Use a same-topology experiment to isolate the intervention. Then allow each method an equally bounded topology search to compare best feasible deployment configurations. Label these results separately rather than crediting an unrelated batch-size or topology change entirely to CSBP.

**Honest cost accounting.** Count all communication, not only the messages eliminated by the intervention. Report warmup policy, repeated timing distributions, device utilization, maximum memory across devices, useful supervised tokens, total processed tokens, and monetary cost under an approved resource contract. Keep parity-check and tuning costs visible. A short communication microbenchmark is not an end-to-end training result.

**Quality checks.** Separate equal-training-exposure evaluation from equal-wall-clock evaluation. Freeze development and confirmation roles before running. Register a preservation margin for old capabilities and score evidence retrieval, cross-evidence reasoning, termination, and unanswerable cases at several supported lengths. Do not label successful input loading or a matching training loss as demonstrated long-context competence.

## Revisit only when

BP-FUT-001 has a qualified workload; an approved multi-device environment exists; and actual profiles identify clean-context replication or attention communication as a relevant cost. Start at the smallest representative context, not at the largest number in the paper.

Remain deferred if the bottleneck is elsewhere, the quality of the underlying model is inadequate, the reference is not qualified, or the expected engineering and resource cost is unjustified. Passing correctness without a useful cost improvement is a valid negative performance finding.

## Evidence and boundary

The experimental plan is a proposed Rineya adaptation, not a reproduction or reported paper result. There is no new backend, configuration, package dependency, benchmark, qualification receipt, training result, or GPU execution in this record. Activation requires the process in [the index](README.md).
