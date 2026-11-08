# Architecture and implementation contract

## Why a separate experiment

The existing dual28 constructor assumes one configuration, equal stack depths, a shared embedding and one official donor. XSC-001 uses separate source configurations, 28 versus 36 native layers, separate embeddings, and an explicit rectangular memory interface. Its Python primitives are imported from three byte-pinned RAB-001 files, unchanged at the reviewed parent. It does not relax or reuse the earlier experiment's source locks or licenses.

Prior cross-scale parameter projection is not positive evidence for this architecture. The preserved transport study at `fde553c908db39e87a15a52ed239d1ba0a68bfd3` tested different operators, withdrew its cross-donor-agreement interpretation, and reported no confirmatory improvement. XSC-001 retains the large decoder and learns a functional interface rather than shrinking donor weights into a small model. Matching token IDs or head dimensions does not establish learned basis alignment.

## Computation

Let E be the full small causal model without using its tied LM readout, and D the ordinary large causal decoder. For each aligned token position:

1. The encoder produces its final normalized state, width 1024.
2. A new RMS normalization (initial weights one) and learned K/V projections form shared memory of 8 heads x 128. The encoder's final norm remains frozen and used; it is not silently dropped.
3. The decoder starts from its own width-2560 embedding. Every original self-attention and SwiGLU block executes unchanged.
4. After each complete decoder block, a new normalized Q projection yields 32 heads x 128. H reads all permitted earlier/current memory positions; P reads only the same physical token with a sigmoid score.
5. Each read is projected back to width 2560 and added to the decoder residual. Decoder final norm and its tied embedding table produce logits.

All 36 reads use native decoder query width 4096. This intentionally accepts a large interface (~757M parameters) for an architecture-first test. Sparse placements, low-rank readers, fewer heads, a tiny memory-slot budget, bridge-width sweeps and backbone compression are separate future experiments, not hidden optimizations in this one.

The initial zero read outputs preserve the ordinary D function on identical decoder inputs. The first optimizer step can update O; Q/K/V gradients wake up after O changes. Tests check this order. A zero-output identity test alone is insufficient: qualification temporarily activates O, verifies all 148 interface gradient tensors, runs causal/cache interventions, and restores the initialization afterwards.

## Position, mask and cache contracts

Encoder self-attention uses its native RoPE and head geometry. The cross-memory writer and decoder readers use the decoder's RoPE geometry. The primitive attention mask enforces physical order **and** absolute positions, padding validity and packed-document segment identity. Repeated position IDs cannot expose a physically later token. Segment IDs must uniquely identify packed documents; repeated use of an ID intentionally makes those tokens part of the same segment.

The three cache groups are validated separately: native encoder K/V, native decoder K/V and decoder-geometry shared memory. Cache identity binds model ownership, parameter pointer/version/dtype/device, configuration and read policy. Mutating a model or changing the read policy invalidates old caches. The forward pass never edits prior cache tensors in place. Callers must treat returned tensors as read-only; frozen dataclasses are not a cryptographic barrier against malicious tensor mutation.

Tail-only mode saves only final-logit projection. It still executes both causal stacks on every input token. Unequal sequence lengths, asynchronous streams, prompt-only encoder caching, eviction and resumption from final shared memory alone are not supported. Gradient checkpointing uses bound layer closures and is tested against the non-checkpointed gradients. Frozen decoder parameters do not imply `no_grad` around decoder computation: gradients must still propagate from later blocks to earlier new reads.

## Sources, persistence and resources

`sources.json` records the exact Supa release and the historical official Qwen3-4B revision, config/tokenizer hashes and all three shard hashes/index. Both loaders reject missing assets, symlinks, unexpected weight formats, altered headers, wrong dtype/shape, wrong shard assignment, extra/missing tensors and custom/quantized configurations. Tensor reads use safetensors, never pickle. Sources are checked again after streaming to reject replacement during loading. The official source is BF16; reference arithmetic is CPU FP32. No source fallback or online custom code exists.

Native comparison uses a read-only Qwen view of the verified tensors to avoid a redundant large weight copy. Independent native forward/cache execution is still required. H/P construction then makes independent backbone copies; trainable interface copies never alias backbone parameters. Arms run sequentially. The smoke deletes the completed autograd graph before allocating the next arm.

Checkpoints contain **only the new interface**, not a complete standalone model. Their manifests bind source/runtime/checkout provenance, architecture, tensor inventory/dtype and file SHA-256. Full validation precedes mutation. Run/checkpoint directories are create-only; failed attempts and partial outputs remain preserved. Checksums detect accidental or unapproved changes; they are not digital signatures and do not authenticate a maliciously fabricated receipt.

Official runners require 64 GiB available RAM (respecting host and cgroup ceilings) before allocation. This conservative prerequisite is not an observed peak-RAM guarantee or a reservation against concurrent jobs. Do not run alongside another memory-intensive workload. GPU, quantized, mixed-precision and deployment claims remain closed.
