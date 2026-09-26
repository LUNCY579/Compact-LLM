# Architecture Compatibility Report

Projection gate: **GO**

## Verified architecture

| Model | Class | Layers | Hidden | MLP | Q heads | KV heads | Head dim | Vocab | Stored dtype | Effective parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| qwen06 | Qwen3ForCausalLM | 28 | 1024 | 3072 | 16 | 8 | 128 | 151936 | bfloat16 | 596049920 |
| supa | Qwen3AbstainForCausalLM | 28 | 1024 | 3072 | 16 | 8 | 128 | 151936 | float16 | 596049920 |
| qwen4b | Qwen3ForCausalLM | 36 | 2560 | 9728 | 32 | 8 | 128 | 151936 | bfloat16 | 4022468096 |
| qwen8b | Qwen3ForCausalLM | 36 | 4096 | 12288 | 32 | 8 | 128 | 151936 | bfloat16 | 8190735360 |

All values above come from checkpoint configs and safetensor headers, not model-card assumptions.

## Small-model identity

- Exact Supa/Qwen3-0.6B state-dict layout match: `True`
- Tokenizer bundle equal across all four models: `True`
- Supa embedding shape: `[151936, 1024]`
- Supa output-head shape: `[151936, 1024]` (tied_to_model.embed_tokens.weight)
- Official 0.6B tied-head verification: `verified_redundant`
- Supa release manifest verification: `verified`
- Supa-A1.1 abstention is a non-weight forward logit offset. It is excluded from parameter deltas and retained only as a separately labeled behavioral control.

## Projection semantics

Safetensor matrices follow the PyTorch linear layout `[out_features, in_features]`. W2 retains complete query/KV heads along attention output-feature axes, then applies the declared residual-channel selection along hidden axes. Embeddings, norms, MLP channels, and output projections use the same deterministic residual-channel index set. Tied output embeddings remain tied.

### qwen4b

Projection GO: `True`

- residual channels require deterministic selection/projection
- MLP channels require deterministic selection/projection
- query heads require complete-head selection
- all KV heads can be retained; residual input width still projects

Depth mappings:

- `D1_prefix`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- `D2_uniform_full_depth`: `[0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 34, 35]`
- `D3_center_preserving`: `[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35]`

### qwen8b

Projection GO: `True`

- residual channels require deterministic selection/projection
- MLP channels require deterministic selection/projection
- query heads require complete-head selection
- all KV heads can be retained; residual input width still projects
- donor output head is untied but target is tied; projection must declare embedding, output_head, or mean collapse policy
- Required tied-collapse policy: `embedding`, `output_head`, `mean`

Depth mappings:

- `D1_prefix`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- `D2_uniform_full_depth`: `[0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 34, 35]`
- `D3_center_preserving`: `[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35]`

## Numerical and loader notes

- Merge arithmetic must cast source tensors to FP32 before any projection or combination.
- Official donors are BF16. Supa-A1.1 is stored FP16; storage-precision comparisons are separate finalist controls.
- Architecture audit reads only headers. Functional GO still requires a full load, finite forward pass, deterministic generation, EOS behavior, and exact post-serialization validation.
- Runtime attention kernels may vary by Transformers/device. Each run manifest must record the resolved implementation.
