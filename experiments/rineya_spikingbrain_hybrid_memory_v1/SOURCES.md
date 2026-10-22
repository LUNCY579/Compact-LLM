# Source and adaptation ledger

Read/verified during implementation on 2026-10-02. Code here is an independent
PyTorch implementation, not vendored from these repositories.

1. **SpikingBrain: Spiking Brain-inspired Large Models**, Pan et al.,
   arXiv:2509.05276v3 (2025-12-01).
   https://arxiv.org/html/2509.05276v3
   Motivation: converted pretrained weights, local/linear hybrid attention and
   separate activation/spike coding. The paper's reported long-context TTFT,
   arithmetic-energy estimates and spike sparsity are not measurements of this
   experiment. Do not transfer its 100x/97% claims to Rineya or this code.

2. **LoLCATs: On Low-Rank Linearizing of Large Language Models**,
   arXiv:2410.10254v2.
   https://arxiv.org/html/2410.10254v2
   Motivation: separate attention-output distillation from low-rank recovery.
   We do not reproduce its full feature maps, kernels, data, scales or results.

3. **Original released SpikingBrain GLA implementation**, inspected source:
   https://github.com/BICLab/SpikingBrain-7B/blob/ef99987167cf7386ab3348312c8c7aa00a6696ee/hf_7B_model/gla_attention.py
   Its hidden_size // heads assumption is NOT portable to the Qwen3-0.6B host.
   This pilot instead validates explicit projection geometry and shares FP32
   state by KV group. Local + recurrent paths are combined within one selected
   layer rather than reproducing its alternating-layer 7B configuration.

4. **Transformers Qwen3 implementation pinned for the native bridge**:
   https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
   Inspected Qwen3Attention and decoder call contract, including head_dim,
   projection shapes, QK norms, host RoPE, DynamicCache and two-item attention
   returns. Fetched Git blob: `5fec83d47888e44a0f3e0ffb2b067ce08e4712d3`.
   Optional integration tests instantiate a tiny RANDOM official implementation;
   no pretrained parameters are fetched by tests. Dependency absence is a skip,
   not native qualification.

5. **Official Qwen3-0.6B public configuration**, read for geometry only:
   https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
   The actual experiment source is NOT implicitly this public model. A separate
   exact local source inventory is mandatory; no model ID auto-download exists.

6. **Rineya parent and historical context**:
   https://github.com/LUNCY579/ChatBubble/tree/b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2
   https://github.com/LUNCY579/ChatBubble/blob/b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2/experiments/supa_long_context_minimal_intervention_v1/README.md
   The previous pilot explicitly distinguishes inference sensitivity from
   training importance. No historical layer ranking is imported as an optimum.
   Broader quality-audit/centralized-system records remain on their own branches;
   they are not merged or superseded by this experiment.

The prior discussion also mentioned SpikingBrain2.0. Its expanded state,
Gated DeltaNet, MHA conversion and large-scale training recipe are NOT
implemented or required by SBHM-001. No follow-up-paper empirical result is
used as a qualification or acceptance threshold here.
