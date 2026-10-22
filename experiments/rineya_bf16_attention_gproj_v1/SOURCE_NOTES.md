# Primary sources and implementation boundary

Consulted 2026-10-04. This code was written independently from the published
mathematical description; no third-party kernel source was copied.

1. Chen et al., *Broken Symmetry in BF16 Attention: Why FlashAttention Gradients
   Blow Up Late in Training*, arXiv:2609.34272v1, 2026-09-28.
   https://arxiv.org/html/2609.34272v1
   Sections 3--4 and Appendix A supply the conservation law and four-key witness;
   Appendix B specifies the correction arithmetic. The consumed score-gradient
   operand `t` and probability operand `r` are BF16. With `rho=sum(t)`, `m=sum(r)`
   and `lambda=rho/m`, correct the contractions using
   `dQ=alpha*(t@K-lambda*(r@K))` and
   `dK=alpha*(t.T@Q-r.T@(lambda*Q))`.
   Their implementation also rounds the key-correction query operand and uses a
   separate key pass. The paper's Hopper/head-dimension-64 kernel is not a
   drop-in native-width Rineya or RTX-4070 implementation. Its reported training
   outcome and overhead are not measured results of this branch.

2. PyTorch 2.10 SDPA API, masks, GQA, dtype behavior and explicit backend selection:
   https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html
   https://docs.pytorch.org/docs/2.10/generated/torch.nn.attention.sdpa_kernel.html
   These contracts guide the native replay. Backend support is probed rather than
   assumed. The reference uses PyTorch's upper-left rectangular causal convention;
   the native training session restricts itself to cache-free square attention.

3. Hugging Face Qwen3 API and attention integration documentation:
   https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3
   https://huggingface.co/docs/transformers/attention_interface
   The optional loader requests SDPA explicitly and verifies actual interception.
   Documentation review is not a native Transformers execution test.

4. Official FlashAttention repository and hardware scope:
   https://github.com/Dao-AILab/flash-attention
   The branch does not install FA3 or patch external FlashAttention packages.

## What the reference deliberately does not emulate

It does not reproduce native LSE reconstruction, tensor-core accumulation order,
atomic reduction order, tiled forward exponentials or a second independently
rounded probability reconstruction. It performs separate corrected contractions
in a dense FP32 reference, with an optional BF16 cast of `lambda*Q`; the FP64
algebra path can omit that cast. Recasting a corrected score matrix is not used.
Native saved O may be supplied to study its centering channel, but native LSE is
unavailable; this cannot be labeled a bitwise-faithful fused-backward replacement.

Precision gates and pilot budgets are proposed Rineya experiment settings, not
thresholds established by these papers or libraries. No literature result is
entered as a local acquisition, retention or throughput measurement.
