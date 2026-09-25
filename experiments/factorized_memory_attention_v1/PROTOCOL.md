# FMA-QUAL-001 — Factorized Memory Attention for compact Rineya

## Status and evidence boundary

This branch is a qualification-first architecture experiment, not an efficacy result. It asks whether the Memory Attention (MA) value-construction mechanism can be made compact enough for a sub-1B Rineya model without confounding the result with a multi-billion-parameter per-layer memory table.

The branch base is 7d9b4fd450bc56405f6cec41459a7e83df7958ae (origin/main at qualification time). Historical research branches are read-only inputs to the audit; no architecture-research branch is inherited as the code base. A branch, unit test, valid conversion, or synthetic smoke test is not evidence that the architecture improves model quality.

Primary paper: Jiale Kang, Memory Attention, arXiv:2609.28399v1 (23 Sep 2026): https://arxiv.org/abs/2609.28399
Reference implementation inspected for QK-normalization ordering: https://github.com/Joluck/memory-attention/blob/main/fla/layers/memory_attn.py

## Paper construction and the Rineya deviation

For hidden states X, standard attention forms Q = X W_Q, K = X W_K, V = X W_V (paper Eq. 1) and applies the usual causal attention operator (Eq. 2). MA introduces, for each layer, a token-indexed table E in R^(N x d_v) and M = Norm(E[s]) (Eq. 7), where s are input token IDs and Norm is RMSNorm applied independently to each token vector within each KV head.

MA then uses Q = X W_Q, K = X W_K, V = K + M (Eq. 8). With RoPE, the paper specifies Q^R = RoPE(Q), K^R = RoPE(K), V = K + M (Eq. 9), so the value path uses K before positional rotation.

Qwen3 additionally has QK RMS normalization. The paper authors' reference implementation constructs v = k + m_norm(m) from the raw k_proj output and only afterward applies q_norm / k_norm for attention scoring. The Rineya retrofit therefore uses the raw key projection, before QK normalization and before RoPE, in the MA value path.

The paper uses a separate full table per layer. This branch intentionally does not reproduce that parameterization first.

## Rineya factorization

The registered baseline is Rineya-Supa-1.7.0-devfreeze. Its repository config gives N=151,936, hidden width 1,024, L=28, 16 attention heads, 8 KV heads, head dimension 128, value width d_v=1,024, and bfloat16 checkpoint dtype.

Instead of an independent E_l in R^(N x d_v) for every layer, FMA uses a shared U in R^(N x r), a per-layer P_l in R^(r x d_v), and:

M_l(s) = HeadwiseRMSNorm(U[s] P_l).

The default qualification rank is r=32. The RMS denominator is computed independently per token and KV head over d_h; one learned d_h scale vector is used per layer and broadcast over KV heads, matching the parameter sharing in the authors' reference implementation, which instantiates RMSNorm(head_dim).

During transition qualification:

V_l = (1 - g_l) * V_standard + g_l * (K_raw,l + M_l(s)),

where K_raw,l = X W_K,l is before QK normalization and RoPE. Gates are non-trainable buffers in this qualification harness. g=0 hard-short-circuits to the untouched standard path. g=1 hard-short-circuits the standard value function, making W_V bypass structural and testable.

The decisive future architecture test is hard-memory g_l=1. A permanently mixed gate is not evidence that token memory can replace the standard value pathway.

## Calibration / teacher option

A calibration-only path is prepared to reduce the shock of replacing pretrained W_V. The pretrained backbone is frozen and only FMA memory parameters are trainable. Using the raw pre-QK-norm/pre-RoPE key:

R_l = X W_V,l - K_raw,l

is the teacher residual and calibration fits HeadwiseRMSNorm(U[s] P_l) ~= R_l.

freeze_backbone_for_calibration() enforces the trainability boundary and calibration_loss() uses a detached teacher target. scripts/calibrate_teacher.py consumes an explicitly unsealed teacher packet. Calibration is a qualification aid; it does not establish final architecture efficacy. No calibration or training was performed in FMA-QUAL-001.

## Parameter budget and controls

For rank 32 over all 28 layers:

- shared U: 151,936 * 32 = 4,861,952 parameters
- layer projections: 28 * 32 * 1,024 = 917,504
- head-wise RMSNorm scales: 28 * 128 = 3,584
- M1 added total: 5,783,040 parameters (0.9702% of the 596,049,920 serialized baseline parameters)
- BF16 storage for added M1 parameters: 11,566,080 bytes
- standard W_V parameters structurally bypassed at g=1: 29,360,128
- if bypassed W_V tensors were physically removed after qualification, total parameters would be 572,472,832

A literal full MA table at every layer requires L * (N*d_v + d_h) = 4,356,312,576 parameters, or 8,712,625,152 BF16 bytes (8.114 GiB) for the MA tables/norm scales alone. It must not be instantiated merely to demonstrate that it exceeds the practical 8 GB GPU budget.

Arms are frozen as:

- M0 — unmodified pretrained baseline, zero added parameters.
- M1 — factorized Memory Attention, rank 32, 5,783,040 added parameters.
- M2 — conventional residual low-rank value adapter, uniform rank 101, 5,791,744 added parameters. Mismatch is +8,704 parameters (+0.1505% relative to M1).

M2 is mandatory because the primary paper explicitly notes that its reported quality results combine the architectural change with additional memory capacity and do not isolate structure from the parameter increase.

## FMA-QUAL-001 checklist

1. g=0 is numerically equivalent to the untouched pretrained baseline. Native Qwen3 g=0 delegates to original attention forward; exact-checkpoint execution remains an exact-host gate.
2. g=1 structurally bypasses the old value projection.
3. Token lookup uses exact input vocabulary IDs.
4. U, P_l, head counts, value width, and layer count derive from model config.
5. Added parameter count is exact and receipt-bound.
6. Changing rank changes parameter count by the analytical formula.
7. Normalization is independent per token/KV head with intended affine sharing.
8. Existing attention masks are not rewritten.
9. Causal behavior is preserved; no future-token leakage in the deterministic toy harness.
10. Generation-cache continuation matches full-prefix execution in the deterministic toy harness; native cache qualification remains an exact-host gate.
11. FMA state serializes and reloads exactly.
12. Constructing the experiment does not mutate the baseline module/checkpoint.
13. Frozen-backbone calibration freezes the backbone and leaves FMA parameters trainable.
14. Teacher residual is hand-verifiable on a tiny tensor.
15. Factorized lookup matches an explicitly materialized dense toy factorization.
16. Dtype/device transfer is supported.
17. Qualification outputs are finite.
18. Deterministic micro-generation smoke executes at g=0 and g=1.
19. Parameter/storage resource probe is produced; RTX 4070 VRAM execution is BLOCKED on a CPU-only agent.
20. No sealed/confirmation data are read by qualification or calibration tooling.

Synthetic/mechanical tests are necessary but not sufficient. Exact native Qwen3 checkpoint execution and RTX 4070 resource measurement must be recorded before FMA-QUAL-001 can be called fully qualified.

## Future efficacy gate

Before final/confirmation evaluation, hard-memory M1 (g_l=1 for every selected layer) must satisfy the preregistered initial noninferiority gate against M0:

- perplexity degradation <= 2%
- downstream aggregate degradation <= 1.0 percentage point

Any improvement must also be compared with M2. These thresholds are frozen here and must not be tuned on the final/confirmation set. Rank, modified layers, gate schedule, calibration recipe, and matched-control choice must be selected without reading sealed or confirmation data.

## Resource and claim policy

The practical target is an RTX 4070 8 GB host. Full layer-specific N x d_v memory tables are costed analytically and are not materialized. Host-RAM or VRAM guards must not be weakened; a resource failure is BLOCKED.

A successful mechanical qualification does not mean Memory Attention improves Rineya. Architecture efficacy requires later preregistered M0/M1/M2 measurements with hard-memory g=1, retention/capability evaluation, and the noninferiority gate above.
