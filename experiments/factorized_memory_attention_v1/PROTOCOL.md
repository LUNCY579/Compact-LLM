# FMA-QUAL-001 — Factorized Memory Attention for compact Rineya

Status: mechanical qualification harness complete; native checkpoint qualification is blocked by the current runtime because the checkpoint weights and CUDA/RTX 4070 host are not present.

This experiment is deliberately a qualification instrument, not an efficacy claim. A passing structural test, a valid branch, or a successful synthetic generation run is not evidence that Memory Attention improves Rineya.

## Scope and provenance

- Repository: `LUNCY579/ChatBubble`
- Base: `origin/main` at `7d9b4fd450bc56405f6cec41459a7e83df7958ae`
- Target branch: `research/factorized-memory-attention-v1`
- Registered model/config: `Rineya-Supa-1.7.0-devfreeze`, `Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/config.json`
- Primary paper: Jiale Kang, “Memory Attention,” arXiv:2609.28399v1, submitted 2026-09-23, <https://arxiv.org/abs/2609.28399>

The named historical branches were inspected read-only for mechanical patterns only. Their tests, conversion infrastructure, and prior receipts are not model-efficacy evidence and are not used as the branch base.

## Exact paper equations used

The paper defines standard self-attention as

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.\tag{1}
\]

For one head, the unchanged attention operator is

\[
\operatorname{Attn}(X)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+C\right)V,\tag{2}
\]

where `C` is the causal mask. Memory Attention removes the independent `W_V` path. For token IDs `s`, it defines a layer-specific table `E` and retrieves

\[
M=\operatorname{Norm}(E[s])\in\mathbb{R}^{T\times d_v},\tag{7}
\]

where normalization is RMSNorm independently for every token and every key/value head. The value construction is

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=K+M.\tag{8}
\]

With RoPE, the paper applies rotation to the query and scoring key but constructs values from the pre-RoPE key:

\[
Q^R=\operatorname{RoPE}(Q),\qquad K^R=\operatorname{RoPE}(K),\qquad V=K+M.\tag{9}
\]

At inference, normalization may be folded into the table:

\[
\overline{E}[i]=\operatorname{Norm}(E[i]),\qquad V=K+\overline{E}[s].\tag{16--17}
\]

The paper's table accounting compares `L(N d_v + p_norm)` MA parameters with `L d d_v` standard value parameters and explicitly notes that the reported quality gains include additional memory capacity. This experiment therefore requires a matched-capacity control.

## Rineya factorization

The compact variant does not allocate a dense `N × d_v` table. It uses one shared token factor table and one projection per modified layer:

\[
U\in\mathbb{R}^{N\times r},\qquad P_l\in\mathbb{R}^{r\times d_v},
\]

\[
M_l(s)=\operatorname{HeadRMSNorm}(U[s]P_l),
\]

with the default qualification rank `r=32`. The token table is shared across layers; the projections and normalization scales are layer-specific. `N`, `d_v`, `d_h`, head counts, and layer count are loaded from the registered repository config, never hardcoded in the model component.

Each modified layer exposes a transition gate:

\[
V_l=(1-g_l)(XW_{V,l})+g_l(K_l+M_l(s)).
\]

`g=0` is an exact untouched-value-path short circuit. `g=1` is a structural short circuit that never calls `W_V`; the old projection may remain resident in the transitional retrofit so that the `g=0` control is testable, but it is not executed by the hard-memory path. The budget reports both transitional residency and the hard-replacement count.

## Calibration option

Calibration is optional and was not run in this qualification. When used, the backbone is frozen before any optimizer is created, only factorized-memory parameters are trainable, and the teacher target is the hand-verifiable residual

\[
R_l=XW_{V,l}-K_l,
\]

with `K_l` taken before RoPE and after the native Qwen3 key normalization. The loss compares `HeadRMSNorm(U[s]P_l)` with `R_l`. Calibration data must be identified by immutable local data IDs; no sealed or confirmation data may be read during calibration, rank selection, layer selection, gate scheduling, or control selection.

## Arms and preregistration

- `M0`: unmodified pretrained baseline.
- `M1`: factorized Memory Attention, rank 32, shared `U`, layer-specific `P_l`, hard decision gate `g_l=1`.
- `M2`: conventional residual low-rank value adapter with the nearest uniform rank to the M1 added-parameter budget.

The future noninferiority gate is preregistered before efficacy work: the hard-memory version must have perplexity degradation no worse than 2% and downstream aggregate degradation no worse than 1 percentage point versus M0, while remaining within the compact parameter budget. Any improvement must also beat or match M2. These thresholds are not tuned on a confirmation/sealed set and are not evaluated by FMA-QUAL-001.

## Qualification matrix

FMA-QUAL-001 covers the following mechanical checks:

1. `g=0` numerical equivalence to the untouched value path.
2. `g=1` structural bypass of `W_V`.
3. Correct token-ID lookup.
4. Config-derived `U` and `P_l` shapes.
5. Exact parameter count and receipt binding.
6. Rank-to-parameter scaling.
7. Head-wise RMSNorm semantics.
8. Unchanged mask path.
9. Causal no-future leakage.
10. Cache continuation parity.
11. Serialization/reload.
12. Baseline non-mutation during construction.
13. Exact frozen-backbone calibration contract.
14. Hand-verifiable teacher residual.
15. Dense-vs-factorized toy equivalence.
16. Dtype/device transfer.
17. No NaN/Inf.
18. Synthetic micro-generation.
19. Guarded memory/resource probe.
20. No sealed-data access.

Native Qwen3 tests are optional and skip when the exact Transformers stack is unavailable. The resource probe refuses to instantiate a literal full memory table when its element count exceeds the small-test guard; the actual full-table cost is calculated only.

## Evidence boundary

The qualification receipt classifies results as `mechanical_qualification_only`. No language-model perplexity, downstream aggregate, calibration efficacy, or architecture-improvement claim is made by this branch. A future efficacy run must bind its own data IDs, checkpoint, seed, package versions, resources, and M0/M1/M2 results before scoring.
