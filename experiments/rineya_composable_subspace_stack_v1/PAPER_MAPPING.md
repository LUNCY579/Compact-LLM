# Paper mapping and experimental interpretation

Verified against primary paper abstract/HTML pages on 2026-09-27. These links document
provenance, not a claim that the five-paper combination has been validated.

| Paper | Primary source | Implemented here |
|---|---|---|
| Shared LoRA Subspaces for almost Strict Continual Learning (Share) | https://arxiv.org/abs/2602.06043 | **Inspired adaptation:** average-projector spectral consensus, shared/private decomposition and frozen residual gain. No complete sequential basis-evolution reproduction. |
| OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting | https://arxiv.org/abs/2510.13003 | Double-sided complement projection of existing effective updates. **Post-hoc adaptation**, not OPLoRA's complete constrained training experiment. |
| Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter | https://arxiv.org/abs/2606.03723 | Norm-adjusted basis fitting, HOSVD initialization and bounded HOOI; original-delta coordinate merging. Dense bounded-matrix implementation, not low-rank-factor acceleration. |
| Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic | https://arxiv.org/abs/2605.18993 | Functional JVP teacher and nonlinear hidden-state distillation with a squared Taylor-residual curvature surrogate. **Inspired**, not a claim of reproducing every published objective/setting. |
| Effective LoRA Adapter Routing using Task Representations (LORAUTER) | https://arxiv.org/abs/2601.21795 | **Inspired baseline:** CAL task prototypes, top-k softmax, ambiguity/base fallback and request-local additive skill weights. Not the full published task-representation pipeline. |

## Matrix definitions

For W = U Sigma V^T, protect k directions with
D_safe = (I - U_k U_k^T) D (I - V_k V_k^T).
D_overlap = D - D_safe is a geometric remainder, not a diagnosis of harmful knowledge.
Projection products are evaluated without allocating full square projector matrices.

For each donor, take nonzero leading singular directions. Concatenating their left/right bases
and thresholding the singular values of that concatenation implements spectral agreement of
average projectors. If L/R are the resulting consensus bases,
D_shared = L (L^T D R) R^T and D_private = D - D_shared.
S applies D_shared + private_gain * D_private. The exact unit-gain identity is a required test.

CtM basis targets use [beta * ||D_i|| + (1-beta) * mean_j ||D_j||] * D_i / ||D_i||.
Zero deltas remain zero. HOSVD initializes L/R and alternating HOOI improves captured energy.
The final update is L [sum_i alpha_i * (L^T D_i R)] R^T, using original deltas.
A fixed L/R commutes with averaging; only different bases, truncation, gains or training create
a distinct intervention. S and C are consequently independently switchable and tested.

## Controls and inference

The existing parent commit `816a1e8f4d1ab1c9d1ab847697d962a141521bac` records an OSIDE-001
null comparison of module routing against scalar attenuation, explicitly development-grade
on an exposed pool. Do not recast that result as established skill independence. CSS-001
keeps scalar doses, full endpoints, scoped endpoints and a norm-matched full-stack control.

The initial 2^3 factorial has eight unique cells. Leave-one-out contrasts reuse its cells.
The norm-matched arm restores the pre-intervention mixture norm **per selected matrix** and is
not a universal dose-matching guarantee. Report per-matrix changes and unchanged scope.

Selection uses only CAL, with the true frozen base as retention reference. The worst retention
domain must meet the registered point-estimate margin; capability gain must exceed the frozen
threshold. Selecting nothing is valid. CHECK accepts only the locked selected artifact; do not
refreeze after inspecting CHECK. Frozen subsets may omit arms, but missing arms cannot support
factorial or leave-one-out attribution. Full factorial contrasts require all eight cells.

Functional interaction is L(A+B)-L(A)-L(B)+L(base) on the same rows, runtime and component doses.
This is nonadditivity on the chosen loss scale, not automatically positive transfer. Matched
components are the scoped **dose** controls for an averaged mixture, not full-dose endpoints.
The implementation refuses misaligned rows, parents, domains, roles, dtypes and common metadata.

Parent-resampled intervals are descriptive and equal-parent weighted; raw row accuracies are
also reported. They are not multiplicity-adjusted, a noninferiority test or proof of no forgetting.
Supplied-ledger disjointness does not establish that omitted training ancestors are clean.

## Stage boundaries

Stage 1: frozen existing checkpoints, post-hoc geometry, one output at a time, CAL/CHECK.
Stage 2: independently train ordinary, tangent and tangent-distilled skills from the same base;
merge each family with identical settings. Ordinary/student optimizer steps are matched, but
teacher overhead makes total compute unequal. Report that cost rather than claiming iso-compute.
Native Qwen3 JVP, tokenizer/masking/EOS, losses and immutable-parameter behavior require exact-host
qualification. The library's generic APIs do not supply or authorize that qualification.
Routing: parallel deployment study, not stacked automatically on a static CtM artifact. CAL
features/task labels fit the router; deployment sees prompt-derived features, not answers.
Oracle headroom/regret is an offline diagnostic only. Live hook banks require serialized requests.

Recurrence, KV compression, RL curriculum and other experiments are intentionally out of scope.
