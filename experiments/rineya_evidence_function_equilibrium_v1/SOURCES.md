# Sources and evidence provenance

Checked September 29, 2026. This ledger distinguishes primary-source metadata,
repository documents actually read, and the supplied prior-agent analysis.

## Primary papers

**P1.** Wang et al., *Random Attention: Rethinking KV Cache Eviction for Efficient
Reasoning*, arXiv:2609.03430v2, revised September 28, 2026.
https://arxiv.org/abs/2609.03430v2
Version history and abstract were checked. The v2 HTML full text did not resolve
in this session; detailed passcode/table claims from the transcript were not
independently re-audited. No published throughput number is used as a Rineya target.
Official code identified by the paper: https://github.com/SalesforceAIResearch/Random-Attention
No upstream code was copied; EFE is not an exact reproduction of its eviction schedule.

**P2.** Csillag et al., *Functional Gradient Descent with Adaptive Representations*,
arXiv:2606.16926v1, June 15, 2026.
https://arxiv.org/html/2606.16926v1
Abstract and method/assumptions were inspected. The paper refines functional-gradient
representations to control approximation error; global convergence requires additional
assumptions. It is not a pretrained-LLM adaptation experiment. EFE's empirical
finite-output projection and coefficient fit do not inherit that theorem.

**P3.** Bai, Kolter, and Koltun, *Deep Equilibrium Models*, 2019.
https://arxiv.org/abs/1909.01377
The abstract describes fixed-point sequence models and implicit differentiation.
Memory independence is with respect to effective depth, not all system memory.

**P4.** Bai, Koltun, and Kolter, *Stabilizing Equilibrium Models by Jacobian
Regularization*, 2021.
https://arxiv.org/abs/2106.14342
The abstract identifies stability/architectural brittleness and Jacobian
regularization. EFE proposes it only as a later controlled option.

## Repository evidence inspected through the GitHub connector

**R1.** The live `post-training-Mimo` branch resolved to
`03db218ceebc61a2bc04f2482cf7e094045e01c1`, tree
`6a1e3b9b16789f7ce211e443654912859dd37ed0`. This is EFE's parent.
https://github.com/LUNCY579/ChatBubble/commit/03db218ceebc61a2bc04f2482cf7e094045e01c1

**R2.** `REHEARSAL_COMPOSITION.md` at R1; observed blob
`a06db2f2a8dd42cf3b838f0f3982a681f1f61b8e`.
https://github.com/LUNCY579/ChatBubble/blob/03db218ceebc61a2bc04f2482cf7e094045e01c1/REHEARSAL_COMPOSITION.md
Read for report-only/quarantine and cross-instrument retention boundaries.
A historical report is not an independently rerun experiment.

**R3.** `experiments/titan_tangent_arithmetic_v1/README.md` on
`research/titan-tangent-task-arithmetic-v1`; observed blob
`7fae26c83ab1394102d85afc9dad1206a3a60e98`.
https://github.com/LUNCY579/ChatBubble/blob/research/titan-tangent-task-arithmetic-v1/experiments/titan_tangent_arithmetic_v1/README.md
Read for interfaces and native qualification boundaries; implementation internals
were not copied or re-audited. The path was absent at R1.

**R4. Unresolved.** The named historical
`research/rineya-composable-subspace-stack-v1` ref returned no commit. EFE does not
assert that its infrastructure is currently available. Resolve a concrete source
commit before a future integration.

## Supplied context, not reverified native evidence

**U1.** The user's pasted prior-agent analysis is the source for INDEP-001 counts,
weight-delta cosine, atomic state-transition findings, the Supa matched-MLP result,
Opallo format/abstention interpretations, HLOC-001's indeterminate result, and the
prior recurrence/composition qualification history. These motivate the proposed
experiments. EFE did not rerun them, inspect checkpoint files, or audit their raw
row-level receipts. No U1 count is used as a fresh confirmation result.
