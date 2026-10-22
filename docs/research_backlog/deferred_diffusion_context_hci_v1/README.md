# Deferred Rineya research notes — diffusion systems and human–AI interaction

**Status: DEFERRED_DOCS_ONLY. Priority: BACKLOG, outside the active RSD/CLARE program.**

Recorded: 2026-09-28. Repository: `LUNCY579/ChatBubble`.

Branch: `research/deferred-diffusion-context-hci-notes-v1`.

Base: `main` at `b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2`.

## Scope and source counting

The user's five screenshots contain four distinct papers: recursive self-distillation, Cognitive Divergence, block-parallel diffusion training, and CLARE. CLARE appears in two screenshots. After reserving recursive self-distillation and CLARE for the higher-priority discussion, there are TWO remaining papers, not three.

For a useful three-topic record, the diffusion paper is explicitly split into its BP and CSBP contributions. This editorial split does not create an additional paper or imply independent experimental evidence for three publications.

| Record | Topic | Primary source | Current disposition |
|---|---|---|---|
| [BP-FUT-001](BLOCK_PARALLELISM.md) | Block-owned distributed training | P1 | Deferred; no Rineya implementation added |
| [CSBP-FUT-001](CONTEXT_SHARDED_BP.md) | Shared-context sharding for the BP workload | P1, same paper | Deferred; depends on a qualified workload |
| [HCI-FUT-001](COGNITIVE_DIVERGENCE.md) | Effective context evaluation and optional human–AI delegation study | P2 | Deferred; two clearly separated proposed evaluation tracks |

Each note records its mechanism or hypothesis, possible repository connection, proposed experiment, confounds, cost/quality endpoints, and criteria for reconsideration. All integration designs are proposals, not measured results.

## Priority and execution boundary

Continue the established RSD and canonical CLARE research independently of this archive. This branch does not add treatments to either experiment, amend their protocols, or enroll itself in centralized E0/E1. The branch exists to preserve ideas without spending their future execution budget now.

This change adds Markdown only. It does not install dependencies, import upstream code, download weights, author or inspect protected evaluation datasets, train models, allocate GPUs, provision cloud resources, recruit participants, collect telemetry, or schedule jobs. There is no executable placeholder, pretend trainer, or synthetic-success receipt. A document's presence is not evidence of native qualification or a model improvement.

No runtime test suite was run for these notes. Publication integrity is checked separately by inspecting the remote files and the branch diff; that check is not an ML experiment.

## Source and repository reconnaissance

Primary sources inspected for this record:

- **P1:** Suresh et al., *Block Parallelism for Efficient Distributed Long-Context Diffusion Language Model Training*, [arXiv:2609.19242v1](https://arxiv.org/html/2609.19242v1). Method definitions: Sections 3–4; workload-specific evidence and limitations: Sections 5–6. No advertised speedup is a Rineya forecast.
- **P2:** Eliav, *The Cognitive Divergence: AI Context Windows, Human Attention Decline, and the Delegation Feedback Loop*, [arXiv:2603.26707v1](https://arxiv.org/html/2603.26707v1). Hypothesis, limitations, and research agenda: Sections 8, 10–11. The proposed Rineya evaluations are not validated by this paper.
- **Candidate upstream code:** [Turbo-dLLM](https://github.com/ScalingIntelligence/Turbo-dLLM). README inspected; source implementation not audited or executed, no dependency pin or compatibility guarantee established. Recheck the actual revision, licenses/notices, and requirements before integration.

Remote branch references were read before branch creation. The main root and docs tree were inspected; these notes use a new directory and leave inherited files unchanged. Candidate integration references were read in the centralized [METHOD_INVENTORY.csv at 5e5fe36](https://github.com/LUNCY579/ChatBubble/blob/5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80/experiments/post_training_mimo_centralized_v1/METHOD_INVENTORY.csv): PROPOSED_04 (Discrete Diffusion), FMA, and NSA. This was a targeted documentation audit, not a repository-wide code review.

That centralized snapshot is a reference, not this branch's parent. Links into it deliberately use immutable commits. Mentioning an experiment there does not mean its implementation is copied into this branch or qualified on the current model/runtime. Resource readings from other experiments are historical, not live measurements.

## Manual reactivation procedure

1. Select one record and state the concrete reason to revisit it. A new paper, a real measured bottleneck, or an actual product-evaluation need can motivate review; none automatically starts work.
2. Refresh relevant sources and repository interfaces. Bind an exact model/tokenizer/runtime for model work, or define the product and participant scope for human work. Preserve prior evidence labels.
3. Write a separately reviewed execution proposal with an owner, explicit data/resource budget, baseline, split rules, primary endpoint, practical effect threshold, stopping conditions, and privacy safeguards where applicable.
4. Create a NEW implementation branch from the then-appropriate qualified baseline. Reference this archive; do not treat its old main parent as an up-to-date execution environment. Keep source data, noisy pilots, and independent confirmation distinct.
5. Start only after explicit authorization. Publish both positive and negative outcomes with total cost and scope. Keep these archival notes unchanged except for clearly dated corrections or links to authorized follow-ups.

There is intentionally no launch command, promised start date, automatic activation trigger, or scheduled review in this branch.
