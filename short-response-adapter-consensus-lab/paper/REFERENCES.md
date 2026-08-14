# Verified Reference Ledger

## Verification policy

Metadata and technical relationships below were checked against primary papers, official proceedings, or author-submitted arXiv records on 2026-08-12. “Relationship” records only the claim supported by that source and the distinction needed for this study. Discovery pages, blogs, and social posts are not evidence. The cite keys correspond exactly to `references.bib`.

## Parameter-efficient adaptation

| Key | Work and primary record | Identifier / status | Claim supported and relationship | Verification |
|---|---|---|---|---|
| `hu2022lora` | Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models” | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685); ICLR 2022 | Freezes base weights and trains low-rank update factors. Upstream adapter parameterization, not inference composition. | PRIMARY VERIFIED |
| `zhang2023adalora` | Zhang et al., “AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning” | [arXiv:2303.10512](https://arxiv.org/abs/2303.10512); ICLR 2023 | Allocates rank adaptively during training. Not a multi-adapter decoder. | PRIMARY VERIFIED |
| `liu2024dora` | Liu et al., “DoRA: Weight-Decomposed Low-Rank Adaptation” | [PMLR 235:32100–32121](https://proceedings.mlr.press/v235/liu24bn.html); ICML 2024 | Decomposes magnitude/direction and applies LoRA to direction. Parameterization method, not voting. | PRIMARY VERIFIED |
| `meng2024pissa` | Meng, Wang, and Zhang, “PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models” | [arXiv:2404.02948](https://arxiv.org/abs/2404.02948); NeurIPS 2024 | Uses principal singular components to initialize low-rank adaptation. Not adapter composition. | PRIMARY VERIFIED |

## Adapter composition, routing, and expert ensembles

| Key | Work and primary record | Identifier / status | Claim supported and relationship | Verification |
|---|---|---|---|---|
| `pfeiffer2021adapterfusion` | Pfeiffer et al., “AdapterFusion: Non-Destructive Task Composition for Transfer Learning” | [doi:10.18653/v1/2021.eacl-main.39](https://aclanthology.org/2021.eacl-main.39/); EACL 2021 | Learns a composition stage over task adapters and mixes representations across layers. | PRIMARY VERIFIED |
| `huang2024lorahub` | Huang et al., “LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition” | [arXiv:2307.13269](https://arxiv.org/abs/2307.13269); COLM 2024 | Gradient-free search/composition of LoRA modules for a target task. Not equal token voting. | PRIMARY VERIFIED |
| `buehler2024xlora` | Buehler and Buehler, “X-LoRA: Mixture of Low-Rank Adapter Experts…” | [arXiv:2402.07148](https://arxiv.org/abs/2402.07148) | Learns hidden-state-dependent, layer-wise, token-level adapter scaling. Continuous learned gating, not independent top-1 plurality. | PRIMARY VERIFIED |
| `wang2024loraflow` | Wang et al., “LoRA-Flow: Dynamic LoRA Fusion for Large Language Models in Generative Tasks” | [arXiv:2402.11455](https://arxiv.org/abs/2402.11455); work in progress | Learns dynamic layer- and token-level fusion weights. | PRIMARY VERIFIED |
| `kong2024loraswitch` | Kong et al., “LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design” | [arXiv:2405.17741](https://arxiv.org/abs/2405.17741) | Uses token-wise Top-2 routing and in-place adapter merging for dynamic serving. | PRIMARY VERIFIED |
| `belofsky2023token` | Belofsky, “Token-Level Adaptation of LoRA Adapters for Downstream Task Generalization” | [arXiv:2311.10847](https://arxiv.org/abs/2311.10847); preprint | Explores gradient-free token-level weighted combinations of LoRA adapters. | PRIMARY VERIFIED |
| `li2025elrea` | Li et al., “Ensembles of Low-Rank Expert Adapters” | [OpenReview l0gZS0sAlf](https://openreview.net/forum?id=l0gZS0sAlf); ICLR 2025 | Closest neighbor: gradient-clustered experts, input-gradient relevance routing, weighted expert preactivation logits, one shared selected token/prefix per step. Establishes that shared-prefix token-level LoRA expert ensembling is prior art. | PRIMARY VERIFIED |

## Ensemble decoding, voting, and rank aggregation

| Key | Work and primary record | Identifier / status | Claim supported and relationship | Verification |
|---|---|---|---|---|
| `lakshminarayanan2017deep` | Lakshminarayanan, Pritzel, and Blundell, “Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles” | [NeurIPS 2017 proceedings](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Independent models improve predictive uncertainty and performance; general ensemble foundation. | PRIMARY VERIFIED |
| `liu2021dexperts` | Liu et al., “DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts” | [doi:10.18653/v1/2021.acl-long.522](https://aclanthology.org/2021.acl-long.522/); ACL-IJCNLP 2021 | Combines base, expert, and anti-expert log probabilities during autoregressive decoding. Continuous score arithmetic, not hard vote. | PRIMARY VERIFIED |
| `gu2024chared` | Gu et al., “CharED: Character-wise Ensemble Decoding for Large Language Models” | [arXiv:2407.11009](https://arxiv.org/abs/2407.11009) | Aligns heterogeneous models in character space and averages marginal distributions. | PRIMARY VERIFIED |
| `phan2025exact` | Phan et al., “Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles” | [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5ed5c3c846f684a54975ad7a2525199f-Abstract-Conference.html) | Maps token distributions to exact next-byte probabilities, enabling cross-vocabulary probability ensembles. | PRIMARY VERIFIED |
| `wang2023selfconsistency` | Wang et al., “Self-Consistency Improves Chain of Thought Reasoning in Language Models” | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171); ICLR 2023 | Samples reasoning paths and votes over final answers. Response-level stochastic voting, not synchronous expert-token voting. | PRIMARY VERIFIED |
| `cormack2009rrf` | Cormack, Clarke, and Büttcher, “Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods” | [doi:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114); SIGIR 2009 | Provides a score-scale-robust ranked-list aggregation rule used here as a comparator. | PRIMARY VERIFIED |
| `durrant2020diversity` | Durrant and Lim, “A Diversity-aware Model for Majority Vote Ensemble Accuracy” | [PMLR 108:4078–4087](https://proceedings.mlr.press/v108/durrant20a.html); AISTATS 2020 | Formalizes dependence of majority-vote accuracy on competence and diversity; motivates coalition/dependence analysis. | PRIMARY VERIFIED |

## Calibration, conditional compute, and merging

| Key | Work and primary record | Identifier / status | Claim supported and relationship | Verification |
|---|---|---|---|---|
| `guo2017calibration` | Guo et al., “On Calibration of Modern Neural Networks” | [PMLR 70:1321–1330](https://proceedings.mlr.press/v70/guo17a.html); ICML 2017 | Establishes temperature scaling as a simple post-hoc calibration baseline. | PRIMARY VERIFIED |
| `roelofs2022calibration` | Roelofs et al., “Mitigating Bias in Calibration Error Estimation” | [PMLR 151:4036–4054](https://proceedings.mlr.press/v151/roelofs22a.html); AISTATS 2022 | Shows bias issues in calibration-error estimation and motivates reporting beyond naive ECE. | PRIMARY VERIFIED |
| `fedus2022switch` | Fedus, Zoph, and Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity” | [JMLR 23(120)](https://www.jmlr.org/papers/v23/21-0998.html), 2022 | Learned sparse routing inside a jointly trained MoE; not a post-hoc committee of adapters. | PRIMARY VERIFIED |
| `schuster2022calm` | Schuster et al., “Confident Adaptive Language Modeling” | [arXiv:2207.07061](https://arxiv.org/abs/2207.07061); NeurIPS 2022 | Adjusts computation depth per token with confidence-based early exits; adjacent adaptive-compute work. | PRIMARY VERIFIED |
| `wortsman2022soups` | Wortsman et al., “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time” | [PMLR 162:23965–23998](https://proceedings.mlr.press/v162/wortsman22a.html); ICML 2022 | Averages weights to create one predictor before inference. | PRIMARY VERIFIED |
| `ilharco2023task` | Ilharco et al., “Editing Models with Task Arithmetic” | [arXiv:2212.04089](https://arxiv.org/abs/2212.04089); ICLR 2023 | Performs arithmetic on task vectors in weight space. | PRIMARY VERIFIED |
| `yadav2023ties` | Yadav et al., “TIES-Merging: Resolving Interference When Merging Models” | [arXiv:2306.01708](https://arxiv.org/abs/2306.01708); NeurIPS 2023 | Trims and resolves sign conflicts before parameter merging; one merged predictor, not decision voting. | PRIMARY VERIFIED |

## Search limitation

The ledger supports a narrow “not found in this documented search” statement only. It does not license an exhaustive novelty claim. ELREA must be cited wherever shared-prefix token-level adapter ensembling is discussed, and the paper must frame the present contribution as an aggregation-rule and mechanism-analysis study.
