# Related-work scope and novelty boundary

This file is a living audit, not a novelty declaration. It records the nearest mechanism families that must be compared before any arXiv claim is written. Bibliographic details must be rechecked against primary sources at manuscript freeze.

## Cascaded and interacting heads

### EfficientViT / Cascaded Group Attention

Liu et al., **“EfficientViT: Memory Efficient Vision Transformer With Cascaded Group Attention,” CVPR 2023**.

Relevant overlap:

- splits the input features among heads;
- feeds information from a previous head into the next;
- motivates cascades partly through redundancy among parallel heads.

Difference to test, not merely assert:

- vision rather than causal language modeling;
- previous-head feature propagation rather than an explicit finite-difference carrier trajectory;
- no central predictor–corrector formulation in which velocity controls Q/K/V and token significance.

This is the mandatory cascade baseline and architectural ancestor.

### Cascaded Head-colliding Attention (CODA)

Zheng, Wu, and Kong, **“Cascaded Head-colliding Attention,” arXiv:2105.14850, 2021**.

Relevant overlap:

- an NLP attention architecture explicitly designed to model interactions among heads;
- hierarchical/cascaded head dependence;
- language-model and translation experiments.

Difference to audit carefully:

- CODA formulates head interaction through a hierarchical variational distribution;
- TGSA’s candidate contribution is an explicit shared-state finite difference used as a learned computational controller.

Because CODA is in NLP, it is a more direct prior-art constraint than vision-only cascading and must be represented fairly in the paper.

### Talking-Heads Attention

Shazeer et al., **“Talking-Heads Attention,” arXiv:2003.02436, 2020**.

Relevant overlap: learned mixing across head dimensions. Difference: cross-head mixing does not inherently serialize heads into a state trajectory.

## Momentum and dynamical-system interpretations

### Momentum Residual Neural Networks

Sander et al., **“Momentum Residual Neural Networks,” ICML 2021**.

Relevant overlap:

- explicit momentum-like state;
- second-order dynamical-system interpretation;
- stronger representational analysis than a plain first-order residual update.

Difference: network-depth dynamics rather than finite differences across sequential attention heads, and no token-level attention-control mechanism.

### Momentum Transformer

Nguyen et al., **“Momentum Transformer: Closing the Performance Gap Between Self-attention and Its Linearization,” MSML/PMLR 2022**.

Relevant overlap:

- momentum inside a Transformer-family architecture;
- adaptive momentum;
- autoregressive experiments.

Difference: momentum is introduced in the optimization-inspired linear-attention/residual computation, not as a shared carrier velocity between cascaded heads. The manuscript must never claim that TGSA is the first Transformer to use momentum.

## Difference-based attention

### Differential Transformer

Ye et al., **“Differential Transformer,” arXiv:2410.05258, 2024**.

Relevant overlap:

- subtraction/difference used to improve attention selectivity;
- long-context and noise-cancellation motivation.

Difference:

- Differential Transformer subtracts two simultaneous softmax attention maps;
- TGSA estimates change across sequential computational depth and uses it to control later computation.

The terms “differential” and “derivative” must therefore be defined precisely to prevent conceptual conflation.

## Dynamic token/head reduction

### SpAtten

Wang, Zhang, and Han, **“SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning,” arXiv:2012.09852, 2020**.

Relevant overlap:

- dynamically scores/prunes tokens and heads;
- cascade-style progressive reduction;
- explicitly addresses hardware cost.

Difference: pruning/accelerator co-design rather than trajectory-conditioned iterative reasoning. SpAtten is nevertheless a required efficiency baseline once TGSA introduces hard compression.

### Differentiable Subset Pruning of Transformer Heads

Li, Cotterell, and Sachan, **arXiv:2108.04657, 2021**.

Relevant overlap: differentiable importance variables and constrained head selection. Difference: selects heads rather than ordering them into a trajectory-guided computation.

## Native LLM attention compatibility

### Grouped-Query Attention

Ainslie et al., **“GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,” arXiv:2305.13245, 2023**.

GQA is not sequential attention, but it defines the K/V-sharing and checkpoint-uptraining constraints required for a credible Qwen-compatible implementation.

## Candidate contribution boundary

The defensible research question is not:

> Are these the first sequential or interacting attention heads?

The defensible question is:

> Does an explicit finite-difference trajectory in a shared carrier space provide useful control information beyond ordinary cascading, when it nonlinearly conditions prediction, Q/K/V construction, token significance, and correction inside a causal language-model attention block?

A possible contribution exists only if all of the following survive controlled experiments:

1. sequential cascading alone is insufficient to explain the gain;
2. a raw linear difference is insufficient to explain the gain;
3. learned trajectory control improves a preregistered quality/compute endpoint;
4. the result replicates across seeds and at least two task families;
5. gains survive parameter-, FLOP-, and latency-aware controls;
6. short-context language modeling and ordinary conversation do not regress beyond the registered guardrails.

## Final literature-search protocol

Before manuscript freeze:

1. search arXiv, Semantic Scholar, OpenReview, ACL Anthology, IEEE/CVF, PMLR, and citation graphs using combinations of “sequential heads,” “cascaded heads,” “inter-head recurrence,” “momentum attention,” “trajectory attention,” “predictor-corrector Transformer,” “head-conditioned attention,” and “adaptive head order”;
2. inspect references and citing papers for every nearest work above;
3. compare equations, not only titles and abstracts;
4. record search date, databases, exact queries, and inclusion decisions;
5. replace every “novel/first” statement with a narrower claim unless the audit supports it;
6. invite at least one independent reader to challenge the novelty table.
