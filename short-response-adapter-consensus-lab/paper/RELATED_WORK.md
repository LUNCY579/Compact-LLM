# Related Work Analysis

## Search scope and conclusion

This review used primary papers, official proceedings, and author-submitted arXiv records available through 2026-08-12. The search covered parameter-efficient adaptation, adapter composition and routing, LoRA expert ensembles, autoregressive ensemble decoding, rank aggregation, conditional computation, model merging, and calibration. Search strings explicitly included variants of *token-level LoRA voting*, *hard vote adapter ensemble*, *shared-prefix adapter decoding*, *mixture of LoRA experts*, *LoRA-Switch*, and *Ensembles of Low-Rank Expert Adapters*. The bibliographic facts and claim-level relationships are recorded in `REFERENCES.md`; verified entries are in `references.bib`.

The closest work found is ELREA [@li2025elrea]. ELREA is already a token-by-token, shared-prefix ensemble of LoRA experts: it trains experts on gradient-clustered subsets, estimates input-dependent relevance using gradient similarity, combines expert preactivation logits with those weights, appends the resulting argmax token, and repeats. Accordingly, this study does **not** claim to invent LoRA ensembles, shared-prefix expert decoding, or token-level adapter composition.

The narrower distinction investigated here is a router-free committee of predefined specialists whose independently evaluated next-token **top-1 decisions** are combined by equal-vote plurality in discrete token space. The empirical contribution is the matched mechanism study: hard vote versus probability and calibrated-probability fusion; temperature and coalition interventions; exhaustive committee ablations; rescue/harm and first-divergence tracing; EOS/tie analysis; adaptive consultation; and cross-corpus confirmation. The primary-source search did not find a paper that jointly matches those design and analysis properties. This is a bounded search result, not proof that no such work exists.

## Parameter-efficient adaptation

LoRA represents weight updates with trainable low-rank factors while keeping the pretrained weights frozen [@hu2022lora]. AdaLoRA allocates rank adaptively under a parameter budget [@zhang2023adalora]; DoRA decomposes magnitude and direction and applies LoRA to directional updates [@liu2024dora]; PiSSA initializes low-rank factors from principal singular components [@meng2024pissa]. These papers motivate the compatible specialist artifacts used in this study, but they change parameterization, initialization, or training. The present work proposes none of those things. Its object of study is post-training inference-time decision composition among already trained, backbone-compatible adapters.

## Adapter composition and routing

AdapterFusion learns a separate composition layer over task adapters and mixes their intermediate representations [@pfeiffer2021adapterfusion]. LoraHub performs gradient-free composition of multiple LoRA modules for a target task [@huang2024lorahub]. X-LoRA learns hidden-state-dependent scaling at token and layer granularity [@buehler2024xlora], while LoRA-Flow learns dynamic layer-wise fusion weights [@wang2024loraflow]. LoRA-Switch uses token-wise routing and in-place adapter merging for efficient multi-task serving [@kong2024loraswitch]. Belofsky studies token-level, gradient-free weighted adaptation over LoRA adapters [@belofsky2023token]. These approaches mix parameters, representations, or continuous outputs through a learned or optimized router; they do not provide the same ordinal, equal-vote decision rule studied here.

ELREA requires the most precise comparison [@li2025elrea]. Both ELREA and the present method retain separate LoRA experts, reevaluate experts at every autoregressive step, choose one shared next token, and condition all experts on the resulting shared prefix. The differences are:

| Property | ELREA | Present study |
|---|---|---|
| Expert construction | Gradient-clustered training subsets | Predefined capability specialists from the frozen upstream study |
| Routing signal | Input-gradient similarity to training clusters | None for the full committee; equal expert votes |
| Combination space | Relevance-weighted preactivation logits | Discrete top-1 token identities |
| Scale sensitivity | Depends on continuous expert scores and routing weights | Top-1 proposals, vote counts, and unique-plurality winners are invariant; the deployed confidence tie-break is not invariant to unequal expert scaling |
| Main evidence | Predictive comparison to LoRA and ensemble baselines | Matched mechanism, rescue/harm, coalition, calibration, and divergence analyses |

Thus the contribution is not a broad architectural novelty claim over ELREA. It is a controlled study of a materially different aggregation rule and the specialist interactions that make that rule help or fail.

## Ensemble decoding and voting

Deep ensembles establish the general predictive value of independently trained models and uncertainty diversity [@lakshminarayanan2017deep]. DExperts combines an expert and anti-expert through log-probability arithmetic at each decoding step [@liu2021dexperts]. CharED aligns heterogeneous language models in character space and averages marginal distributions [@gu2024chared]. Exact byte-level ensembling maps tokenized models into a shared byte-level probability space, enabling probability aggregation across different vocabularies [@phan2025exact]. These are continuous score or probability combinations, whereas plurality deliberately discards score magnitudes after each expert's top-1 decision.

Self-consistency samples multiple reasoning paths from one model and votes over final answers [@wang2023selfconsistency]. That is response-level voting over stochastic trajectories, not synchronous next-token voting by independently specialized adapters sharing one prefix. Reciprocal rank fusion provides a score-scale-robust way to combine ranked lists [@cormack2009rrf], and is therefore included here as a rank-based comparator rather than treated as equivalent to plurality.

Majority-vote theory predicts that diversity and individual competence jointly determine ensemble accuracy; dependence can reverse the usual intuition that adding voters helps [@durrant2020diversity]. This motivates exhaustive coalition and diversity analyses, but classifier-level theory is not by itself evidence that autoregressive token voting will improve sequence accuracy. Prefix feedback, EOS decisions, and early divergence create dependencies absent from a one-shot classifier vote.

## Calibration and ordinal invariance

Temperature scaling is a standard post-hoc calibration method [@guo2017calibration]. Calibration error estimates such as binned ECE can themselves be biased and should not be treated as complete measures of probabilistic quality [@roelofs2022calibration]. In this study, calibrated-probability fusion is therefore evaluated with held-out DEV temperatures and multiple proper or discrimination-oriented diagnostics rather than ECE alone.

Positive temperature scaling preserves an expert's argmax, so top-1 proposals, vote counts, and unique-plurality winners are unchanged. That elementary fact is not claimed as a new theorem. The implemented rule resolves plurality ties with summed log probability, however, so unequal expert scaling can move the selected token at that boundary. The contribution is using this qualified contrast as a falsifiable mechanism probe: probability fusion can move at any position when confidence scales change, whereas hard plurality can move only at tied vote leaders under its deployed secondary rule.

## Conditional computation and adaptive consultation

Switch Transformers use learned sparse routing inside a jointly trained mixture-of-experts model [@fedus2022switch]. CALM adjusts computation depth per token using early exits [@schuster2022calm]. These works motivate conditional-compute comparisons, but their experts or layers are internal components of one trained model. The adaptive policy here begins with a frozen core committee and consults additional, separately trained adapters only when a DEV-frozen agreement rule triggers. It is an inference policy over existing specialists, not a new MoE training scheme.

## Weight-space merging

Model soups average model weights [@wortsman2022soups], task arithmetic manipulates task vectors [@ilharco2023task], and TIES-Merging trims, resolves sign conflicts, and merges parameters [@yadav2023ties]. Each produces one combined predictor before decoding. The committee studied here keeps adapters separate and combines their token decisions after independent forward passes. Weight merging is consequently an adjacent efficiency and composition family, not an interchangeable implementation of the same mechanism. No superiority claim over those families is made without a matched merging experiment.

## Supported gap statement

Prior work has established low-rank adaptation, learned adapter fusion, token- and layer-wise LoRA routing, gradient-routed LoRA expert ensembles, probability/logit ensemble decoding, rank aggregation, response-level voting, conditional computation, and weight-space merging. Within the primary-source search documented here, we did not find a directly matched study of router-free, equal-vote top-1 plurality among predefined specialist adapters under a shared autoregressive prefix **together with** matched soft-fusion calibration controls and specialist-level rescue, harm, coalition, and divergence analysis. The contribution is therefore a reproducible mechanism study of that narrow design—not a claim to invent adapter ensembling, majority voting, or shared-prefix token-level expert decoding.
