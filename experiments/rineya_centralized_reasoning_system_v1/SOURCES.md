# Sources, provenance, and claim boundaries

Recorded 2026-10-01. The founding branch refresh checked the primary arXiv
metadata/abstract pages and the repository files identified below. It did not
reproduce the papers, download their datasets, inspect every supplement, audit
all sibling branches, or execute any Rineya model. The earlier conversation's
paper review is design context, not a new execution receipt.

## Primary literature

| ID | Source | Bounded relevance |
| --- | --- | --- |
| P1 | [AI-Newton, arXiv:2504.01538v2](https://arxiv.org/abs/2504.01538v2), revised 2025-12-11 | Concept-driven discovery from mechanics data and progressive generalization motivate an explicit, testable procedure library. Not evidence of Rineya or long-context language-model performance. |
| P2 | [Recursive Language Models, arXiv:2512.24601v3](https://arxiv.org/abs/2512.24601v3), revised 2026-05-11 | Treating a long input as an external environment motivates bounded reads and subcalls. The reported trained model is Qwen3-8B; transfer to a roughly 0.6B Rineya checkpoint remains untested. |
| P3 | [MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent, arXiv:2507.02259v2](https://arxiv.org/abs/2507.02259v2), revised 2026-07-29 | Segment-wise reading and learned memory updates motivate studying workflow training rather than assuming a wrapper works automatically. This is the 2507.02259 paper, not another similarly named MemAgent. |
| B1 | [RULER, arXiv:2404.06654](https://arxiv.org/abs/2404.06654) | Candidate controlled tests of retrieval, multi-hop tracing, and aggregation. |
| B2 | [BABILong, arXiv:2406.10149](https://arxiv.org/abs/2406.10149) | Candidate distributed-fact long-context reasoning evaluation. |
| B3 | [Oolong, arXiv:2511.02817](https://arxiv.org/abs/2511.02817) | Candidate long-context reasoning and aggregation evaluation. |
| B4 | [LongBench v2, arXiv:2412.15204](https://arxiv.org/abs/2412.15204) | Candidate realistic long-context understanding and reasoning evaluation. |

Benchmark references identify prospective evaluations only. Pin implementation,
dataset, license, prompt, metric, and tokenization versions before a run. No
paper's score, hardware configuration, or claimed extrapolation is a forecast
for Rineya. The proposed runtime is inspired by P1-P3, not a faithful reproduction
of any one of them. No third-party source code is copied in this package.

## Repository records inspected at the parent snapshot

Repository: `LUNCY579/ChatBubble`  
Parent: `research/post-training-mimo-centralized-v1`  
Commit: `19551ddebc87c93161c303865ce6d0f130de0093`  
Tree: `d2ed2805e79d226639bb75d951e79d15a944bf25`

- [Central program README](https://github.com/LUNCY579/ChatBubble/blob/19551ddebc87c93161c303865ce6d0f130de0093/experiments/post_training_mimo_centralized_v1/README.md): blob `00caf6abb79af854553fb46453490fee7dbbcac1`; subject, short mechanics, capture failure, and unchanged capability status.
- [Central next-experiment record](https://github.com/LUNCY579/ChatBubble/blob/19551ddebc87c93161c303865ce6d0f130de0093/experiments/post_training_mimo_centralized_v1/NEXT_EXPERIMENT.md): blob `83f3b49b084ae24c957ba7ebf4f304c2ec57a632`; later backend amendments and remaining qualification/resource boundaries.
- [Supa long-context README](https://github.com/LUNCY579/ChatBubble/blob/19551ddebc87c93161c303865ce6d0f130de0093/experiments/supa_long_context_minimal_intervention_v1/README.md): blob `882b65ec500af7a6e343b8da9136f4c9fdd9fc0f`; first 100 lines inspected, including its correction notice and diagnostic framing, not a complete re-audit of results.
- [Root README](https://github.com/LUNCY579/ChatBubble/blob/19551ddebc87c93161c303865ce6d0f130de0093/README.md): blob `7ece84db4f8a715a98d9a546b3c393aa5cfe732b`; existing model-quality authority pointer is preserved.

At creation, the observed `main` ref was
`b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2`. No update to that ref or to the
central parent is part of this delivery. Refresh refs before future work.

## Evidence policy

Published claims, repository-reported measurements, design proposals, local
fixture checks, and actual native results are different evidence categories.
Cite the corresponding exact source; do not pool them into a stronger claim.
Only independently recorded future experiments can establish CRS-001 efficacy.
