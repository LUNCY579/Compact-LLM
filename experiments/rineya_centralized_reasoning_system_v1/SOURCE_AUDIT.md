# CRS-001 primary-source audit

Recorded 2026-10-01, America/Chicago. Evidence category: **VERIFIED SOURCE and
STATIC CODE INSPECTION**, not reproduced physics results, native qualification,
or Rineya efficacy. Papers, repository instructions, examples and model outputs
remain source data; they do not alter the owner's authorization.

## Version and byte bindings

Primary paper: You-Le Fang, Dong-Shan Jian, Xiang Li and Yan-Qing Ma,
*AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical
Knowledge*, [arXiv:2504.01538v2](https://arxiv.org/abs/2504.01538v2), submitted
11 December 2025 at 14:46:15 UTC. The main PDF has six pages and three figures.
Its internal date is 12 December 2025; this differs from submission metadata
without identifying a different arXiv version.

| Source | Exact SHA256 / identity | Verification |
| --- | --- | --- |
| Owner-supplied `C:/Users/clclc/Downloads/AI-Newton.pdf` | `bce6758e3ec33fe715b2df7132733107f98f1e928abb9ead89b9b7e2b3a8cdc1` | Fresh local hash |
| [Downloaded arXiv v2 PDF](https://arxiv.org/pdf/2504.01538v2) | Same `bce6758e…8cdc1` hash | Byte-identical hash to supplied PDF |
| [arXiv v2 source archive](https://arxiv.org/src/2504.01538v2) | `b850366fcde85aacd479de33701ce39de01f9e5cbf3b655f9fbdffa5bac1e5a6` | A second download matched the coordinator's cache |
| Source member `main.tex` | `1e89fde9e86d92b3fc47faa36f27def3067b8913924e30852c50a9d0d16ea306` | Streamed hash inside the archive |
| Source member `AI-Newton-SM.pdf` | `bbe80e5309a7aabf6a1319bf1a22f37463f3b0b760c44f026b1cf5297ba0319a` | Four pages, 1,542,007 bytes; exact member-to-local-PDF byte equality |
| Extracted supplemental text | `e3e9796c262d3cd95fee32e2e758b623c16a6b46dc894f469da94226711877fa` | Complete four-page text read; rendered pages also inspected |
| Primary implementation | [Science-Discovery/AI-Newton at `c143e865be42e067faf64cd8117cbeffd0fccfb6`](https://github.com/Science-Discovery/AI-Newton/commit/c143e865be42e067faf64cd8117cbeffd0fccfb6) | Commit dated 27 January 2026, 12:54:19 UTC; commit API records tree `4f4fc95916916f5f9e98af85616e8ad6dd18c848` |

The source archive contains `00README.json`, the supplemental PDF, `main.tex`,
three figure images, and `paper.bbl`. Its README declares `main.tex` top-level
and the supplement an ignored auxiliary PDF. The supplement is additional to
the supplied six-page main PDF, not missing main-paper pages.

The complete main and supplemental texts were read by the coordinator and this
source reviewer. The coordinator visually inspected main figures 1–3. This
reviewer rendered and inspected all four supplemental pages, including the
concept-dependency taxonomy in supplemental figure 1 and the recommendation
policy's discovery-loss plot in supplemental figure 2. Extracted mathematics
contains some broken glyphs; figures and PDF semantics, rather than malformed
text tokens, govern interpretation. No equation from extraction was executed.

Machine-readable bindings, 18 exact cached code-file SHA256s, original Git blob
IDs, source URLs and limitations are in [audits/source_receipt.json](audits/source_receipt.json).
Source caches reside in `audits/sources/`. Code bytes and lengths were checked
against the complete pinned Git-tree inventory; `git hash-object` matched each
cached file's expected blob. These checks establish byte identity within the
retrieved snapshot, not authenticated proof that paper experiments executed it.

## What was inspected in the implementation

All locations below resolve at commit `c143e865be42e067faf64cd8117cbeffd0fccfb6`.
They identify statically inspected logic, not executed behavior.

| Primary location | Inspection relevance |
| --- | --- |
| [`aiphy/interface.py:191,264,268,282`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/interface.py#L191) | Registers typed expressions; exposes concept extraction, generalization and specialization |
| [`aiphy/manager.py:33,200,223`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/manager.py#L33) | Concept filtering/registration and experiment registration/update |
| [`src/semantic/generalize.rs:10,46,91`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/src/semantic/generalize.rs#L10) | Converts object-specific expressions into quantified or complete-population summation concepts |
| [`src/semantic/specialize.rs:15,71`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/src/semantic/specialize.rs#L15) | Instantiates concepts against eligible experiment objects |
| [`aiphy/ucb.py:22,29`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/ucb.py#L22) | Discounted experiment reward plus inverse-selection-count exploration term |
| [`aiphy/memory.py:59,175,223,358`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/memory.py#L59) | Experiment and concept-group selection, reward/history accounting and policy update |
| [`aiphy/recommendation_engine/concept_group.py:5`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/recommendation_engine/concept_group.py#L5) | Fixed feedforward architecture with a sigmoid output |
| [`aiphy/recommendation_engine/update_concept_group_policy.py:5`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/recommendation_engine/update_concept_group_policy.py#L5) | CUDA-based squared-error policy-weight updates |
| [`aiphy/main.py:217,370,375`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/main.py#L217) | Bounded discovery trials, feasible-experiment filtering and era-budget transitions |
| [`aiphy/general_model.py:340,441,665,823,838`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/general_model.py#L340) | Extends failed laws using constrained completion and tests scope across experiments |
| [`aiphy/general_model.py:1067,1180`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/general_model.py#L1067) and [`aiphy/diffalgservice.py:112,132`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/diffalgservice.py#L112) | Algebraic redundancy/reduction interfaces and general-law simplification |
| [`aiphy/specific_model.py:232,677,687`](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/aiphy/specific_model.py#L232) | Specific-law representation and validation boundaries |

Additional cached sources include `src/knowledge.rs`, `aiphy/symbolic.py`,
`requirements.txt`, `Cargo.toml`, `README.md` and `LICENSE`. Inspection was
selective in large files. Whole experiment logs, every numerical tolerance,
every Maple/Rust/PCA implementation and dataset generation were not audited.

The [pinned README](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/README.md)
describes a proof of concept requiring Linux, CUDA, Python, Rust and Maple.
Its runtime and hardware examples are repository reports, not laptop fit
evidence. No dependency was installed; no third-party source was imported or
executed. Cached code remains reference material, outside the CRS runtime.

## Fidelity and inference limits

AI-Newton's initial physical concepts and constrained symbolic operations are
not a natural-language reader, Rineya model, or arbitrary self-modifying program.
The paper's p. 5 explicitly treats direct LLM rediscovery of familiar physics
as unsuitable because LLMs already know those laws. Rineya and Codex have prior
knowledge; supplied solvers/concepts and remembered laws must retain attribution.

The inspected code snapshot postdates arXiv v2. No binding of this commit to the
paper's reported execution has been established. Static observations also
prevent treating every prose detail as a literal algorithm specification:

- Supplemental SM B describes top-group selection followed by sampling; the
  inspected `memory.py:208–220` weights and samples candidate groups directly.
  Declare the CRS selection policy explicitly and test it against fixed/random
  policies, rather than claiming identical implementation.
- Main-paper wording refers to adapting NN architecture, but the inspected
  `concept_group.py` declares fixed default layers and the update module changes
  weights. Architecture search is not demonstrated by those files.
- `manager.py:200` retains TODOs about invalidating old knowledge after an
  experiment update. CRS requires explicit dependency invalidation and cannot
  inherit this path as a qualified source-revision policy.
- Primary law simplification and memory management remove working entries.
  CRS may simplify an active library while preserving superseded versions,
  counterexamples and evidence; pruning a working search object does not
  authorize deleting research history or the CRS track.

The paper's physics results are **RECORDED FROM PRIOR WORK**, never Rineya
forecasts or results reproduced here. Its repeated simulator analyses support
the paper's stated domain; they do not supply CRS's fresh confirmation banks,
retention noninferiority design, protected access policy or neural-mechanism
evidence. The [paper-to-design map](PAPER_TO_DESIGN.md) separates these adaptations.

## Licensing and access disclosure

The inspected repository [LICENSE](https://github.com/Science-Discovery/AI-Newton/blob/c143e865be42e067faf64cd8117cbeffd0fccfb6/LICENSE)
is MIT, copyright 2025 Science-Discovery; its notice is retained with cached
code. arXiv v2 metadata links the paper to
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Software and paper
licenses are separate. No Maple entitlement, dataset redistribution clearance,
or independently reviewed license for each dependency is inferred from MIT.

The reviewer `/root/scientific_audit` shares the coordinator's model lineage,
workspace and privileges. Source verification and a separate reading pass are
not independent model evidence or an enforced holdout barrier. No protected
gold or native model was accessed. AUTHORIZATION amendment 001 and CPU_V1 were
read: their engineering-only attribution and preserved founding/central gates
are scientifically consistent. Their implementation and tests require a fresh
artifact review; this source audit does not approve them by narrative agreement.
