# Sources, provenance and dependency boundaries

Checked on **2026-10-02**. Commit pins are historical snapshots, not promises
about future branch heads or qualification. Links are recorded for future
review. No external source text, paper figures, weights or datasets are copied
into this package.

## P1: paper motivating the experiment

Panagiotis Theodoropoulos, Nan Jiang, Xintong Duan, Ali Hasan, Yuriy Nevmyvaka,
Evangelos A. Theodorou and Wei Deng. *Explore Broadly, Reason Sharply: Push Small
Models toward the Frontier via Sampling*. arXiv:2609.38104v1, submitted
2026-09-29.

- Versioned abstract: https://arxiv.org/abs/2609.38104v1
- Versioned full-text HTML: https://arxiv.org/html/2609.38104v1
- Review targets for an implementation: sections 2-3; appendices B-C for
  transition, horizon and exactness conventions; section 4 and appendix H for
  experimental interpretation.

The paper supplies the method motivation, not evidence that Rineya 0.6B benefits.
No published frontier comparison is imported as a Rineya result. Pilot sizes,
resource choices, architecture contracts and promotion rules in this package
are our proposed adaptations. No official implementation revision has been
selected, imported, audited or reproduced by this founding change.

## R1: parent, CRS-001

Branch: `research/rineya-centralized-reasoning-system-v1`  
Commit: `7459b614bcb017ae4f0160b15e0ccd53473bb990`  
Tree: `9223d7b22e4bbbab867107a241c67ea7410a0ede`

https://github.com/LUNCY579/ChatBubble/commit/7459b614bcb017ae4f0160b15e0ccd53473bb990

The root charter and AGENTS guidance were read. This is the selected base
because it records centralized evidence/procedure research and owner-directed
preservation. Its contents are inherited unchanged. The original CRS question
and its execution boundaries remain authoritative for CRS-001.

## R2: independent-resampling reference, NOT merged here

Branch: `research/rineya-verified-resampling-v1`  
Commit: `ac9b79ac07ad650d8815c76eb4194a180f23e8d4`

https://github.com/LUNCY579/ChatBubble/commit/ac9b79ac07ad650d8815c76eb4194a180f23e8d4

Reference for a future reuse review and baseline comparison. A remote ref was
verified; this delivery does not re-audit or import its runtime or transfer any
qualification. Fetch its exact files and instructions before implementing a
shared adapter. Preserve the baseline's experiment identity and receipts.

## R3: related centralized synthesis track, NOT merged here

Branch: `research/rineya-synth-centralized-reasoning-v1`  
Commit: `23cebe93a8684302eda3a114337482b1c95a26ba`

https://github.com/LUNCY579/ChatBubble/commit/23cebe93a8684302eda3a114337482b1c95a26ba

Its founding commit records complementary centralized reasoning/distillation
notes. PPT-001 is a sibling from CRS-001, not its successor or replacement.
Potential trace sharing remains a later controlled integration.

## R4: centralized MiMo ancestor

Branch: `research/post-training-mimo-centralized-v1`  
Commit: `19551ddebc87c93161c303865ce6d0f130de0093`

https://github.com/LUNCY579/ChatBubble/commit/19551ddebc87c93161c303865ce6d0f130de0093

Inherited through CRS-001. This snapshot does not supply a new PPT launch grant.
The founding work does not alter native gates, weights, protected banks or the
central program's acquisition/retention objective.
