# Rineya PPT: enduring complementary centralized reasoning research

**Experiment:** PPT-001 / Rineya Parallel Power Tempering v1  
**Established:** 2026-10-02  
**Repository:** `LUNCY579/ChatBubble`  
**Branch:** `research/rineya-ppt-centralized-reasoning-v1`  
**State:** `DESIGN_RECORDED_NATIVE_NOT_IMPLEMENTED`  
**Start here:** [experiment package](experiments/rineya_ppt_centralized_reasoning_v1/README.md)

## PPT-ADR-001: owner intent and durable preservation

The owner requested a new dedicated branch with specialized notes for future
centralized system design and reasoning experiments for Rineya. The owner
explicitly emphasized that this research complements existing discoveries and
must not simply be obsoleted or discarded, including in the near future.

**PPT-001 is a complementary, enduring research track, not a replacement.**
It investigates whether interacting inference-time searches improve how a frozen
compact model uses its existing capabilities. Its value includes diagnosing
search and selection failures, even when a particular sampler does not improve
accuracy. It does not claim that the model has acquired a new skill or that
likelihood is a correctness verifier.

There is **no automatic sunset, near-term expiry, or automatic obsolescence**.
New papers, more capable models, a newer branch, partial conceptual overlap,
inactivity, hardware constraints, or a single negative pilot are not sufficient
reasons to discard the track. Preserve successful findings, null results,
negative evidence, unsuccessful implementations, and unresolved questions.
Do not force-push away its provenance.

Parking or deprioritizing the work must retain its exact source tip, open
questions, dependencies, reason for parking, and concrete restart condition.
Retirement, archival, branch deletion, replacement, or an obsolescence label
requires **explicit owner approval** and a disposition note that identifies:

1. Which questions were answered, rejected, left open, or transferred.
2. Which successor covers each transferred responsibility, with actual evidence.
3. Exact commits and artifact hashes needed to reconstruct the work.
4. The dependency handoff and a reversible preservation route.

If archival is later approved, preserve the agreed immutable reference and
required evidence before any deletion. This policy is repository guidance;
**GitHub branch protection or technical deletion prevention is not configured
by this change**.

Scientific hypotheses remain falsifiable. A bad method may be rejected and a
costly run stopped without erasing the research track. Preservation is not an
instruction to conceal negative evidence or to keep spending compute.

## Scope and scientific identity

**Primary question:** Can bounded, interacting sequence-level search improve a
frozen Rineya 0.6B checkpoint over independent sampling and single-chain
refinement under comparable computational budgets?

In a future centralized system, PPT is an optional search operation. It does not
replace the controller, evidence store, retrieval, memory, reusable procedures,
verification, abstention, or stopping policy. It does not supersede native
recurrence/TAPS, post-training, retention, skill composition, compression, or
architecture research. Preserve their separate objectives and evidence.

The initial comparison must isolate sampler behavior. Keep weights, prompt,
evidence, inference conventions, verifier access, and final-answer readout
fixed. Any later controller integration, recurrence combination, or distilled
training result requires its own declared comparison and provenance.

Keep four distinct reports: ordinary model decoding; frozen-model search;
tool-assisted centralized reasoning; and a separately trained distilled model.
System/search gains are not automatically model-weight gains. External evidence
processing does not extend a model's native context window.

## Base selection and relationships

PPT-001 starts from `research/rineya-centralized-reasoning-system-v1` at
`7459b614bcb017ae4f0160b15e0ccd53473bb990`, tree
`9223d7b22e4bbbab867107a241c67ea7410a0ede`, verified remotely on 2026-10-02.
This preserves CRS-001's existing charter and notes unchanged. It is a new
sibling research direction, not a replacement version of CRS-001.

CRS-001's parent is the centralized MiMo snapshot
`19551ddebc87c93161c303865ce6d0f130de0093`. The separate SYNTH-CRS and verified
resampling branches are pinned as related work in the [sources](experiments/rineya_ppt_centralized_reasoning_v1/SOURCES.md),
not silently merged or imported. No claim is made that their latest code or
qualification is present here.

## Founding delivery boundary

This change adds notes, a descriptive registration, and CPU-only registration
checks. It does not implement PPT transitions or a centralized runtime, load
weights, generate a corpus, launch inference or training, read protected
CHECK/FINAL banks, use paid compute, upload models, change launch permissions,
merge branches, or change the disposition of other research.

Passing documentation/registration checks is not sampler correctness,
checkpoint qualification, or evidence of a reasoning improvement. See the
[status and handoff](experiments/rineya_ppt_centralized_reasoning_v1/STATUS.md).
