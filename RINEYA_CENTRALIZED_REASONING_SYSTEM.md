# Rineya Centralized Reasoning System: enduring complementary research track

**Track:** CRS-001 / Rineya Centralized Reasoning System v1  
**Established:** 2026-10-01  
**Branch:** `research/rineya-centralized-reasoning-system-v1`  
**Repository:** `LUNCY579/ChatBubble`  
**State:** `DESIGN_RECORDED_EXECUTION_NOT_AUTHORIZED`  
**Start here:** [experiment package](experiments/rineya_centralized_reasoning_system_v1/README.md)

## CRS-ADR-001: owner intent and preservation

The owner requested this branch for centralized system design and reasoning
experiments for the Rineya family, explicitly emphasizing that it complements
existing discoveries and must not simply be obsoleted or discarded, including
in the near future. This decision is an enduring research-priority and
preservation instruction, not a claim of scientific success.

**CRS-001 is complementary, not a replacement.** Its purpose is to investigate
shared evidence, memory, reasoning execution, verification, and reusable
procedures around Rineya models. Existing native reasoning, post-training,
retention, skill-composition, retrieval, attention/cache, recurrence, and
architecture research retains its independent value and evidence boundaries.

A more capable model may strengthen this system; a better system may provide
validated training traces for that model. Neither development automatically
supersedes the other. Conversely, failure of one external procedure does not
invalidate an unrelated model or architecture discovery.

### Non-discard policy

No automatic sunset or near-term expiry is assigned to this track. Inactivity,
resource constraints, lack of immediate benchmark gains, new papers, and partial
overlap with other branches are not sufficient reasons to delete it or mark it
obsolete. Preserve negative results, failed attempts, unresolved questions, and
successful components alike. Do not force-push away its history.

Deprioritization or parking must record the reason, remaining questions, exact
source tip, dependencies, and a concrete restart condition. Retirement,
archival, replacement, branch deletion, or an obsolescence designation requires
**explicit owner approval** plus an evidence-based disposition note identifying:

1. Which questions are answered, rejected, still open, or transferred.
2. Which successor covers each responsibility and where its evidence lives.
3. Exact source tips and durable provenance references; preserve any relevant
   external artifacts and their hashes before changing branch disposition.
4. Outstanding dependencies and a reversible handoff path.

If the owner later authorizes archival, preserve the exact tip with an agreed
immutable reference and migrate all necessary evidence before any deletion.
This founding change does not create archive tags, modify repository rulesets,
or make the branch technically undeletable. It records the governing policy.

Scientific hypotheses remain falsifiable. Preserving a research track does not
require retaining a disproven claim as true or continuing an unproductive run.

## Scientific purpose

**Question:** Can a compact Rineya model solve harder long-context problems by
using verified, reusable reasoning procedures, beyond a matched system with the
same model, retrieval, primitive tools, verifier, and resource limits?

The proposed system combines an immutable evidence store, bounded per-question
working state, typed executable operations, a controller, and a versioned
procedure library. AI-Newton motivates explicit reusable concepts and iterative
validation; it does not supply a demonstrated long-context result for Rineya.
See [sources](experiments/rineya_centralized_reasoning_system_v1/SOURCES.md).

Maintain three separate scoreboards: **model-only**, **tool-assisted system**,
and **distilled-model** performance. Do not claim that externally processing a
long source extends a model's native attention window or changes its weights.

## Provenance and relationship to the central program

The branch starts from `research/post-training-mimo-centralized-v1` at
`19551ddebc87c93161c303865ce6d0f130de0093` (source commit dated 2026-09-30).
Its base tree is `d2ed2805e79d226639bb75d951e79d15a944bf25`.

The existing centralized Titan program concerns qualified native execution,
capability acquisition, and retention. CRS-001 concerns an external reasoning
runtime that may later connect to that program. The similar word
"centralized" does not merge their experiments, relax their gates, or transfer
qualification between them. The parent remains an independent source branch.

## Founding scope

This delivery records architecture, experiment design, integration boundaries,
source provenance, continuation guidance, and preservation policy. It does not
implement or run the runtime, load a model, generate a dataset, open protected
banks, train an adapter, publish weights, or claim a benchmark improvement.
No existing research file or experiment is replaced by the founding package.
