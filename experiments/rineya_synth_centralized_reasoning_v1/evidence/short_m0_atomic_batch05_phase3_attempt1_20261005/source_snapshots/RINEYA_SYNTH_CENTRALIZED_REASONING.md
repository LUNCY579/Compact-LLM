# Rineya SYNTH-inspired centralized reasoning: durable complementary track

**Experiment:** SYNTH-CRS-001  
**Established:** 2026-10-02  
**Branch:** `research/rineya-synth-centralized-reasoning-v1`  
**Repository:** `LUNCY579/ChatBubble`  
**Status:** `DESIGN_REGISTERED_NATIVE_UNQUALIFIED`  
**Start here:** [package and handoff](experiments/rineya_synth_centralized_reasoning_v1/README.md)

## ADR-SYNTH-CRS-001: owner intent and preservation

The owner explicitly requested a new dedicated branch for centralized system
design and reasoning experiments for Rineya, with durable notes emphasizing
that the work complements existing discoveries and must not simply be obsoleted
or discarded, including in the near future. This charter records that intent.

**This is a complementary research track, not a replacement for CRS-001 or for
Rineya's existing model, data, retention, composition, or architecture research.**
It studies whether a centralized pipeline can turn grounded tasks and verified
solutions into useful learning signals for a compact, approximately 0.6B student.
It also preserves the separate question of useful external reasoning at inference.

A stronger model may improve the system; the system may produce better training
data for the model. Neither success automatically supersedes the other. New
papers, inactivity, hardware constraints, overlap with another branch, a negative
pilot, or improvement elsewhere are NOT sufficient reasons to delete this track,
archive it, replace it, or label it obsolete. There is no automatic sunset date.

Retirement, archival, replacement, branch deletion, or an obsolescence designation
requires **explicit owner approval** and an evidence-based disposition note.
Record answered/rejected/open questions, exact source tips, preserved artifact
hashes, dependencies, successor responsibilities, and a reversible handoff.
Parking is allowed with a reason, remaining questions, and concrete restart
conditions. Preserve negative results; reject disproven hypotheses honestly.
Preservation does not require indefinite computation or pretending a null is a gain.

This is an agent/research-governance instruction. It does not configure GitHub
branch protection or make the branch technically undeletable. Do not force-push
away history or discard other discoveries in favor of this one.

## Identity and provenance

This branch is based on the existing CRS-001 tip
`7459b614bcb017ae4f0160b15e0ccd53473bb990`, with base tree
`9223d7b22e4bbbab867107a241c67ea7410a0ede`.
Its source branch is `research/rineya-centralized-reasoning-system-v1`.
That source descends from the centralized training snapshot
`19551ddebc87c93161c303865ce6d0f130de0093`.

CRS-001 asks whether reusable procedure memory improves an otherwise matched
external reasoning system. SYNTH-CRS-001 asks whether verified task amplification
and, later, verified representations improve what the student learns. These are
different hypotheses. The original CRS-001 charter and protocol remain intact.
The similarly named centralized Titan/MiMo training program is also independent;
its original objective, qualification checks, and protected banks remain binding.

## Research contract

Keep three scoreboards separate: **model-only**, **tool-assisted system**, and
**distilled-model**. System success is not evidence of changed model weights;
external source processing is not a native context-window extension; teacher
success is not student success. Record total costs as well as student costs.

The paper inspiration is *It's All Training: A Fully Synthetic Single-Stage
Recipe for LLMs*, arXiv:2609.37891v1. This proposal adapts grounded synthetic
supervision to an existing Rineya checkpoint. It is not a reproduction of
from-scratch single-stage pretraining and does not assume all synthetic traces
are correct. See [sources](experiments/rineya_synth_centralized_reasoning_v1/SOURCES.md).

## Founding delivery boundary

This delivery adds a charter, architecture and experiment design, data contracts,
a descriptive manifest, and CPU-only tests of the registration's invariants.
It does NOT implement a reasoning runtime, construct a training corpus, run a
model, train a student, open CHECK/FINAL banks, publish weights, or establish
capability gains. Native launches need a separately completed and approved run
manifest; documentation tests do not qualify a model or authorize execution.
