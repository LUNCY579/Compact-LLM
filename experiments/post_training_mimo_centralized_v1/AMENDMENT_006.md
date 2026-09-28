# Amendment 006: language DEV/CAL candidate inputs

Prospective data-only scope, 28 September 2026. The prior turn completed the E1
plan. Native E0 remains blocked; this amendment permits original language input
authorship, assistant semantic review, tokenizer-only measurement and unassigned
review-packet mechanics. It grants no native collection, training, qualified
reference labels, grading authority, paid provider calls or held-out access.
The original LANGUAGE_DEV_PROPOSAL.json remains immutable as the proposal record.

## Fixed populations and role assignment

Retain its eight strata: explanation, meaning_preserving_rewrite, synthesis,
clarification, supportive_conversation, disagreement_and_compromise,
contextual_recommendation and followup_and_closure. Each has eight DEV worlds
and two CAL worlds: 64 DEV and 16 CAL parents, not 80 independent mechanisms.
IDs are LDEV-{stratum}-01 through -08 and LCAL-{stratum}-01 through -02.
DEV 01-04 are single turn; 05-08 are multi turn. CAL 01 is single turn and 02
is multi turn. Multi-turn conversations have 2-4 user turns with genuine context
evolution; they are not one question padded with irrelevant dialogue.

First author only each world's actors, facts, timeline, constraints, intent,
decisive semantic structure, potential confusions and exclusion tags. Assign
the parent and all derivatives to its fixed role before rendering or solving.
Review and commit these specifications and byte/hash inventory before authoring
conversations, exemplar replies or CAL control responses. Any subsequent spec
repair must retain the original and record a new revision before re-rendering.
Do not replace difficult worlds after tokenizer fit or student outputs are seen.

This is a small, versioned original source package. Derived JSONL, token arrays,
packet mappings and rendering outputs remain local; committed source texts and
hash receipts allow reproduction. No existing V1/V2/V3 split or source template
is repurposed as a fresh world. Hash and semantic reviews have explicit coverage
limits and cannot establish unknown-pretraining exclusion or independence by ID.
CHECK remains 128 count reservations, with no payloads or newly opened content.
FINAL remains closed and unspecified.

## Authorship and semantic review

Assistant /root/baseline_enrollment authors the first four strata;
/root/e1_review authors the last four. Root reviews the pre-render specifications.
After rendering, each author reviews the other author's complete conversations,
contracts and CAL controls. These are identified assistant source reviews, not
qualified human judgments or statistical scorer calibration. No qualified owner,
human reviewer, adjudicator or model grader is currently assigned.

Use short English conversations whose decisive facts are fully supplied. Messages
alternate user/assistant and end with the scored user request; no additional system
message is introduced. Preserve prior assistant context as authored context, not
as generated Titan behavior. Each DEV world receives a non-exhaustive illustrative
reply and separate semantic acceptance requirements. Exact matching is insufficient.

Each CAL world receives two distinct intended acceptable replies and four candidate
defect replies, primarily targeting adequacy, relevance, fluency and role consistency.
This yields 96 candidate controls. Dimensions can interact: an off-topic response
can also fail adequacy. A primary intended defect is not a guaranteed isolated
single-dimension failure, and no complete reference score vector is assigned here.
Intended annotations remain unqualified author intent until an independently
qualified rubric owner validates and freezes labels. Preserve rejected drafts and
reviews; repair transparently before final input freeze, not by silent replacement.

## Measurement and packet boundaries

After semantic review, use the existing exact Titan tokenizer/template in the
pinned Python 3.11.15 environment without Torch or model calls. Keep complete
prompts at most 896 tokens and references/controls at most 128 including EOS;
no truncation. Measure actual tokens and bind source/template/tokenizer hashes.
Record every failed preparation attempt and measured wall time where available.

Unassigned CAL packet templates contain the exact conversation and candidate reply,
opaque randomized IDs and 16 hidden repeats (one per CAL world). Keep all author
intent, parent mapping, repeat flags and model/recipe labels out of the rating view.
Preserve a separate immutable coordinator mapping; access separation is a workflow
requirement, not a cryptographic guarantee. Packet templates do not assign reviewers
or constitute a real calibration run. Four-dimension rating imports must preserve
missing/unjudgeable cases and provenance and remain diagnostic without qualified
identity, independent labels, assignments and calibration evidence. Never fill
synthetic ratings into production packets.

Original proposed calibration thresholds, baseline-first enrollment, blinded paired
review, drift rules and native scientific/resource gates are unchanged. Numerical
native wall/GPU ceilings remain unmeasured E0-dependent blockers. No authored or
tokenized candidate bank changes source-correct membership, current best recipe,
training eligibility or E1 launch authority.
