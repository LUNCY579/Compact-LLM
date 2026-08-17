# Data Card — Rineya Short-Response Behavior Corpus v1.0

## Purpose

This package teaches **short-response behavior**, not broad world knowledge. It targets a model that reads the available context and returns the shortest natural reply that remains correct, useful, context-aware, and safe.

The design responds to the v2.3.1 finding that raw example count was not the dominant bottleneck. The corpus therefore emphasizes explicit behavioral contracts, class balance, multi-turn context, deterministic correctness, prompt uniqueness, and targeted negative examples rather than merely repeating more templates.

## Contents

- **10,000 new SFT rows** across 25 response classes.
- **2,000 preservation rows** sampled only from the existing training split.
- **12,000-row combined SFT** file recommended for the first run.
- **5,000 DPO pairs** sampled at exactly 50% of each new response class.
- Schemas, validation tooling, manifests, checksums, leakage evidence, and a stratified audit packet.

## Behavioral coverage

The new data covers factual compression, exact arithmetic, acknowledgments, confirmations, celebrations, empathy, natural follow-ups, clarification, direct preferences, gentle disagreement, uncertainty, knowledge boundaries, multi-turn recall and updates, pronoun resolution, practical first steps, concise explanations, explicit tradeoff comparisons, apology repair, de-escalation, safety boundaries, complex-task escalation, output-only contracts, roleplay, and informal texting.

## Trainer compatibility

Every SFT record:

1. is one JSON object on one line;
2. begins with a `system` message;
3. alternates `user` / `assistant`;
4. ends with the exact `assistant` reply to learn;
5. has `metadata.split = "train"`;
6. omits the top-level `question` field, so Phase 9 does **not** rebuild or replace its stored prompt.

Every DPO record contains `prompt`, `chosen`, and `rejected`. `prompt` is pre-rendered in Qwen chat-template form and ends at the assistant generation prefix.

## Preservation design

The preservation slice contains all 550 original core Phase 1 training rows plus 1,450 roleplay training rows. Roleplay selection is balanced to 72–73 rows per response class and 72–73 rows per persona. No repository DEV or SEALED row is copied.

## Quality controls

Automated checks passed for schema validity, role order, final assistant targets, unique IDs, unique rendered prompts, target length, arithmetic correctness, numerical confirmation correctness, explicit comparison criteria, absence of hidden-reasoning tags, and train/evaluation isolation.

The leakage scan compared the full 12,000-row combined SFT against the repository's 510 DEV and 110 SEALED rows. Exact rendered-prompt overlap, exact latest-user overlap, template-family overlap, and RapidFuzz ratio ≥94 near-overlap were all zero.

## Important limitation

`human_review_status` remains **pending**. Automated validation cannot guarantee every conversational wording choice is ideal. Before production training, review the included stratified audit packet and give special attention to safety, empathy, roleplay boundaries, and DPO negatives.

## Recommended use

Start with the 12,000-row combined SFT for one epoch and evaluate the existing untouched DEV split. Compare against the current Rineya checkpoint on brevity, completeness, factual/context integrity, safety, preservation, and category tails. Increase training only when the held-out evidence supports it.

Use the 5,000 DPO pairs only after an SFT candidate is selected. Keep the repository's existing DPO DEV file unchanged.
