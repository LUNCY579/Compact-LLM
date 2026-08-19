# Data Card: Rineya Corpus v3.0

## Purpose

Train-only supervised and preference data for compact Rineya models that must reason from a full conversational thread and return a few natural words or one compact sentence. The corpus is synthetic, procedurally generated from reviewed capability grammars, and marked for human review.

## Composition

- 110,000 new SFT rows across 40 retained v2 class IDs.
- 14,000 deterministic v1/v2 preservation rows.
- 124,000-row recommended combined SFT.
- 16,000 DPO pairs, 400 per class.

## Intended use

QLoRA/LoRA chat SFT and post-SFT DPO for the Rineya model family. The final assistant message is the target. `enable_thinking=False` is assumed. No row contains a top-level `question` field.

## Not intended for

Training on evaluation splits, medical diagnosis, licensed legal advice, real-person data, autonomous harmful action, or public claims without human audit and sealed evaluation.

## Known limitations

Procedural generation cannot prove that every social reply is the uniquely best conversational response. Some exact short answers naturally repeat in constrained reasoning tasks. The anti-collapse rule is enforced for 3–6-word natural replies within shards, while deterministic answers are evaluated by correctness. Automated near-duplicate analysis is sampled rather than a quadratic all-pairs comparison.

## Review status

`human_review_status` is `pending` for every generated row and DPO pair.
