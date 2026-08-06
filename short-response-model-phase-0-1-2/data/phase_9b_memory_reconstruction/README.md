# Phase 9B memory-reconstruction data

`memories.sample.jsonl` demonstrates the external memory schema. Every item keeps the reusable content together with its historical source state, validity conditions, provenance, timestamp, and trust level.

`reviewed.sample.jsonl` is a smoke-test packet containing human-approved ACCEPT, ADAPT, REJECT, CONFLICT, INSUFFICIENT, counterfactual, and retrieval-injection cases. It is intentionally too small for a research claim. Production training should be built from the accepted Phase 9 corpus plus a separately reviewed Phase 9B packet.
