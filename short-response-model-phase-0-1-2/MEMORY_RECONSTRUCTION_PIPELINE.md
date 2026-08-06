# Contextual memory reconstruction — Phase 9B design

## Purpose

Phase 9 proves that a smaller student can answer from supplied evidence. Phase 9B tests a different question: can that student determine whether retrieved information still applies before using it?

Ordinary RAG performs candidate retrieval. Phase 9B performs an inference-time applicability transformation:

```text
semantic/lexical relevance
        ≠
current-state applicability
```

The external memory database is preserved. The model-native skill is the ability to compare `source_state` and `validity_conditions` with the current state and produce one of five auditable decisions: `accept`, `adapt`, `reject`, `conflict`, or `insufficient`.

## Training architecture

A reviewed canonical row contains:

```json
{
  "question": "...",
  "current_state": {"...": "..."},
  "memories": [
    {
      "memory_id": "...",
      "text": "...",
      "source_state": {"...": "..."},
      "validity_conditions": {"...": "..."},
      "source_id": "..."
    }
  ],
  "approved_reconstruction": {
    "decision": "adapt",
    "source_ids": ["..."],
    "applicable_facts": ["..."],
    "invalid_assumptions": ["..."],
    "guidance": "...",
    "confidence": "high"
  },
  "approved_answer": "..."
}
```

Each row yields explicit reconstruction SFT, answer-from-guidance SFT, direct one-pass distillation, and an optional DPO preference pair. The direct task prevents the two-pass research mechanism from becoming unusable in Phase 10's ordinary one-pass architecture evaluations.

## Runtime architecture

Phase 11 can enable the same selected checkpoint in `prompted` or `trained` reconstruction mode. The first pass returns compact JSON; the second pass receives only reconstructed guidance. Invalid JSON falls back to `insufficient` and the canonical abstention rather than raw replay.

## Evaluation

The decisive project-specific metric is negative transfer:

```text
P(memory-assisted answer is worse than the same model with memory disabled)
```

The gate also records raw-failure rescue, raw-success harm, decision accuracy, source-ID preservation, parse validity, functional answer success, latency, prompt tokens, generated tokens, and peak VRAM. No fixed universal quality floor controls promotion. The measured reconstruction benefit must exceed its measured runtime/token cost, followed by human review.
