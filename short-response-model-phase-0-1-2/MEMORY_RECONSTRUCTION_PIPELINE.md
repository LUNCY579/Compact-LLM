## v1.7.4 dedicated reconstruction curriculum

The default Phase 9B research corpus is now `data/phase_9b_memory_reconstruction/reconstruction_corpus_v1`. It supplies explicit `current_state`, memory `source_state`, validity conditions, five-way reconstruction targets, counterfactual pairs, curated DPO negatives, and evaluator calibration candidates. The authored 440/80/80 topic-family split is preserved: TRAIN supports SFT/KD/preference alignment, DEV selects and freezes one reconstruction strategy, and TEST is sealed until that freeze exists.

Only approved reconstruction/answer targets are SFT data. Raw replay, over-rejection, wrong-source, and evaluator alternatives are negative examples for preference/evaluator work only. Unknown validity predicates fail closed rather than being executed. Full details are in `PHASE_9B_MEMORY_RECONSTRUCTION_CORPUS_INTEGRATION_v1.7.4.md`.

# Contextual memory reconstruction — Phase 9B design


## v1.7.3 corpus integration amendment

`data/phase_9_student_distillation/rag_reasoning_v2` is the default Phase 9B source. Its stale/current, injection, conflict, missing-evidence, procedure, and ambiguity cases are converted into reconstruction candidates while preserving authored train/dev/test intent. Reconstruction first removes untrusted and superseded memories, evaluates `validity_conditions`, and only then detects unresolved conflicts among still-applicable sources. Consequently a superseded old fact plus a current replacement is normally `adapt`, not `conflict`.

The integrated corpus is intentionally enriched with `current_state`, `source_state`, `validity_conditions`, `memory_type`, trust/version metadata, and source provenance where those concepts are applicable. Human-reviewed reconstruction JSON remains mandatory before training; the deterministic conversion is a conservative proposal, not an automatic label authority.

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
