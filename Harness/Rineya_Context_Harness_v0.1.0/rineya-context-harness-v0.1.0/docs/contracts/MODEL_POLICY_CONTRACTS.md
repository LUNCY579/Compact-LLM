# Model Policy Contracts

## Query planning

Output: `QueryPlan`

```json
{
  "gap_id": "primary",
  "query": "narrow search query",
  "reason": "resolve the highest-priority information gap"
}
```

## Reconstruction

Output: `ReconstructionOutcome`

```json
{
  "claims": [
    {
      "statement": "Atomic factual claim.",
      "evidence_ids": ["ev_..."],
      "support_status": "supported",
      "required_for_answer": true,
      "freshness_required": false
    }
  ],
  "memory_additions": [
    {
      "kind": "resolved_evidence",
      "summary": "Compact future-usable fact.",
      "payload": {"claim": "..."},
      "source_ids": ["ev_..."],
      "dependencies": [],
      "temperature": "hot",
      "pinned": false
    }
  ],
  "memory_updates": [],
  "deletion": {
    "memory_ids": ["mem_ev_..."],
    "reason_code": "source_reduced_to_claim",
    "reason": "Raw passage is no longer needed in active context.",
    "replacement_state_ids": []
  },
  "unresolved_questions": [],
  "summary": "One supported fact reconstructed."
}
```

## Grounded answer

Output: `AnswerDraft`

```json
{
  "answer": "Concise supported answer.",
  "citations": ["ev_..."],
  "confidence_note": null
}
```

Citations are evidence IDs, not raw URLs. The presentation layer may map IDs to URLs and titles.
