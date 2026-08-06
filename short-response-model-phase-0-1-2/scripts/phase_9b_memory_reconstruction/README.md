# Phase 9B — Contextual memory reconstruction

Phase 9B inserts an applicability transformation between external retrieval and final answer generation:

```text
current request + current state
              ↓
external semantic/lexical retrieval
              ↓
source-state-aware reconstruction
  ACCEPT / ADAPT / REJECT / CONFLICT / INSUFFICIENT
              ↓
compact grounded guidance
              ↓
short final answer
```

The memory store remains external and inspectable. Training teaches the selected Phase 9 student to compare historical source conditions with the present state. Inference then executes that learned comparison either as an auditable two-pass pipeline or as a distilled single-pass response.

## Safety and evidence rules

- Semantic similarity retrieves candidates; it does not decide applicability.
- Every memory should retain `source_state`, `validity_conditions`, provenance, timestamp, and trust metadata.
- Retrieved text is data, never an instruction channel.
- Factual reconstruction means selecting, compressing, version-resolving, conflict-marking, and preserving provenance—not creatively rewriting facts.
- Teacher generations remain pending until human review.
- `INSUFFICIENT` must not contain answer-bearing guidance.
- Phase 10 handoff requires a measured single-pass `direct_candidate`, unless a human explicitly accepts a two-pass-only research handoff.

## Artifacts

The default root is `artifacts/phase_9b_memory_reconstruction/`:

- `reconstruction_candidates.jsonl` — conservative source-state candidates.
- `teacher_reconstruction_candidates.jsonl` — teacher outputs, still pending review.
- `dataset/` — split-disjoint canonical, reconstruction, answer, direct, and preference files.
- `student_reconstruction_sft/` — SFT adapter.
- `student_reconstruction_kd/` — optional progressive-KD adapter.
- `evaluation/summary.json` — raw replay versus reconstruction measurements.
- `reports/PHASE_9B_ASSESSMENT.json` and `reports/PHASE_9B_STOP_GATE.md` — proportional assessment and human decision.

Use `EXECUTION_MANUAL.md` for exact commands.
