# File responsibilities — Phase 9

| File | Responsibility | Must not be treated as |
|---|---|---|
| `run_phase_9.ps1` | Orchestrates bootstrap, reviewed pilot training, evaluation, optional on-policy KD, and the pending gate. | Permission to skip human teacher-data review or Phase 8 approval. |
| `phase9_common.py` | Data validation, routing, prompt construction, SQLite retrieval, evidence pre-gating, grounding diagnostics, splitting, and footprint helpers. | A semantic entailment oracle or universal answerability classifier. |
| `create_student.py` | Creates depth-copied, random-control, or pretrained-clone initializations with manifests. | A claim that a random narrow student inherited language competence. |
| `build_knowledge_index.py` | Builds the low-memory FTS5 index and provenance manifest. | A neural reranker or semantic retrieval guarantee. |
| `generate_teacher_data.py` | Produces teacher candidates, diagnostics, provenance, and a review packet. | Automatically approved training data. |
| `build_pilot_dataset.py` | Converts accepted teacher rows into document/topic-disjoint train/dev/test splits. | Authorization to include pending or rejected outputs. |
| `train_student.py` | Runs completion-preserving SFT or exact-tokenizer online KL distillation. | Cross-tokenizer logit alignment. |
| `generate_on_policy_data.py` | Mixes teacher references with student-generated trajectories and disables hard-label reinforcement on student rows. | A replacement for teacher feedback. |
| `calibrate_evidence_gate.py` | Selects a development-calibrated lexical pre-gate threshold using explicit product costs. | A universal threshold portable to another retriever/index/domain. |
| `evaluate_rag.py` | Evaluates closed/oracle/real/adversarial RAG, grounding, abstention, retrieval, quality, latency, and memory. | A substitute for human factual/safety review. |
| `check_phase_9_gate.py` | Uses the shared proportional opportunity engine and writes a pending human gate. | Automatic production promotion. |
| `README.md` | Explains the architecture, tracks, normal commands, and interpretation. | Execution evidence. |
| `EXECUTION_MANUAL.md` | Gives detailed D0–D10 commands and review order. | Proof that training or quantization succeeded. |
| `FILE_RESPONSIBILITIES.md` | Defines ownership and boundaries. | Runtime code. |
