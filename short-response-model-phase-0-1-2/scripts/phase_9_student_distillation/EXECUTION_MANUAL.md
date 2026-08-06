# Phase 9 execution manual

## Prerequisites

1. Complete the teacher Phase 0–8 pipeline and identify the exact accepted full checkpoint.
2. Preserve the teacher evaluation, selected-reference manifest, tokenizer, and response-control contract.
3. Prepare versioned knowledge, procedure, and prompt JSONL files.
4. Install the repository requirements with a CUDA-enabled PyTorch build appropriate for the target computer.
5. Run `python -m pytest -q` and `python -m compileall -q scripts tests`.

## D0 — Baseline and initialization

Create a warm-started intermediate first:

```powershell
python .\scripts\phase_9_student_distillation\create_student.py `
  --source .\artifacts\<accepted-teacher> `
  --output .\artifacts\phase_9_student_distillation\student_initialization `
  --mode depth_copy `
  --num-hidden-layers 12
```

The aggressive narrow Qwen track is an explicit control:

```powershell
python .\scripts\phase_9_student_distillation\create_student.py `
  --source .\artifacts\<accepted-teacher> `
  --output .\artifacts\phase_9_student_distillation\tiny_random_control `
  --mode custom_qwen_random `
  --num-hidden-layers 16 `
  --hidden-size 384 `
  --intermediate-size 1024 `
  --num-attention-heads 6 `
  --num-key-value-heads 2 `
  --head-dim 64 `
  --acknowledge-random-control
```

Do not promote a random-control checkpoint without substantial language-readiness evidence.

## D1 — Knowledge/procedure index

```powershell
python .\scripts\phase_9_student_distillation\build_knowledge_index.py `
  --knowledge .\data\phase_9_student_distillation\knowledge.sample.jsonl `
  --procedures .\data\phase_9_student_distillation\procedures.sample.jsonl `
  --output .\artifacts\phase_9_student_distillation\knowledge.sqlite3
```

## D2 — Teacher-data generation and human review

```powershell
python .\scripts\phase_9_student_distillation\generate_teacher_data.py `
  --teacher .\artifacts\<accepted-teacher> `
  --prompts .\data\phase_9_student_distillation\prompts.sample.jsonl `
  --knowledge-db .\artifacts\phase_9_student_distillation\knowledge.sqlite3 `
  --retrieval-mode oracle `
  --output .\artifacts\phase_9_student_distillation\teacher_candidates.jsonl
```

Review every row. Set `review_status` to `accepted` and either `selected_candidate_index` or `approved_answer`. Preserve rejected candidates for auditability.

## D3 — Document/topic-disjoint pilot

```powershell
python .\scripts\phase_9_student_distillation\build_pilot_dataset.py `
  --reviewed-input .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  --output-dir .\artifacts\phase_9_student_distillation\pilot_dataset `
  --seed 42
```

Confirm train/dev/test groups and support-document IDs do not overlap.

## D4 — Calibrate the external evidence pre-gate

```powershell
python .\scripts\phase_9_student_distillation\calibrate_evidence_gate.py `
  --benchmark .\artifacts\phase_9_student_distillation\pilot_dataset\dev.jsonl `
  --knowledge-db .\artifacts\phase_9_student_distillation\knowledge.sqlite3 `
  --output .\artifacts\phase_9_student_distillation\evidence_gate_calibration.json
```

The selected threshold is tied to this retriever, index, query normalization, and development distribution. Recalibrate after any of those change.

## D5 — Matched SFT versus completion-only KD

Ordinary SFT:

```powershell
python .\scripts\phase_9_student_distillation\train_student.py `
  --student .\artifacts\phase_9_student_distillation\student_initialization `
  --objective sft `
  --train-file .\artifacts\phase_9_student_distillation\pilot_dataset\train.jsonl `
  --dev-file .\artifacts\phase_9_student_distillation\pilot_dataset\dev.jsonl `
  --output-dir .\artifacts\phase_9_student_distillation\student_sft `
  --max-steps 200
```

Same-initialization KD:

```powershell
python .\scripts\phase_9_student_distillation\train_student.py `
  --student .\artifacts\phase_9_student_distillation\student_initialization `
  --teacher .\artifacts\<accepted-teacher> `
  --objective kd `
  --train-file .\artifacts\phase_9_student_distillation\pilot_dataset\train.jsonl `
  --dev-file .\artifacts\phase_9_student_distillation\pilot_dataset\dev.jsonl `
  --output-dir .\artifacts\phase_9_student_distillation\student_kd `
  --max-steps 200
```

Direct KL refuses tokenizer mismatch. For a SmolLM challenger, use sequence-level SFT first; a cross-tokenizer span-alignment experiment remains separate from the official primary gate.

## D6 — Four-condition evaluation

Run each candidate with:

```powershell
python .\scripts\phase_9_student_distillation\evaluate_rag.py `
  --model .\artifacts\phase_9_student_distillation\student_kd\model `
  --benchmark .\artifacts\phase_9_student_distillation\pilot_dataset\test.jsonl `
  --knowledge-db .\artifacts\phase_9_student_distillation\knowledge.sqlite3 `
  --gate-calibration .\artifacts\phase_9_student_distillation\evidence_gate_calibration.json `
  --conditions closed_book,oracle_rag,real_rag,adversarial_rag `
  --output-dir .\artifacts\phase_9_student_distillation\evaluation_kd
```

## D7 — Optional on-policy KD

Generate mixed trajectories and then run KD again. Student-generated rows carry hard-label weight zero. Begin around 20% student trajectories, then increase only if evaluation improves.

## D8 — Preference alignment

Use the existing Phase 3 DPO implementation only after evidence-conditioned SFT/KD is stable. Evaluate the aligned adapter with `evaluate_rag.py` and include its summary in `check_phase_9_gate.py`.

## D9 — Gate and handoff

```powershell
python .\scripts\phase_9_student_distillation\check_phase_9_gate.py `
  --teacher-summary .\artifacts\phase_9_student_distillation\evaluation_teacher\summary.json `
  --sft-summary .\artifacts\phase_9_student_distillation\evaluation_sft\summary.json `
  --kd-summary .\artifacts\phase_9_student_distillation\evaluation_kd\summary.json
```

Human review must name the exact selected artifact and classify it as production candidate, intermediate teacher, or research-only control.

## D10 — Quantization and total-process measurement

Quantize only the accepted dense/full student or merged adapter checkpoint. Measure generator weights, KV cache, retriever, reranker if any, index, application overhead, prompt latency, generation latency, and total process RSS. An idealized INT4 parameter estimate is not a measured deployment footprint.
