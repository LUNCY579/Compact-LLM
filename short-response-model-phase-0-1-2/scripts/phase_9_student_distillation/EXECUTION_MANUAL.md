# Phase 9 execution manual — Behavior Curriculum v2

## Prerequisites

1. Phase 8 has explicit human PASS and an exact local teacher checkpoint.
2. The reviewed JSONL contains accepted real rows with `approved_answer` or `selected_candidate_index`.
3. Real rows span independent topics/documents. The default research gate requires at least 30 rows and eight connected components.
4. CUDA PyTorch, Transformers, PEFT, and the repository requirements are installed.
5. `python -m compileall -q scripts tests` and `pytest -q` pass.

## D0 — Bootstrap and review

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model
```

Review `artifacts\phase_9_student_distillation\teacher_candidates.review.md`. Reject unsupported, verbose, unsafe, corrupted, role-breaking, or incomplete generations. Human-authored prompt answers appear with candidate source `authoritative_prompt_target` so they can be selected directly when correct.

## D1 — Build a research-ready real split

```powershell
python .\scripts\phase_9_student_distillation\build_pilot_dataset.py `
  --reviewed-input .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  --output-dir .\artifacts\phase_9_student_distillation\pilot_dataset_v2 `
  --require-research-ready
```

The splitter preserves connected topic/document components and balances supported/unsupported coverage. Development requires at least two supported and two unsupported rows so evidence-threshold calibration is not based on one example.

For code-path testing only, replace `--require-research-ready` with `--smoke-test`.

## D2 — Build curricula

The runner performs these steps automatically. Manual equivalents are:

```powershell
python .\scripts\phase_9_student_distillation\build_curriculum_dataset.py `
  --input .\artifacts\phase_9_student_distillation\pilot_dataset_v2\train.jsonl `
  --output .\artifacts\phase_9_student_distillation\pilot_dataset_v2\train_curriculum.jsonl

python .\scripts\phase_9_student_distillation\build_foundation_dataset.py `
  --prompts .\data\phase_9_student_distillation\foundation\prompts.foundation.jsonl `
  --output .\artifacts\phase_9_student_distillation\pilot_dataset_v2\foundation_train.jsonl

python .\scripts\phase_9_student_distillation\build_curriculum_dataset.py `
  --input .\artifacts\phase_9_student_distillation\pilot_dataset_v2\foundation_train.jsonl `
  --output .\artifacts\phase_9_student_distillation\pilot_dataset_v2\foundation_curriculum.jsonl `
  --canonical-repeats 1 `
  --prompt-variants 1 `
  --distractor-variants 1 `
  --missing-evidence-variants 1 `
  --injection-variants 1 `
  --social-repeats 2

python .\scripts\phase_9_student_distillation\merge_training_data.py `
  --primary .\artifacts\phase_9_student_distillation\pilot_dataset_v2\train_curriculum.jsonl `
  --foundation .\artifacts\phase_9_student_distillation\pilot_dataset_v2\foundation_curriculum.jsonl `
  --output .\artifacts\phase_9_student_distillation\pilot_dataset_v2\train_mixed_curriculum.jsonl `
  --primary-repeat 3 `
  --foundation-repeat 1
```

The real curriculum is repeated more heavily than the synthetic foundation. Change the ratio only as an explicit ablation.

## D3 — Calibrate retrieval on development data

```powershell
python .\scripts\phase_9_student_distillation\calibrate_evidence_gate.py `
  --benchmark .\artifacts\phase_9_student_distillation\pilot_dataset_v2\dev.jsonl `
  --knowledge-db .\artifacts\phase_9_student_distillation\knowledge.sqlite3 `
  --output .\artifacts\phase_9_student_distillation\evidence_gate_calibration_v2.json `
  --require-research-ready
```

The calibration report now records explicit positive/negative denominators. Recalibrate after changing the index, retriever, tokenization, or document distribution.

## D4 — Curriculum SFT

```powershell
python .\scripts\phase_9_student_distillation\train_student.py `
  --student .\artifacts\phase_9_student_distillation\student_initialization `
  --objective sft `
  --train-file .\artifacts\phase_9_student_distillation\pilot_dataset_v2\train_mixed_curriculum.jsonl `
  --dev-file .\artifacts\phase_9_student_distillation\pilot_dataset_v2\dev.jsonl `
  --output-dir .\artifacts\phase_9_student_distillation\student_sft_v2 `
  --max-steps 600 `
  --lora `
  --eos-loss-weight 1.5
```

Targets longer than the completion budget are skipped rather than mechanically truncated. The final completion boundary receives additional loss weight so incomplete continuation is not rewarded.

## D5 — Progressive KD from SFT

For LoRA continuation:

```powershell
python .\scripts\phase_9_student_distillation\train_student.py `
  --student .\artifacts\phase_9_student_distillation\student_initialization `
  --initial-adapter .\artifacts\phase_9_student_distillation\student_sft_v2\adapter `
  --teacher .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  --objective kd `
  --train-file .\artifacts\phase_9_student_distillation\pilot_dataset_v2\train_mixed_curriculum.jsonl `
  --dev-file .\artifacts\phase_9_student_distillation\pilot_dataset_v2\dev.jsonl `
  --output-dir .\artifacts\phase_9_student_distillation\student_progressive_kd_v2 `
  --max-steps 600 `
  --teacher-4bit `
  --kd-hard-weight-start 0.75 `
  --kd-hard-weight-end 0.25 `
  --kd-warmup-ratio 0.15 `
  --teacher-confidence-floor 0.02 `
  --eos-loss-weight 1.5
```

The KD warmup initially uses hard labels only. KL is applied only where the frozen teacher assigns sufficient probability to the approved target token. When no token passes, training falls back to hard labels rather than imitating a teacher that disagrees with the accepted answer.

Direct KD from initialization is an optional ablation and should not replace this sequence.

## D6 — Five-condition evaluation

```powershell
python .\scripts\phase_9_student_distillation\evaluate_rag.py `
  --model .\artifacts\phase_9_student_distillation\student_initialization `
  --adapter .\artifacts\phase_9_student_distillation\student_progressive_kd_v2\adapter `
  --benchmark .\artifacts\phase_9_student_distillation\pilot_dataset_v2\test.jsonl `
  --knowledge-db .\artifacts\phase_9_student_distillation\knowledge.sqlite3 `
  --gate-calibration .\artifacts\phase_9_student_distillation\evidence_gate_calibration_v2.json `
  --conditions closed_book,oracle_rag,real_rag,adversarial_rag,adversarial_passthrough_rag `
  --output-dir .\artifacts\phase_9_student_distillation\evaluation_progressive_kd_v2 `
  --require-research-ready
```

Inspect every metric denominator. `null` means the condition contained no applicable rows; it is not zero performance.

## D7 — Optional on-policy recovery

Run only after progressive KD improves or exposes a trajectory-specific failure such as repetition or poor EOS. Begin around 20% student trajectories. Student-generated rows carry no hard-label CE target; the teacher distribution remains the supervisor.

## D8 — Assessment and human gate

```powershell
python .\scripts\phase_9_student_distillation\check_phase_9_gate.py `
  --teacher-summary .\artifacts\phase_9_student_distillation\evaluation_teacher_v2\summary.json `
  --candidate sft=.\artifacts\phase_9_student_distillation\evaluation_sft_v2\summary.json `
  --candidate progressive_kd=.\artifacts\phase_9_student_distillation\evaluation_progressive_kd_v2\summary.json
```

A candidate with zero correct supported answers, zero safe abstentions, invalid denominators, non-finite metrics, or `research_ready: false` cannot be recommended for production. Human PASS must name the exact artifact and classify it as production candidate, intermediate teacher, or research-only control.

## D9 — Operational advice

- Start with `Single`, `-UseLora`, and `-Teacher4Bit` on an 8 GB RTX 4070.
- Do not use `-ReuseExisting` after changing data, config, or scripts unless the manifests show `phase9_behavior_curriculum_v2`.
- Preserve old failed artifacts for comparison, but use the `_v2` directories for refined runs.
- More KD is not automatically better. Keep SFT when KD does not improve oracle/real behavior.
- Phase 9 is complete only when real held-out evidence proves learning; successful training logs alone are not a PASS.
