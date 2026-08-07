## v1.7.4 reconstruction-corpus handoff

The dedicated Phase 9B reconstruction corpus is now validated downstream whenever it is relevant. Phase 10 validates it when an approved `phase_9b_student_reference.json` is the architecture-reduction parent. Phase 11 validates it at startup and can benchmark the complete two-pass reconstruction runtime, not just the persistent service.

For an approved Phase 9B-derived model, Phase 11 can measure trained reconstruction with:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Full `
  -MemoryReconstructionMode trained `
  -ReconstructionMaxNewTokens 128
```

Use `prompted` only as an inference-time protocol baseline; use `trained` only when the selected lineage actually includes approved Phase 9B reconstruction training. `off` remains the ordinary-RAG control. The production benchmark reports reconstruction validity, added tokens, added latency, and full end-to-end request cost.

# Complete Phase 10–11 workflow


## v1.7.3 shared corpus and sealed-test amendment

Phase 10 and Phase 11 now default to the v1.7.3 Phase 9 grounded-recovery dataset and its shared `rag_reasoning_v2` knowledge/procedure corpus. Phase 10 profiles, screens, recovery-trains, and chooses the final architecture entirely on **dev**, freezes one candidate, and then evaluates exactly the reference and frozen candidate on the **sealed test**. Phase 11 calibrates answerability on dev and uses the test split for final runtime/release comparison only.

Procedure answerability is derived from normalized decision semantics rather than `mode == procedure`; answerable procedures therefore count as supported behavior, while missing-input and genuine-conflict procedures remain abstention cases. Production metrics separately report supported correctness, safe conflict handling, procedure correctness, evidence calibration, stale/current precedence, prompt-injection task success, social usability, and their real denominators.

## Phase 9B handoff amendment

Phase 10 now prefers `artifacts/references/phase_9b_student_reference.json` when it exists and has an explicit Phase 9B human PASS. Otherwise it continues to use the Phase 9 reference. The Phase 9B manifest preserves the Phase 9 model as rollback.

Phase 11 can run source-state reconstruction before final answer generation:

```powershell
python .\scripts\phase_11_production_rag\serve_production_rag.py `
  --model .\artifacts\phase_11_production_rag\deployment_artifact `
  --knowledge-db .\artifacts\phase_11_production_rag\knowledge_snapshot\knowledge.sqlite3 `
  --answerability-calibration .\artifacts\phase_11_production_rag\answerability_calibration.json `
  --memory-reconstruction-mode trained
```

Each JSONL request may include a structured `current_state` object. The result records reconstruction validity, decision payload, source IDs, extra prompt/generated tokens, and two-pass latency. `off` remains the backward-compatible default.


Run from the repository root in the activated Python 3.11 environment.

## Prerequisite: finish and approve Phase 9

Phase 9 must complete its SFT/KD comparison and have an explicit human `PASS` in `reports/PHASE_9_STOP_GATE.md`.

Materialize the exact approved Phase 9 candidate:

```powershell
python .\scripts\phase_10_student_architecture\finalize_phase_9_selection.py `
  --candidate kd_warm_344m `
  --role intermediate_teacher `
  --reason "Approved Phase 9 warm-profile KD student for architecture reduction."
```

Use the exact candidate key from `reports/PHASE_9_ASSESSMENT.json`. Legacy single-suite keys (`sft`, `kd`, `on_policy`, `preference`) and hybrid keys (`sft_warm_344m`, `kd_warm_344m`, and profile-qualified narrow controls) are all accepted. A candidate listed in `research_controls` must normally use `--role research_control`; promoting one requires the explicit `--allow-research-control-promotion` acknowledgement after separate language-readiness review.


### Hybrid Phase 9 handoff

Phase 10 does not assume a fixed Phase 9 candidate vocabulary. The handoff validates the requested key against the assessment packet and records whether it was the automated recommendation, a production-track candidate, or a research control. This prevents the new hybrid profile names from conflicting with the Phase 10 reference manifest.

## Phase 10 — pilot architecture search

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model
```

Pilot mode:

1. validates Phase 9 approval;
2. evaluates the selected student reference;
3. profiles residual/MLP/attention usage;
4. builds the bounded pilot structural matrix;
5. fresh-reloads every candidate;
6. evaluates every raw candidate;
7. recovery-distills the bounded top set under the original teacher;
8. reevaluates recovered candidates;
9. writes the Phase 10 registry, report, and pending gate.

Resume safely:

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReuseExisting
```

## Phase 10 — full ladder and optional compact challenger

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 `
  -Mode Full `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -MaxSteps 1000 `
  -RecoveryTopK 4
```

Optional pretrained compact challenger:

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 `
  -Mode Full `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ChallengerSource "HuggingFaceTB/SmolLM2-135M-Instruct" `
  -ChallengerName smollm2_135m `
  -ReuseExisting
```

The challenger starts with sequence-level SFT because its tokenizer differs from Qwen. It is compared on the same closed-book, oracle-RAG, real-RAG, and adversarial-RAG conditions.

## Phase 10 human decision and handoff

Review:

- `artifacts/phase_10_student_architecture/candidate_registry.json`
- `reports/PHASE_10_CANDIDATE_REPORT.md`
- raw and recovered `results.jsonl` files;
- each `phase_10_candidate_manifest.json`;
- `reports/PHASE_10_STOP_GATE.md`.

Set the gate's human decision to `PASS` only after naming the exact candidate. Then materialize it:

```powershell
python .\scripts\phase_10_student_architecture\finalize_phase_10_selection.py `
  --candidate recovered_combined_small_target `
  --role deployment_candidate `
  --reason "Best measured quality-per-byte candidate after recovery."
```

Use the exact registry name. A raw candidate, recovered candidate, compact challenger, or retained Phase 9 reference may be selected only when represented in the registry and human-approved.

## Phase 11 — prepare immutable knowledge and answerability policy

Replace sample knowledge/procedure paths with production-curated files before a release run.

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Prepare `
  -Knowledge .\data\phase_9_student_distillation\knowledge.sample.jsonl `
  -Procedures .\data\phase_9_student_distillation\procedures.sample.jsonl
```

Preparation builds a hashed immutable snapshot and calibrates the answerability threshold on the development set. Rebuild calibration whenever the snapshot, query normalization, filtering, or top-k policy changes.

## Phase 11 — dense baseline and deployment candidate

Dense negative-control release experiment:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Full `
  -QuantizationBackend dense_copy
```

Bitsandbytes NF4 experiment:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Full `
  -QuantizationBackend bitsandbytes_nf4 `
  -ReuseExisting
```

llama.cpp GGUF experiment:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Full `
  -QuantizationBackend llamacpp_gguf `
  -LlamaCppConvertScript C:\path\to\llama.cpp\convert_hf_to_gguf.py `
  -LlamaCppQuantizeExe C:\path\to\llama-quantize.exe `
  -LlamaCli C:\path\to\llama-cli.exe `
  -GgufType Q4_K_M `
  -ReuseExisting
```

The Phase 11 comparison measures the complete request path, including retrieval, answerability, generation, post-grounding replacement, index bytes, current/peak process memory, VRAM, and latency.

## Phase 11 local production process

After preparation, run one question:

```powershell
python .\scripts\phase_11_production_rag\serve_production_rag.py `
  --model .\artifacts\phase_11_production_rag\deployment_artifact `
  --knowledge-db .\artifacts\phase_11_production_rag\knowledge_snapshot\knowledge.sqlite3 `
  --answerability-calibration .\artifacts\phase_11_production_rag\answerability_calibration.json `
  --question "When will Nookumi be released?" `
  --plain
```

Persistent JSONL process for desktop integration:

```powershell
python .\scripts\phase_11_production_rag\serve_production_rag.py `
  --model .\artifacts\phase_11_production_rag\deployment_artifact `
  --knowledge-db .\artifacts\phase_11_production_rag\knowledge_snapshot\knowledge.sqlite3 `
  --answerability-calibration .\artifacts\phase_11_production_rag\answerability_calibration.json
```

Write one JSON object per stdin line:

```json
{"id":"request_1","question":"When will Nookumi be released?","mode":"grounded_factual"}
```

The process returns one JSON object containing the answer, route, evidence/procedure IDs, gate reason, grounding action, timings, and memory measurements. Use `--plain` for answer-only interactive output.

## Phase 11 human approval and release package

Review:

- `artifacts/phase_11_production_rag/release_candidate_registry.json`
- `reports/PHASE_11_RELEASE_CANDIDATE_REPORT.md`
- `reports/PHASE_11_STOP_GATE.md`
- both dense and deployment `results.jsonl` files;
- retrieval misses and post-grounding replacements;
- quantization manifest;
- rollback loading.

After explicit `PASS`, package the exact registry candidate:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Package `
  -PackageCandidate deployment_candidate `
  -PackageMode copy
```

`manifest` mode packages provenance/evidence only. `copy` mode embeds the selected model and immutable knowledge snapshot.

## Valid negative outcomes

- No Phase 10 structural candidate survives: retain the Phase 9 student as the architecture result.
- The 80M–120M target is too damaged: approve an intermediate 100M–300M tier.
- The compact challenger loses: preserve it as a research control.
- Quantization loses after total-process measurement: release the dense student.
- Retrieval is the bottleneck: improve the index/reranker and recalibrate before changing the generator.
