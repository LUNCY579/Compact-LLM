# Release verification — Phase 9 hybrid synthesis with Phase 10 and Phase 11

## Verification status

| Check | Result |
|---|---|
| Authoritative merge base | PASS — latest `Output(10).zip` Phase 0–9 source preserved |
| Python source compilation | PASS |
| Complete pytest suite | PASS — 159 tests |
| Phase 9 hybrid-profile regression coverage | PASS |
| Phase 9-to-10 arbitrary candidate-key handoff | PASS |
| Research-control promotion safeguard | PASS |
| Evaluator v4.5 comprehensive packet | PASS — 639/639 |
| Evaluator reporting categories | PASS — 31 |
| Evaluator scored labels | PASS — 63 |
| Real-collapse regression coverage | PASS — 150/150 |
| Relative-opportunity policy audit | PASS |
| Response-control verification | PASS |
| Phase 2B–11 runner-reference audit | PASS — 11 runners |
| Phase 11 sample snapshot build | PASS |
| Phase 11 sample answerability calibration | PASS |
| ZIP integrity | PASS — final archive tested with Python `zipfile.testzip()` |

## Phase 9 compatibility verification

The latest Phase 9 source adds a hybrid suite with one warm depth-copy profile and three narrow same-tokenizer research controls. Its candidate assessment keys are no longer limited to `sft`, `kd`, `on_policy`, and `preference`.

The synthesized Phase 10 handoff now:

- accepts any exact candidate key present in `reports/PHASE_9_ASSESSMENT.json`;
- supports names such as `kd_warm_344m` and `sft_narrow_130m`;
- normalizes relative model and adapter paths against the repository root;
- records whether the selected candidate was the Phase 9 automated recommendation;
- records whether the candidate belongs to the production track or research controls;
- rejects a research-control candidate assigned a deployment/intermediate role unless the reviewer supplies `--allow-research-control-promotion` deliberately.

The newer Phase 9 config, runner, gate, student creation logic, tests, and existing reports were preserved rather than replaced by the older Phase 10/11 source base.

## Functional source evidence

Phase 10 unit tests instantiate a toy decoder-only model and verify that MLP slicing, GQA-group slicing, and hidden-coordinate slicing reduce actual tensor shapes and total parameter count. Runner tests verify the canonical recovery naming, intermediate/small/deployment parameter bands, and hybrid Phase 9 handoff safeguards.

Phase 11 unit tests cover routing, calibrated false-answer preference, curated conflict handling, unsupported entity/number rejection, context-budget orchestration, and source completeness. A real SQLite FTS5 snapshot and answerability calibration were built from the bundled Phase 9 sample records. The sample calibration selected a threshold with zero false answers on that tiny smoke packet; this is implementation evidence only, not a production-quality claim.

## Reviewed integration corrections

- rebased Phase 10/11 onto the latest hybrid-capable Phase 9 repository;
- removed the fixed Phase 9 candidate-name restriction in the Phase 10 handoff;
- added explicit research-control role enforcement and acknowledgement;
- resolved candidate summary, model, and adapter paths consistently;
- updated root phase boundaries and execution documentation through Phase 11;
- updated Phase 10 workflow/manuals for profile-qualified Phase 9 keys;
- retained all prior Phase 10 physical slicing, recovery, challenger, and opportunity logic;
- retained all prior Phase 11 snapshot, routing, answerability, grounding, quantization, runtime, and packaging logic.

## Not executed in this packaging environment

The archive does not contain the user's accepted Phase 9 checkpoint, CUDA runtime, RTX 4070, bitsandbytes-supported Windows stack, llama.cpp executables, or production knowledge base. Therefore the following claims are intentionally **not** made:

- that any Phase 9 hybrid candidate should receive human PASS;
- that any Phase 10 candidate preserves sufficient quality;
- that the 80M–120M target is achievable for this teacher/data;
- that recovery KD fits within a particular VRAM budget;
- that NF4, INT8, or GGUF is faster or smaller in total process memory;
- that the production retriever has adequate recall;
- that Phase 10 or Phase 11 should receive human PASS.

Those results are produced only by the target-machine runs and their human gates.
