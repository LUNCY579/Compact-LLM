# Changed files — v1.7.4 Phase 9B memory reconstruction corpus integration

## New data

- `data/phase_9b_memory_reconstruction/reconstruction_corpus_v1/**`
  - normalized 300-memory / 600-example corpus;
  - authored train/dev/test splits;
  - 120 audited counterfactual pairs;
  - curated preferences and evaluator candidates;
  - source and normalized hash manifests.

## New Phase 9B utilities

- `scripts/phase_9b_memory_reconstruction/prepare_memory_reconstruction_corpus.py`
- `scripts/phase_9b_memory_reconstruction/validate_memory_reconstruction_corpus.py`
- `scripts/phase_9b_memory_reconstruction/verify_memory_reconstruction_evaluator.py`
- `scripts/phase_9b_memory_reconstruction/freeze_phase_9b_dev_selection.py`
- `scripts/phase_9b_memory_reconstruction/run_memory_reconstruction_preference_refinement.ps1`

## Modified Phase 9B implementation

- `scripts/phase_9b_memory_reconstruction/phase9b_common.py`
  - bounded validity-condition DSL;
  - state-aware deterministic reconstruction;
  - conservative unknown-state handling;
  - bounded reconstruction-candidate diagnostics.
- `scripts/phase_9b_memory_reconstruction/build_reconstruction_dataset.py`
  - authored split preservation;
  - curated three-per-row DPO preferences;
  - separate generated preference fallback.
- `scripts/phase_9b_memory_reconstruction/evaluate_memory_reconstruction.py`
  - flexible functional contracts;
  - safe conflict handling vs explicit conflict recognition;
  - scenario, negative-transfer, and counterfactual metrics.
- `scripts/phase_9b_memory_reconstruction/check_phase_9b_gate.py`
  - counterfactual/negative-transfer benefit components;
  - locked DEV strategy for sealed TEST.
- `scripts/phase_9b_memory_reconstruction/run_phase_9b.ps1`
  - dedicated corpus default;
  - corpus/evaluator validation before expensive work;
  - SFT → optional KD → optional curated DPO;
  - DEV selection → freeze → sealed TEST.
- `configs/phase_9b_memory_reconstruction.json`
  - v1.7.4 corpus, DSL, split, evaluation, and handoff contract.

## Downstream integration

- `scripts/phase_10_student_architecture/run_phase_10.ps1`
  - validates the reconstruction corpus when consuming a Phase 9B reference.
- `configs/phase_10_student_architecture.json`
  - records the Phase 9B corpus/handoff contract.
- `scripts/phase_11_production_rag/benchmark_production_runtime.py`
  - measured optional reconstruction pass;
  - records validity, latency, prompt tokens, generated tokens, and full request cost.
- `scripts/phase_11_production_rag/run_phase_11.ps1`
  - validates the reconstruction corpus;
  - exposes `-MemoryReconstructionMode` and `-ReconstructionMaxNewTokens` to the benchmark.
- `configs/phase_11_production_rag.json`
  - v1.7.4 trained/prompted/off runtime contract and reconstruction metrics.

## Tests and documentation

- `tests/test_phase_9b_memory_reconstruction_v174.py`
- `PHASE_9B_MEMORY_RECONSTRUCTION_CORPUS_INTEGRATION_v1.7.4.md`
- `CHANGED_FILES_PHASE_9B_MEMORY_RECONSTRUCTION_v1.7.4.md`
- `RELEASE_VERIFICATION_PHASE_9B_MEMORY_RECONSTRUCTION_v1.7.4.md`
- `README.md`
- `MEMORY_RECONSTRUCTION_PIPELINE.md`
- `PHASE_10_11_WORKFLOW.md`
- `scripts/phase_9b_memory_reconstruction/EXECUTION_MANUAL.md`
