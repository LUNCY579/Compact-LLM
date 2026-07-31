# Phase 2C Source Verification

## Verification performed

- `python -m compileall -q scripts tests` — **PASS**
- `python -m unittest discover -s tests -v` — **PASS, 31 tests**
- Phase 2C deterministic dataset regeneration — **PASS**
- Dataset structure, unique IDs, assistant targets, source/output SHA-256 values, and exact train/dev prompt-target leakage check — **PASS**
- Generated counts:
  - train: 3,074 records (2,950 existing + 124 Phase 2C)
  - dev: 570 records (510 existing + 60 Phase 2C)
  - exact prompt-target overlap: 0
- Default limited-prefix coverage:
  - first 64 dev records: 21 Phase 2C cases spanning all eight added categories
  - first 180 dev records: all 60 Phase 2C cases plus 120 original controls
- Modified PowerShell runner references — **38 local Python references checked, 0 missing**
- No generated `artifacts/` directory included in the source package.

## Regression coverage added

Tests cover deterministic data separation/interleaving, importance ranking, required control construction, strict quality/resource gates, category semantic collapse, new-major-issue rejection, profile selection, and human/seed-gated profile resolution.

## Not executed in this environment

- Windows PowerShell execution
- CUDA/QLoRA training
- Qwen checkpoint loading
- GPU VRAM or wall-clock benchmarking
- Actual Phase 2C candidate quality or human review

Those measurements must be produced on the target Windows/CUDA machine. This package does not invent training or benchmark results.
