# Release verification — v1.7.4 memory reconstruction integration

This record distinguishes repository validation from unrun GPU/model-quality experiments.

## Dataset validation

- normalized memory records: 300
- canonical reviewed rows: 600
- topic families: 60
- train/dev/test: 440 / 80 / 80
- decisions: ACCEPT 96 / ADAPT 204 / REJECT 120 / CONFLICT 60 / INSUFFICIENT 120
- counterfactual pairs: 120
- curated preference pairs: 1,800
- evaluator candidates: 2,400
- topic-family split leakage: none detected by corpus validator
- normalized hash validation: PASS

## Evaluator calibration smoke

DEV candidate calibration:

- approved mean structural/semantic diagnostic: 1.0000
- verbatim replay mean: ~0.2255
- over-rejection mean: ~0.2212
- wrong-source mean: ~0.3087
- approved-vs-known-negative ordering violations: 0

This diagnostic is not a learned judge and is not a production promotion score; it verifies that the repository's bounded calibration features rank the known synthetic errors below their accepted targets.

## Dataset materialization

Research-ready build succeeds with authored split preservation and curated preferences:

- canonical: 440 / 80 / 80
- combined SFT rows: 1,320 / 240 / 240
- curated preferences: 1,320 / 240 / 240

The TEST preference file remains evaluation evidence and is not used by the official training runner.

## Code verification

See final release command output for:

- `python -m compileall -q scripts tests`
- full `pytest -q`
- ZIP CRC integrity
- SHA-256 of the delivered archive.

## Not claimed

No CUDA SFT, KD, DPO, Phase 9B strategy winner, Phase 10 compression result, Phase 11 runtime speedup, or human PASS is fabricated by this release. Those require the user's actual hardware runs and human stop gates.

## Final local verification result

- `python -m compileall -q scripts tests`: **PASS**
- `pytest -q`: **213 passed in 22.53s**
- dedicated corpus validator: **PASS**, 0 problems
- DEV evaluator calibration verifier: **PASS**, 0 ordering violations
- research-ready reconstruction dataset materialization: **PASS**
- PowerShell executable: unavailable in the packaging environment; `.ps1` contracts are covered by repository static/regression tests and manual argument/path inspection.
