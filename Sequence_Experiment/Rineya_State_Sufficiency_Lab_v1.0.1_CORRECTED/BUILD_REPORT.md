# Build and Validation Report

**Release:** Rineya State Sufficiency Lab v1.0.1 CORRECTED  
**Build date:** 2026-08-19 UTC  
**Purpose:** Standalone training and evaluation repository for comparing full-context reasoning supervision with unordered compact-state transition supervision.

## Correction

The previously supplied archive, `Rineya_State_Sufficiency_Lab_v1.0_STANDALONE.zip`, was invalid: it was 22 bytes and contained zero ZIP entries. This release was rebuilt under a new filename and validated from both its source tree and a clean extraction of the final archive.

## Included payload

- Complete Python package and command-line interface.
- FULL and SPLIT LoRA/QLoRA training configurations.
- Local Transformers, OpenAI-compatible HTTP, replay, and deterministic reference inference backends.
- Direct-solve, oracle-transition, state-ablation, and autonomous generated-state rollout evaluations.
- Strict and lenient response parsing, typed deterministic scorers, paired statistics, reports, and audit sampling.
- PowerShell and Unix launch scripts.
- Documentation, JSON schemas, tests, and release-verification tooling.
- The complete paired dataset archive:
  - 48,000 source problems.
  - 189,922 split transitions.
  - Dataset archive SHA-256: `47febce0911e99c12ba374f51ba6989d18d8a2d13b5cdf85a36ee3df26c53880`.

## Executed checks

### 1. Automated test suite

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -o addopts='' -p no:cacheprovider
```

Result: **31 passed**.

### 2. Python syntax validation

All Python files under `src/`, `tests/`, and `tools/` were parsed with `ast.parse` without error.

Result: **44 files passed**.

### 3. Configuration validation

All distributed YAML and JSON configurations were parsed successfully.

Result at validation time: **18 YAML files and 4 JSON files passed**.

### 4. Command-line entry points

The package-level CLI and `doctor` command help paths executed successfully using `PYTHONPATH=src`.

### 5. Release-structure verification

`tools/verify_release.py` confirmed that required source, configuration, and dataset files were present and nonempty; the bundled dataset ZIP passed CRC testing and hash verification.

### 6. Complete dataset validation

The actual bundled 60,556,304-byte dataset archive was safely extracted and validated in strict mode.

Results:

- **48,000** unique source problems.
- **189,922** unique transitions.
- TRAIN/DEV/TEST counts matched the manifest.
- Every transition chain was aligned to its source problem.
- No transition-chain discontinuities were found.
- No adjacent same-problem rows were present in the globally shuffled files.
- File hashes matched.
- **0 errors; 0 warnings**.

### 7. End-to-end deterministic reference smoke test

The complete prepare → validate → evaluate → compare path was executed on a bounded sample using the deterministic reference backend.

Results:

- Direct-solve strict accuracy: **1.0000**.
- Oracle-transition strict accuracy: **1.0000**.
- Generated-state rollout strict accuracy: **1.0000**.
- Paired comparison verdict: **EQUIVALENT**.

This verifies the data, protocol, parsing, scoring, reporting, and comparison plumbing. It is not a scientific model result.

### 8. Final archive validation

After packaging, the release process additionally performs:

- ZIP CRC verification with `ZipFile.testzip()`.
- member-count and uncompressed-size checks;
- forbidden-artifact checks;
- clean extraction into a new directory;
- source-hash verification against `RELEASE_MANIFEST.json`;
- syntax, test-suite, CLI, release-verifier, dataset-validation, and reference-smoke reruns from the extracted archive.

The final archive verification report is distributed beside the ZIP so it can describe the immutable final archive without creating a self-referential hash cycle.

## Deliberately not claimed

A real LoRA or QLoRA adapter was **not** trained in this build environment. The runtime available for release construction did not include the optional Transformers/PEFT/Accelerate training stack or CUDA. Full adapter training must therefore be run on the target machine after installing `.[train]`. The repository does not present the deterministic reference smoke test as evidence for the research hypothesis.

## Evidence files

Exact machine-readable and textual source-tree outputs are retained under `release_evidence/`, including:

- `test_results.txt`
- `syntax_validation.txt`
- `config_validation.txt`
- `release_verification.json`
- `full_dataset_validation.json`
- `reference_smoke_test_report.json`

## Reproduction

Start with `README.md`, then follow `docs/RUNBOOK.md`. Before opening the sealed TEST split, freeze the base model, adapter settings, generation settings, parser version, primary metric, and non-inferiority/equivalence margin in `docs/PREREGISTRATION_TEMPLATE.md`.
