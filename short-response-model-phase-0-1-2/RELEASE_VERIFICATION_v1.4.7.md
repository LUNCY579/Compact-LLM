# Release Verification — v1.4.7

## Source reviewed

- Attached repository: `Output(6).zip`
- Detected baseline: v1.4.6 Incremental Reporting
- Review focus: Phase 2C correctness, Windows reliability, resume semantics, artifact integrity, and downstream handoff safety

## Automated verification

- Full Python compilation: PASS
- Full pytest suite: 132 tests PASS
- Evaluator v4.5 calibration verification: 639/639 PASS
- Evaluator coverage: 31 reporting categories, 63 scored labels, 150/150 real collapse cases
- Relative-opportunity policy audit: PASS; no absolute promotion floors found

## Targeted regression coverage

The release verifies:

- transient atomic replacement errors are retried;
- progress-sidecar failure does not discard completed evidence;
- equivalent path spellings resume successfully;
- incompatible manifests still fail with explicit differences;
- generation and teacher-forcing resume are propagated;
- malformed JSON sentinels are quarantined;
- manifestless generation evidence is rejected, while legacy Phase 2C NLL rows require the exact approved contract-only migration and are then manifested;
- evaluator script hashes are recorded in new generation and NLL manifests;
- duplicate or partial output-ID coverage is rejected;
- v1.4.6 Phase 2C contract migration accepts only known hashes;
- unrelated run-config changes remain rejected;
- saved wrong arithmetic replies are rescored below acceptable quality;
- completed summaries are reopened when migration rescoring is required;
- the PowerShell wrapper validates and builds the selected sibling dataset directory;
- deterministic objective rows require explicit strict contracts.

## Manual/GPU evidence still required

This packaging environment did not retrain the user’s Phase 2C candidates. After installation, the existing run must complete an identical `-Resume` execution so saved replies are rescored and remaining candidates/seeds finish. The final Phase 2C artifact audit, report review, and human stop gate remain mandatory.

## Packaging policy

The full source ZIP excludes generated `artifacts`, `playground_artifacts`, caches, bytecode, and nested ZIP files. The hotfix ZIP contains only changed/new v1.4.7 files and is intended to be extracted over the existing repository while preserving artifacts.
