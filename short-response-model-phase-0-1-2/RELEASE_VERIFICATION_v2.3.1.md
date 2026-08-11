# v2.3.1 Release Verification

## Release

**Short Response Model v2.3.1 — Grokking Dynamics Observatory**

Source of truth: `shortresponsev2.3revised.zip` supplied by the user.

Source archive SHA-256:

```text
535857065aab57c41480e300e98bdc9f6da1ee75fb43987cc214e8678471331c
```

v2.3.1 preserves the revised v2.3 repository and expands its existing diagnostic-completion helpers into an end-to-end delayed-generalization / capability-dose dynamics experiment.

## Verification performed

```text
python -m compileall -q scripts tests
pytest -q

285 passed
```

The packaged v2.3 direct/procedural corpus was revalidated:

```text
status: PASS
direct rows: 24,300
direct domains: 27
direct operations: 12
procedural rows: 7,200
procedural families: 8
problems: []
```

The new v2.3.1 variant builder was also exercised end-to-end on a synthetic residual/preservation fixture. It successfully produced deterministic training variants and train/DEV/OOD probes without access to a sealed path.

The v2.3.1 PowerShell orchestrator passed a delimiter/lexical static scan and an explicit sealed-path audit. Windows PowerShell is not installed in the packaging environment, so the `.ps1` runner itself could not be executed here.

## Important runtime statement

No GPU long-horizon fine-tuning was performed in the packaging environment. Therefore this release does **not** claim that grokking, delayed generalization, or any new capability improvement has already been observed.

The package implements the experiment needed to distinguish:

- capability-dose/readout thresholds;
- ordinary feature learning;
- memorization without abstraction;
- grokking-like delayed generalization;
- reproducible delayed generalization across seeds.

## Sealed-data verification

The v2.3.1 master runner has **no sealed-test switch** and contains no `fresh_sealed` path. Its validator and checkpoint/representation/procedure evaluators reject sealed benchmark paths.

The consumed v2.2 sealed evaluation and fresh v2.3 sealed generation remain outside this research branch.

## Storage/resume design

Long-horizon runs use two checkpoint layers:

1. a small rolling set of full Trainer checkpoints with optimizer state for interruption recovery;
2. lightweight LoRA adapter snapshots at every save event for temporal analysis.

This avoids keeping dozens of optimizer-state copies while retaining the full capability trajectory.

## Source lineage

The source archive contains 831 files. Before adding this verification document and package manifest, the cleaned v2.3.1 tree contained 832 non-cache files:

- 814 paths common with the source archive;
- 808 common files byte-for-byte unchanged;
- 6 existing files intentionally modified;
- 18 new v2.3.1 files at that point;
- 17 source-only paths removed, all disposable `__pycache__/*.pyc` files;
- **0 non-cache source files removed**.

After adding `RELEASE_VERIFICATION_v2.3.1.md` and `PACKAGE_MANIFEST_v2.3.1.json`, the final cleaned tree is expected to contain 834 files.

See `CHANGED_FILES_v2.3.1.md` and `PACKAGE_MANIFEST_v2.3.1.json` for exact details.
