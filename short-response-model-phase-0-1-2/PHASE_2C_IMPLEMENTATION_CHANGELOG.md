# Phase 2C Implementation Changelog

## Added

- Complete Phase 2C PowerShell runner and Python orchestration.
- Deterministic concise-reasoning augmentation with disjoint entities, grammar handling, leakage checks, deterministic shuffle/interleaving, and SHA-256 manifests.
- Surviving-layer first-order Taylor sensitivity, temporary layer-bypass target-NLL ranking, and optional final-stage Phase 2B structural evidence.
- Required all-remaining, top-k, complementary, random, low-ranked, attention-only, and MLP-only controls.
- Strict candidate comparison against both all-surviving-layer recovery and the frozen reduced model.
- Category-level heuristic, semantic, lexical-reference, completion, critical, and major metrics in the shared evaluator.
- Same-seed all-remaining controls and iterative repeated-seed validation for every final recommended profile.
- Trainable-parameter, adapter-size, training-VRAM, runtime, generation, NLL, semantic, category, and output-deviation evidence.
- Exact resume configuration plus model/data/catalog/registry/benchmark SHA-256 fingerprints.
- Human-gated model catalog, hardened Phase 3 profile resolver, automated evidence checker, read-only artifact auditor, and regression tests.

## Modified

- Phase 1 SFT manifests now record trainable/total parameters, serialized adapter bytes, LoRA/training settings, and an optional run label.
- The evaluator now publishes category semantic-pass, reference-F1, completion, critical-case, and major-case metrics while preserving existing fields.
- Phase 2 runner can invoke `TargetedRecoveryOnly` after Phase 2B approval.
- Phase 3 accepts mutually exclusive Phase 2B or Phase 2C profiles and writes isolated branch artifacts.
- Root workflow documentation and source verification include Phase 2C.

## Evidence boundary

Source-level compilation and unit tests are included. GPU training, model quality, VRAM, and runtime measurements must be produced on the target Windows/CUDA machine; no benchmark result is fabricated by this package.
