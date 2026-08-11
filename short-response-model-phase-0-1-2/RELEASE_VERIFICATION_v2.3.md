# v2.3 Release Verification

## Release identity

- Version: **2.3**
- Name: **Preservation-Constrained Residual Capability Distillation**
- Source repository ZIP: `Output(20260811-003531).zip`
- Source ZIP SHA-256: `f1863f2086c554721966de650bb0bbd75ab1026c4e38fb8f1fab3183277e9ab1`
- Governing implementation document: `RESIDUAL_CAPABILITY_V2_3.md`

## Source-lineage preservation

The v2.3 repository was created by normalizing and modifying the exact latest attached source archive. Relative to that archive, the final release is expected to contain:

- 777 source files;
- 764 source paths retained in common;
- 751 common files byte-for-byte unchanged;
- 13 intentionally modified existing files;
- 45 new v2.3 files (including this verification record, the changed-files record, and the package manifest);
- 13 removed files, all disposable `__pycache__/*.pyc` artifacts from the source archive.

No source/config/dataset/documentation file from the input archive is intentionally removed.

## Automated repository verification

Final pre-packaging commands:

```text
python -m compileall -q scripts tests
pytest -q
```

Result:

```text
277 passed
```

The test suite includes v2.3 checks for packaged corpus counts/structural holdout, topic-family split isolation, explicit-locator `query_aware_v23` fail-closed behavior while preserving `query_aware_v21`, answerable-gain preservation-aware selection, row-specific KL anchoring, and the end-to-end runner's fresh-sealed contract.

## Fresh corpus validation

Command:

```text
python scripts/phase_9_student_distillation/validate_residual_capability_v23.py \
  --direct-root data/phase_9_student_distillation/residual_capability_v23/direct_reasoning \
  --rag-root data/phase_9_student_distillation/residual_capability_v23/procedural_rag \
  --require-direct-rows 24000 \
  --require-rag-rows 7000
```

Result:

```text
status: PASS
direct_rows: 24300
direct_domains: 27
direct_operations: 12
rag_rows: 7200
rag_families: 8
problems: []
```

Additional packaged-data checks:

- direct corpus: **24,300 / 24,300 unique user prompts**;
- direct answerable rows: **20,250**;
- direct abstention rows: **4,050**;
- direct split counts: 11,340 discovery / 4,860 DEV / 4,860 OOD-DEV / 3,240 fresh sealed;
- fresh procedural split counts: 4,320 discovery / 1,080 DEV / 1,080 OOD-DEV / 720 fresh sealed;
- direct manifest hash audit: **7 tracked artifacts, 0 mismatches**;
- procedural manifest hash audit: **8 tracked artifacts, 0 mismatches**.

## Retrieval-policy verification

A temporary fresh v2.3 SQLite index was built from the packaged procedural knowledge/procedure files and calibrated using only the packaged discovery calibration subset.

`query_aware_v23` calibration result:

```text
minimum_retrieval_score: 0.34375
answerable cases: 400
unanswerable cases: 400
TP / FP / TN / FN: 400 / 0 / 400 / 0
supported recall: 1.0
unsupported rejection: 1.0
answer precision: 1.0
weighted cost: 0.0
research_ready: true
```

Fresh v2.3 procedural DEV support audit:

```text
supported cases: 720
support hits: 720
support recall: 1.0
misses: 0
status: PASS
```

This verification specifically protects the new missing-explicit-record behavior. Historical Phase 9/v2.1.1 runs retain `query_aware_v21` so v2.3 does not silently rewrite historical results.

## Sealed-data isolation

`data/phase_9_student_distillation/residual_capability_v23/SEALED_REGISTRY.json` records that the v2.2 sealed TEST is consumed. The v2.3 PowerShell runner never assigns historical `OriginalTest` or `V211Test` variables and its sealed stage references only:

1. `direct_reasoning/splits/fresh_sealed.reviewed.jsonl`;
2. `procedural_rag/splits/fresh_sealed.reviewed.jsonl`.

Fresh procedural sealed evaluation reuses the discovery-frozen gate calibration. No sealed recalibration, retraining, coefficient change, merging, or candidate switching is implemented.

## PowerShell orchestration verification

Windows PowerShell is not installed in the packaging environment, so the `.ps1` runner could not be executed here. It was checked statically and through repository tests:

```text
PowerShell lexical balance: PASS
Runner line count: 274
Placeholder ellipsis: absent
```

The runner is designed to execute the full discovery -> residual mining -> two residual trainings -> dose sweeps -> compositions -> DEV/OOD selection -> diagnostics sequence unattended. `-ReuseExisting` is the intended resume mode.

## Runtime dependency limitation

The packaging container does not include the project's full GPU training stack (`transformers`, PEFT/CUDA runtime), so GPU model fine-tuning and model-generation evaluation were not fabricated or claimed here. The release verifies deterministic data, retrieval, selection utilities, static orchestration, compilation, and repository tests. The user's environment must perform the actual v2.3 model experiment.

## Recommended first run

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_residual_capability_v23.ps1 `
  -V22RunRoot ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200" `
  -OutputRoot ".\artifacts\phase_9_residual_capability_v23" `
  -RequireCuda
```

Do **not** pass `-RunFreshSealedTest` on the first run. Review residual-mining, dose, selection, and diagnostics evidence first.
