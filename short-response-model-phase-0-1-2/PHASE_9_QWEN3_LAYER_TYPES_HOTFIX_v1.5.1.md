# Phase 9 Qwen3 `layer_types` hotfix — v1.5.1

## Failure fixed

The accepted Phase 8 teacher contains 19 physical Qwen3 decoder layers and a
matching 19-entry `config.layer_types` list. Phase 9's default `depth_copy`
student retained 12 physical blocks and changed `num_hidden_layers` to 12, but
left `layer_types` at 19 entries. Recent Transformers validates these fields at
save time and rejected the checkpoint:

```text
num_hidden_layers (12) must be equal to the number of layer_types (19)
```

## Code change

`scripts/phase_9_student_distillation/create_student.py` now delegates physical
depth reduction to the repository's shared `reduce_model_depth()` helper, which
was already proven in Phases 2B and 4B. The helper:

- retains the matching `layer_types` entries;
- updates `num_hidden_layers`;
- reindexes layer and attention cache indices;
- validates unexpected per-layer metadata lengths;
- preserves the selected teacher-layer lineage in the Phase 9 manifest.

## Resume

No valid Phase 9 student checkpoint was published by the failed run. Rerun the
same Bootstrap command. The orchestrator will intentionally retry student
creation and continue to the knowledge-index and teacher-data steps.
