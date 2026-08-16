# Bundled Quality Evaluator v4.5 Provenance

The scoring implementation in this directory was copied from the user-supplied repository archive:

`Short_Response_Model_Full_v2.1.1_Preservation_First_Procedural_Recovery(1).zip`

Source files:

- `scripts/common/quality_evaluation.py`
- `scripts/common/observable_quality.py`
- `scripts/common/response_control.py`
- `configs/quality_evaluation.json`
- `configs/quality_issue_taxonomy_v4_5.json`
- `configs/response_control.json`

Only two packaging changes were made:

1. The internal import of `observable_quality` was changed to a relative package import.
2. The repository-root calculation and taxonomy path were adjusted to this standalone layout,
   including a package-resource fallback for wheel installations.

The scoring rules and version identifier remain v4.5. `checksums.json` records the packaged files. For bit-for-bit use of the user's official phase runner, configure `evaluation.backend: official_phase_v45`; that backend launches the original `scripts/phase_1_dense_sft/evaluate_model.py` from a user-specified official repository root.

`SOURCE_CHECKSUMS.json` records both the original archive hashes and the packaged hashes. Files whose hashes differ are limited to the documented import/path packaging adaptations; unchanged files are marked identical.
