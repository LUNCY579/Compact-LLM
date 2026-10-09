# Strict scientific and retained checkpoint wrappers

The reviewed wrappers now validate exact integer schema version1 and matching
outer/inner kinds separately for scientific and retained archives. Scientific
metadata uses the actual17-field production schema and rejects retained-only
fields. Shared byte/cursor/source/permission checks no longer change a retained
descriptor's kind to reuse scientific validation.

Resume descriptors, envelopes and retained records now reject booleans, floats,
strings, missing or unsupported schema versions. Final adapter restore checks
the scientific metadata before archive reads/deserialization and again when
validating the loaded payload. The strong backend restore, scientific9/18/27
boundaries, full state, transaction, reference and cumulative accounting checks
remain intact.

[Preserved correction](evidence/study_retained_schema_reviewed_cpu_20261009/ROOT_PRESERVATION.json)
contains the frozen source, predecessors, independent review and CPU receipts.
All21 authored and independent controls pass:8 core,7 terminal and6 targeted
schema controls. Root's targeted6 replay passes. No corpus, bank, framework,
model, native resume or export parity work occurred. The legacy authored scalar
fixture gains its two missing inventory fields; an AST-only check confirms17
fields without executing its data-bound setup.

The [original point-in-time finding](evidence/retained_snapshot_kind_finding_20261009/ROOT_PRESERVATION.json)
and earlier successful controls remain preserved at their exact source pins.
This resolves wrapper validation gaps; it does not establish native training,
GD export parity or the M0/M1/M2 transfer/retention result.
