# Exact registered per-update retention bindings

[ClosedPartial001](evidence/retained_schedule_binding_reviewed_cpu_20261009/current/CLOSED_PARTIAL_SCHEDULE_001.json)
contains54 cumulative rows for the two27-update arms. Each row is derived by
walking the actual registered20-occurrence update group and its per-update
input/target counts, LR and exposure vector, then hashing the exact accumulated
row-ID prefix. No per-update dose is inferred from chunk or final totals.

The walk verifies all27 LR fractions, all9/18/27 chunk identities/local hashes,
counters and exposures, full540 occurrence/cohort hashes and matched parent
vectors. The actual producer `retained_binding` API agrees on all54 rows.

| Arm | Through9 input/target | Through18 input/target | Through27 input/target |
| --- | ---: | ---: | ---: |
| M1 | 42,592 / 1,916 | 85,422 / 3,747 | 130,562 / 5,614 |
| M2 | 42,476 / 1,882 | 85,309 / 3,759 | 130,574 / 5,615 |

Five authored/Root and independent controls pass. Resealed missing, duplicated,
reordered, token/exposure/LR and chunk mutations reject. [Preserved evidence](evidence/retained_schedule_binding_reviewed_cpu_20261009/ROOT_PRESERVATION.json)
records exact inputs, current producer pins, independent calculation and receipts.
Reads were registration metadata and selected source only: no training arrays,
prompt/target labels, models, banks or native execution.

Native trajectory/recipe/source/runtime/data/base/tokenizer identities, fresh
start/inventory attestations and GD policy/probe/parity remain pending. Their
fields stay `None`; the external schedule validator refuses this candidate.
The historical registration's data freeze is not current native qualification.
This finishes registered counter binding, while preserving closed admission.
