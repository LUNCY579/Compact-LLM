# Frozen v3 statistical-design artifacts

Produced 2026-08-29, **before any confirmatory outcome exists**. Nothing here
reads a checkpoint, a prediction file, or any TGSA outcome; every number comes
from the outcome-blind simulator in `tgsa/phase1_simulation.py` driven through
the registered estimators in `tgsa/phase1_inference.py`.

| file | what it is |
|---|---|
| `power_audit_full_grid.json` | Full grid: 6 seed counts x 7 effects x 4 seed SDs, 10,000 trials/cell. Establishes the v2 conjunction gate's structural 50% ceiling. |
| `power_audit_seed_refinement.json` | Seed-count refinement: 11 counts x 3 effects x 2 seed SDs, 20,000 trials/cell, registered gate. Selects the replication level. |
| `inference_calibration.json` | Null calibration and coverage for all four estimators, plus the stratum-misspecification probe that explains the v2 ~7% figure. |

Narrative: `PHASE1_V3_POWER_REPORT.md`, `PHASE1_V3_CALIBRATION_REPORT.md`.
