# Phase 10 data boundary

Phase 10 does not duplicate Phase 9 records by default. It consumes the human-reviewed, document-connected train/dev/test split under `artifacts/phase_9_student_distillation/pilot_dataset/` and writes activation profiles and candidates under `artifacts/phase_10_student_architecture/`.

Place only additional architecture-calibration records here when they are separately versioned and remain disjoint from the final test set.
