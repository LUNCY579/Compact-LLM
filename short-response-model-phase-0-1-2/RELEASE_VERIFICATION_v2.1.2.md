# v2.1.2 Release Verification

This release is built from the user-supplied `Output(20260810-023627).zip` lineage and adds the broad general-reasoning experiment without removing v2.1/v2.1.1 recovery infrastructure.

Verification performed before packaging:

- v2.1.2 corpus generation completed deterministically.
- Static corpus validator passed.
- 13,200 canonical examples generated across 33 families.
- Train/DEV/TEST split: 10,560 / 1,320 / 1,320.
- Stage A view: 4,800 general-reasoning rows.
- Stage B view: 6,000 rows = 3,600 general reasoning + 2,400 train-only preservation replay.
- No chain-of-thought appears in assistant targets.
- Python compile check passed.
- Repository pytest suite passed: **259 passed**.
- Dataset manifest hash audit passed: **8 tracked files, 0 mismatches**.
- ZIP integrity and SHA-256 recorded during packaging.

No GPU model training was performed in the packaging environment. v2.1.2 is training-ready, not a claim that the new curriculum already improves the student.
