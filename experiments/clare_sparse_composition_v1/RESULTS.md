# CLARE-RINEYA-001 results — CPU mechanics only

Source commit: `76230adc4f7edc89b5992fbe93b2f5c6839e487b`.
Exact parent: `816a1e8f4d1ab1c9d1ab847697d962a141521bac`.
All 24 new source/config/test/document files match the GitHub source subtree
`91148ef0876ecdbb39ad0f3bfc0eae17056d99bf` byte-for-byte (Git tree equality).

## MEASURED

**87 tests passed, 0 failed, 0 errors, 0 skipped** in the recorded final CPU run.
Compilation, whitespace/diff check, three exact configs and result JSON schema
validation passed. This is the new package's CPU suite, not repository-wide or
remote CI validation. The tests cover all five arms and skill-boundary resume.

A separate synthetic nonlinear MLP run executed all five arms, four stand-in
skills each: 20 coordinates, seed 17, 4 discovery plus 8 retraining steps per
skill, 240 optimizer steps total. All 20 discovery resets were exact. D and E
allocated identical per-skill budgets `[4, 3, 3, 2]`. Across their 12 delta pairs,
maximum support overlap and absolute parameter dot product were both zero.
These are verified mechanics, not language-model retention/composition evidence.

`evidence/results.json`, `allocation_by_layer.csv` and `delta_geometry.csv` hold
those measurements. Empty retention/composition CSVs mean **NOT RUN**, not zero
accuracy. The synthetic command also saved five full non-pickle state files and
manifests with explicit masks and per-skill delta hashes to the local validation
output; these tiny checkpoint artifacts are included in the downloadable delivery
bundle, not promoted as Rineya weights or committed as model exports.

## BLOCKED / NOT MEASURED

Native preflight returned BLOCKED before any model, tokenizer or real dataset
load. This runtime is Linux, Torch 2.10.0+cpu; Transformers and Accelerate are
absent. Exact Windows lease host, paths, module pool, runtime binding and smoke
row are unbound. There were **zero native training steps** and no native accuracy,
retention, composition, throughput, VRAM peak or rank-structure measurement.
No whole-corpus audit, native smoke or full study was performed here.

The generic masked training engine is implemented, but a real skill objective
and matched native collector still need integration/qualification. Do not treat
missing host inputs or a passed mechanics suite as native experiment readiness.

## DERIVED and HYPOTHESIZED

Zero dot products follow from exact disjoint support. They do not imply independent
logits, positive transfer, retention or successful composition. A unit counterexample
explicitly verifies nonlinear interaction despite disjoint parameter coordinates.
Any native advantage of learned over random masks remains HYPOTHESIZED.

## Next single experiment

Run the bound one-row rollback-only native smoke described in `HOST_HANDOFF.md`.
Do not start the rho sweep or a full training run. After native loss and collector
qualification, the proposed first efficacy comparison remains A/D/E, rho=.80,
order R/U/S/P and repeated seeds 17/29/41.
