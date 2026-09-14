# R3/R4: depth sweep and full-B28 checkpoint

The full B28 checkpoint is saved and independently hash-verified. It scores **73/150**, retaining **41/47** source successes. Its gains split **21 wrong-answer /11 format-only**; all six losses are wrong-answer losses.

These are finite-panel observations under the registered permissive scorer. They do not establish general capability or authorize an official release.

| Arm | Trainable parameters | Correct /150 | Retained /47 | Wrong-answer gains | Format gains | Runaway gains | Losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| B28last8 | 125,847,552 | 47 | 46 | 1 | 0 | 0 | 1 |
| B28last12 | 188,771,328 | 52 | 44 | 8 | 0 | 0 | 3 |
| B28last16 | 251,695,104 | 55 | 40 | 9 | 6 | 0 | 7 |
| B28last20 | 314,618,880 | 67 | 40 | 17 | 9 | 1 | 7 |
| B28 | 596,049,920 | 73 | 41 | 21 | 11 | 0 | 6 |

Every arm used seed 17, lr 1e-5, cosine warmup .03, beta2 .999, and 5,233 actual targets /41 updates. Tail arms freeze embeddings and final norm; full B28 trains them, so the full endpoint is not a pure layer-count intervention. All five fail the zero-loss retention rule.

## Registered paired comparisons

Exact two-sided McNemar tests use the same 150 IDs, with Holm familywise correction over the fixed ten comparisons. Five comparisons pass adjusted p < .05:

| Comparison | Better arm | Adjusted p |
|---|---|---:|
| B28last8~B28last20 | B28last20 | 0.00406608 |
| B28last8~B28 | B28 | 0.00012913 |
| B28last12~B28 | B28 | 0.00406608 |
| B28last16~B28last20 | B28last20 | 0.02508545 |
| B28last16~B28 | B28 | 0.00109005 |

Full B28 versus last-20 is not distinguished (adjusted p=0.57255745). Last-20 versus last-12 also does not pass (adjusted p=0.05336920). A higher observed total alone is not a demonstrated effect. These tests concern total typed correctness; the source-based acquisition strata above remain separately reported.

## Replication and preservation

Retry2 matches the historical 73 correct /41 retained /32 gained /6 lost aggregate. Retry1 and retry2 also match on all 150/150 final text-and-termination records. Historical row outputs and tensors do not exist, so neither observation establishes identity with the historical weights or general training determinism.

Two failed attempts remain preserved: the first stopped after one update on host RAM; retry1 completed the final probe but failed the full CPU-copy save. The explicit run registry selected retry2 for a saving-resource fix, with the known score recorded rather than hidden. The failed-save observation remains incomplete as a run and is not represented as a successful checkpoint.

The working execution uses bounded optimizer file I/O, a 1.5 GiB GPU workspace reserve, and unused-cache release. Saving releases optimizer state and writes at most 1 MiB of parameter data per chunk. Exact CPU/CUDA optimizer and safetensors roundtrip tests passed.

Saved checkpoint:

`experiments/architecture_benchmark_001/outputs/arm-checkpoints/B28_lr1e-05_b5200_seed17_retry2/trainable.safetensors`

- Parameters: 596,049,920; F32; full model.
- Bytes: 2,384,231,152.
- Independently verified SHA-256: `d96937545d2f5de88430044e689a96f2f57d4bdafdb6061a9e28f6160fd85e44`.
- Checkpoint reconstruction against all saved final answers is the alpha-one control in the next experiment.

## Audit corrections and remaining work

The inherited Hlast4 result passes only the registered permissive retention rule; strict scoring loses one reference answer. Its two gains are one wrong-answer and one format-only row. Hlast4 and B28last8 match on 146 correctness decisions but only 70 exact responses. The six-pair WSD comparison has no Holm-significant pair; the claimed simultaneity mechanism is not causally established. See AUDIT_2026-09-13.md for the artifact audit.

R5 interpolation is next: all five fixed alphas, endpoint controls first, no alpha selected on the panel. The in-domain dev split and first 60 selection rows are already frozen. Remaining soup candidates and the greedy soup procedure retain their preregistered recipes and held-out selection rule.

Evidence: outputs/DEPTH_FRONTIER_COMPLETE_20260913.json, run_registry.json, outputs/R4_RETRY1_SAVE_FAILURE_20260913.json, and the linked ARM_DOSE artifacts. The core implementation digest remains a329eb7bb06fc003a9220ea41713e721fba575c478008240749c425774b2b58e. Mechanical checks passed 306 package tests and 76 exploratory tests; these tests are not model-quality evidence.
