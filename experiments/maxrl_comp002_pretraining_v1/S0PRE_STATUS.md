# COMP-RS-001 pre-S0 CPU qualification status — 2026-09-27

**Status: CPU mechanics only.** This is not a training grant, not native qualification and not S0.
- **No model activity.** No model or tokenizer was loaded, nothing was generated, and no GPU or GPU lease was used.
- **No sealed data.** No TEST, CHALLENGE, guard_test or guard_challenge file was opened; the open log below
  records this.

Governing text: `COMP_RS_001_AMENDMENT_1.md`, §§1–3, 8, 9, 11, 12 and Clarifications 1–9 (§15).

## What exists

| file | amendment | what it does |
|---|---|---|
| `scoped_scorer.py` | §8.0, C8 | **Scorer.** Strict scoring comes from the unchanged `corpus.reward`. Scope receipts are mandatory for CALIBRATION, DEV and TEST, and every signature is bound. Boundary rows are refused unless the call is flagged diagnostic. The code bundle covers the scorer's whole 14-file import closure and is checked against `source_lock.json`. |
| `content_scoring.py`, `failure_taxonomy.py` | §8.4, C4–C6 | **Content scoring.** Adds the legacy content path, the `multiple_candidates_in_object` rejection, the strict-failure subtyper and the RS response classifier. |
| `canary.py` | §8.5 | **Canary.** Runs through the single scoring entrypoint and covers every reference index, every applicable wrapper per signature, derangement, lookup-blind certificates and all 8 legacy families. It takes a frozen reference receipt and an expected code bundle; the legacy section is required unless an exemption is recorded. |
| `stats_exact.py`, `h3b_null_cell.py` | §3, §9, C1–C3, C5, C7 | **Statistics.** Exact one-sided McNemar and Clopper–Pearson, nd_min, and Holm with m pinned at 12 and GATE_FAILED. Also the fixed sequence, π_blind/e, `h3b_candidate`, `h3c_state`, and the sign-flip and design-randomization tests. The null cell accepts only registered counts (≥ 2,000 re-draws, 10^6 flips), and role test is refused before any file is opened. |
| `arm_build.py` | §2 | **Arm builder.** FP32 deltas, a streamed arm build in registered order, arm digests, a STOP on rebuild-digest mismatch, and a delta writer/reader with a data-section digest. Tested on synthetic tensors only. |
| `data_gates.py`, `scope_receipt.py` | §11, §1, C9 | **Data gates and scope receipts.** Spec-only data gates; at freeze on TEST only the floor and the triple structure are computed. Create-only scope receipts for CALIBRATION, DEV (R, S, P, RS) and TEST (`--spec-only-ack`). |
| `s0pre_evidence.py` | DS-8 | **Evidence runner.** Regenerates every receipt below in-process, under an audit hook that refuses sealed filenames, and writes the open log. |
| `tests/conftest.py` + `tests/test_s0pre_*.py` | — | **Tests.** A sealed-file guard (wrappers plus an audit hook, covering any path under the data root) and the fixtures. |

## Evidence (create-only, `evidence/`)

| receipt | sha256 (prefix) | headline |
|---|---|---|
| `S0PRE_TESTS_WINDOWS.txt` | `ddac487180cdf961` | Default python: **486 passed**; with the dataset absent, 460 passed and 26 skipped. `.venv-mq001`: **485 passed, 1 skipped** (the optional scipy cross-check); with the dataset absent, 459 passed and 27 skipped. |
| `S0PRE_OPEN_LOG.json` | `79dbe122f9545b79` | **PASS.** 11 expected data files opened, **0 sealed files opened**, 0 refusals, 0 writes to the data root. The sealed-route positive control refused 12/12. |
| `S0PRE_CALIBRATION_SCOPE.json` | `1acd416dea7bcf61` | Core R, S and P, 150 rows each, on the same 150 parents; no RS. |
| `S0PRE_DEV_SCOPE.json` | `89af04f58f073b0c` | Core R, S, P and RS, 250 rows each, on the same 250 parents. R, S and RS equal `Q1_DEV_SCOPE.json`; P is added for the H1 P gate. |
| `S0PRE_DATA_GATES.json` | `1a1be2152d43c74f` | DEV lookup-blind floor 0.444 at k=2 (84 rows) and 0.293 at k=3 (166 rows); first-lex blind subset passes 83/250. S→RS leak 62/250 and 33/250. S shortcut TRAIN 424/3000, CALIBRATION 19/150, DEV 36/250. Overlap DEV S 1/250, RS 5/250. Triple structure PASS. |
| `S0PRE_CANARY_DEV.json` | `e8ded988ff4dc381` | **PASS**: R, S, P and RS plus guard_dev legacy. Every composition signature is complete, with only bare UNKNOWN NOT_APPLICABLE. Code bundle `df351b69…`. |
| `S0PRE_CANARY_CALIBRATION.json` | `4df404e7e503db7f` | **PASS**: R, S, P plus legacy, bound to the CALIBRATION scope receipt. |
| `S0PRE_CANARY_GUARD_DEV.json` | `623a553309cadfce` | **PASS**: 8 families. |
| `S0PRE_H3B_NULL_CELL_DEV.json` | `100f0248dde53714` | At the registered counts (2,000 re-draws, 10^6 flips, seed 20260926), false rejection is 0.0435 (upper bound 0.052) and 0.0465 (upper bound 0.055). Rule-compliant; outcome SIGNFLIP_REFERENCE. |

The superseded receipts from the pre-fix code were moved, not deleted, to
`C:\Users\clclc\BusinessProject\s0pre_superseded_evidence_20260927\`, with a SHA256SUMS file.

## Review record

**Second review round.** Four lenses (spec, test adequacy, data safety, numerics) reported 9 MAJOR and 36 MINOR
findings against the first implementation. All were closed. Four decisions were made pre-data and recorded as
Clarifications 1–9.

**What the orchestrator verified first-hand:**
- the full import-closure code bundle;
- mandatory scope receipts and boundary refusal;
- the H4 state logic under C1/C2;
- the null-cell count enforcement and the role-test refusal;
- `h3b_candidate` and `h3c_state`;
- the canary's reference-receipt, code-bundle and legacy-exemption parameters;
- two mutations that previously survived the whole suite (dropping the R_BLIND_S_VALID sum check; moving the
  H3b switch bound to 0.10) are now killed.

## Not established here

- **Native behaviour of any kind.** Loader, generation, logprob parity, training and the GPU path of the arm
  builder are all S0 (§16).
- **TEST.** The TEST scope receipt is built separately before any donor read (§1).
- **Guard scope receipts.** Guard banks are bound by the manifest-checked bank sha (C8); guard scope receipts are
  future work.
- **Stage-runner wiring.** Running `score_with_canary` in the same pass as every scored batch, with the S1/S2 code
  bundle pinned at TEST, is S0 work.
