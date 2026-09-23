# SOUP-MINI-001 Phase A — materialization results

Date: 2026-09-22. Machine-readable companion: `PHASE_A_BUILD_VALIDATION.json`.

**Status: all five registered primary soup artifacts are built and bitwise-verified.
No model was loaded, no token was generated, no row was scored. No quality, retention,
acquisition, equivalence or release claim is made or supported by anything below.**

## 1. Endpoint weights located and pinned

All four endpoints were found on this machine and every SHA-256 matches the value pinned in
`soup.py:PINS`. Nothing was inferred from a directory name.

| Endpoint | Directory | SHA-256 vs pin |
|---|---|---|
| `supa` | `Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze` | match |
| `titan` | `Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Experimental-1.0.0-devfreeze` | match |
| `titan_sparse` | `Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze` | match |
| `opallo` | `ChatBubble-ced-native-loader-qualification/Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze` | match |

**Opallo is not in the main checkout.** A sweep of `Chat-Bubble` found no Opallo directory at
all, and of the ten worktrees carrying an `Rineya-Opallo-1.0.0-devfreeze` folder, nine hold
metadata only. The single copy with real weights is the one above; its digest equals the pinned
`e726d6be...`, so it is the exact registered artifact irrespective of which worktree stores it.

`local_ft` was left unbound. No local fine-tuned child was enrolled, so the conditional arms
`local_parent_50`, `local_supa_50` and `local_opallo_50` were not built.

### Read-only discipline

Source digests were taken before any build, `soup.py` re-hashed every source after each build
(`sources_rehashed_after_build: true` in all five receipts), and all four were hashed again
afterwards. **All four are byte-identical and their mtimes all predate this session.** Every
new byte written by this session lives under
`experiments/titan_model_soup_mini_v1/`. Nothing in `Chat-Bubble`,
`ChatBubble-ced-native-loader-qualification`, or any other directory was created, modified or
deleted.

## 2. Preflight and test suite

`python -m pytest -q` → **124 passed, 0 failed** (66 inherited mechanics + 58 screening).

`python soup.py preflight --manifest manifest.local.json` → exit 0,
`CHECKPOINT_COMPATIBILITY_ONLY`. 310 tensors, 596,049,920 parameters, identical keys/shapes and
identical canonical config across all four roots, tied head verified by value where present.

One thing worth recording rather than glossing: Titan-Sparse ships a **different `config.json`
byte hash** (`25ace026...`) from the other three (`660db3b7...`). This is the expected
`torch_dtype` field — Titan-Sparse stores FP32, the others BF16 — and `canonical_config()`
ignores exactly that field, so the canonical bodies are identical. Not a mismatch.

## 3. The five arms

Built one at a time (~40 s each) into `local-runs/arms/`, FP32, coefficient exactly 0.5,
arithmetic `left + alpha*(right-left)`.

| Arm | Formula | Output SHA-256 (first 16) | changed elems vs left | ‖out − left‖ |
|---|---|---|---|---|
| `supa_titan_50` | 0.5·Supa + 0.5·Titan | `844ceb0c1d2d8649` | 123,933,332 | 0.401735 |
| `supa_sparse_50` | 0.5·Supa + 0.5·Titan-Sparse | `aef35a8df84060c7` | 78,654,720 | 0.347028 |
| `supa_opallo_50` | 0.5·Supa + 0.5·Opallo | `f0670c2f553fadcb` | 170,582,038 | 0.701913 |
| `titan_opallo_50` | 0.5·Titan + 0.5·Opallo | `2d7e85e0832870bc` | 180,395,926 | 0.807880 |
| `sparse_opallo_50` | 0.5·Titan-Sparse + 0.5·Opallo (**primary**) | `df5bf6c8c1c35c7b` | 206,624,315 | 0.782334 |

Each is 2,384,235,000 bytes with the five pinned sidecars byte-identical to `soup.ASSETS` and a
`config.json` carrying `torch_dtype: float32`. No `INCOMPLETE.json`, no `FAILED.json`, no stray
files in any output directory.

### Independent verification

The builder's own output is not evidence that the builder is right, so every arm was re-derived
with a **separate implementation sharing no code path with `soup.py`**: a hand-written
safetensors header parser, a hand-written BF16→FP32 decode (`uint16 << 16`, reinterpreted), and
numpy arithmetic — no torch anywhere.

Across all five arms, **2,980,249,600 element comparisons, 0 bitwise mismatches**, 0 non-finite
values, 0 header-layout problems (offsets contiguous, ascending in written order, payload exactly
filling the file, 8-byte aligned). The independently computed `‖out − left‖ / ‖right − left‖` is
**exactly 0.500000** for all five: these are true midpoints.

A float64 restatement of the primary arm's identity
(`output == supa + 0.5·dSparse + 0.5·dOpallo`) flagged 23 of 596,049,920 elements above a 1e-6
absolute tolerance. All 23 are layernorm weights of magnitude ≈ 44–48, and every one is **exactly
0.50 float32-ULP** — the float32 storage grid, not builder error. The bitwise test against the
implemented formula was already 0/596,049,920.

## 4. Weight-space geometry (diagnostics only)

Per `EVIDENCE.md` item 5, everything in this section is a diagnostic. Small norms, near-zero
cosines and chance-level sign agreement are **not** evidence of functional independence, stable
downstream behaviour, lossless composition, retention, or broad capability.

Deltas against exact Supa: ‖dT_stored‖ = 0.8035, ‖dSparse‖ = 0.6941, ‖dOpallo‖ = 1.4038.

| Pair | cosine | overlapping support | sign agreement |
|---|---|---|---|
| dT_stored, dOpallo | **0.00249** | 96,537,552 | 50.20% |
| dSparse, dOpallo | **0.00218** | 42,612,284 | 50.29% |
| dT_stored, dSparse | 0.74546 | 56,243,142 | — |

The two donor deltas are near-orthogonal in weight space with sign agreement at chance. As a
consequence the primary soup's displacement is essentially Pythagorean: measured 0.783692 versus
0.783013 for the exact-orthogonality prediction.

A fact worth flagging for whoever designs the evaluation: **`sparse_opallo_50` sits at 0.7837
from Supa while stored Titan alone sits at 0.8035** — 97.5% of the same distance — but in a
substantially different direction (cos = 0.332 between the two displacements). Same step size,
different bearing. That makes stored Titan a natural magnitude-matched reference for the primary
arm, which the registered arm table does not currently exploit.

Displacements from Supa: `supa_sparse_50` 0.3470, `supa_titan_50` 0.4017, `supa_opallo_50` 0.7019,
`sparse_opallo_50` 0.7837, `titan_opallo_50` 0.8096.

### Three collisions with the recorded census

These were computed from raw bytes with the independent reader, then compared against numbers
recorded before this session. All three agree:

| Measured here | Recorded |
|---|---|
| Titan-Sparse support = **exactly 0.250000** of a 220-tensor, 314,618,880-param footprint in blocks 8–27 | `CORE.json`: top-25%-magnitude support, 220 trained tensors, blocks 8–27 |
| Stored-Titan BF16 erasure over that footprint = **60.6084%** | `AMENDMENT_001`: **60.61%** |
| Whole trained tensors erased by BF16 = 220 − 196 = **24** | `AMENDMENT_001`: **24/220** |

This validates the reader and the delta arithmetic against known quantities. It is provenance
confirmation, not capability evidence.

One consequence is concrete and easy to get wrong: **dSparse's support is not a subset of
dT_stored's.** 22,411,578 of dSparse's 78,654,720 non-zero elements (28.5%) sit where the stored
BF16 Titan is exactly zero. Anyone treating stored Titan as a superset of Titan-Sparse, or
treating `sparse_opallo_50` as a masked version of `titan_opallo_50`, will be wrong.

### Norm-convention flag

‖dT_stored‖ measures **0.8035** here. The retention regression `retained ≈ 35.50 − 6.28·ln(‖dΘ‖)`
was fitted on a ladder whose dense trained delta has norm **1.8464**. It is not established that
this is even the same delta, let alone the same norm convention — and it probably is not: erasing
the *smallest* 60.61% of elements cannot take a delta from 1.8464 to 0.8035, because the surviving
largest 39.4% would have to carry only 19% of the squared norm. **Do not evaluate that regression
at the norms in this record** to predict retention for any arm. Identifying which delta the ladder
was fitted on is separate work; a borrowed scale is not a scale.

## 5. What was *not* done, and why

The five artifacts exist. They have not been evaluated, and I did not evaluate them. Phase A
scoring is blocked by gates that are not mine to open:

- **No admissible bank.** CAL/CHECK require fresh rows parent-disjoint from *every* fine-tuned
  ancestor. The existing banks are parent-contaminated against these donors
  (`BANK_SPLIT_IS_CONTAMINATION_20260920`: 275/384 overlapping parents, 128/128 on
  `prerequisite_plan`), and the confirmation instruments are spent. Re-splitting already-inspected
  rows is explicitly forbidden.
- **No qualified collector.** `score.py` consumes externally scored rows; no native runner exists
  in this package, and no collector code/runtime/decoding/tokenizer receipts are bound.
- **No frozen runtime or decoding identity.**
- **No whole-ancestry contamination audit.**
- **No SOUP-MINI-001 resource lease or native-read grant.** `protocol.json` still reads
  `native_authorized: false`.
- **No candidate lock.** `screen-freeze.local.json` is neither populated nor externally frozen.

Nothing above was relaxed, amended, or self-granted, and `protocol.json` is unmodified.

As a positive control that the selection gate can actually refuse rather than merely never being
reached: `python screen.py CAL --freeze screen-freeze.example.json ...` exits **2** with
`{"status": "REFUSED", "error": "unbound SHA-256"}` and creates no output file. The 58 screening
tests exercise both the passing and failing branches of the same rule.

## 6. Next action

The cheapest thing that can now fail informatively is **endpoint load-and-decode parity**: load
all four endpoints and the five arms through one FP32 path and confirm identical tokenization and
deterministic greedy decoding before any bank is built. That needs the native-read grant this
package does not hold. Building a fresh admissible CAL bank is the long pole and is a separate
registration, not a step inside this directory.

## Reproduction

```powershell
cd experiments/titan_model_soup_mini_v1
python -m pytest -q
python soup.py preflight --manifest manifest.local.json
python soup.py merge --manifest manifest.local.json --arm sparse_opallo_50 --output <NEW_DIR>
python local-runs/verify_independent.py     # bitwise re-derivation, no torch
python local-runs/geometry.py               # delta geometry
```

Environment actually used — recorded, and **not** the pinned set in
`requirements-mechanics.txt` (python 3.13.5 / torch 2.10.0+cpu / safetensors 0.7.0 /
numpy 2.3.5 / pytest 9.0.2):

| | this run | pinned |
|---|---|---|
| python | 3.11.15 | 3.13.5 |
| torch | 2.6.0+cu124 (CUDA present, unused) | 2.10.0+cpu |
| safetensors | 0.8.0 | 0.7.0 |
| numpy | 2.4.6 | 2.3.5 |
| pytest | 9.1.1 | 9.0.2 |

The drift is recorded rather than waived. It is immaterial to the bytes produced here because the
independent no-torch reimplementation reproduced every output element bitwise — but a native
evaluation must not inherit that conclusion, since it exercises code paths this run never touched.
