# RAB-001 continuation file

Compact resume state. Written 2026-09-11. Read `PROGRESS_LOG.md` for the evidence behind
every claim here. Local files plus verified artifacts are the source of truth.

## Identity

| | |
|---|---|
| Worktree | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Branch | `research/ced-rab001-hotfix001-qualification-v1` |
| Branched from | `2f65f9e` (`research/ced-native-loader-qualification-v1`) |
| Commits | `8b042d4` baseline → `e0ec051` ladder → `e7f760b` engineering → `fb0e2b8` audit fix → `3a604d0` requal → `f77792c` HOTFIX-002 → `0da7073` negative result → `16cac5c` final requal |
| `implementation_sha256` | `fc44701e3f78d337d9a3a54ea06fdfb3251c4773d5bcf14dc3466a0505bec04d` |
| digests qualified | `eab80e9f` → `8d54f540` → `fc44701e`, all three run end to end, all agreeing to the element |
| Qualification runtime | `C:\Users\clclc\BusinessProject\.venv-rab001` |
| Source weights | `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze` |
| `audit_digest` | `97cd76af0fb2d6064c5cb954906ac493d21e40429cebe55f0bed0000b23d9ef8` |

Runtime: python 3.11.15 / Windows-10-10.0.26200-SP0 / torch 2.10.0+cpu /
transformers 4.51.3 / safetensors 0.7.0 / numpy 2.3.5 / tokenizers 0.21.4.
Tests: **186 passed, 0 skipped**. Hardware: RTX 4070 **Laptop** 8 GB (unused — CPU runs
only), 15.2 GB RAM, 113 GB free disk.

**`run.py:117` hard-fails any gate unless `transformers == 4.51.3`.** The shared
`Chat-Bubble/.venv` is `2.12.1+cu126 / 5.14.1` and will not run this experiment. Always use
`.venv-rab001`. Never create a venv under `experiments/architecture_benchmark_001/` —
`implementation_manifest()` globs `ROOT.rglob('*.py')` and would absorb it into the digest.

## Four-arm state

| arm | checkpoint / lineage | qualification at `eab80e9f` | numerics | engineering | next blocker |
|---|---|---|---|---|---|
| **B28** | frozen official source, no checkpoint | **G0 fresh PASS** 8/8 | FP32 4 violating elements total; FP64 clean (3.8e-13) | **`G2_CACHED_B28` PASS 9/9**, 358.5 s, 3,134 targets | none; it is the frozen teacher and the C28 control |
| **D40** | `outputs/ladder-eab80e9f-v1/G1_WIDTH/attempt-0001/` (from G1_IDENTITY, bitwise exact) | **G1_WIDTH fresh PASS** 7/7 | FP32 **0 violations on every path and row** | **`G2_CACHED_D40` PASS 9/9**, 319.9 s, 3,134 targets | not a *recovered* donor; recovery recipe unfrozen |
| **C28** | `outputs/ladder-fc44701e-v1/G1_CED_C28/attempt-0001/` (donor B28, layer 14, 596,311,424 params) | **FAIL** 12/15 | FP32 401,863 violations / 8.2e-4; FP64 clean (4.1e-12) | **BLOCKED** by failed parent | see CED disposition — no cheap fix exists |
| **C40** | `outputs/ladder-fc44701e-v1/G1_CED_C40/attempt-0001/` (donor D40, layer 20, 594,279,808 params) | **FAIL** 12/15 | FP32 726,228 violations / 5.9e-3 | **BLOCKED** by failed parent | same |

Status vocabulary: every PASS/FAIL above is a **fresh** result at the current implementation
digest, produced this session. The older `evidence/g*-official-supa-1.7.0/` receipts are
**historical** (schema 1, digest `dc212c9f`) and cannot qualify current code.

## The one finding that drives everything — now with its mechanism

CED is **mathematically correct but structurally forfeits the numerical margin** that lets
a plain model pass a strict 1e-4 elementwise logit gate in FP32.

- FP64: both arms clean (B28 3.8e-13, C28 4.1e-12), every path. No logic, causality,
  masking or cache bug. `causal_suffix_isolation_exact` is True for every row of every arm
  at every precision.
- FP32: D40 passes the identical grid with **zero** violations; C28/C40 fail with 401,863
  and 726,228.
- **CED creates no extra drift.** B28 per-layer chunk drift matches C28 encoder drift
  digit-for-digit (3.815e-06 at layer 00, 1.678e-04 at layer 13), and B28 final layer
  drifts *more* (8.8e-03) than the C28 encoder does (3.9e-03).
- **The difference is the path to the logits.** A plain arm's drift reaches the final
  RMSNorm, which suppresses the correlated component: 8.8e-03 hidden becomes 1.7e-04
  logit. CED freezes the *unnormalized mid-network* encoder output into a shared K/V bank
  (dV = 4.6e-04, already 4.6x outside tolerance) and re-injects it into all 28 decoder
  layers coherently, before the final norm, so it never receives that suppression.

### CED disposition: the cheap fix was tried and failed

`--memory-precision float64` (HOTFIX-002) raises accumulation for the shared-memory read.
Measured: C28 401,863 -> 399,698 (0.5%, max error worse); C40 726,228 -> **803,104**
(worse). Falsified. Precision at the read cannot recover information already wrong in the
bank, and raising the bank's *projection* precision cannot help either because its input
`h` is already perturbed. Detail in `evidence/HOTFIX_002_DIAGNOSTIC_RESULT.md`.

Three routes remain, all architecture-level, none taken:
1. Build the shared bank from a full-context forward instead of incrementally — changes
   incremental-decoding semantics and cost.
2. Run the encoder at higher precision — that is most of the model; a throughput/memory
   decision, not a tolerance tweak.
3. Prospectively amend the equivalence policy for architectures that expose a mid-network
   tensor to cache-order comparison. Must be written **without** citing this failure as
   its justification, and reviewed separately.

Each changes what a later CED-vs-ordinary comparison can claim, so this is a decision to
take deliberately, not a repair to apply.

`decode_choices_exact` passes on both CED arms — which is exactly why the rejected
argmax-based rule recorded them as PASS. Argmax agreement does not establish harmlessness.

## What is NOT established

- No quality, recovery, efficacy, latency, memory, throughput or deployment claim.
- No arm has been compared to any other on any task.
- No GPU/VRAM measurement at all. CUDA would be a different runtime identity and would
  require requalifying the whole ladder; devices were not silently switched.
- **A 1M-token pilot is not feasible on this machine as configured.** Measured ~9 supervised
  targets/s on CPU = ~31 h per arm, ~62 h per matched recipe; and a 1M-token teacher cache
  is ~4.1 GB against 15.2 GB RAM with a ~7 GB training process.
- G3/G4/G5 remain closed. `recovery-readiness` still reports BLOCKED with 18 missing items.
- The workflow audit of the package was **not a clean review**: 60 of 123 agents died on a
  usage limit. Its `receipt-integrity` and `data-tooling` dimensions completed verification;
  `ced-parity-gate`, `numerics`, `teacher-cache` and `cli-tests-g3` findings are unverified
  second-pass observations only.

## Data state

Local fixtures: `pilot.jsonl` 24 rows (open diagnostic, never training),
`calibration.jsonl` 156 rows, `train-smoke.jsonl` 25 rows. **No scientific recovery data.**

Candidate pool (local, no download needed):
`Chat-Bubble/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0` — 110,000 SFT
rows, 16,504 template families (none >8 rows), validation/leakage/near-duplicate reports all
PASS (0 near-dup hits at threshold 90, zero exact overlap with sealed/dev/RAG splits).

Open data questions, all of which must be settled before any budget is frozen:
1. **Supervision policy is unregistered.** Assistant-only gives ~454k shifted targets from
   the whole pool (1M unreachable); the delivered `prepare_data` emits no label mask and
   counts every window token as a shifted target, giving ~14.9M (1M needs ~7,400 docs).
   These are different experiments. Measured: mean 135.6 tokens/row rendered, mean 4.13
   final-assistant target tokens, 0/2000 rows exceed the 512 context.
2. **Human review is pending for every row.** A 520-row audit packet and 40,000-ID pool
   exist. `data_tools.py:67` accepts `review_status: "approved"` verbatim with no reviewer
   identity, timestamp or attestation — so that field is a declaration, not evidence.
3. ~~Leakage re-check against current sealed/dev~~ **CLOSED.** Re-ran against the live
   assets (1,019 distinct eval keys vs 124,000 corpus rows): **0 last-user-turn and 0
   full-prompt collisions**, using normalized hashes so no sealed content entered any
   training process. Exact-match only; near-duplicate evidence remains the corpus's own
   sampled audit.
4. v3 rows carry `messages`, not the flat `text` field `prepare-data` requires — a rendering
   step is needed either way.

## Queued code fixes (each changes the digest — fix, then requalify, then proceed)

1. ~~`review.py:28` legacy-key-only strict-parity reporting~~ **FIXED** in `fb0e2b8`,
   requalified in `3a604d0`; the audit now reports 401,863 / 726,228.
2. `ladder.py:101` derives the next attempt number from attempt *directories* only, while
   the attempt *log* is created with exclusive open — a child failing before `_directory`
   leaves a log that collides on retry.
3. `qualify_identity` (`run.py:276`) uses one `assert_close` on a single halves split over
   second-half positions; `G1_WIDTH` tests no cache parity at all. The ordinary arms are
   held to a materially weaker numerical gate than the CED arms. Unequal gates across arms
   is an experimental-design defect independent of the CED result.
4. `engineering.py:213` `matched_control_verified` is tautologically True for non-C arms.

Do **not** apply these while relying on the receipts in `evidence/ladder-fc44701e-v1/` —
they bind `fc44701e`. The pattern used twice this session works: change, run the suite,
requalify, and verify the numerics reproduce to the element before trusting the result.

## Single next action

**This is now a decision for the operator, not a repair.** Route (a) from the previous
handoff — higher-precision accumulation for the shared memory — was implemented, measured
and falsified. The mechanism is understood and no cheap numerical remedy exists.

Choose one of the three architecture-level routes above, or conclude that CED's
cache-order behaviour is acceptable as-is and that the gate measures a property CED cannot
have. That last option is legitimate but requires a prospective, separately reviewed
amendment — not a post-hoc relabelling of this run.

Ordinary B28/D40 work is independently eligible and may continue under any route. Note
that the ordinary arms are still held to a materially weaker numerical gate (queued fix 3),
which should be corrected before any cross-arm numerical claim is made.

## Exact safe resume commands

```bash
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
SRC=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
E=experiments/architecture_benchmark_001

# 1. Confirm identity before trusting anything (expect 186 passed, 0 skipped)
$PY -m pytest $E/tests -q
$PY -c "from experiments.architecture_benchmark_001.gates import implementation_digest as d; print(d())"
# expect fc44701e3f78d337d9a3a54ea06fdfb3251c4773d5bcf14dc3466a0505bec04d

# 2. Re-read the existing verdicts WITHOUT recomputing (nothing here re-runs a gate)
$PY -m experiments.architecture_benchmark_001 audit-evidence --root $E/outputs/ladder-fc44701e-v1
$PY -m experiments.architecture_benchmark_001 status --source $SRC
```

Re-running the ladder is **not** needed while the digest is unchanged; use a **new** `--out`
root if you do (`execute` refuses to reuse a root whose session context changed, and
`--resume` only continues an identical session). Never delete `ACTIVE.lock` without first
confirming no process is running.
