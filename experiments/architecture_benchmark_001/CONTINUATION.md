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

## The one finding that drives everything

**Neither conversion produces a usable model, and no gate in the ladder tested for it.**

| arm | CE | perplexity | pilot hits | behaviour |
|---|---:|---:|---:|---|
| B28 frozen source | 5.546 | 256 | **6/24** | works; answers 3, 5, 7, 9 |
| D40 ordinary width | 5.360 | 213 | 0/24 | degenerate `</think>` loop |
| C28 CED | **12.585** | 292,267 | 0/24 | token salad, never terminates |
| C40 CED | **12.960** | 425,015 | 0/24 | token salad, never terminates |

`ln(151936) = 11.93`, so both CED arms are **worse than uniform**. The ordinary control is
broken too, which invalidates the four-arm design rather than only the CED arms. Only the
frozen source works.

`G1_WIDTH`, `G1_CED_*` and `G2_CACHED_*` are all satisfied by a model emitting token salad:
they test tensor accounting, save/reload, parameter counts and cache-order parity. Parity is
self-consistency, not quality — D40 passes the full parity grid with **zero** violations
while being degenerate.

### The governing law for any KV-sharing scheme

Call a layer *displaced* if it attends to a K/V bank it did not produce. Then CE is a
function of the displaced **count**, essentially independent of topology, and every removed
bank displaces exactly one layer by construction:

| scheme | displaced | CE | KV ratio |
|---|---:|---:|---:|
| source / local group 1 *(control)* | 0 | 5.5458 | 1.000x |
| ced last 4 | 3 | 6.4065 | 0.893x |
| ced last 8 | 7 | 8.3245 | 0.750x |
| ced last 14 | 13 | 12.5854 | 0.536x |
| local group 2 | 14 | 12.9136 | 0.500x |
| local group 4 | 21 | 16.8612 | 0.250x |

Two structurally unrelated schemes land on one curve at 13 and 14 displaced layers, at
roughly **0.5 nats per bank removed**. Local cross-layer sharing was hypothesised to beat
CED and is **falsified** — it is worse at matched budget. There is no clever topology to
find; only training can move the frontier.

## The earlier parity work, in its proper place

Still true, still correctly measured, but secondary: both CED arms fail strict FP32
cache-order parity (401,863 and 726,228 violations) while D40 passes with zero. Cause
located — divergence starts at encoder layer 0, the shared bank inherits dV = 4.6e-04
before any decoder read, and CED forfeits the final-RMSNorm suppression a plain model gets.
FP64 is clean for both arms (3.8e-13 / 4.1e-12); `causal_suffix_isolation_exact` and
`decode_choices_exact` hold everywhere, so there is no causality or cache bug. The
read-side precision fix (HOTFIX-002) was implemented and **falsified**. All of it is a
hygiene property of a model at perplexity 292,267, so the ordering of work was wrong.

## What is NOT established

- No efficacy, latency, memory, throughput, 8-GB-fit or deployment claim.
- No arm has been compared to another **after training**. No recovery or distillation
  result. Every quality number is a single deterministic CPU run, no seeds.
- The generation probe is 8-24 open pilot prompts of one narrow arithmetic family: a smoke
  test, not a capability benchmark.
- All quality/architecture diagnostics live in `experiments/rab001_exploration/`, outside
  the hashed root, so they produce **no receipts** and open no gates.
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

The parity question is no longer the blocker and should not be resumed. The open question
is **trainability**: can any of these initializations be optimised at all? A bounded
overfitting probe (`trainability.py`, 25 rows, AdamW) is running against C28; if CE moves
substantially the problem is budget, and if it barely moves the design has an optimisation
pathology.

Then, in order:

1. **Add a quality gate** before any further architecture work. Forward-only, on the
   already-open calibration/pilot splits: bounded CE increase, bounded KL to the frozen
   source, plus nondegeneracy (termination rate, distinct-output count). It would have
   failed D40, C28 and C40 immediately. This is the single highest-value protocol change.
2. **Retire D40 as the ordinary control.** A degenerate control supports no comparison; the
   frozen source is the only valid reference today.
3. **Stop the CED-by-conversion arm** on capability grounds, independent of parity.
4. **Do not pursue another sharing topology** — the exchange rate is a property of sharing
   itself, confirmed by two unrelated schemes on one curve.
5. If KV cost is the real target, the honest options are a design trained from the start, or
   orthogonal methods that do not repurpose these weights (quantised or paged KV), which are
   outside this experiment's registered scope.
6. If capability is the target, the function-preserving zero-gated augmentation
   (`gated_memory.py`) is the only variant that starts at source quality — but it **adds**
   KV rather than saving it, and its value is as a training scaffold, not a recovery target.

Do not run a 1M-token recovery on this hardware: 31 h/arm measured, and from CE 12.6 it
would be training a new architecture rather than recovering one.

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
