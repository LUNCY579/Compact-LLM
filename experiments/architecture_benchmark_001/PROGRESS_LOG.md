# RAB-001 progress log (append-only)

Each entry records verified facts only. "Verified" means produced by a command run in this
worktree and reproducible from the recorded paths. Prior chat summaries are not evidence.

---

## 2026-09-11 — Session start: state audit after CED_RAB001_full_hotfix001

### Environment established

| Item | Value |
|---|---|
| Worktree | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Starting commit | `2f65f9e96f8649a86c5d2f4369181df5b8b21aa5` |
| New branch | `research/ced-rab001-hotfix001-qualification-v1` |
| Baseline commit | `8b042d4` (applied package, verbatim, no local correction) |
| GPU | NVIDIA GeForce RTX 4070 **Laptop** GPU, 8188 MiB, driver 610.62, ~894 MiB already in use |
| Free disk (C:) | 113 GB of 927 GB |
| RAB-001 jobs running at start | none (checked `Win32_Process`; only Claude extension processes) |

### Pinned qualification runtime — BUILT (new, isolated)

The shared `Chat-Bubble/.venv` had drifted to `torch 2.12.1+cu126 / transformers 5.14.1`,
which is **not** the pinned qualification runtime and is shared with other experiments.
Rather than install over it (forbidden), a separate environment was created:

    C:\Users\clclc\BusinessProject\.venv-rab001

Deliberately placed **outside the repository**: `implementation_manifest()` globs
`ROOT.rglob('*.py')`, so a venv anywhere under `experiments/architecture_benchmark_001/`
would inject thousands of files into the implementation digest.

Resolved identity, matching `requirements-native.txt` exactly:

    python 3.11.15 | Windows-10-10.0.26200-SP0 | torch 2.10.0+cpu
    transformers 4.51.3 | safetensors 0.7.0 | numpy 2.3.5 | tokenizers 0.21.4

### Test suite — PASS, zero skips (requirement met)

    181 passed in 41.95s

against the vendor's own reported `162 passed, 2 skipped` (their Linux sandbox had
`transformers: not_installed`, so every native test was skipped there). All native tests
execute here.

### Implementation digest — verified platform-independent

    implementation_digest() = eab80e9f99494268a113fba381f4178818aab759e3acbc6193ea936d33560fe6

Identical to the vendor's Linux-computed `tested_implementation_sha256`. The hotfix's
POSIX-path-key change works as claimed. Manifest covers 28 files.

### Source audit — PASS on real weights

    audit_digest = 97cd76af0fb2d6064c5cb954906ac493d21e40429cebe55f0bed0000b23d9ef8
    parameters   = 596,049,920   real_weights_present = true

All seven registered assets matched `source_lock.json`. Weights live only at
`Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/` (the worktree has none).

---

### Finding 1 (BLOCKER, resolved by re-running) — no pre-existing evidence qualifies the current implementation

Every receipt under `evidence/` is **schema 1** and all six are self-hash-consistent:

    g0-official-supa-1.7.0/G0.json                   schema 1 PASS
    g1-identity-official-supa-1.7.0/G1_IDENTITY.json schema 1 PASS
    g1-width-official-supa-1.7.0/G1_WIDTH.json       schema 1 PASS
    g2-official-supa-1.7.0/G2.json                   schema 1 PASS
    g1-ced-c28-official-supa-1.7.0/G1_CED_C28.json   schema 1 PASS
    g1-ced-c40-official-supa-1.7.0/G1_CED_C40.json   schema 1 PASS

`gates.py` docstring: "Schema-1 receipts are preserved as historical evidence, not upgraded
in place." `review.py` stamps every record `reusable_as_current_qualification: False`.
All six carry `implementation_sha256 dc212c9f…`, not the current `eab80e9f…`.

**Consequence:** the *entire* ladder had to be re-executed. This is not a defect in the
package — it is the package working as designed after a code change.

### Finding 2 (BLOCKER for the CED arms) — shipped CED receipts use the acceptance rule the amendment rejects

`evidence/g1-ced-c28-*` and `g1-ced-c40-*` declare `"status": "PASS"` on checks named
`tail_only_within_scaled_tolerance` / `chunked_cache_within_scaled_tolerance`.

That string appears in **no Python file in the repository** — only inside those two JSON
files. The current code (`run.py:414`, `gates.py:110`) emits and requires instead:

    strict_tail_only_parity, strict_chunked_cache_parity, causal_prefix_parity,
    decode_choices_exact, pilot_unchanged, calibration_unchanged

So those receipts were produced by the **pre-hotfix** implementation, whose acceptance rule
`HOTFIX_001.md` explicitly rejects: "The uploaded CED acceptance rule used exact argmax plus
a tensor-global scale bound after strict elementwise failures were observed. This amendment
rejects that substitution."

Their own recorded diagnostics, read under the binding strict rule, are failures:

| arm | tail_only violations | chunked_cache violations | strict verdict |
|---|---:|---:|---|
| C28 | 19 / 151,936 | 4,114 / 2,734,848 | **FAIL both paths** |
| C40 | 0 / 151,936 | 1,251 / 2,734,848 | **FAIL chunked path** |

They also lack the required `numerics.json` artifact and are schema 1 where the amendment
demands schema 2. They must not be read as CED qualification. `review.py` already reports
them under `recorded_strict_parity_failures`.

### Finding 3 (documentation only — NOT fabrication) — "retained byte-for-byte" is loosely worded

The package modified four already-committed receipts, including measured `wall_seconds`
(G0 35.771→46.918, G2 167.279→197.977) and `implementation_sha256`
(`c5b952d8…` → `dc212c9f…`). A changed *measured* value initially looked like an edited
historical record. It is not. Reconciled from three independent sources:

- `OFFICIAL_GATE_LADDER.md` states the whole ladder was regenerated from scratch once
  `convert-ced` changed the implementation digest, "exactly as AGENTS.md requires".
- `REVIEW_REPORT.md` records that the uploaded ZIP's gate-ladder markdown "was already
  rewritten in the ZIP", and that `dc212c9f` is "exactly the value in the historical
  receipts" *as received*.
- Therefore commit `2f65f9e` holds an **older** ladder run (digest `c5b952d8`, before
  `convert-ced` existed) and the ZIP holds a **newer, genuinely re-executed** ladder run
  (digest `dc212c9f`). The differing wall times are two real runs, not an edited number.

No restoration performed; both generations are preserved (the older in git history at
`2f65f9e`, the newer in the working tree). The only defect is wording: the amendment's
sentence "Original evidence files are retained byte-for-byte" is true of git history but
not of the working tree, which invites exactly the misreading above.

### Finding 4 — `verification/*.json` is not local evidence

`verification/status.json`, `MECHANICAL.json`, `recovery-readiness.json` and both junit files
describe `Linux-6.18.35-x86_64`, `torch 2.10.0+cpu`, `transformers not_installed`,
`cuda_available: false`. That is the vendor's sandbox, not this machine. They are supplier
claims and carry no weight as local qualification. Superseded here by the fresh run.

---

## Fresh ladder execution — implementation `eab80e9f`

    out: experiments/architecture_benchmark_001/outputs/ladder-eab80e9f-v1
    cmd: python -m experiments.architecture_benchmark_001.ladder \
           --source <Supa-1.7.0-devfreeze> \
           --calibration experiments/architecture_benchmark_001/data/calibration.jsonl \
           --out experiments/architecture_benchmark_001/outputs/ladder-eab80e9f-v1 --device cpu

| stage | verdict | wall | receipt |
|---|---|---|---|
| G0 | **PASS** (fresh, schema 2) | 41.3 s | `G0/attempt-0001/G0.json` |
| G1_IDENTITY | **PASS** (fresh, schema 2) | 29 s | `G1_IDENTITY/attempt-0001/` |
| G1_WIDTH | **PASS** (fresh, schema 2) | 88 s | `G1_WIDTH/attempt-0001/` |
| G1_CED_C28 | RUNNING | — | — |
| G1_CED_C40 | pending | — | — |

G0 metrics: rows 24, hits 6 (both correct and incorrect present), parse_rate 1.0,
termination_rate 1.0, `max_abs_native_reference_error` 1.736e-4 (within `1e-4 + 1e-4·|a|`).

---

## Data inventory (Phase 4) — audited before obtaining anything

### Already in the experiment

| file | rows | role |
|---|---:|---|
| `data/pilot.jsonl` | 24 | open engineering diagnostic (synthetic arithmetic chains). Never training data. |
| `data/calibration.jsonl` | 156 | G1_WIDTH calibration, 4 per class × 39 classes |
| `data/train-smoke.jsonl` | 25 | exactly the 25 engineering steps |

None is scientific recovery data.

### Candidate broad recovery pool (local, no download needed)

`Chat-Bubble/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0`

- 110,000 SFT rows (+14,000 preservation slice), 16,504 template families, no family > 8 rows
  — families line up with the protocol's `resampling_unit: template_or_document_family`.
- `reports/validation_report.json`: PASS, 110,000 unique ids / prompts / latest-user turns.
- `reports/leakage_report.json`: PASS, scanned against `sealed.jsonl`,
  `sealed_roleplay_v2_DO_NOT_TRAIN.jsonl`, `dev.jsonl`, `dev_roleplay_v2.jsonl` and the RAG
  dev/test/preference splits. Zero exact overlap.
- `reports/near_duplicate_audit.json`: PASS, 12,000 sampled vs 42,000 prior, threshold 90,
  **0 hits**.
- Spot-check of the human audit packet: 3/3 arithmetic targets independently recomputed correct.

`RECOVERY_DATASETS.md` itself concludes an external download is not mandatory when suitable
local material exists. It does.

### Measured token budget — MEASURED BLOCKER for a 1M-token pilot

Official Supa tokenizer, 2,000-row sample of `rineya_v3_reasoning_train_50000.jsonl`,
`enable_thinking=False`:

    full rendered sequence : mean 135.6 tok, p95 198, max 258
    rows exceeding 512 ctx : 0 / 2000          (<=512 regime satisfied with wide margin)
    final-assistant target : mean 4.13 tok, max 10
    all-assistant targets  : mean 9.67 tok, max 29

Extrapolated over the whole 110,000-row pool:

| supervision policy | tokens available | rows needed for 1M |
|---|---:|---:|
| final assistant turn only | **~454,000** | 242,395 (pool has 110,000) |
| all assistant turns | ~1,063,000 | 103,434 |

**Reachability of the 1M budget depends entirely on a supervision policy that is not yet
frozen.** Two legitimate readings, with very different consequences:

- *Assistant-only chat SFT supervision* (supervise only the model's own turns): the whole
  110,000-row pool yields ~454k shifted targets (final turn) or ~1.06M (all assistant turns).
  1M is unreachable, or reachable only at ~1 epoch with no held-out margin. And 4-token
  targets inside 136-token contexts supervise ~3% of positions — sparse signal for
  recovering a structurally transplanted backbone.
- *Full-LM windowing*, which is what the delivered pipeline actually implements:
  `data_tools.prepare_data` emits `input_ids` windows with **no label mask** (verified at
  data_tools.py:80-88) and its own accounting counts `shifted_targets = len(input_ids)-1`
  (data_tools.py:101), i.e. every token is a supervised target. That yields ~14.9M shifted
  targets from the pool, so 1M needs only ~7,400 documents.

So the budget is not blocked — but the policy choice is a real, unregistered design decision
that changes what the experiment trains on. Full-LM windowing over rendered chat text trains
on system and user turns as well as assistant turns; assistant-only preserves the product
contract but starves the token budget. This must be chosen, justified and frozen *before*
any budget is set, not discovered afterwards. Note also that the v3 rows carry `messages`,
not the flat `text` field `prepare_data` requires, so a rendering step is needed either way.

### Outstanding data gaps

1. **Human review is pending for every row** of the v3 corpus (stated in its own README and
   `CALIBRATION_PROVENANCE.json`). A 520-row audit packet and a 40,000-ID audit pool exist.
   `review_status` must not be flipped to satisfy a validator.
2. The corpus leakage scan ran against a **build-time snapshot**. Its README requires
   re-running against the current local repo. Current assets do exist at
   `short-response-model-phase-0-1-2/data/{sealed,dev}/`. A hash-based re-check keeps sealed
   contents out of any training process.

---

## Constraint discovered, relevant to any future G3 work

`implementation_manifest()` = `ROOT.rglob('*.py')` + `protocol.json` + `source_lock.json` +
`requirements*.txt` + amendments named in `protocol.json`.

Therefore **adding any `.py` file anywhere under the experiment — including `tests/` —
changes the implementation digest and invalidates every implementation-bound receipt.**
A G3 runner must be written, tested and frozen *before* the qualification receipts it
depends on are regenerated. Markdown at the experiment root (this file) does not affect the
digest; files under `amendments/` do, once listed in `protocol.json`.

---

## Fresh ladder result — BOTH CED ARMS FAIL STRICT PARITY

    LADDER-0001.json -> FAIL

| stage | verdict | checks | wall |
|---|---|---|---|
| G0 | **PASS** | 8/8 | 41.3 s |
| G1_IDENTITY | **PASS** | 5/5, `max_abs_logit_error = 0.0` (bitwise exact) | 29 s |
| G1_WIDTH | **PASS** | 7/7 | 88 s |
| G1_CED_C28 | **FAIL** | 12/15 | 115 s |
| G1_CED_C40 | **FAIL** | 12/15 | 124 s |

Both CED arms fail exactly the same three checks, and no others:

    strict_tail_only_parity, strict_chunked_cache_parity, causal_prefix_parity

Everything structural passes on both arms: `registered_dimensions`, `exact_parameter_count`,
`registered_donor_layer`, `shared_encoder_final_memory`, `decoder_local_attention_absent`,
`finite_converted_logits`, `save_reload_exact`, `donor_state_unmodified`, `source_unchanged`,
`pilot_unchanged`, `calibration_unchanged`, and `decode_choices_exact`.

`causal_suffix_isolation_exact` is **True for every row of every arm at every precision**.
`causal_prefix_parity` fails only because it ANDs that (true) causality result with the
prefix *numerical* comparison. **There is no causality defect.**

## Matched ordinary controls — the decisive comparison

The ladder holds the arms to unequal gates: `G1_CED_*` uses the full `compare_paths` grid
(tail-only, chunk 1/7/16, halves, prefix, suffix isolation, all positions), while
`G1_IDENTITY` uses one `assert_close` on a single halves split over second-half positions
only, and `G1_WIDTH` tests **no** cache parity at all. So a CED failure could not be
attributed to CED from the ladder alone.

`diagnose-numerics` applies the *identical* `compare_paths` grid to the ordinary arms.
Same `DenseReference` class, same official weights, same 8 pilot rows, same policy.

### FP32 — maximum absolute error and violating-element counts (worst row per path)

| arm | tail | chunk 1 | chunk 7 | chunk 16 | halves | prefix | verdict |
|---|---|---|---|---|---|---|---|
| **D40** ordinary | 2.7e-5 / 0 | 8.1e-5 / 0 | 6.2e-5 / 0 | 4.5e-5 / 0 | 5.1e-5 / 0 | 1.8e-5 / 0 | **ALL PASS** |
| **B28** official | 1.3e-5 / 0 | 1.7e-4 / **4** | 8.3e-5 / 0 | 3.5e-5 / 0 | 1.2e-4 / 0 | 1.9e-5 / 0 | 4 elements over, 1 row |
| **C28** CED | 2.6e-4 / 184 | 8.1e-4 / 51,579 | 8.2e-4 / 63,801 | 5.2e-4 / 6,741 | 4.9e-4 / 9,555 | 4.9e-4 / 7,447 | **FAIL** |
| **C40** CED | 2.8e-4 / 286 | 3.5e-3 / 132,432 | 5.9e-3 / 134,197 | 9.5e-5 / 0 | 5.9e-3 / 127,892 | 1.8e-4 / 50 | **FAIL** |

### FP64 — same grid, whole model cast to float64

| arm | tail | chunk 1 | chunk 7 | halves | prefix | verdict |
|---|---|---|---|---|---|---|
| **B28** official | 5.5e-14 | 3.8e-13 | 3.6e-13 | 2.6e-13 | 8.9e-14 | **ALL PASS** |
| **C28** CED | 1.0e-12 | 4.1e-12 | 3.3e-12 | 2.2e-12 | 6.2e-13 | **ALL PASS** |

### What this establishes

1. **The CED transplant is mathematically correct.** At FP64 it reproduces cache-order
   invariance to ~1e-12, eight orders of magnitude inside the 1e-4 tolerance, and passes
   every path including prefix and causal isolation. There is no logic, causality, masking
   or cache-ordering bug.
2. **The strict gate is achievable.** D40 passes it outright with zero violations on every
   path and every row; B28 misses by 4 elements out of ~5.5 million in a single row. The
   gate is not an impossible standard for fp32 inference.
3. **The FP32 failure is therefore CED-specific and real.** Against its own matched ordinary
   control, CED inflates worst-case cache-order error by ~10-70x and violating-element
   counts by three to five orders of magnitude. The FP64 lane shows the same ordering —
   C28's FP64 error is ~10x B28's — so CED is genuinely less well-conditioned, plausibly
   because one shared encoder-final K/V bank feeds all decoder layers, letting a single
   rounding difference propagate everywhere instead of being re-derived per layer.
   (Mechanism is a hypothesis consistent with the data; the receipt still records
   `cause: UNDETERMINED` and no causal attribution is claimed.)
4. **The pre-hotfix `parity_note` is empirically falsified.** It claimed "Cache-order
   differences are fp32 accumulation, not a behaviour change". The first half is right —
   FP64 proves it is accumulation, not a behaviour change. The conclusion drawn from it was
   wrong: the matched ordinary control passes the very same test, so the discrepancy is not
   a generic property of fp32 decoding and cannot be waved through on that basis. This is
   exactly the substitution HOTFIX_001 rejected, and the rejection is now backed by
   measurement rather than by argument.
5. **FP64 success does not unblock anything.** HOTFIX_001: "FP64 is an observation-only
   comparison lane, not permission to deploy or train FP32 after a failed FP32 gate."

### Consequence

At the registered FP32 precision the CED arms are **not qualified**. Per the standing rule
that a CED strict-parity failure blocks its optimizer lane, `G2_CACHED_C28` and
`G2_CACHED_D40->C40` are closed. `gates.py` enforces this without intervention: their parent
stages are `G1_CED_C28` / `G1_CED_C40`, which now hold FAIL receipts.

No tolerance was widened, no historical result relabelled, and no permissive substitute
implemented. B28 and D40 remain independently eligible and proceed.

---

## Leakage re-check against the CURRENT local evaluation assets — PASS

The v3 corpus README requires re-running its leakage scan against the live repository,
because its shipped `leakage_report.json` was built against a snapshot
(`/mnt/data/_repofiles/...`). Done, using normalized hashes only so that **no sealed content
is read into any training or tuning process**:

    eval sources : short-response-model-phase-0-1-2/data/sealed/sealed.jsonl          110 rows
                   .../sealed/sealed_roleplay_v2_DO_NOT_TRAIN.jsonl                   400 rows
                   .../dev/dev.jsonl                                                  510 rows
                   .../dev/dev_roleplay_v2.jsonl                                      400 rows
    distinct eval keys : 1,019 last-user-turn / 1,020 full-prompt
    corpus rows scanned: 124,000 (110k new v3 SFT + 14k preservation slice)

    LAST-USER-TURN exact collisions : 0
    FULL-PROMPT     exact collisions : 0

Normalization is NFKC + casefold + whitespace collapse. This is exact-match contamination
detection, **not** semantic or near-duplicate detection; for that the only evidence remains
the corpus's own sampled `near_duplicate_audit.json` (12,000 sampled vs 42,000 prior,
threshold 90, 0 hits). Open data question 3 is closed; 1, 2 and 4 remain open.
