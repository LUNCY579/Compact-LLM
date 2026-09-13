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

---

## Matched cached-teacher engineering (Phase 3) — B28 and D40 PASS, C28/C40 blocked by design

`G2_CACHED_C28` and `G2_CACHED_C40` declare `G1_CED_C28` / `G1_CED_C40` as parents. Those
now hold FAIL receipts, so `validate_receipt` refuses those lanes without any intervention
from me. The two ordinary arms are independently eligible (`G2_CACHED_B28` parents `G0`;
`G2_CACHED_D40` parents `G1_WIDTH`) and were executed.

    outputs/engineering/teacher-cache/TEACHER_CACHE.json
    outputs/engineering/g2cached-b28/G2_CACHED_B28.json   PASS 9/9
    outputs/engineering/g2cached-d40/G2_CACHED_D40.json   PASS 9/9

Identical recipe on both arms, as the protocol requires: exactly 25 steps, SGD without
momentum at 1e-5, alpha 0.5, temperature 1.0, `head_chunk_tokens` 8, gradient checkpointing,
seed 17, same teacher cache, same 25 training rows, same order.

| | B28 | D40 |
|---|---|---|
| status | PASS 9/9 | PASS 9/9 |
| student_training_tokens | 3,159 | 3,159 |
| supervised_shifted_targets | 3,134 | 3,134 |
| wall_seconds | 358.5 | 319.9 |
| throughput | 8.7 targets/s | 9.8 targets/s |
| teacher backbone tokens **inside** the student loop | **0** | **0** |
| head projection positions (student / teacher) | 6,268 / 6,268 | 6,268 / 6,268 |
| peak CUDA allocated | null (CPU) | null (CPU) |
| loss first -> last (min..max) | 2.838 -> 2.621 (2.189..3.317) | 3.460 -> 3.291 (2.894..4.104) |

Losses are noisy and essentially flat. That is the expected and correct outcome of 25 SGD
steps at 1e-5 — **these are engineering checks, not recovery results**, and nothing here is
evidence about either architecture's quality. The two arms' losses are also not comparable
to each other: different shapes, different initialisations, no registered pairing between
them (the registered pairings are C28↔B28 and C40↔D40, and both CED arms are blocked).

`G2_CACHED_B28` was additionally re-validated through `validate_receipt` directly:
recursive parent DAG and artifact hashes verify against G0.

### Measured resources on this machine

| quantity | measured |
|---|---|
| teacher cache | 12,981,798 B for 3,159 tokens = **4,110 B/token** (hidden 1024 x fp32) |
| checkpoint per arm | 2.3 GB fp32 |
| process RAM, 25-step CPU smoke | **~7.0 GB** peak working set (of 15.2 GB total) |
| total `outputs/` after this session | 12 GB (99 GB free remains) |
| student throughput | ~9 supervised targets/s, CPU, 16 threads |

### Extrapolations that matter for any G3 plan

- 1,000,000 supervised targets -> teacher cache **~4.1 GB**; 10M -> ~41 GB. A 4.1 GB cache
  cannot be held resident beside a ~7 GB training process on a 15.2 GB machine, so
  bounded-memory indexed shards are a hard requirement, not a nicety.
- 1,000,000 supervised targets at ~9 targets/s = **~31 hours per arm on CPU**, so ~62 hours
  for one complete matched recipe. **A 1M-token CPU pilot is not feasible on this machine.**

### GPU: not measured, and deliberately not attempted

The RTX 4070 Laptop (8 GB, ~0.9 GB already in use) was never used. `peak_cuda_allocated_bytes`
is null in both receipts because the qualified runtime is `torch 2.10.0+cpu`. Running on CUDA
means a different `torch` build, therefore a different runtime identity in `context()`, so
`validate_receipt` would reject every parent receipt and the entire ladder would need
requalifying under a CUDA runtime first. Devices were not silently switched. **No 8 GB fit,
VRAM, latency or throughput claim is made.**

---

## Fix 1 applied: `audit-evidence` now reports schema-2 strict failures (digest 8d54f540)

`review.py:28` read violation counts only from `metrics.cache_order_parity`, the schema-1
CED layout. The schema-2 writer records named strict checks instead and keeps per-path
counts in the `numerics.json` artifact, so every schema-2 CED FAIL receipt was reported as
`recorded_strict_parity_failures: {}`.

The existing field keeps its exact meaning; three were added — `failed_strict_parity_checks`,
`strict_parity_violation_totals`, and `any_recorded_strict_parity_failure` as the
generation-independent answer. Per-path counts are read from `numerics.json` **only** when
it is a declared artifact whose recorded hash still matches, so an unbound or edited
sidecar contributes nothing; a tamper case is in the test.

Verified against real receipts — the audit now reports what it previously hid:

    G1_CED_C28  FAIL  any_recorded_strict_parity_failure=True  totals 401,863
    G1_CED_C40  FAIL  any_recorded_strict_parity_failure=True  totals 726,228

and the schema-1 historical receipts still surface 19/4,114 and 1,251 while showing
`declared_status: PASS` — the contradiction is now visible in a single view.

### Requalification at 8d54f540 — the fix is proven inert

Digest moved `eab80e9f` → `8d54f540`, so the whole ladder and both engineering controls
were re-executed. Every measured quantity reproduced **exactly**:

| stage | eab80e9f | 8d54f540 | checks identical | per-path numerics identical |
|---|---|---|---|---|
| G0 | PASS | PASS | yes | — |
| G1_IDENTITY | PASS | PASS | yes | — |
| G1_WIDTH | PASS | PASS | yes | — |
| G1_CED_C28 | FAIL 12/15, 401,863 | FAIL 12/15, **401,863** | yes | **yes** |
| G1_CED_C40 | FAIL 12/15, 726,228 | FAIL 12/15, **726,228** | yes | **yes** |
| G2_CACHED_B28 | PASS 9/9 | PASS 9/9 | yes | — |
| G2_CACHED_D40 | PASS 9/9 | PASS 9/9 | yes | — |

Identical to the element, not merely to the verdict. The `eab80e9f` evidence is retained
alongside as the baseline that inertness is measured against.

## Fix 2 applied: HOTFIX-002 shared-memory accumulation diagnostic (digest fc44701e)

Registered in `protocol.json` under `G1.parity_policy.memory_accumulation_diagnostic`,
default `inherit`, scoped to the `ReadBlock` attention over the shared bank only.
`SelfBlock` — and therefore the encoder and every ordinary arm — is untouched.
`memory_accumulation_dtype` is runtime-only and not part of `Config`, so parameter counts,
`config` equality against `presets()` and saved `config.json` are unaffected;
`_signature()` includes it so a cache filled in one lane cannot be reused in the other.

186 passed, 0 skipped. No tolerance, acceptance rule or gate threshold changed.

---

## HOTFIX-002 result: the high-precision shared-memory read does NOT work — and why

### Measurement (same C28/C40 checkpoints, FP32, 8 pilot rows, identical grid)

| arm | `--memory-precision` | max abs err | total violations | failing paths |
|---|---|---:|---:|---:|
| C28 | inherit | 8.23e-04 | 401,863 | 6 |
| C28 | **float64** | 1.00e-03 | **399,698** | 6 |
| C40 | inherit | 5.89e-03 | 726,228 | 5 |
| C40 | **float64** | 4.72e-03 | **803,104** | 5 |

C28 improves by 0.5% (noise, and its max error gets *worse*); C40 gets 10.6% **worse**.
The intervention fails. The same six/five paths still fail on each arm. **The hypothesis
that the amplification is created in the read contraction is falsified.**

### Where the divergence actually starts

Per-layer hidden-state comparison, full forward vs chunked, C28 checkpoint:

    chunk 7: enc00 3.815e-06 -> enc13 1.678e-04   shared bank  dK 2.012e-04  dV 4.597e-04
    chunk 1: enc00 3.576e-06 -> enc13 3.906e-03   shared bank  dK 1.073e-04  dV 2.565e-04

Divergence begins at **encoder layer 0** and amplifies ~1000x through encoder depth. The
shared memory bank inherits it: **dV = 4.6e-04, already 4.6x outside the 1e-4 tolerance
before a single decoder read happens.** No precision applied at the read can recover
information that is already wrong in the bank. That is exactly why the intervention failed.

### The ordinary control explains the whole effect

Same probe on the unmodified official B28 (its first 14 layers *are* C28's encoder):

    chunk 7: layer00 3.815e-06  layer13 1.678e-04  layer27 8.545e-04   (224x growth)
    chunk 1: layer00 3.576e-06  layer13 3.906e-03  layer27 8.789e-03   (2458x growth)

B28's per-layer drift is **identical** to C28's encoder drift at layer 00 and layer 13 —
to the digit — and by its final layer B28 drifts *more* (8.8e-03) than C28's encoder does
(3.9e-03). Yet B28's logits show 4 violating elements and C28's show 401,863.

So CED does **not** create more internal drift. Both architectures carry ~1e-3 mid-network
chunk-order drift in FP32; that is simply what FP32 transformer inference does.

**The difference is what happens to that drift on the way to the logits.** In a plain
model the drift reaches the final RMSNorm, which suppresses the correlated component, and
1e-3 of hidden drift becomes 1.7e-4 of logit error. CED instead converts the *unnormalized
mid-network* encoder output into a shared K/V bank and re-injects it into all 28 decoder
layers, coherently, before the final norm. The bank's 4.6e-4 error is already out of
tolerance and never gets the benefit of that suppression.

### Consequence

This is an **architectural property of CED, not a defect and not a fixable rounding
artifact**. The strict 1e-4 elementwise logit gate is passed by plain models largely
because the final RMSNorm cleans up accumulated drift; CED forfeits that by construction.
No cheap numerical remedy exists at the read, and raising the *bank* projection precision
cannot help either, because the bank's input `h` is already perturbed.

The remaining routes are genuinely larger decisions, and none is taken here:

1. Build the shared bank from a full-context forward instead of incrementally. Changes
   incremental-decoding semantics and cost; needs its own design and registration.
2. Run the encoder at higher precision. That is most of the model; it is a throughput and
   memory decision, not a tolerance tweak.
3. Prospectively amend the equivalence policy so cache-order parity is evaluated in a way
   that is meaningful for an architecture which exposes a mid-network tensor. This must be
   written **without** citing this run's failure as its justification, and reviewed
   separately.

`HOTFIX_002.md` stays in the protocol as the registered record of a diagnostic that was
tried and did not work. The option remains default `inherit`, so nothing in the qualified
path changed. Negative result, reported as such.

---

## Final requalification at digest `fc44701e` — HOTFIX-002 default proven inert

| stage | 8d54f540 | fc44701e | checks identical | per-path numerics identical |
|---|---|---|---|---|
| G0 | PASS | PASS | yes | — |
| G1_IDENTITY | PASS | PASS | yes | — |
| G1_WIDTH | PASS | PASS | yes | — |
| G1_CED_C28 | FAIL, 401,863 | FAIL, **401,863** | yes | **yes** |
| G1_CED_C40 | FAIL, 726,228 | FAIL, **726,228** | yes | **yes** |
| G2_CACHED_B28 | PASS 9/9 | PASS 9/9 | yes | — |
| G2_CACHED_D40 | PASS 9/9 | PASS 9/9 | yes | — |

Three digests have now been qualified end to end on official weights — `eab80e9f`,
`8d54f540`, `fc44701e` — and all three agree to the element. Adding the audit fix and the
registered diagnostic changed no measurement, which is the property both changes were
designed to have and the reason each requalification was run rather than assumed.

Current qualified state: **`fc44701e`**, 186 tests passed, 0 skipped.

---

## Correction: the digest moved again after HOTFIX-003

The entry above ends "Current qualified state: `fc44701e`, 186 tests passed, 0 skipped."
That was true when written and is left in place, because this log is append-only.

It is no longer the current state. Adding the `G1_QUALITY` gate (HOTFIX-003) changed
`protocol.json`, `gates.py`, `run.py` and the test suite, all of which are inside
`implementation_manifest()`. Measured:

    implementation_digest() = 7d48de215b08f564f4578b52b4fb89a85ce972c812c1ce43c74064c146d6f4cd
    manifest files          = 30   (amendments/HOTFIX_003.md included)
    tests                   = 189 passed, 0 skipped

Consequences, recorded rather than glossed:

- The receipts under `evidence/ladder-fc44701e-v1/` and `evidence/engineering-fc44701e-v1/`
  bind `fc44701e` and are **stale relative to the working tree**. They remain valid evidence
  of what was measured at that digest; they are not current qualification.
- **Requalification at `7d48de21` is PENDING, not done.** HOTFIX-003 only adds a gate and
  touches no existing arithmetic, so the prediction is that every number reproduces exactly —
  as happened for the two previous digest moves. That prediction is explicitly unverified.
- The quality/architecture diagnostics in `evidence/arm-quality/` and
  `evidence/trainability/` are unaffected either way: they run from
  `experiments/rab001_exploration/`, outside the hashed root, and bind no digest.

---

## HOTFIX-004, three harness bugs, and the rebuilt evaluation

### Digest moved again

    implementation_digest() = 8b47b39595dcdd7ee82512de16e4ea30abee4642063de7f5119be76ed6d6621f
    manifest files          = 31   (amendments/HOTFIX_004.md included)
    tests                   = 191 passed, 0 skipped

Requalified at `8b47b395`: G0 / G1_IDENTITY / G1_WIDTH PASS, both CED arms FAIL, identical to
`fc44701e` to the element. **Fifth digest qualified end to end.**

### HOTFIX-004 — the quality gate was under-calibrated

`G1_QUALITY_D40` now fails on **three** checks, not two: the nondegeneracy floor scales as
`max(2, ceil(0.25·prompts))`, so at 24 prompts the floor is 6 and D40's 4 distinct outputs
fail. Previously an absolute floor of 2 passed a `</think>` loop — the exact case the check
exists for.

HOTFIX-003's claim that "no plausible setting of these thresholds changes any arm's verdict"
was **false** for D40's top-1 (0.5032 against a 0.5000 floor) and is corrected. **The top-1
floor was deliberately not moved**: tuning a threshold after seeing a measured outcome is
what this programme has criticised elsewhere. A test pins it at 0.5.

### Three harness bugs

1. **The seed consumed nothing.** Seeds 17 and 29 gave bit-identical output — dev CE
   5.524462 both, task 38/100 both, trajectories matching to nine decimals. Fixed data order,
   fixed weights, no dropout, greedy decode. Reporting those as three seeds would have been
   one run reported three times. The seed now shuffles TRAIN; overlap of the 40 rows actually
   used is 1/40, 3/40, 1/40 across seeds.
2. **Leading-integer scoring.** 31 of 50 `time_duration` answers are clock times, so
   `10:55 AM` matched `10:25 AM`; decimals collapsed identically. Now compares the full
   number sequence. Unit-checked on 11 cases.
3. **A print could kill a run.** Three C28 runs died on cp1252 console encoding *after* their
   compute. All 12 exploration scripts now reconfigure stdout/stderr to UTF-8.

Superseded runs retained under `outputs/rigorous/superseded_vacuous_seed/` as the evidence
for bugs 1 and 2 rather than deleted.

### Evaluation rebuilt on measured capability

`arithmetic_chain` — the obvious verifiable probe — is **unusable**: the frozen source scores
0/20. Measured baselines by class: percentage_reasoning 65%, ratio_proportion 50%,
time_duration 30%, estimation_sanity 30%, counting_sets 10%, sequence_pattern 5%,
unit_conversion 5%, arithmetic_chain 0%.

Probe = the three classes above 30%, held out of TRAIN/DEV so the task measures **retention**
against a real baseline. Splits: TRAIN 800 / DEV 150 / TASK 150, family-disjoint, zero
overlaps, rendering verified byte-for-byte against the existing prepared data. The selection
was **not re-fitted** after the scorer correction moved `time_duration` to exactly 30.0%.

### GPU was evaluated and rejected for this recipe

AdamW fp32 on 596M parameters needs 8.89 GB (params + grads + two moments); the RTX 4070
Laptop has 6.86 GB free. Only SGD-no-momentum or Adafactor fit, and changing the optimizer
would break the matched comparison. Ada also defaults fp32 matmul to TF32. Recorded as a
measured constraint, not an untested assumption.

---

## 2026-09-12 — Training retention design checkpoint and evidence reconciliation

Read CONTINUATION, this log, COMPREHENSIVE_ANALYSIS and REQUALIFICATION_REVIEW in order,
then the agent contract and referenced implementation/runbook materials. Starting HEAD
`e7285e5`, requested worktree/branch unchanged. Two bounded read-only internal agent reviews
completed; no model job ran and no implementation/protocol file was edited.

### State verification

The digest reproduces exactly:
`8b47b39595dcdd7ee82512de16e4ea30abee4642063de7f5119be76ed6d6621f`, 31 files.
To avoid importing torch under memory pressure, the exact stdlib-only repository function
bodies were executed via AST extraction. All 14 inspected current live/archive receipt JSON
hashes match; present bound artifacts match. Live PASS DAGs validate against their recorded
contexts. CED retains the three strict-parity failures; D40 quality retains its three failures.
Archived evidence copies omit model binaries and retain relocated parent paths, so their
original live outputs remain necessary for authorization.

**Fresh tests are pending, not passed in this session.** `hostbudget.require(4.0, ...)`
refused before torch/pytest import: 0.06 GiB free, 99% load. Subsequent snapshots reached only
0.40-1.04 GiB free. The user explicitly chose "Record tests as pending". The prior 191/0
result is preserved as historical, and native/runtime execution was not reverified. Installed
dependency metadata matches the pinned versions. No lock or process was removed.

### Newly verified defects and corrections

- The binding G3 primary endpoint is already exact/executable capability verification.
  The claim that both registered endpoints are exclusively likelihood-shaped is unsupported.
  Executable retention enforcement and the other frozen campaign fields remain absent.
- G1_QUALITY is a leaf: the existing G2/G2_CACHED parent DAG does not consume it.
- C28's 500k artifact records 468 optimizer updates but evaluates update 467; the final
  seven-row partial update occurs afterward. The token-only final-evaluation guard misses it.
- Longbudget resume omits pending gradients/microbatch count. The first mismatched applied
  update is step 117 (125,834 versus 126,070 tokens, different learning rates). The control's
  3,743 consumed rows exceed its 467 full eight-row windows by seven rows. This is consistent
  with discarded accumulation on resume. Exact matched-final claims are not established.
- Full paired capability vectors were not saved: only eight truncated final samples remain.
  The legacy 2/60 versus 26/60 scores cannot be independently rescored in full. Their CE-gap
  arithmetic does reproduce: 0.115811 nats, 98.4157% closure, reply concentration 4.27223x.
- Both longbudget JSONs are committed and byte-identical to their live copies. C28's JSON
  lacks a TRAIN fingerprint. Recorded times are 5.14 h and 6.50 h; 10.3 h refers to the C28
  1M extrapolation. Timers omit initialization/cache and discarded crash work. Pair count
  1,000,198 exceeds a 1,000,000 pair ceiling by 198, contrary to the historical prose.
- Legacy scorer self-test: 11/11 pass when explicitly invoked, but extra smoke cases accept
  wrong AM/PM, wrong units and negation, and reject equivalent duration units. A retained
  question asks for sensor tags while its reference says spare fuses. Typed scoring requires
  reviewed labels, not simply a stricter regex. Tests do not automatically run at import.
- Current large TRAIN has 10,000 families. TRAIN/DEV/TASK are mutually disjoint, but overlap
  with calibration remains (12 TRAIN families/one source row; two DEV and two TASK families).
  Reused TASK families across seeds are not independent observations for pooled Wilson CIs.
- Commit `c8b9e28` records a completed 28-agent internal review. Earlier universal claims that
  no review completed are stale; independent external review is still not established.

The old JSON evidence is preserved. These findings qualify the exact matching and endpoint
claims; they do not erase the recorded exploratory likelihood/capability discrepancy or the
earlier motivation for a retention gate. See the new design document for precise scope.

### Proposed rules, before implementation or new candidate scores

`RETENTION_GATE_DESIGN.md` records an open 150-row monitor with zero losses from the union
of frozen-source-correct and initialization-correct items, separately by class; termination
floor 80% per class; step-zero, every 5,000 applied-token boundary after an optimizer update,
and terminal evaluation after the final update. Gains cannot offset lost source successes.

Separate prospective confirmation proposes 300 fresh families/class and seeds 17/29/43,
one terminal look, a 90% source-success retention floor, exact binomial forgetting UCB <=0.10
at alpha 0.05/9, and at least 50 source successes/class. The sample is explicitly a bounded
feasibility design, not guaranteed power. Missing evidence yields INSUFFICIENT, never a
threshold change. Historical scores are already known and cannot provide preregistered
validation. CE/reply CE/KL remain diagnostics, and G3-G5 remain closed.

The user's pending CONTINUATION update was corrected to reflect these findings before
commit. The design and audit records do not change the implementation digest. Next: deliver
the requested design checkpoint; complete pending baseline tests when memory permits before
implementation, then freeze/scorer-test/amend, run the suite, requalify and verify inertness.

Artifacts: `evidence/retention-design-v1/{BASELINE_AUDIT,RECONCILIATION}.json`.
