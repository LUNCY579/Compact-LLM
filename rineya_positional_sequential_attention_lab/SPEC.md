# PSA Lab — Implementation Contract

Binding interface spec. Every module in `src/rineya_psa/` implements against this
document. Do not change a signature that another module imports without noting it
here first.

## 0. Environment (measured 2026-08-28)

| Fact | Value |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.12.1+cu126 |
| transformers | 5.14.1 |
| numpy / scipy | 2.4.6 / 1.17.1 |
| GPU | NVIDIA RTX 4070 Laptop, 8.59 GB VRAM |
| CPU | 32 logical cores |
| OS | Windows 11 (win32) |
| Interpreter | `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe` |

Repo root: `C:/Users/clclc/BusinessProject/Chat-Bubble`.
Lab root: `<repo>/rineya_positional_sequential_attention_lab`.

**VRAM is the binding constraint.** 8.59 GB total, ~7.5 GB usable. A 0.6B model in
bf16 is ~1.2 GB. Never materialise a full `[B, 16, T, T]` attention tensor at
T >= 4096 in fp32. Chunk over query positions (see §5.3).

## 1. Model panel

All four checkpoints are **Qwen3-0.6B architecture** and mutually shape-compatible:

```
num_hidden_layers   28
hidden_size       1024
intermediate_size 3072
num_attention_heads  16   (query heads)
num_key_value_heads   8   (KV heads)  -> GQA group size 2
head_dim            128
vocab_size       151936
rope_theta          1e6
tie_word_embeddings true
attention_bias     false
q_norm / k_norm    RMSNorm over head_dim (Qwen3-specific)
```

| key | path (relative to repo root) | store dtype | loader notes |
| --- | --- | --- | --- |
| `qwen3` | `Qwen3-0.6B` | bfloat16 | plain `Qwen3ForCausalLM` |
| `halo` | `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze` | float32 | plain |
| `nimbus` | `Rineya_Published_Cores/Rineya-Nimbus-1.6.0-devfreeze` | bfloat16 | plain |
| `supa_a11` | `Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze` | float16 | `trust_remote_code=True`, `Qwen3AbstainForCausalLM`, **must load in float16** |

**Supa-A1.1 is not a plain Qwen3.** It adds a constant `+5.15625` logit offset on
token `51378` (`UNKNOWN`) inside `forward`. Two consequences the lab must respect:

1. Loading without `trust_remote_code=True` silently disables the gate. Every
   loader asserts `type(model).__name__ == "Qwen3AbstainForCausalLM"` for this key.
2. For **mechanistic** comparisons (gold-answer logprob, significance matrices) the
   abstention offset is a confound: it shifts one vocabulary row only. The lab
   therefore reports Supa-A1.1 in two modes, both recorded:
   - `gated` (default, as published) — used for accuracy / preservation.
   - `ungated` (`abstain_logit_bias=0.0`) — used for teacher-forced NLL and all
     causal significance, so that `Delta` reflects attention, not the gate.
   `load_model(..., ungate=True)` sets the config field to 0.0 in the loaded
   instance only. The checkpoint on disk is never touched.

Checkpoints are **read-only**. No code in this lab opens a file under
`Qwen3-0.6B/` or `Rineya_Published_Cores/` for writing. `tests/test_immutability.py`
enforces this by SHA-256 before/after.

## 2. Package layout

```
src/rineya_psa/
  paths.py            # repo/lab roots, MODELS registry, artifact path helpers
  types.py            # frozen dataclasses shared across modules
  loading.py          # load_model / load_tokenizer / device + dtype policy
  model_audit/        # Phase 1
  benchmark/          # Phase 2
  hooks/              # attention interface registry, capture, interventions
  significance/       # Phase 4-5, 7  (I, B, D matrices; SVD)
  activation_analysis/# Phase 6 (probes, patching)
  sequential_attention/ # Phase 10 (Arms A-E)
  interventions/      # Phase 8, 11 (PSA / SGSA / cheap corrections)
  merging/            # Phase 8 (checkpoint arithmetic, position-aware soup)
  evaluation/         # scoring, generation, preservation
  statistics/         # bootstrap, paired tests, effect sizes
  figures/            # matplotlib figure builders
```

Import style: absolute, `from rineya_psa.hooks.registry import ...`.
No module imports `torch` at import time except under `src/rineya_psa/hooks/`,
`loading.py`, `sequential_attention/`, `activation_analysis/`, and
`interventions/runtime*.py`. Everything else must stay importable on a torch-free
interpreter so CPU analysis and tests run fast.

## 3. Core types (`types.py`) — authoritative

```python
@dataclass(frozen=True)
class ModelSpec:
    key: str                 # "qwen3" | "halo" | "nimbus" | "supa_a11"
    display_name: str
    path: Path
    store_dtype: str
    trust_remote_code: bool
    expected_arch: str
    ungatable: bool          # True only for supa_a11

@dataclass(frozen=True)
class Arch:
    num_layers: int; hidden_size: int; num_q_heads: int
    num_kv_heads: int; head_dim: int; vocab_size: int
    @property
    def kv_group_size(self) -> int          # num_q_heads // num_kv_heads
    def q_heads_for_kv(self, kv: int) -> tuple[int, ...]

@dataclass(frozen=True)
class PositionGrid:
    values: tuple[float, ...]  # normalized targets, e.g. (0.02,...,0.98)

@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str            # unique, deterministic
    scenario_id: str        # SHARED across positional variants of one source example
    family: str             # "exact_retrieval"|"semantic_single_hop"|"multi_hop"|"state_constraint"
    fmt: str                # "natural"|"shuffled"|"list"|"json"
    length_bucket: str      # "2k"|"4k"|"8k"|"16k"|"32k"
    target_positions: tuple[float, ...]   # requested normalized positions (1 or 2 for multi-hop)
    local_position: str     # "first"|"middle"|"last"|"na"
    split: str              # "calibration"|"validation"|"sealed"
    system: str
    user: str
    answer: str             # gold surface string
    answer_aliases: tuple[str, ...]
    distractors: tuple[str, ...]
    evidence_sentences: tuple[str, ...]
    seed: int
    meta: Mapping[str, Any]  # includes construction params; JSON-serialisable

@dataclass(frozen=True)
class TokenizedItem:
    item_id: str
    input_ids: tuple[int, ...]        # full prompt after chat template, no answer
    answer_ids: tuple[int, ...]       # gold continuation
    evidence_spans: tuple[tuple[int,int], ...]  # [start, end) in input_ids
    actual_positions: tuple[float, ...]         # measured evidence midpoint / n_ctx
    n_tokens: int
```

Every `BenchmarkItem` variant of the same underlying example shares `scenario_id`.
All paired statistics pair on `scenario_id`.

## 4. Loading (`loading.py`)

```python
def load_tokenizer(key: str)
def load_model(key: str, *, device="cuda", dtype=None, attn="sdpa",
               ungate=False, eval_mode=True)
```

- `dtype=None` -> the checkpoint's own store dtype, except `halo` (float32 on an
  8.6 GB card is 2.4 GB; load it in bfloat16 and record that as an approximation
  in the manifest — **note in results**, do not hide it). Explicit override wins.
- `attn` in `{"sdpa", "eager", "psa"}` where `"psa"` is this lab's registered
  interface (§5).
- Asserts architecture class matches `ModelSpec.expected_arch`.
- Returns `(model, arch: Arch)`.

## 5. Hook framework (`hooks/`) — the single most important contract

### 5.1 Why a custom attention interface, not monkeypatching

`Qwen3Attention.forward` resolves its kernel through

```python
ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
```

so registering `"psa"` in `ALL_ATTENTION_FUNCTIONS` and setting
`config._attn_implementation = "psa"` gives a fully reversible, per-forward hook
with access to `module` (hence `layer_idx`), `query`, `key`, `value`,
`attention_mask`, and `scaling`. Restoring `"sdpa"` restores exact baseline
behaviour. This is the mechanism the lab uses.

### 5.2 `PSAContext`

```python
@dataclass
class PSAContext:
    enabled: bool = False
    logit_bias: LogitBiasSpec | None = None    # Mode 1
    value_scale: ValueScaleSpec | None = None  # Mode 2
    head_scale: HeadScaleSpec | None = None    # Mode 3
    capture: CaptureSpec | None = None         # statistics collection
    position_bins: PositionBinning | None = None
```

Used as a context manager:

```python
with psa_session(model, ctx):     # swaps _attn_implementation, restores on exit
    out = model(input_ids)
```

**Mandatory invariant** (`tests/test_baseline_equivalence.py`): with
`PSAContext(enabled=False)` — and also with every spec present but coefficient 0 —
logits must match the unmodified `sdpa` run to `atol=2e-3, rtol=0` in bf16 and
`atol=1e-5` in fp32, on at least a 512-token prompt.

### 5.3 Intervention semantics (exact definitions)

Let `j` index source tokens, `b(j)` its position bin, `l` layer, `h` KV head,
`g` query head with `kv(g) = g // kv_group_size`.

- **Mode 1 / logit** `z'[g,i,j] = z[g,i,j] + lam * S[l, kv(g), b(j)]`.
  Implemented as an additive term folded into `attention_mask` before softmax, so
  it composes with SDPA. Causal masking must be applied *after* (never unmask).
- **Mode 2 / value** `V'[h,j,:] = S[l,h,b(j)] * V[h,j,:]` before `repeat_kv`.
- **Mode 3 / head output** `O'[g] = gamma[l,g] * O[g]` applied to the per-head
  attention output before `o_proj` reshape.
- **Attenuation for causal significance** is Mode 2 with
  `S = 1 - eps * indicator(bin == b)`, default `eps = 0.25`, applied to a single
  `(l, h, b)` cell.

`position_bins` maps absolute token index -> bin. Two binnings exist and are kept
separate: `AbsoluteBinning(n_bins)` over `j / T`, and `DistanceBinning(edges)` over
`(i - j) / T`. Distance binning is query-dependent and therefore only valid in
capture and in Mode 1/2 when the query index is available (it is: `i` is the row).

### 5.4 Capture

`CaptureSpec(layers, what)` where `what` ⊆ `{"attn_by_bin", "attn_entropy",
"gold_span_attn", "qk_logit_mean", "head_out_norm"}`. Capture is chunked over
query rows (`chunk=256`) and reduces inside the chunk — never returns a full
`[T, T]` tensor. Results land in `ctx.captured: dict[(layer, tensor_name), np.ndarray]`.

### 5.5 GQA mapping

`Arch.q_heads_for_kv(kv)` is the only sanctioned mapping. Any code that writes
`head` without saying `q_head` or `kv_head` is a bug.
`tests/test_gqa_mapping.py` pins `q_heads_for_kv(3) == (6, 7)` for this arch.

## 6. Metrics (`evaluation/metrics.py`)

```python
accuracy(pred, item) -> bool          # normalised exact match over answer + aliases
edge_middle_gap(acc_by_pos) -> float  # (A(min_p)+A(max_p))/2 - A(closest to 0.5)
worst_position(acc_by_pos) -> float
position_robustness_index(acc_by_pos, eps=1e-9) -> float
ushape_fit(positions, acc) -> UShapeFit  # beta0, beta1, beta2 + stderr
gold_nll(model, tok_item) -> float    # teacher-forced, sum over answer_ids
answer_margin(...) -> float           # logP(gold first tok) - logP(top competitor)
```

Accuracy normalisation reuses the existing project convention: casefold, strip,
collapse whitespace, drop trailing punctuation. It does **not** do fuzzy matching.

## 7. Statistics (`statistics/`)

- Paired bootstrap over `scenario_id` (not over positional variants), 10 000
  resamples, seed fixed per call and recorded.
- Report mean, sd, paired mean difference, BCa 95% CI, Cohen's dz.
- McNemar exact for binary paired accuracy.
- The three-way interaction contrast of §48 gets its own function
  `interaction_contrast(seq_sm, seq, par_sm, par)` returning point estimate + CI.

## 8. Splits

Deterministic by `scenario_id` hash: `sha256(scenario_id) % 100` ->
`< 40` calibration, `< 70` validation, else sealed. Written once into
`data/position_split_manifest.json` and never recomputed on the fly.
**Sealed is opened exactly once, at Phase 14.** `tests/test_split_discipline.py`
asserts no script outside `scripts/phase14_*.py` reads `data/sealed/`.

## 9. Artifacts

All under `results/` and `manifests/`, names exactly as listed in the task brief
(§46). Every result file is accompanied by a sibling `*.provenance.json` carrying
git commit, config hash, seeds, model SHA-256s, and package versions
(`statistics/provenance.py:stamp()`).

## 10. Determinism

`torch.manual_seed`, `numpy.default_rng(seed)`, `torch.use_deterministic_algorithms(False)`
(SDPA kernels are not bit-deterministic on this card; we record that rather than
pay the speed cost). Generation is **greedy, do_sample=False** everywhere.

---

## 11. Compute budget (ADDENDUM, measured 2026-08-28 — binding on Phase 4+)

### 11.1 The memory wall is the logit tensor, not attention

Measured on this card, Qwen3-0.6B bf16, SDPA, batch 1:

| T | peak VRAM, full logits | wall clock |
| --- | --- | --- |
| 2048 | 1.94 GB | 1.04 s |
| 4096 | 3.91 GB | 6.87 s |
| 8192 | thrashes / OOM | — |

`logits[1, T, 151936]` in bf16 is 2.5 GB at T=8192 before the float32 upcast. The
attention matrix is comparatively cheap. **Every teacher-forced call must pass
`logits_to_keep=k`** (k = len(answer_ids) + 1), which is all `gold_nll` and
`answer_margin` need. `evaluation/runner.py` must default to this and only compute
full logits when a caller explicitly asks. Greedy generation is unaffected
(it already keeps one position).

### 11.2 The significance matrix is not affordable by brute force

Brute-force causal significance costs `n_items * (1 + L*H*B)` forwards. At
`n_items=256, L=28, H=8, B=8` that is 458 k forwards — days. The lab therefore uses a
**two-tier estimator, and reports both tiers plus their agreement**:

**Tier 1 — attribution (linear response), all cells at once.**
For the Mode-2 gate `s` on cell `(l, h, b)`, `V'[l,h,j] = s * V[l,h,j]` for
`j in bin b`, the exact first derivative at `s = 1` is

```
dL/ds |_(l,h,b)  =  sum_{j in bin b}  < dL/dV[l,h,j] , V[l,h,j] >
```

so `I_hat[l,h,b] = eps * dL/ds` recovers `L(1) - L(1-eps)` to first order, and **one
forward + one backward per item yields every `(l,h,b)` cell simultaneously.**
Cost: `n_items` backward passes, i.e. ~3x a forward. This is standard attribution
patching; it is an approximation and is labelled as such everywhere it appears.

**Tier 2 — true ablation, on a selected subset.**
Real `eps = 0.25` Mode-2 attenuation, one cell per forward, run on:
- the top-K cells by |Tier 1| (K ~ 64), and
- a random control sample of the same size drawn from the remaining cells.

**The validation output is the Tier-1/Tier-2 rank correlation (Spearman) and the
Bland-Altman-style agreement plot on the union set.** If Tier 1 does not track
Tier 2, Tier 1 is discarded and the study falls back to coarse-to-fine ablation on
the six representative layers only — say so in the report rather than proceeding.

`significance/causal.py` exposes both as
`estimate_causal_significance_attribution(...)` and
`estimate_causal_significance_ablation(...)`, plus
`validate_attribution_against_ablation(...)` returning the correlation, its CI, and
the subset used.

### 11.3 Screening scale

Per the brief's §30, mechanistic screening runs on the **calibration split at 4K**
with the five-point grid (0.05, 0.25, 0.50, 0.75, 0.95) and 256–512 examples. The
seven-point grid and the 2K/8K buckets are for the baseline fingerprint (Phase 3)
and the confirmatory runs, not for every intervention sweep. Any script that would
exceed ~2 GPU-hours must print its `cost_estimate()` and require an explicit
`--yes-i-costed-this` flag.

### 11.4 Confirmed environment facts

- All four tokenizers are behaviourally identical: same vocab (151669 addressable),
  same chat template render, same token ids on probe strings. Cross-model positional
  comparison is therefore on **byte-identical token sequences** — a confound removed,
  and worth stating in the report.
- `UNKNOWN` is the single token id 51378, as Supa-A1.1's core metadata claims.
- The tokenizer is fast and returns `offset_mapping`, so evidence spans can be
  located exactly; `benchmark/tokenize.py` should use offsets, not incremental
  re-encoding.
- `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS` is an `AttentionInterface`
  with a `register(name, fn)` method reflected into a global mapping. `register`
  is the sanctioned way to install the `"psa"` kernel.

---

## 12. Kernel design correction (ADDENDUM 2, measured — SUPERSEDES §5.1's eager framing)

### 12.1 The SDPA backend must be forced, everywhere

Measured, Qwen3-0.6B bf16, batch 1, `logits_to_keep=8`:

| T | default dispatch | forced cuDNN/mem-efficient |
| --- | --- | --- |
| 2048 | 0.64 s / 1.94 GB | **0.119 s / 1.26 GB** |
| 4096 | 2.30 s / 3.91 GB | **0.288 s / 1.31 GB** |
| 8192 | 54.2 s / 11.6 GB (spills) | **0.834 s / 1.42 GB** |
| 16384 | OOM | **2.27 s / 1.65 GB** |
| 32768 | OOM | **7.06 s / 2.10 GB** |

Backend availability on this machine: FLASH **unavailable**, MEM_EFFICIENT ok
(4.5 ms), CUDNN ok (1.8 ms), MATH ok but 82 ms and 2.7 GB. Left to itself the
dispatcher picks MATH, which is 8x slower and 3x heavier and is what makes 8K look
impossible. Note also that `enable_gqa=True` forces the MATH path here — **do not
use it**; expand with `repeat_kv` instead.

Therefore every forward in the lab runs inside

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    ...
```

`evaluation/runner.py` owns this context manager and every other module goes
through the runner. 16K and 32K confirmation runs (brief §5) are affordable after
all and should be planned in.

### 12.2 Batch size is 1

At T=4096, bs=1 costs 0.288 s/item; bs=8 costs 0.362 s/item. Batching is a net loss
on this card — it is already compute-bound — and it drags in left-padding, which is
a positional confound in a study about position. **Run batch size 1, unpadded, with
`attention_mask=None`.** `runner.py` keeps its batching code but defaults to 1 and
must document why.

### 12.3 The PSA kernel is SDPA-based, not eager

An eager kernel costs 4.0 GB at T=4096 and 12.3 GB at T=8192 — it cannot carry the
intervention experiments. All three modes are expressible through SDPA:

- **Mode 2 (value scale)** — scale `value` per `(kv_head, bin)` *before* `repeat_kv`,
  then call SDPA normally. Exact, free.
- **Mode 1 (logit bias)** — under absolute binning the bias depends only on
  `(q_head, j)`, so it is an additive mask of shape `[1, n_q_heads, 1, T]` that
  broadcasts over query rows. Add it to the causal mask and pass as `attn_mask`.
  Measured: an explicit mask with the backend forced still runs at 0.36 s / 1.31 GB
  at T=4096, so this costs essentially nothing.
- **Mode 3 (head output scale)** — applied to the SDPA output before reshape. Exact.

**Limitation to document, not to hide:** under *distance* binning the Mode-1 bias is
query-dependent and would need a full `[1, H, T, T]` mask. That is affordable at 2K,
not at 8K. Distance-binned Mode 1 is therefore restricted to T <= 2048 and the
restriction is reported. Distance-binned Mode 2 has no such problem (V scaling is
still per-key) — except that a query-dependent V scale is not expressible at all, so
distance-binned Mode 2 is likewise 2K-only. Absolute binning, which is the primary
grid, is unrestricted.

Eager remains in the codebase for exactly one purpose: **capture** (§5.4), which
genuinely needs the probabilities. It runs chunked over query rows, on one layer at
a time, and never on the critical path of an intervention sweep.

### 12.4 Attribution (Tier 1) budget

Backward, batch 1, `logits_to_keep`, params **frozen**:

| T | no checkpointing |
| --- | --- |
| 2048 | 0.51 s / 5.69 GB |
| 4096 | 12.2 s / 9.87 GB (spills) |
| 8192 | OOM |

So Tier-1 attribution is comfortable at 2K and marginal at 4K. Two mitigations, in
order of preference: freeze all parameters (`requires_grad_(False)`) so no weight
gradients are allocated, and enable gradient checkpointing — which requires
`model.train()`, safe here only because Qwen3-0.6B has `attention_dropout=0.0` and no
other dropout. `significance/causal.py` must **assert train/eval bitwise equality**
before relying on that, and fall back to 2K-only attribution if the assertion fails.

---

## 13. Instrument calibration (ADDENDUM 3, measured on Qwen3-0.6B — binding on Phase 3+)

A pilot of 162 items at 2K/natural, greedy, `enable_thinking=False`, was run before
Phase 3 to check that the benchmark has dynamic range. It does not have it on
accuracy, and it does on NLL. Both facts change the plan.

### 13.1 Exact match is at floor; gold NLL is not

Teacher-forced gold NLL per answer token, `exact_retrieval`, n=6 per cell:

| p | 0.02 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 0.98 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NLL | **0.221** | 0.408 | 0.502 | 0.482 | 0.426 | 0.449 | **0.197** |
| strict EM | 2/6 | 0/6 | 0/6 | 1/6 | 0/6 | 0/6 | 2/6 |

The NLL curve is U-shaped with an edge-middle gap of ~0.27 nats. The accuracy column
carries the same signal at far lower resolution. **Gold NLL and answer margin are the
primary metrics for every mechanistic phase; accuracy is reported alongside but is
never the quantity an intervention is selected on.** This is what the brief's §10
anticipated ("exact accuracy is too coarse for small perturbations"); the pilot
converts that from a precaution into a measured requirement.

### 13.2 Two scoring artefacts, both real, both to be reported not patched away

Observed failure modes that are *not* retrieval failures:

- **Format non-compliance.** Gold `the loom overseer`, prediction
  `The answer is: **The loom overseer**.` The model found it; strict EM scores 0.
- **Truncation.** The model restates the question before answering and runs out of
  budget.

`evaluation/metrics.py` therefore reports **three** accuracy columns, always
together, never collapsed into one headline: `strict` (normalised exact match),
`contains` (gold appears as a normalised span in the prediction), and `first_line`
(current behaviour). `contains` is a *ceiling*, not the score — it is inflatable by a
verbose model and must never be used alone to accept an intervention.

### 13.3 Multi-hop is at floor and must be made tractable

`multi_hop` scored 0/36 on both strict and contains, at NLL 4.4–5.1 — the model has
no signal at all. The brief's §41 requires "at least one genuine multi-hop
middle-context condition must improve", which is unmeasurable against a floor.

`benchmark/families.py` gains a **`multi_hop_guided`** variant: the same two-fact
chain, but with explicit linking language and the bridge entity repeated verbatim in
both evidence sentences, so the chain is followable by a 0.6B model. The hard variant
is retained and both are reported. If the guided variant is still at floor, that is
recorded as a capability limit of the panel and the §41 multi-hop criterion is
declared unmeetable **in the report**, not quietly dropped.

### 13.4 Exact-retrieval IDs are a transcription test, not a retrieval test

Full UUID-like answers are ~13–16 tokens; observed errors include `902eb5c6…` for
gold `402eb5c6…` — a single wrong hex character scoring identically to retrieving
the wrong record entirely. That conflates two distinct failure modes the study needs
to tell apart.

`exact_retrieval` answers become **short nonce keys (6–8 characters, one to three
tokens)**. This stays inside the brief's §7A menu ("nonce keys; IDs; UUID-like
values; arbitrary labels"). The *lookup keys* inside the context stay long and
UUID-like, so the retrieval problem is unchanged; only the string the model must
reproduce gets shorter. Answer token length is recorded per item so the
transcription-length confound can be checked rather than assumed away.

### 13.5 The `<think>` trap

Qwen3 is a hybrid reasoning model. Rendering the chat template *without*
`enable_thinking=False` makes the model emit `<think>` and reason for the whole
generation budget, scoring **0/27** on a pilot where the correct protocol scores
non-zero. The project's canonical protocol is already
`rineya_long_context_demo_v1_bf16_greedy_no_think_24_4096`; the lab matches it.
`evaluation/runner.py` sets `enable_thinking=False` and additionally strips any
`</think>` prefix defensively. Any script that renders a prompt without going through
the runner is a bug — `tests/` should assert the rendered prompt contains the
pre-closed empty think block.
