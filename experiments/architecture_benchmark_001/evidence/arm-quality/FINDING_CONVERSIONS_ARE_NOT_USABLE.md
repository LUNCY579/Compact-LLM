# Initialization finding: neither CED conversion is usable *before training*, and the gate ladder never tested for it

> **Scope — untrained conversions only.** Every measurement below is taken at step 0, before
> any gradient step. The original title said "decisive"; it is not decisive about the design,
> only about its initialization. Two forward-looking statements here have since been measured
> false: "they are not degraded but recoverable" (§1) and "nothing in the existing evidence
> suggests 1M tokens would be close to sufficient" (§3). See
> `evidence/trainability/RECOVERABILITY_FRONTIER.md` and the completed 500,099-token run.

Exploratory diagnostics at implementation `fc44701e`, pinned runtime, official
Supa 1.7.0-devfreeze weights. Run **outside** the experiment package so the implementation
digest and every existing receipt stay valid. Not receipts; these authorize nothing.

## 1. Teacher-forced quality on the open calibration split

24 calibration rows, 3,271 scored target tokens, CPU FP32.

| arm | mean CE (nats) | perplexity | KL to source | top-1 agreement with source |
|---|---:|---:|---:|---:|
| B28 frozen source | 5.546 | 256 | 0 | 1.000 |
| D40 ordinary width | 5.360 | 213 | 1.468 | 0.500 |
| **C28 CED** | **12.585** | **292,267** | 9.154 | **0.014** |
| **C40 CED** | **12.960** | **425,015** | 9.878 | **0.004** |

`ln(151936) = 11.93`. **Both CED arms score worse than a uniform distribution over the
vocabulary.** As *initializations* they carry negative information about the next token.

That is an initialization deficit, not a verdict on recoverability, and the stronger reading
(originally written here as "they are not degraded but recoverable") has since been falsified
for small displacements: `ced_last4`, at 3 displaced layers, recovers to 69/180 against a
matched control's 73/180 — overlapping 95% intervals. What does **not** recover is C28 at 13
displaced layers: 0/180 after 40 steps, and only 2/60 after a completed 500,099-token run.
See `evidence/trainability/RECOVERABILITY_FRONTIER.md`.

## 2. Greedy generation on the open pilot split

24 new tokens max, greedy, official chat template.

| arm | exact hits | terminated | distinct outputs | behaviour |
|---|---:|---:|---:|---|
| B28 frozen source | **6/24** | 24/24 | 13 | works — answers `3`, `5`, `7`, `9` correctly |
| D40 ordinary width | 0/24 | 18/24 | 4 | **degenerate**: repeats `</think>\n` indefinitely |
| C28 CED | 0/24 | 0/24 | 24 | multilingual token salad, never terminates |
| C40 CED | 0/24 | 0/24 | 24 | multilingual token salad, never terminates |

Verbatim C28 output for `pilot-d1-v0` (expected `3`):

    ``fredou间距تك pred usamovereito以下-added itdagioventions fleck itfersатьwapish

## 3. What this means

**The ordinary control is broken too.** This is the part that invalidates the existing
four-arm design, not just the CED arms. D40's CE looks *better* than the source (5.360 vs
5.546) while its free generation collapses into a `</think>` loop — the classic signature
of a model that survives teacher forcing and fails autoregressively. Only the frozen
source B28 is a working model.

**The gate ladder never tested capability.** `G1_WIDTH` PASSes on
`registered_dimensions`, `all_tensors_accounted`, `finite_converted_logits`,
`save_reload_exact`, `source_unchanged`, `calibration_unchanged`, `exact_parameter_count`.
Every one of those is satisfied by a model that emits token salad. `G1_CED_*` adds parity
checks, but parity is a *self-consistency* property: D40 passes the full `compare_paths`
grid with **zero** violations while being degenerate, and C28 fails it while being equally
degenerate. Neither fact bears on quality.

**Therefore this session's numerical work measured a property that does not matter yet.**
The FP32 cache-order parity question — the shared bank bypassing final-norm suppression,
the failed read-side precision fix — is all real and correctly measured, but it is a
hygiene property of a model that cannot produce text. Chasing a 1e-4 logit tolerance on a
model at perplexity 292,267 is the wrong order of work. Recorded plainly because it was
the instruction, and because the negative result still rules out a route.

**"Recovery" is the wrong word for what these arms need.** A 1M-token budget was framed as
recovering a structurally-transplanted model toward its teacher. From CE 12.6 — worse than
uniform — that is not recovery, it is training a new architecture from a bad initialization.
The ~9 supervised targets/s behind this estimate came from the 25-step *cached-distillation*
smoke, which also projects 6,268 teacher head positions per step. A dedicated plain-supervised
long-budget run on the same CPU FP32 runtime measures **27.0 supervised targets/s** — 500,099
targets in 18,496 s including nine evaluation checkpoints, i.e. ~10.3 h per 1M per arm, not
~31 h. That run has since been done, and it did **not** restore capability (0/60 → 2/60), so
the conclusion below survives while its arithmetic does not.

The original estimate (~9 supervised targets/s, ~31 h per 1M tokens per arm) makes
that infeasible here, and nothing in the existing evidence suggests 1M tokens would be
close to sufficient.

## 4. Why the conversions destroy the model

`to_ced` keeps encoder layers 0..13 of the donor, forms one shared K/V bank from layer 13's
output, and turns donor layers 14..27 into `ReadBlock`s. A `ReadBlock` has **no
self-attention**: its `q_proj` queries only the shared encoder-final bank. So weights
trained to attend over their own layer's keys are asked to attend over a completely
different key space, and decoder positions can no longer see each other at all. The
information flow is changed radically while the weights are reused unchanged. `to_ced`
already records `function_preserving: false`; these numbers are the size of that.

By contrast `G1_IDENTITY` — pure depth expansion — is **bitwise exact**
(`max_abs_logit_error = 0.0`). Lossless structural change is achievable in this codebase.
The losses come from (a) the 1024→896 width reduction in `calibrate_d40` and (b) the
self-attention→shared-memory replacement in `to_ced`.

## 5. The design implication

Any CED-like architecture worth training should be **function-preserving at
initialization**, so it starts at source quality and can only improve. The standard way to
get that is a zero-initialized gated read: keep the decoder's own self-attention, add the
shared-memory cross-attention alongside it, and zero-initialize the added output projection
so the contribution is exactly nothing at step 0. Then the KV saving is realized later, by
training the gate up and pruning per-layer attention, rather than assumed at step 0 and
paid for in destroyed quality.

That is a staged program with a measurable first milestone — CE and parity identical to the
source at init — rather than a conversion whose starting point is worse than random.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY <scratchpad>/arm_quality.py 24     # table 1
$PY <scratchpad>/arm_generate.py       # table 2
```

Both scripts live outside `experiments/architecture_benchmark_001/` deliberately: adding
them under `ROOT` would change `implementation_digest()` and invalidate every receipt.
Their content is reproduced in this branch's history alongside this file.
