# Dataset and retention review — 2026-09-15

**Subsequent correction:** RRP-005 found missing procedural evidence and dropped assistant
history in the inherited preservation bank. Its hash is now blocked from new execution.
The RRP-004 training plan below was never run and must not be launched; see
[`../rrp005/PLAN.md`](../rrp005/PLAN.md) for the corrected bank and required new source baseline.

## Decision

The goal remains **ACTIVE, NOT ACHIEVED**. No new model-quality gain has been measured in
this task. The useful next experiment is a corrected, shuffled, bounded Forge v2 control
from the exact ungated Supa 1.7 checkpoint, followed by a matched preservation-replay
contrast. A result must clear a task shortcut floor and demonstrate retention; a lower
training loss or an improved format score is insufficient.

The review concentrated on the dataset/training/evaluation paths and their upstream
evidence. It is not a claim that every file in this large multi-project repository was
reviewed. Findings below were reproduced with executable checks, not inferred from prose.

## Remote reconciliation and model identity

Starting checkout: `research/model-quality-targeted-gpu-v1@8c3ed88`. `git fetch --prune`
showed that upstream branch had been deleted. The live reasoning successor
`research/rineya-reasoning-posttraining-v1@d417e100ac10d225145f9701914854d10c55b43f`
was **32 commits ahead, zero behind**. Created `codex/dataset-retention-audit-v1` from the
starting head and fast-forwarded it to that successor. Existing local branches, worktrees,
models, untracked release contents and historical grants were preserved.

Other live additions reviewed at their branch boundaries:

| Branch/head | Decision |
|---|---|
| `research/opallo-native-collector-v1@171a301` | Useful bounded collection/instrument work; 19 added/changed files relative to its merge base. Remains isolated pending its own native qualification. It is not a trained model and does not repair Opallo's reasoning. |
| `research/cross-scale-stitch-graft-framework-v1@3059e57` | Distinct architecture programme; 532-file branch delta includes extensive historical evidence. No blanket merge into the dataset experiment. Weight grafting would introduce another independent intervention before establishing a sound data control. |
| `origin/main@a6ee824` | Historical programme-audit branch, not the newest posttraining state. |

Fresh authenticated Hugging Face metadata distinguished **local registration** from Hub
publication:

| Model | Verified location / identity | Interpretation |
|---|---|---|
| Supa 1.7 | Private `Luncy1978/rineya-published-cores`, revision `d11707ee75481c245ef44491079737fde1f6b910`; weight SHA256 `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` | Exact ungated 596,049,920-parameter parent supported by the qualified trainer. Actual complete local assets are in `Supa/`; the published-core metadata directory in this checkout lacks its tokenizer payload. |
| Supa-A1.1 | Private `Luncy1978/Rineya-Supa-A1.1`, revision `199eed212ab3a3db487270dc6f97b16756f7bff0`; weight SHA256 `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026` | Separate FP16/custom-abstention serving model. Do not substitute it for the ungated source or inherit its scores. |
| Opallo 1.0 | Local `Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/CORE.json` | Registered experimental Supa descendant. Absent from the current account model listing and family-vault file inventory; current Hub publication was **not verified**. |

The inherited evidence reports Opallo's targeted panel gain (47/150 → 73/150 for stored
weights) alongside losses: 6 of the source's 47 successes, long-context strict 161→135/256,
and weaker puzzle/Forge probes. These are historical development results under differing
protocols, not a global ranking. See `MODEL_QUALITY_STATUS.md` and the original raw records.

## Concrete defects and corrections

| Finding | Reproduction / effect | Correction |
|---|---|---|
| Bootstrap resampling was invalid for power-of-two cluster counts | LCG `% k` cycles through each cluster once for k=8/16/32. Eight effects split equally between +1 and −1 produced interval `[0,0]`. At k=32 the code even labelled this interval reliable. | Independent with-replacement MT19937 draws, RNG recorded; tests against the closed-form binomial distribution at all three k values. Exact sign-flips now enumerate only nonzero effects while retaining the original denominator. Historical statistics are not rewritten. |
| Arm data cannot reach native evaluator | Real `compile_training()` output lacks `reference`; native probe raises `KeyError('reference')`. Existing tests used hand-built rows and missed the seam. | Emit canonical `reference` alongside the released `reference_text`; real builder-to-evaluator integration test. |
| Held-out evaluation dies after training | `ins` was imported only in the baseline branch but used in the training branch. Executing the held-out block reproduced `NameError`. | Dedicated exercised helper with its own import; keep window-qualification receipt; check candidate parameters unchanged and bank bytes unchanged. |
| A1 grant cannot start its own first training run | Registered adaptive budget is zero; `Journal.start('train')` reserves one round and refuses before launch. | Preserve A1 and its refusal. Register RRP-004 separately with **one** accounted round and a fresh, data/plan-bound grant. No old permit is consumed. |
| New code/data could train without earning a matching source qualification | Training dispatch did not require or execute qualification; historical passes have different checkout/data bindings. | RRP-004 performs zero-update native/mask/cache/gradient/2048-shape qualification in the supervised worker before allocating the optimizer. |
| Leakage audit can pass without checking its claim | Empty banks accepted; malformed prompts skipped; `parent_id` alias not checked. All three reproduced. | Reject empty/malformed banks; check parent/group/ID identity. Fresh audit of 726 TRAIN worlds versus all 1,488 registered DEV rows is clean on the seven declared axes. This does not prove grammar independence. |
| Nominal order seed was not used to mix families | Prepared A1 contains runs of 60 examples from one family/status bucket. | Separate experiment preserves the selected world set and hash-shuffles globally with seed 17. Longest bucket run becomes 3. This is a declared intervention, not an exact A1 replication. |
| Proposed candidate evaluation differs from preservation baseline | Source preservation run uses 2048 context / 64 generated tokens; proposed arm used 4096 / 128 for every bank. | Freeze per-bank hashes, windows and caps; prohibit CLI overrides or bank substitution for RRP-004. Forge v1/v2: 4096/128; preservation: 2048/64. |
| Qualification call budget understated work | Reserved 2 calls despite a chunked cache loop and five other top-level forwards. | Reserve `5 + ceil(native_probe_tokens/3)`; preserve processed-token reservation. Backward checkpoint recomputation is covered by worker time, not described as a separate evaluation request. |

The inherited suite passed 226 tests before these reproductions. New targeted tests first
failed on five concrete defects. After correction, the RRP suite passed **239 tests** and
the reused MQ suite passed **113, skipped 1** (Windows symlink privilege). These are CPU
mechanics tests, not pretrained-model improvement evidence.

## Data quality and efficient use of the new releases

Forge v1 has 10,000 canonical worlds; Forge RL v2, created September 14, has 20,000,
including 16,000 TRAIN worlds. v2's final/reasoning SFT, preference, process, episode and
layout views inherit parent IDs. They are **not additive independent tasks**. The new
preparation byte-verifies the v2 archive against its pinned SHA256 and all 95 release
files, invokes the release's solver on selected TRAIN targets, and reproduces native
tokens/masks. Every selected target passes: **726/726**.

Actual exposure: **1,161,808 processed tokens; 9,169 supervised tokens; 726 updates; one
exposure per selected world**. Only **0.789%** of processed tokens carry loss. The sparse
signal motivates compact, verifier-checked intermediate-state targets in a later arm.
It does not justify deleting evidence or truncating a reasoning trace.

The 2048 training window excludes most long Forge v2 worlds. In the selected set,
`evidence_chain` has **2 ANSWER + 4 UNKNOWN**, not the full eligible pool's 4 + 9.
All other twelve family/status buckets have 60 rows each. No meaningful standalone
evidence-chain gain can be inferred from six examples. This is a **short-context dose
probe**, not a comprehensive long-context training run.

Use existing corpora by function, with explicit admission checks:

| Source | Intended role | Admission / limitation |
|---|---|---|
| Forge v2 TRAIN | Planning, execution, constrained optimization, structured answer contracts | Verifier gold checks; parent-group splits; report length/family skew; no blind DPO/RL import. |
| Forge v1 TRAIN | Arithmetic/state/relational primitives and cross-grammar transfer | Independently solved TRAIN rows; existing test/validation panels stay out of training. |
| General reasoning/natural communication v3 + short-response TRAIN | Ordinary dialogue, concise instructions, grounded answerability replay | Audit actual generator families, targets, rights and overlap against preservation and older panels. A released answer is not automatically verified gold. |
| `residual_v23` TRAIN | Evidence-grounded reasoning and procedural retrieval replay | Keep DEV/OOD-DEV/test and all descendants quarantined; pair missing-evidence and answerable scenarios together. |
| Puzzle reasoning TRAIN | Exact-checked combinatorial/arithmetic expansion | Useful supplementary domain, not a replacement for broad preservation. Historical tests are exposed DEV. |
| v2 process/trace/preference/episode views | Compact steps and causal state learning, later verifier-qualified RL | Share parent partitions with final-only rows; validate every compressed step; no future-state leakage; no reward-credit from output syntax alone. |

Replay is a research-supported candidate for retention, not a guarantee. GeRe tests general
sample replay and several regularizers ([paper](https://arxiv.org/abs/2508.04676)); I-LoRA
tests replay/interpolation in parameter-efficient continual learning
([paper](https://arxiv.org/abs/2402.18865)). The empirical continual-finetuning study measures
forgetting across knowledge, reasoning and comprehension
([paper](https://arxiv.org/abs/2308.08747)). Their results motivate our controlled contrast;
none supplies evidence that Rineya has improved.

## Bounded experiment ladder and acceptance

1. **RRP-004 A1 corrected control**: exact Supa, FP32, blocks 8–27, response+EOS loss,
   inherited 1e-5 WSD/AdamW recipe; newly verified/shuffled 726-world bank. One supervised
   worker; fresh in-run native qualification; preserve update-2 and final full-state
   checkpoints; evaluate all three registered DEV banks under matched protocols. No
   recipe selection within this arm. Report failed launches and all resource costs.
2. **Replay contrast, separately frozen before launch**: target processed-token proportions
   40% preservation/grounded language, 20% Forge v1 primitives, 40% Forge v2. Same source,
   recipe and approximately matched processed-token ceiling as control. Enforce actual
   full-row token quotas and pair integrity, not advertised pool percentages. This is a
   hypothesis; do not silently select ratios after reading outcomes. Its data admission,
   code, plan and budget are not yet qualified.
3. **Compact verified reasoning contrast** only if replay retains skills: match parent
   worlds, compare direct answers with bounded evidence/operator/state/check targets.
   Qualify active-thinking serialization and new token budget. Count all emitted reasoning
   tokens and latency. Do not confuse an 80-character final answer with cheap reasoning.
4. **Confirmation and transfer**: freeze recipe and margins, repeat independently selected
   TRAIN draws, then use a prospectively protected independently authored instrument and
   the intended serving dtype. Existing Forge holdouts are developer-visible; calling a
   new seed independent would be false. Other programmes' `fresh_sealed` files remain unread.

Inherit the predeclared RRP-002 P1–P4 primary contrasts and Holm family; document the RNG
correction before candidate outcomes. Report row- and cluster-weighted effects, ANSWER /
UNKNOWN / IMPOSSIBLE separately, retention per sub-bank, minimal-pair evidence sensitivity,
format, termination, tokens, wall time and memory. Non-significance is **not** equivalence.
Source baselines already fail the shortcut floor on v2 and ordinary preservation, so a
candidate must improve useful behavior as well as its numerical baseline.

The actual preservation evaluation includes **29 complete pairs**, not all 40 in the bank.
The inherited +6-pair practical requirement remains +6; report denominator 29 and apply
the registered multiplicity adjustment. The prose assertion that six moving pairs can
always establish P4 at the family-wise level is incorrect (Holm may require more).

This instrumentation does not cover every capability. Coding, multilingual behavior,
tool use, open-domain factuality, safety, creative dialogue and long contexts beyond the
qualified window have **no new validated gains**. They require independent retention
panels before a broad family-model release claim.

## Compute and current limitation

Host: RTX 4070 Laptop 8 GiB, Ryzen 9 7940HX (16 cores / 32 threads), 15.22 GiB physical RAM.
Use one GPU model worker and one-thread numerical CPU work to preserve RAM/headroom.
No cloud/teacher spend, process termination, checkpoint deletion or source-weight changes.

RRP-004 ceiling: 12,000 model-worker/GPU seconds, 16,000 owned wall seconds, 4,000,000
processed training tokens including failures, 3,000 logical evaluation/qualification calls,
262,144 reserved generated tokens, 10 GiB new disk, two launches, one retry, one model-selection
round. These are conservative **agent-proposed/instantiated limits**, not numbers quoted
from the user's message. A grant must bind the actual prepared hashes before execution.
One planned attempt is roughly 45 minutes training plus evaluation, subject to actual speed.

RAM observations during review ranged roughly 3.8–5.0 GiB after transient lows. The unchanged
training preload minimum is **6.0 GiB after native runtime import**. No other process was
closed and no guard lowered. GPU availability alone does not make a run admissible. The
launcher must record an actual resource refusal or pass, never substitute these observations
for a live check.
