# CRS-001 exact-prefix reader qualification design review

Recorded 2026-10-02, America/Chicago. Reviewer `/root/scientific_audit`.
Primary-checkout, source-only review: no framework/model imports, binary hashing,
new cases, protected labels or native calls. Read
`contracts/READER_QUALIFICATION_DRAFT_V1.json`, original capture helpers/source,
central Amendments 015/016 and the retained native D2 evidence/reviews. Shared
model lineage/workspace privileges remain disclosed.

Disposition: **CONDITIONALLY VIABLE SEPARATE SERVING CONTRACT; DRAFT NOT
LAUNCH-READY.** An exact-prefix deterministic serving implementation can be
qualified as a distinct endpoint if the controls below pass. It cannot qualify
the failed bulk capture, bulk training likelihood, existing central E1 or any
reader simply by replaying its own probabilities.

The parent's latest native validation was refused before dispatch for insufficient
RAM and unregistered Photos/Unity GPU processes. Those recorded admission failures
remain blockers; this design review supplies no fit forecast, resource exception,
restart permission, process termination or budget reset.

## 1. Scope decision and historical boundary

The original capture gate compared online prefix model logprobs with one full
prompt-plus-generated teacher-forced sequence. It failed maximum 0.03125 and
mean 0.01 limits on the mechanical 128-token prompt. D1 reproduced the discrepancy
upstream of the head; changing the head alone to FP32 did not resolve it. D2
recorded fixed-shape future-suffix invariance for the tested 131-token trace.
These facts do not establish length/shape invariance or a GEMM/attention cause.

Amendment 015 explicitly limits canonical per-prefix replay: it cannot relabel
the failed bulk gate or authorize that capture. Amendment 016 allows a later
separately registered exact-prefix scoring scope after causal controls, retaining
the original limits/failure and prohibiting bulk-training equivalence or stochastic
RL inference. The owner subsequently authorized bounded CRS implementation and
qualified local inference; that instruction does not waive numerical gates.

Therefore distinguish two questions in the new contract:

| Question | Permissible decision |
| --- | --- |
| Does the new deterministic endpoint faithfully implement the pinned stock no-cache full-prefix greedy path? | May earn a new scoped serving qualification after independent forward/generate/reference controls and complete native supervision |
| Does bulk teacher forcing agree with online prefix likelihood under the original limits? | Remains FAILED/UNRESOLVED; no successful exact-prefix replay substitutes for this gate |
| Does any trained model acquire/retain capability or internalize procedures? | Remains a separately registered, unqualified learning/training question |

Use a new namespace such as `CRS_EXACT_PREFIX_GREEDY_SERVING_576_V1` and an
immutable qualification receipt identifying its allowed operations. Keep
`capture_qualified`, existing central launch flags and training/RL permissions
closed. Do not update the original gate or call the new endpoint a repaired bulk
capture. A caller-provided PASS dictionary or D2 receipt cannot open it.

The existing retained native worker observations permit development/registration
of this proposal. They do not supply complete-path serving qualification: the
old supervisor failed and the newest validation was admission-refused. Before
DEV inference, require a fully admitted/contained/complete new qualification path
under the final reviewed supervisor, or an independently reviewed equivalent
complete qualification result. Registration and CPU preparation can proceed now.

## 2. Required reference and parity controls

Freeze exact assets, source commit, runtime/source-file identities, BF16 arithmetic
flags, eager/eval/no-gradient settings, one resident model, batch one and no cache.
Use no adapters. Exchange structured source/token fields; no arbitrary code or
scorer/gold inputs enter a worker.

1. **Loader/native forward identity.** Bind every weight/config/tokenizer/template
   asset and loaded tensor inventory; verify tied embeddings/head and frozen
   parameters before/after. Reproduce the known mechanical prefix and failed
   bulk reference endpoints from original immutable evidence. Bind actual masks,
   positions, dtype, shape and finiteness. A baseline drift stops its affected
   scope; do not refresh expected bits after outcomes.
2. **No-op serving wrapper.** Compare direct stock forward and the owned wrapper
   on identical token streams/shapes/settings, including the maximum planned
   shape. Require raw same-dtype logits/hidden parity, not merely equal argmax
   or numeric `torch.equal`. CPU zero-operation fixtures cannot replace exact-
   checkpoint parity. If an adapter is later introduced, it needs its own
   zero-adapter qualification; this no-adapter contract does not grant it.
3. **Independent generation-loop reference.** Compare the owned manual greedy
   loop with the pinned library's stock `generate`, sequentially on the same
   resident checkpoint. Explicitly match masks/positions, cache=false,
   do_sample=false, one beam/return, EOS/pad IDs, maximum new tokens, template,
   and the complete effective logits-processor/stopping configuration. Require
   identical emitted IDs, stop/EOS/rendering and raw lowest-ID argmax evidence.
   A different active processor, hidden min-length policy, speculative decoding,
   repetition penalty or cache path changes the experiment.
4. **Exact-prefix offline replay.** For emitted token j, independently reconstruct
   `prompt + generated[:j]`; predict token j from that prefix's last row. Replay
   uses exactly the realized prefix geometry and the same retained-logit policy.
   Bind full token IDs/masks/positions, selected logits/model logprobs and all
   attempts. Require same-prefix raw parity where declared plus the unchanged
   0.03125/0.01 probability-error limits. Replay alone establishes only replay/
   extraction consistency, not an independently qualified decoder or bulk repair.
5. **Keep bulk comparison separate.** Reproduce/measure bulk-prefix disagreement
   as a disclosed numerical diagnostic with the original limits. Any failure
   keeps that bulk operation ineligible. Predeclare separate serving and bulk
   outcomes before execution; never decide after seeing failure that a formerly
   required gate is merely diagnostic.

Installed source inspected without importing Transformers:
`generation/utils.py:2508–2512` sets logits_to_keep=1 when supported;
`:2800–2810` retains raw float32-upcast logits separately from processed scores;
`:2831` selects argmax in the deterministic path. Qwen3 forward explicitly slices
retained hidden rows before the tied head. Thus the reference must use **raw
logits**, not processed `output_scores`, and compare dtype-normalized raw values
with explicitly registered conversion when stock generate returns FP32 copies
of BF16 logits. No FP32 model/head intervention is implied by that copy.

Model softmax logprob and deterministic behavior logprob differ: verified greedy
behavior has point-mass logprob zero for its selected token, while raw model
logprob usually does not. Neither is semantic confidence/calibration. Stochastic
sampling, RL rewards/reference likelihood and bulk CE remain separate gates.

## 3. Tokenizer, EOS and maximum-shape contract

Before native execution, qualify tokenizer/template mechanics on CPU when resources
permit; this review does not run them. Bind exact rendered UTF-8 bytes, token IDs,
tokenizer/template versions, system/user/assistant prefix, thinking setting,
special IDs, EOS list and decode settings. Reject unsupported turns or overflows.
The 512 input tokens include all instructions, source, state, procedure/schema and
chat-template overhead. Never truncate decisive evidence or a verified target.

An input of 512 with 64 emissions ordinarily performs prediction forwards at
prefix lengths 512 through 575. The final complete sequence has 576 tokens;
add an explicitly non-emitting 576-token forward to qualify that maximum shape.
Do not accidentally emit a 65th token or call the final forward an extra answer.

The draft's “64-step sustained generation even if natural EOS occurs” must be
split into distinct conditions:

- **Natural serving:** preserve first emitted EOS and stop; prompt EOS does not
  stop a new response. Record actual output count and raw/visible text. A short
  natural rollout cannot be called an observed 64-token rollout.
- **Worst-shape/sustained mechanics:** use a frozen non-evaluation token continuation
  to run all 64 full-prefix geometries and the final 576 shape. This tests
  sustained resource/shape behavior, not unconstrained model generation. Suppressing
  EOS or injecting continuations changes the treatment and must retain that label.
- **Cap/EOS code branches:** independently test both EOS IDs, EOS at the cap,
  empty/malformed output and exact cap enforcement on CPU. Require at least one
  native natural EOS if the contract claims its observation. If native natural
  cap-stop is not observed, report it as unobserved; do not fabricate it from
  forced prefixes. Any qualification statement must separate observed native
  branches, tested engineering policy and mechanically supported maximum geometry.

At the maximum shape, add preregistered same-shape suffix-causality controls with
passive mask/position/normalized-hidden witnesses, retaining the existing bitwise
rule. These cannot infer invariance over every length or token distribution.
Every actual input/output/logits-retention shape belongs in the receipt; a bulk
diagnostic retaining N+1 rows has a different output envelope from keep=1 serving.

## 4. Concrete bounded qualification schedule to finalize

One possible compact engineering schedule is below. This is a prospective upper
envelope to compile and independently review, **not a launcher or fit claim**.
It includes necessary path/reference/resource controls; do not expand it into
a hyperparameter or efficacy sweep.

| Control block | Maximum forwards | Maximum forwarded input positions |
| --- | ---: | ---: |
| Direct versus no-op wrapper at 128,512,576 | 6 | 2,432 |
| Historical natural 128-prefix trace: owned/reference/replay, frozen three emissions | 9 | 1,161 |
| 512-input owned/reference/replay, at most 64 emissions per natural rollout | 192 | 104,352 |
| Frozen sustained 64 prefixes 512–575 plus non-emitting 576 probe | 65 | 35,360 |
| Historical bulk131 and long bulk≤576 diagnostics | 2 | 707 |
| Maximum576 baseline plus three fixed-shape suffix interventions | 4 | 2,304 |
| **Prospective total** | **278** | **146,316** |

With the existing one-construction/one-load/310-copy definition, this is at most
590 logical model/load/copy calls; manual/reference natural emissions total at
most 134. Mechanical continuations/replayed targets are separately counted
input exposure, not newly generated model tokens. Length-dependent counts must
use actual natural N, retaining aborted/unavailable blocks without reducing
planned denominators. The historical three-emission requirement is a frozen
reference condition; drift stops it rather than extending a search for passing
outputs.

Freeze complete token streams, reference settings, output-retention geometry,
operation accounting and any smaller equivalent schedule before launch. Retain
600 seconds per job, one worker, 2 GiB RSS cap, 1.25 GiB system reserve,
3.3125 GiB pre-worker budget, GPU4/0.5 GiB and disk4 GiB or stricter measured
limits. Qualification/failures count within 7,200 seconds/epoch and 14,400
seconds/rolling24h. No dummy calls/tokens, tolerance change, precision change,
paid service or resource-floor reduction earns admission. If this actual schedule
cannot fit/time-complete, register a bounded segmented qualification with explicit
aggregate costs and unchanged requirements, or leave the larger envelope closed.

## 5. Alternative when the old numerical gate is mandatory for the intended use

If the intended “reader” operation consumes bulk likelihood, trains with a claimed
equivalent bulk objective, implements stochastic likelihood-dependent control,
or inherits a protocol explicitly requiring the old capture PASS, exact-prefix
replay would bypass a required gate. **Keep that operation closed.** It must
repair/qualify the original comparison or register a genuinely different
implementation with its own complete numerical/serving contract; it cannot use
a renamed receipt to pass the original study.

A concrete separately registered diagnostic alternative is full stock eager
**FP32 backbone execution from the same BF16 source tensors upcast on GPU**,
compared with its own independent stock reference at matched prefix/bulk
geometries under the unchanged likelihood limits. It addresses the observed
upstream discrepancy rather than repeating the failed head-only FP32 comparator.
It is a different arithmetic representation and needs its own loader/no-op,
gradient-independent, resource and serving checks; no result is forecast.
Weights alone require about 2.22 GiB VRAM before activations/driver/state, which
is an analytic lower bound, not fit evidence. Do not allocate a full CPU copy,
overwrite source weights or infer BF16 qualification from an FP32 pass. A single
predeclared reduction-flag intervention is another possible mechanism diagnostic,
but likewise changes math and requires a separate frozen comparison. Neither
alternative is executable under the current failed host admission.

## 6. Reader diagnostic and remaining scientific gates

After scoped serving qualification, separately register exactly 16 open DEV
controlled-grammar reader cases. These are source-interpretation/extraction
diagnostics, not natural-language universality, retrieval success, native
attention extension, capability ceilings, autonomous discovery or system efficacy.
Record extraction fields, validity, negation/modality/time/aliases/cancellation,
raw output/termination, errors and independent grades separately from final owner
correctness and formatting. Calibration remains unmeasured without an explicit
instrument. Case families/targets/scorer feedback must not leak through prompts,
retrieval, state or logs into deployed workers.

Compile each required output before the diagnostic freeze. A complete multi-event
JSON target may exceed 64 output tokens. If so, use a prospectively bounded
per-span/per-field multi-call diagnostic with common source access and full costs,
or register/qualify a larger output envelope. Do not silently truncate events,
backfill model extraction with the supplied parser, give gold event records, or
change scoring after results. Preserve source byte offsets, tokenizer identity
and instance resets. Native failures stay failures, not correct UNKNOWN fallbacks.

Reader implementation, validated-procedure discovery and microtraining remain
separate studies. A source-faithful serving PASS permits only its qualified
deterministic endpoint and registered DEV diagnostic. It establishes no B/C
mechanism gain, weight learning, retention, internal mechanism, broad preservation
or publication/release qualification. The exact-prefix contract and launch
validator should enforce these narrow allowed operations rather than opening
all old `native_qualified`/capture/training flags.
