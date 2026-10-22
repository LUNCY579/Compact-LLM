# Prospective RSD-001 protocol and falsification plan

## Scientific object

Test whether an answer-privileged copy of Titan can teach unaided Titan at the
states Titan actually visits. This is a constrained post-training adaptation of
DCE/SRCL, not a reproduction of the paper's model sizes, training budget, long
reasoning regime, curriculum or benchmark results. It makes no frontier-level
capability claim.

For a student-generated response y, each prediction position uses the same
response prefix y_<t in both roles. The student has problem x; the teacher has x
plus verified solution g and the registered transition. The full-vocabulary
objective is mean_t sum_v q(v)*(log q(v)-log p(v)). Teacher probabilities are
detached. The frozen teacher always uses the common initial adapter; the evolving
teacher uses the current pre-update adapter. One update includes fresh student
rollouts, privileged scoring, gold-free rewrites where enabled, losses, and exactly
one optimizer step. No multi-epoch reuse or delayed teacher refresh is implemented.

Short rewrites receive only x and the original response. They enter CE only when
naturally terminated, strictly shorter in tokenizer tokens including EOS, and
accepted by the registered endpoint/format verifier. The student CE input is the
ordinary problem, not the verbose attempt. Concise CE is token-mean over accepted
rewrite tokens. Guidance, normal SFT and replay are each sequence-mean. Replay has
a fixed independent loss weight and count across treatments. Every component's
weighted loss and supervised-token denominator are recorded.

## Variables and controls

Primary student: exact 596,049,920-parameter Titan; BF16 frozen base, FP32 rank-8
q/v LoRA (56 sites, alpha16). No layer/rank unlocking. Every treatment starts at
the same reviewed SFT adapter and resets AdamW state. Fixed plan seed controls
parent ordering and independently namespaced generation/rewrite/evaluation RNG
streams. Tokenizer, prompt wrapper, answer placement, decoding, replay, optimizer,
context and response cap are bound. Generation is categorical temperature1,
full vocabulary, no KV cache; evaluation is greedy at the same frozen cap.

| Contrast | Hypothesis | Important alternative explanation |
|---|---|---|
| evolving minus frozen | Updating a privileged teacher transfers useful improvements. | No effective guidance gap; moving targets become unstable; extra mistakes are amplified. |
| evolving minus continued SFT | Fresh same-prefix guidance helps beyond ordinary verified demonstrations. | Ordinary SFT is sufficient or better at this budget. |
| evolving_srcl minus evolving | Accepted concise rewrites improve the accuracy/length tradeoff. | Extra CE/compute or endpoint shortcuts, not useful compression. |
| frozen_srcl minus frozen | Concise learning helps without teacher co-evolution. | The benefit is primarily rewriting, independent of recursion. |
| combined versus srcl_only | Guidance contributes beyond accepted-rewrite CE. | Rewrite supervision alone explains the apparent improvement. |

Keep frozen/evolving privilege placement identical. The paper changes prompt
placement as well as teacher dynamics compared with original OPSD; our matched
control deliberately separates these factors. An optional original-OPSD placement
arm is not implemented and must be a separate registered amendment.

## Suggested sequence

1. Pass exact E0 and fit the actual task/solution tokens without truncation.
2. Run or supply a normal-SFT common parent and review its unaided behavior and
   preservation. The optional `parent` mode is a separate RSD parent recipe, not E1.
3. Run the eight-parent TRAIN_SELECTION privilege probe; treat it as diagnostic.
4. Qualify and screen SFT/frozen/evolving with one seed and frozen task order.
5. Add SRCL factorial/only arms after confirming the representation and costs.
6. Replicate finalists with independently registered seeds/training draws; then
   perform untouched confirmation outside this adaptive CLI.

Do not add recurrence, another RL loss, pruning, merging or dynamic curriculum to
the first screen. Those changes would obscure why an effect appeared.

## Data and measurement

Split by global parent before generating variants. Preserve source versions,
licenses/provenance and genuine references. Import historical exclusion identifiers
without reading protected test content. Exact/normalized prompt exclusion is not
semantic decontamination. Ensure no common-parent SFT training data contaminate
confirmation panels; a manifest/reviewer attestation cannot establish that alone.

The first native path supports deterministic text endpoint graders only. A
verified final answer is not a verified proof. Audit intermediate reasoning and
rewrite content independently, especially answers that appear correct but rely on
invalid deductions or references to removed context. Do not claim general language
retention from an arithmetic-only retention panel. Broad retention graders require
separate qualification; this package does not silently borrow unqualified labels.

Primary operational measure: parent-macro single-attempt correctness at the fixed
response budget. Report truncation, length, malformed/empty/wrong answers, repairs
and damages, and every retention source, not just pooled accuracy. An answer that
happens to be right but fails the prescribed EOS/format contract does not pass.
The eight-parent continuation probe controls the current prefix but is too small
and selected to establish causal generalization. A larger independent wrong-prefix
recovery bank and step validators are registered follow-ups, not implemented
measurement guarantees.

## Statistical interpretation

Freeze the comparator, evaluated checkpoint, token cap and minimum useful effect
before outcomes. Default minimum useful gain is 0.02 (two percentage points); this
is a design proposal requiring task-specific justification and power planning.
A small feasibility panel has limited power. Bootstrap parent means rather than
pretending paraphrases are independent. Calibration is repeatedly monitored, so
its percentile intervals are not sequentially valid confirmatory intervals.

The CLI labels an interval wholly above the practical threshold as a candidate
requiring untouched confirmation. An upper bound below the threshold means the
claimed practical gain is not supported on that panel/model/budget. Otherwise the
result is inconclusive. None is an unrestricted statement that recursive learning
is proven or impossible. Use paired seed-level effects for replicated arms; do not
pool row-by-seed observations as independent samples. Multi-seed/hierarchical and
multiplicity-adjusted confirmation analyses are not automated here.

| Observed result | Defensible action |
|---|---|
| Resource/loader/parity fails | Diagnose implementation/runtime; no conclusion about learning efficacy. |
| Probe has little privilege advantage | Review references, task level and prompt usability; avoid an expensive blind launch. |
| Natural termination too rare | Stop this configuration; do not use forced length to hide instability. |
| Evolving is not better than frozen | Teacher refreshing lacks evidence in this condition; preserve the negative result. |
| Combined improves only after more work | Run a separate compute-matched comparison before claiming efficiency. |
| Accuracy rises but retention fails | Do not promote; retain the failed treatment and investigate a separate preservation design. |
| Shorter output loses correctness | Compression failed the accuracy/length objective, regardless of token savings. |
| All exploratory gates look good | Replicate and test untouched parents; do not certify from reused calibration. |

## Cost and stopping contracts

The initial comparison matches optimizer-step ceilings, parent schedule, frozen
parameter scope and replay. On-policy trajectories and accepted rewrite counts
naturally differ. Optimizer-step matching is not compute matching. Failed rewrite
generation, teacher scoring, evaluation and discarded attempts must remain visible.
The projection implementation recomputes full-vocabulary logits during backward;
measure wall time and memory rather than assuming chunking is free.

Source-specific observed retention drop >0.02 or truncation >0.25 stops the default
plan. These are conservative operational thresholds, not confidence guarantees.
No signal, clear truncation failure, finite-value failure or runtime limit stops
rather than silently substituting replay-only updates or broadening context.
There is no automatic deployment promotion or rollback. Preserve all intermediate
checkpoints and the terminal result. Do not select the best checkpoint after
looking at confirmation labels.

## Known engineering boundaries

CPU tests validate real autograd and a small causal model interface; they do not
validate actual Titan memory fit. Native qualification is specific to one arm,
plan, input catalog, adapter, runtime and code identity. The process watchdog is
reused, not independently requalified in this implementation session. There is no
Linux/macOS native escape hatch; ports require a new containment qualification.

Public DTOs and loss masks prevent accidental gold routing through the supported
API, not hostile Python code reading local files. Only trusted local loaders and
reviewed catalogs should run in this process. File hashes detect accidental drift,
not a malicious actor rewriting all artifacts consistently. Inputs/receipts are
operator-owned research evidence, not cryptographic proof of a scientific claim.

On interruption, unknown spending and partially committed work are quarantined.
Only clean PAUSED checkpoints can resume. Do not erase locks/INFLIGHT to create the
appearance of a continuous run. Native resume parity is actually tested with
three disposable updates in each arm's qualification. No higher-order gradient,
trainable output head, distributed training, cached generation or quantized-base
path is supported by the custom loss.
