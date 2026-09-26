# Audit-driven corrections fixed before new Phase 1 adapter outcomes

The same-LCR prior-training hypothesis was refuted by exact source matches of
cached Stage A CC0/replay rows. Phase 0's causal ceiling claims were withdrawn.
Phase 1 remains a fixed empirical adaptation test. RESULT.md records the audit.

- MAX_LENGTH becomes 3840: the actual seeded bounded training maximum is 3698;
  2560 drops 641/2400 training rows and 30/160 dev rows. Preserve all sampled
  packs, without a new dataset or length sweep.
- Keep the repository's actual counting_sets/conditional_rules preservation
  classes and scorer. The earlier preregistration's arithmetic_chain entry
  disagrees with source and prior smoke. Disclose baseline-informed selection
  and limited coverage; no broad preservation claim.
- One process per arm, isolated output directories, refuse stale arm contents,
  complete compatible A/B/C records before combine can select a winner. Sum
  resource cost over arms; partial stages are not trained nulls.
- Load A1 from its actual read-only bfloat16 core. The old runner ignored its
  core argument. Hash the actual selected canonical core before and after.
- Mandatory gradient checkpointing, detached logged loss, guard nonfinite
  updates, count attempts/applied/skipped updates, schedule only applied updates.
  The outer bounded max-steps remains 300 attempts to match B/C data exposure.
  Disclose unequal applied budgets; no matched-update superiority claim when
  they differ. Do not silently extend budgets to compensate for overflow.
- Gate off during backward, restored during validation and generation.
  Version state hashes; never compare unversioned historical state digests.

Keep layers 0/1/2 frozen and unchanged. Attention only on 17/18/19, rank 4,
alpha 8, dropout 0, seed 17, AdamW, LR 1e-4, accumulation 8, original cosine
schedule and fixed data order. No rank/layer/LR sweep. Gate thresholds unchanged.

Run both requested cores. Prior expectation, before new results: A1 should
closely reproduce A1.1 because Phase 0 inference results are close. Similar
inference does not guarantee similar training; this expectation is falsifiable.
Phase 2 is only the specified conditional expansion; composition stays forbidden.


## Memory repair after mechanics smoke, before bounded training

The first repaired A/B smoke completed, with B applying all six updates. Its
peak reserved memory exceeded the physical GPU capacity. The controller stopped
C and independently hashed the canonical checkpoint after interruption; unchanged.
This incomplete smoke is preserved in audit_20260905/pre_answer_logits_smoke,
and is not a scientific null. No bounded adapter result was seen.

Project vocabulary logits only for the last n_answer_tokens + 1 positions,
using matching cropped labels, while leaving full decoder input/context and all
attention unchanged. The preceding prompt position supervises the first answer;
EOS remains supervised. Delete output/loss references between microbatches.
The canonical forward/loss implementation supports this exactly equivalent
answer-token objective. CPU equivalence tests at short and maximum sequence
lengths across fp32/fp16/bf16 preserve adapter gradients; floating loss-reduction
rounding is documented. This changes allocation, not data, model architecture,
attention, rank, layers, LR, precision, or optimizer. Repeat all A/B/C smoke with
this final source before bounded execution. Evidence and executable equivalence
check: results/audit_20260905/answer_only_logits_* and verify_answer_only_logits.py.


## Fixed allocator policy before bounded runs

Set both PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF to
`backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8`
for every bounded arm on both cores, and the A1 smoke. Installed parser accepted
this without initializing CUDA. The fixed allocation cap leaves GPU headroom;
garbage collection releases unused allocator blocks. Live tensors, decoder KV
behavior, numerical operations, model settings, curriculum, and optimizer remain
unchanged. This follows observed reserved-versus-live memory pressure in smoke;
actual OS paging was not measured. Record exact policy in each stage's
allocator_environment.json. A live requirement above the cap can still fail,
and any such mechanical failure must remain incomplete, not become a null result.


## Synchronized execution recovery after driver resets

No bounded adapter evaluation was completed or inspected before this change.
After an interrupted153-update run and a four-update CUDA/TDR failure, a short
16-microbatch zero-update diagnostic passed with CUDA_LAUNCH_BLOCKING=1. The final
retry uses that setting for all A/B/C on both cores, regenerating the frozen A
baseline as well. This changes launch scheduling only; frozen Python/model code,
corpus, rank/layers/LR/seed, optimizer, and300-attempt budget remain fixed. Record
actual cost under this setting; do not compare its speed as though the earlier
asynchronous runs used the same execution conditions. Successful short diagnostics
do not establish full-run stability or a scientific positive/negative outcome.


## Durable recovery after user-confirmed Codex restarts

The user confirmed restarting Codex. Read-only Windows tests found kill-on-close
jobs: explicit breakaway was permitted from the immediate job but left a remaining
job that disallowed further breakaway. No limits or permissions were changed and
no unsupported escape was attempted. Ordinary hidden controllers are therefore
not represented as restart-independent.

The reviewed checkpoint prototype is now adopted. It preserves only trainable
LoRA values plus AdamW/scaler/scheduler, RNG, original data cursor, committed
attempt/applied/skipped counts, complete trajectory and pending validation. Atomic
latest-state commits occur after each whole attempt and scheduled validation;
recovery always uses the latest valid transaction, never the best score. The
budget remains300 total attempts, not300 additional attempts. Uncommitted work is
replayed from the previous state; committed updates cannot be duplicated.

Each arm has one active GPU process. If interrupted, a fresh process resumes its
same logical course; every physical segment, model load and resource cost is
preserved separately. Costs that could not be measured at a hard termination are
explicitly unknown or lower bounds. This is a documented recovery exception to
one uninterrupted process per arm, not concurrent arms or a new training recipe.
Original source/data/base/initializer identities must match before restoration.

No completed bounded adapter evaluation has been seen. The model, corpus, layer
set, rank, LR, initialization, precision, full-input length, gate and total budget
remain fixed. The integrated151-test CPU suite passed on the required runtime,
including actual hard-exit restoration and independent accounting checks. Repeat
the smoke on the adopted source before bounded work, and separately verify a
completed smoke checkpoint's GPU restoration without new optimizer updates.
