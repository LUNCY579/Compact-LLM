# Evaluation protocol

See RESEARCH_PROTOCOL for frozen thresholds and parent-level statistics. Independent
roles are TRAIN, TRAIN_SELECTION, DEV/CAL, CHECK and FINAL. Assign canonical parents
before variants, solving, teacher calls or difficulty estimates. Track parent,
source/revision, normalized and tokenized prompt hashes and all derivative ancestry.
Exact deduplication cannot prove semantic or pretraining decontamination.

E0 fixtures are mechanics-only and never become scored bridge parents. DEV can
select the continuation decision. CHECK compares only unchanged base and frozen
final endpoints. CHECK does not tune anything. FINAL stays closed until a separate
confirmation plan and winner are frozen; a gate failure leaves later banks closed.

Report acquisition by family, wording and composition; preservation by source-correct
loss/repair, instruction, retrieval, long-context, abstention and language where
represented; robustness via malformed/EOS errors, length and distractors. Missing
panels remain UNMEASURED. No aggregate gain cancels a family regression.

Strict answer envelopes and independently implemented deterministic verifiers must
reject malformed, wrong, extra-field, duplicate-key and type-confused answers.
Truncated verified targets are ineligible. Count unsuccessful and rejected attempts.
