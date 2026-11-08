# RRP-005 — repair the retention instrument before training

## Finding and immediate consequence

**RRP-004's training launch is superseded and blocked in code.** The bank it would use for
retention, SHA256 `0dcbec17a5842d94f6157db34d888efda3650b5b73a6aaf1b266cc643b89a975`,
omits the procedures and factual evidence from **all 192 procedural-RAG rows**. The source
corpus stores these separately from `messages`. Copying messages alone creates a
question-only task whose original answer is unsupported. This is an instrument defect,
not a finding about the model. Old records remain unchanged.

Example: the saved prompt asks to apply a procedure to APP23766, record 4, and expects
`Tier 2.` It contains neither the $1,391 / low-risk record nor the rule selecting Tier 2
for amount ≥ $1,000 or high risk. The previous aggregate preservation result cannot serve
as a valid retention baseline. The 29-pair v3 diagnostic had a different source and is not
invalidated by this omission alone; its restricted single-turn coverage remains a limit.

The original `preserve._prompt()` also removes **all assistant turns**, not only the final
target. It therefore discards earlier conversational evidence. Restoring complete histories
and adding a separately requested final-turn-only serialization admits 31 formerly refused
multi-turn rows. The historical single-turn compiler keeps its original contract.

## Independently checked procedural targets

`grounded_replay.py` renders a whitelist of model-visible procedures/evidence and computes
results from that visible text. It does not receive gold, decision labels, verification
metadata or generation seeds. Eight literal rule grammars are supported: latency, capacity,
threshold, semantic versions, priority, approval, backoff, and quota. Unknown rules,
incomplete records and unresolved ties are refused. Exact audit/record binding is required.

The generator's conflict variant changes an input but **does not always change the requested
result**. Examples include risk changes above the $1,000 approval threshold and retry counts
that both saturate at 60 seconds. The question asks for one certain result, yet such rows
demand the conflict sentence. They are unsuitable targets without adjudication.

Fresh full discovery-split audit: **4,189 verified, 129 same-result conflict targets and
2 unresolved ties quarantined, out of 4,320 rows**. No training occurred; no targets were
silently corrected. The protected `fresh_sealed` split was not read. A separate read of the
permitted full DEV split found 1,035 verified, 44 same-result conflicts and one unresolved
tie out of 1,080. The selected old bank contains 11 of these unresolved/inconsistent rows.

For nonprocedural bank rows, the original targets are preserved and explicitly labelled
**SOURCE_TARGET_PRESERVED_NOT_INDEPENDENTLY_SOLVED**. Feeding a scorer its own gold proves
scorer mechanics, not that the question supports that gold. The older broad “768/768 positive
control” description overstated the strength of that evidence.

## Corrected DEV and causal diagnostic

- **757 main-bank rows**: original selected IDs minus 11 quarantined procedural rows; no
  selection based on model outputs. Full conversational history preserved. Procedures and
  evidence explicitly rendered. Canonical missing/conflict output rules stated uniformly.
- **96 matched evidence-removal pairs (192 views)**: start from the 96 answerable procedural
  rows; remove their single decisive evidence document for the negative sibling. The
  question, record number, system instructions and procedure stay identical. The independent
  solver verifies both outcomes. This defeats the old missing-record-number and conflict
  wording shortcuts within these pairs.
- **853 unique model inputs**: 757 main rows plus 96 negative siblings. Positive probes reuse
  main-bank outputs only after exact prompt/target equality is proved. Never count aliases as
  independent examples. Main-bank and diagnostic results are separate; pooled accuracy is
  uninterpretable.
- All 757 main rows and 192 diagnostic views tokenize successfully at **2048**, with no
  truncation. Largest full main sequence: **414 tokens**; largest supervised target: **15**.
  Earlier conservative cluster IDs are retained; extra rendered evidence is not a license
  to inflate independence. Diagnostic pairs also carry their original parent topic IDs.

Corrected artifacts are in
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/preservation-correction-002`.
The earlier `001` construction remains preserved; `002` adds explicit input reuse and
partition accounting. The original RRP-004 726-world TRAIN selection and its weights remain
unchanged and untrained.

## Source-only measurement, registered before outputs

Experiment: **RRP-005-PRESERVATION-BASELINE**. Exact ungated Supa 1.7; pinned FP32 runtime;
greedy, 2048 input window, 64 generated tokens. New bank hash and dialogue serialization
are mandatory in the frozen plan. Fresh in-run 2048 inference qualification verifies finite
logits, cached/full agreement, causal behavior, and source parameter identity, with zero
backward passes and zero optimizer updates. This is not training qualification.

Report:

1. Main-bank strict/tolerant response-contract accuracy per sub-bank; explicit answerable,
   missing-information and source-conflict counts; wrong abstention kind; termination/caps.
2. Procedural-only independently verified accuracy. Report the exclusion count beside it.
3. All 40 original v3 minimal pairs when evaluable, separately from the 96 new procedural
   pairs. Within each pair: both correct; evidence-sensitive; always answers; always abstains;
   inverted. A correct negative alone does not establish reasoning.
4. Constant-abstention, constant-conflict and question-only shortcuts on exactly the same
   rows. The paired diagnostic's questions are identical within each pair, so a fixed
   question-only answer cannot be correct on both sides.
5. Native source digest before/after, counts, time, peak resources and the full raw outputs.

No old preservation prediction can substitute for a corrected-prompt baseline. Differences
between old and new prompt results would be **input-restoration effects**, not learning.
No candidate, optimization or model selection is authorized in this lane. An improved
training successor will need a newly bound plan/grant and matched corrected evaluations.

Ceiling: 3,600 model-worker seconds, 4,500 owned wall seconds, 1,000 logical
evaluation/qualification calls, 65,536 reserved generated tokens, 256 MiB new evidence,
two launches and one retry. Training, adaptive model selection, RL and paid spend are zero.
These are agent-instantiated conservative numeric limits under the current user's explicit
local-compute authorization, not numbers attributed to the user.
