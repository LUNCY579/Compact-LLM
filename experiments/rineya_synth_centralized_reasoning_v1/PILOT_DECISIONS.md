# First development pilot decisions — 2026-10-03

The owner delegated checkpoint, trainable-scope and token-budget decisions after
the CPU compiler implementation. These choices are frozen for a **small
development dose screen** in [PILOT_DECISIONS.json](PILOT_DECISIONS.json).
This is not a complete run manifest or native launch qualification.

| Choice | Frozen value | Reason |
| --- | --- | --- |
| Student / M0 | Unchanged dense Titan Experimental 1.0.0-devfreeze, 596,049,920 BF16 parameters | Central protocol's intended primary; avoids composed-checkpoint and scale variables |
| M1/M2 trainables | Rank-8 q/v LoRA on all 28 layers, alpha 16, dropout 0; 112 FP32 matrices, 1,146,880 parameters | Existing short CE512 precedent; established confined adapter scope |
| First dose | At most 131,072 tokenizer-measured non-padding input positions per trained arm, including prompt, answer and EOS | Bounded screen, about 19.3% of the central candidate's recorded 680,532-token dose; no claim of a sufficient acquisition dose |
| Matching | Each arm 130,560–131,072 positions; difference at most 512; target 50/50 R/S tokens, each skill within 512 of half | Whole-example tolerance; no padding or target truncation |
| Optimizer | AdamW 1e-4, betas .9/.999, eps 1e-8, decay 0, clip 1; 5% warmup then linear decay | Keep the central q/v adapter recipe consistent |
| Execution | Microbatch 1, accumulation 20 sequences; completion+EOS mean per sequence then accumulation mean; no packing/padding; 512-token cap | Existing short operational route; current physical driver and sustained fit still need qualification |
| Seed / selection | Seed 101; final checkpoint only | Exploratory dose first; 211/307 require separately frozen continuation |
| Warmstart / replay | Neither for this first atomic screen | Isolate the intervention; disjoint prior-correct guards remain mandatory |

The exact [adapter inventory](pilot_trainable_inventory.json) comes from recorded
static shapes. Each layer has q A/B `[8,1024]` / `[2048,8]`, and v A/B
`[8,1024]` / `[1024,8]`. This is prospective inventory, not native tensor evidence.
Existing local model, config and tokenizer asset hashes were refreshed by CPU
file hashing. No weights were deserialized, framework imported or model worker
started. Read-only host/GPU inventory observations are not resource admission.

M1/M2 must share source parents, complete per-parent exposure vectors, optimizer
update count, initialization by seed, acceptance policy and answer format. Actual
tokenization must find a schedule satisfying **all** constraints, including full
20-sequence accumulations. Otherwise record `BLOCKED_MATCHING`; do not drop
parents, top off only one arm, truncate answers or change the ceiling. Report
actual supervised positions too. Current CPU fixtures do not supply this native
schedule. Cross conditions/renderings and balance answer positions before
constructing a training corpus.

Parameters and dose are now fixed; a measured speed or monetary estimate still
requires qualified throughput, driver overhead and sustained resource fit. Do
not extrapolate hours from a disposable short fixture.

The [central protocol](../post_training_mimo_centralized_v1/RESEARCH_PROTOCOL.md)
and [handoff](../post_training_mimo_centralized_v1/NEXT_EXPERIMENT.md) supply the
rationale. The [CE512 receipt](../post_training_mimo_centralized_v1/receipts/LAPTOP_NATIVE_STAGE_005_RESULTS.json)
supports only its short mechanics scope. The
[capture failure](../post_training_mimo_centralized_v1/receipts/CAPTURE_SHORT_V1_NATIVE_FAILURE.json)
and [geometry diagnostic](../post_training_mimo_centralized_v1/receipts/CAPTURE_GEOMETRY_D1_NATIVE_RESULT.json)
retain the failed likelihood gate. Two-update backend math/optimizer/restore,
sustained training, fresh baseline enrollment, language/long-context preservation,
fresh atomic and RS families, statistics/power, effect/retention margins and a
complete run-specific launch manifest remain gates. No checks are lowered.

M2 must beat M1 on fresh families while preserving prior-correct behavior, with
RS reserved for transfer. Familiar-template gains cannot answer the question.
A null in this small screen cannot reject synthetic supervision or condition
diversity generally. Preserve dose, uncertainty and failure analysis; register
any larger successor before fresh evaluation. COMP-RS's acquisition-bottleneck
premise was supplied by the owner, not recomputed from protected banks here.
