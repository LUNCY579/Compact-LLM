# Model-quality status and decision memo

**Authoritative continuation for this successor, audited 2026-09-15 UTC.**

**Decision: improve the measurement and training data first; retain dense tail adaptation as the ordinary control; evaluate a larger standalone model before expensive architectural recovery. No overall model-quality winner is established.**

This memo supersedes current-status recommendations in the inherited continuation files. It does not amend historical registrations, results, checkpoints, published-core metadata or their acceptance thresholds. Historical document bytes are preserved under [history](docs/history/model-quality-audit-2026-09-15/README.md). Other repositories' historical branches remain unchanged; their contradictory summaries are reconciled here with immutable source citations.

## Scope and evidence contract

The audit read live remote state before choosing an intervention: **50 heads** across ChatBubble (15), SupaExperiment (12), RineyaArchitecture (15), RineyaBeta (6) and directly relevant RineyaRAG (2). Full remote SHAs and immediate parents are in [remote-heads-before.json](docs/model-quality/audit-2026-09-15/remote-heads-before.json). Scientific lineage, questions, actual code/executions, blockers and dispositions are in these matrices:

- [ChatBubble: RAB, ordinary adaptation, published cores, CED and scale transfer](docs/model-quality/audit-2026-09-15/chatbubble-results-audit.md).
- [SupaExperiment: calibration, stateless/recurrent adapters, symmetric generators, WSD and memory](docs/model-quality/audit-2026-09-15/supa-audit.md).
- [Architecture, Beta and RAG: capacity, distillation controls, replay, looped flow and later corrections](docs/model-quality/audit-2026-09-15/architecture-beta-audit.md).
- [MQ raw-record recomputation](docs/model-quality/audit-2026-09-15/mq-audit.json), [underlying generator-family audit](docs/model-quality/audit-2026-09-15/corpus-payload-audit.json), [113 archived local records and byte manifest](docs/model-quality/audit-2026-09-15/historical-mq-archive-manifest.json).

Repository discovery found RineyaWorkspace and RineyaWorkspace-bare application/refactor heads, with no additional directly relevant model-quality research branch. A directory named `RineyaBeta-supa-fwa-v1` actually points to ChatBubble and has pre-existing changes; it was left untouched and the real RineyaBeta remote was inspected separately. Deleted Supa remote branches are reported as **local historical evidence**, not live remote heads.

Facts below are attributed to retained repository/local artifacts. This audit ran offline checks and CPU fixtures; it did not rerun historical model measurements. Checksums establish byte integrity, not independent proof that the described model executed. Missing evidence is explicit.

### Evidence vocabulary

| Classification | Meaning here |
|---|---|
| Mechanically implemented | Code/fixture behavior exists; no pretrained capability implied |
| Native qualified | Passed stated gates for a specific loader, source, runtime and test surface; distinguish random native-model fixtures from actual-checkpoint qualification; not portable to another implementation |
| Official-weight executed | Exact pretrained assets used in recorded forward/training calls; does not imply efficacy |
| Exploratory model result | Inspected development instrument; useful for diagnosis/selection |
| Replicated result | Specify what varies: ordering, independent training draws, or independent families |
| Confirmatory result | Prospectively frozen question, valid endpoint/statistics and protected evaluation; narrow scope only |
| Release candidate | Complete relevant serving/quality/retention gates; **none established by this audit** |
| Published experimental core | Registered/materialized artifact; Opallo is in this category, with explicit regressions |
| Blocked | Missing execution prerequisite, qualification, evidence or infrastructure |
| Superseded | Interpretation/status replaced by identified later evidence; original bytes retained |
| Historical negative result | Executed unfavorable result preserved; not rewritten as an unrun plan |

## Decisions: the twelve questions

### 1. Strongest current model-quality candidate

**No candidate dominates all measured dimensions.** Opallo is the best documented **published ungated Supa-derived arithmetic candidate**. Its BF16 stored weights score 73/150 with 41/47 source successes retained when evaluated in FP32. Its FP32 parent scores 74. Full B28 WSD reaches 76 but retains only 37, loses ten source successes and trains all 596M parameters. Neither count establishes general superiority.

SupaExperiment's CAPABILITY-1 matched MLP is a worthwhile small, stateless comparator, but it uses a **different FP16 gated Supa-A1.1 checkpoint** and different evaluation panels. It is not an established replacement for Opallo. Existing comparisons cannot rank the two.

### 2. What limits it

Opallo loses six of 47 source arithmetic successes, regresses on the historical long-context strict endpoint (161→135/256), becomes longer, and has poor missing-evidence behavior. The exact literal `UNKNOWN` count is **13→0/64**, not the historically reported 13→2: Opallo's two substring matches are `UNKNOWN.`. There is also one explicit prose abstention; a blind semantic/context review is still needed. Zero literal false abstentions on answerable questions does not establish zero semantic false abstention.

General reasoning/transfer gains are unestablished. Forge and puzzle observations are limited by source, truncation/exclusion, generation settings and exposed instruments. There is no untouched broad confirmation or native BF16 serving qualification. Training selection over roughly 40 examples varies materially across draws; that is evidence to improve data/measurement, not proof of an architectural capacity limit.

### 3. Obsolete, overstated or superseded conclusions

The corrections table below is authoritative. In particular: WSD comparisons also changed beta2 and warmup; pooled repeated-row p-values do not gain independence from new training seeds; nonsignificance does not prove equivalence; reasoning-data failures do not close all reasoning/data work; native loader PASS cannot be transferred between RAB and XSC; and MQ native/training work is no longer merely pending.

### 4. Is registered abstention repair the correct next run?

**Not as the next clean confirmatory training experiment.** ChatBubble's registration remains unexecuted in the audited artifacts, but its target and inference assumptions have known defects before outcomes: natural-language abstention training is judged through an ambiguous UNKNOWN counter; treatment/control differ in examples, targets and updates; the pooled row×order McNemar argument is invalid. The claimed 17+11 decomposition is not a disjoint partition of the net 26 losses. Failure would not prove abstention unteachable.

First complete the existing [inference-only screen's native prerequisites](https://github.com/LUNCY579/ChatBubble/blob/d59729433fae8632d0e2bbe6bf99e185fb4c2694/experiments/opallo_behavior_screen_v1/PROTOCOL.md). Its B/F/G/FG design already separates strict format and literal UNKNOWN effects and requires retention/false-abstention checks. No qualified native collector exists in that branch yet.

Preserve `REPAIR_PREREG.md` verbatim. If its historical experiment is run, report its original thresholds unchanged as that historical question. For model-quality inference, freeze a separate successor **before training**, with explicit literal versus semantic endpoints, blind semantic review, matched exposure/update groups and family/draw-aware analysis. This is a justified correction, not a redundant replacement of a valid completed experiment. SupaExperiment's CALIB series is already executed and should not be rerun as pending repair.

### 5. Is MQ-001 technically ready?

**The historical harness has executed real-source CUDA qualification and training; the next screen is not ready to launch.** Four local qualification envelopes and eight completed cell result sets survive, with valid recorded hashes. [Independent provenance checks](docs/model-quality/audit-2026-09-15/mq-provenance-recheck.json) also reproduce both historical code digests from their exact Git commits and verify the local Supa/Opallo weight hashes. All eight final checkpoint directories are absent. The first pair had supervisor receipts; six later cells deliberately bypassed the watchdog. They cannot certify complete supervised/restart-safe execution. The 20-token native qualification does not certify maximum 2,048-token fit, long-context evaluation or official-source resume equivalence.

This successor repairs the operational supervisor and disables the deletion/bypass wrappers, while retaining scientific configuration, source locks, losses, optimizer, masks and resource minima. It earns **CPU mechanics only** until fresh exact-commit native qualification. Corpus/scorer review and a corrected prospective design remain required. See [technical readiness](#mq-technical-readiness).

### 6. What exact training corpus should MQ use?

**No existing audited bank is ready as the requested broad quality-training corpus.** `mq-data-003` is a 2,000-row candidate pool sampled from v3.0; its trained prefix is 238 rows. It is not the proposed 800/600/400/200 quality mixture, contains no explicit UNKNOWN contracts or matched answerability pairs, and its DEV is a 51-scoreable/64-row style-sensitive development instrument. Forty-three of 64 DEV rows share underlying base generator families with the full 2,000-row TRAIN candidate pool. Twenty-five of 64 (22 of the 51 scoreable rows) share them with the 238 rows actually consumed. Declared v3 family IDs are disjoint. See the [consumed-prefix audit](docs/model-quality/audit-2026-09-15/corpus-consumed-prefix-audit.json). No exact source-index collision was found; this is not a claim of identical full prompts. A concrete near-duplicate pair repeats the same battery-choice task with different contextual names.

Use the **proposed QTB-001 specification below**, with separately generated solver-verified tasks and reviewed preservation/dialogue examples. It names exact source revisions, counts, partition rules and release gates. QTB-001 has **not been built, reviewed or frozen**; no invented hash or approval is supplied. Existing operator approval of the old pool is recorded, but does not approve a changed corpus or replace the missing shortcut audits.

### 7. Which six-cell experiment first?

After the corrected bank, evaluator, resource plan and native qualification are frozen, run `{whole, assistant-response+EOS} × {3e-6, 1e-5, 3e-5}` as **one exploratory screen**, each from exact ungated Supa. Keep blocks8–27, WSD20/60/20 to0.1×peak, betas(.9,.95), epsilon1e-8, decay0, clipping1, rows/order/full sequences and full-row update groups fixed. Every matched cell processes the same inputs; response training gets no extra rows to match supervised tokens.

The central-rate pair is the first integration check on the new bank. Finite updates/frozen tensors/termination and prospectively measured resource limits are health checks; comparing first-versus-last minibatch loss is invalid because the rows differ. The existing central-rate runs are historical development evidence and do not fill cells for a new data/code recipe. Preserve them without calling a fresh screen a replication of identical conditions.

### 8. Can a learning-rate/schedule cell be removed now?

**No one of the three dense LRs is ruled out by the available evidence.** Only1e-5 was run under MQ's recipe. Adapter gate multipliers and old dense schedules are different interventions. The economical action is to defer the grid until its instrument is useful, not remove a rate based on an unmeasured outcome. Stop a cell under predeclared numerical/resource rules; retain the failed attempt. Schedule is fixed in this grid, so there are no additional cosine/WSD cells to run now.

### 9. Priority of a larger standalone model

**High priority as a baseline, before expanding the Supa tuning programme or starting cross-scale recovery.** Use already relevant official Qwen3-4B with an independently qualified serving path and the same corrected open evaluation contracts, then compare targeted adaptation if feasible. It is a continuity choice, not a claim about the best model currently available globally.

Old NF4 Showcase observations (8B53/64,4B43/64 versus older Supa-A1.145/64) contradict the withdrawn no-capability-gap claim but do not rank the current candidates. New evaluation must pin tokenizer/thinking/precision/context/output limits and measure latency/memory. A 4,022,468,096-parameter source has about7.49GiB of BF16 weights alone; this is an analytic lower bound, not an8GiB runtime fit certificate. Register adequate hardware or a separately measured quantized baseline. No cloud spend/job was initiated here.

### 10. Is sparse cross-scale CED recovery justified now?

**As a bounded conditional hypothesis, yes; as the next expensive run, no.** The d82 full-width XSC reference adds757,169,792 interface parameters across36 readers. Native source loading/RoPE qualification and resource fit are unresolved; the issue4 RAB branch does not repair this loader. Hlast4's comparatively mild single-dose behavior does not establish an optimal site count or a useful encoder contribution.

After standalone4B and ordinary4B adaptation, qualify a separate sparse native-width successor retaining every decoder block and reading in a registered final4–8 sites. Site sparsification is one intervention; compact reader width is a later one. R0 function-preserving initialization, R1 frozen-backbone interface recovery, then measured nearby-layer unfreezing need matched controls. Efficacy requires H/P, removed/shuffled/counterfactual memory, retention and cost. Active memory/lower loss are not capability claims. Functional alignment by held-out ridge/activation analysis is conditional on an active but ineffective interface, not justification from matching dimensions.

### 11. Active versus parked work

| Pursue now | Retain as comparator / conditional | Park or preserve as history |
|---|---|---|
| Correct evaluators and corpus; qualified behavior collector; MQ operational repair; frozen ordinary baselines; standalone4B measurement | Opallo and fullB28 frontier; exact-source matched MLP; clean schedule contrast after data/objective selection; sparse XSC native repair | Broad CED/depth sweeps, full36-reader recovery, recurrence/norm/capacity scaffolds without native results, EAM/replay/looped-flow/FGP efficacy, repeated soup/weight-transfer nulls |
| Freeze independent training-draw replication and fresh confirmation | LoRA with fair tuning opportunity, then DoRA if useful; verified teacher data | Hybrid-memory failed promotion; R2 frozen-HLGA negatives; old leaked shortcut positives; pruning before a stronger model warrants it |

Parking is a research recommendation. No branch, artifact, checkpoint or remote ref was deleted or archived by this task.

### 12. Minimum defensible experiment sequence

1. Finish evidence/evaluator corrections and implement and qualify the missing native collector for the existing registered behavior screen; replay preserved outputs for literal/format diagnostics and conduct blind semantic review. Freeze untouched scenario/family allocation without opening confirmation outputs.
2. Build/verify QTB-001 and its development instruments. Measure frozen Supa, Opallo and a qualified standalone4B on the same contracts; include exact-source small-adapter comparison only with its own qualification. If standalone4B already meets quality/cost needs, prioritize its ordinary adaptation.
3. Qualify this operational successor at zero updates on the actual source/runtime with representative and maximum planned shapes; validate full-state resume on a bounded engineering attempt before relying on restart behavior. Keep engineering updates outside the scientific screen.
4. Run the bounded six-cell objective×LR development screen. Record acquisition, paired losses, abstention dimensions, length, termination and measured costs; no winner from one seed.
5. Freeze at most two recipes and repeat on at least two additional **independently selected training draws**, paired within draw. Keep ordering seeds separate. Freeze effect/retention margins and family-aware uncertainty before confirmation; repeatedly inspected DEV remains DEV.
6. If the selected recipe meets the intended quality/cost needs, freeze it and confirm once on fresh protected families/scenarios and the intended exported dtype. Optional bounded schedule, layerwise LR, PEFT or verified-distillation studies belong on DEV before this final freeze; they are not required to complete the minimum path. Any later recipe change requires a new protected confirmation instrument. Stop when the model meets the required quality/cost tradeoff rather than accumulating experiments.

## Ordinary-model baseline ledger

Exact source identities must be retained: ungated Supa weight SHA256 `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`; Opallo stored weight SHA256 `e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b`. Both local files were independently hashed in this audit and matched their manifests. Opallo FP32 trainable checkpoint SHA256 is `915af77f4ad2a1813df46a5b5a65b9dddd138b820e2e9e53ad8845bdf8d0db0d` as recorded historically; no checkpoint load was performed here.

| Candidate | Total / trainable parameters | Examples; processed input / target tokens; updates | Recipe | Task outcome; source retained/gained/lost | Evidence class |
|---|---:|---|---|---|---|
| Frozen ungated Supa |596,049,920 /0|0;0/0;0|No adaptation|47/150;47/0/0;148 stops|Official-weight development baseline|
| FullB28 WSD, seed17 |596,049,920 /596,049,920|41;5,274/5,233;41|1e-5,WSD20/60/20,betas.9/.95|76/150;37/39/10;150 stops|Exploratory; full update costs and retention tradeoff|
| B28last20 WSD FP32, seed17 |596,049,920 /314,618,880|41;5,274/5,233;41|Same|74/150;41/33/6;150 stops|Exploratory best documented parent|
| Opallo BF16 storage |596,049,920 /0 at serving|Materialized from preceding FP32 parent|BF16-rounded weights upcast toFP32 for reported measurements|73/150;41/32/6;150 stops|Published experimental core; not native BF16 serving qualification|
| MQ central-rate whole, three orders |596,049,920 /314,618,880|238;24,910/24,672;13|1e-5,WSD20/60/20,betas.9/.95,group normalization|7,7,8/51;3 retained,4/4/5 gained,0 lost vs its own3/51 source DEV|Order-repeated exploratory; different task instrument|
| MQ central-rate response, three orders |same|238;24,910/1,944;13|Same except target mask|13/51 each;3 retained,10 gained,0 lost;64 stops each|Same row set; no independent-draw confirmation|
| CAPABILITY1 matchedMLP |596,206,145 /156,225|3,600 presentations; token totals not recorded;450 updates|FP16 gatedA1.1 base;FP32 sharedMLP;3e-4,gateLR×10,AdamW defaults,decay.01|OwnDEV answerables285→339/480,75 gained/21 lost;unanswerables120→231/240 at bias0|Different exact source and panels; exploratory transfer, not directly rankable|

RAB inputs are whole-sequence CE despite the misleading historical `response_only` artifact tag. The derived5,274 input tokens count the entire forwarded sequences; the original budget/counter is5,233 next-token targets. All dense training uses FP32, norm clip1, decay0, epsilon1e-8. Historical RAB uses one row/update; MQ uses token-normalized full-row groups≤2,048, including the final partial group. Trainable blocks8–27 exclude embeddings, tied output table, final norm and blocks0–7. Last8/12/16 and additional draws/doses, exact row/hash/LR traces and costs are in the [expanded cards](docs/model-quality/audit-2026-09-15/chatbubble-results-audit.md).

SupaExperiment uses model revision`199eed212ab3a3db487270dc6f97b16756f7bff0`, FP16 weight SHA256`a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`, custom abstention readout; it is not the ungated BF16 source. Sidecar SHA256`f2ccf8faa099ba9e0dc5b07474c6a3a4ec42f5bee9141422dd795d9ae6612d79`. Its transfer +93 answerables is at canonical bias5.15625; −41.23% proxy loss is at bias0. Neither evaluates deployed bias3.75. MLP shipping qualification remains blocked.

### Behavior, dtype and cost limits

- Opallo's storage cast changes10/150 output texts, loses one correctness item and preserves150/150 termination decisions. This quantifies **storage rounding**, not BF16 arithmetic.
- Long-context RAB first-line score161→135/256; post-hoc answerable containment149→150/192; full literal UNKNOWN13→0/64; historical punctuation/substring UNKNOWN13→2; false literal UNKNOWN0/192 both. Mean generated tokens2.9453125→4.3046875;256/256 terminate. Semantic abstention, semantic false abstention and unsupported-answer rates remain **not measured with a qualified rubric**.
- Puzzle102→96/450; Forge IID32→30/200 and transfer44→36/200 for the FP32 parent, with overlong-row exclusions and a24-token cap. These are exposed development probes, not broad no-regression certificates.
- RAB recorded timed train/probe sections: last20WSD308.57s/5.894GiB peak allocatedGPU; fullB28WSD735.17s/7.605GiB. Full host peaks and whole-program wall time are absent. Analytical two-moment FP32 storage is2,516,951,040bytes for last20 and4,768,399,360 for fullB28, excluding weights/gradients/staging.
- MQ recorded between-update intervals102.97–164.73s for corrected cells, GPU peak-allocation snapshots about3.50–3.58GiB and processRSS snapshots about3.99GiB. These exclude whole-run timing and can miss final-step/host peaks; they are not complete serving/training cost measurements. Exact values are in `mq-audit.json`.
- No qualified inference-latency comparison, comprehensive output-format/unsupported-answer card, fresh transfer confirmation, or useful4B serving-cost result exists for this common candidate set. Do not fill missing fields with zero.

## MQ technical readiness

Historical subject: `research/model-quality-targeted-gpu-v1@8c3ed88a821b83e531ab682b3201879ea4fc928b`. Original implementation parent`a36de96b30eb03265db3a18e464a2e2a23ee67bf`; current RAB sibling`f53cf4a56998e76e79df19f81b6d5141633f9c7d` adds a memory note and is not silently substituted as MQ's parent. First pair bound to`131f0af71536489308248bb2bca52c76e157821c`; corrected order runs bound to`1e41cc5d9f8f59b62559c0e58384fce77423b8f4`. The local archive preserves plans, reviews, all failed attempts, raw outputs, step logs and qualification envelopes.

| Surface | Audit result / remaining limit |
|---|---|
| Response masks, EOS, empty thinking stub | Exact Qwen single-turn text/token prefix checked; answer+EOS included; header/stub/prompt and postEOS newline excluded |
| Padding, truncation, packing | No padded batches or packing implemented; one full row at a time; oversized/unsupported multi-turn rows rejected; no silent truncation |
| Loss values/gradients | Chunked sum vs ordinary CE values/gradients tested on CPU; native receipt checks loss/cache/active finite tail gradients on20 tokens, not full native gradient-equivalence at max shape |
| Grouping/normalization | Same examples/order/groups per objective pair; group-global supervised denominator; final partial group included; only target set and LR differ across cells |
| Schedule/Adam | Token-indexed WSD boundaries and CPU/offloaded FP32AdamW parity tested; no scientific algorithm change here |
| RNG/resume | Python/TorchCPU/CUDA streams serialized, clean update cursor and moments validated; fixture resume equivalence tested. NumPy is not used by the training randomness. No official-source restart equivalence established; historical checkpoints now missing |
| Integrity | Source/hash/runtime/code/checkout/plan/review and inventory checks; create-only files; tamper/incomplete/source/plan mismatch tests. Checksums are not attestations |
| Native loading | Actual Supa assets/runtime recorded: torch2.10.0+cu126,transformers4.57.1,RTX4070Laptop; qualification0updates,220trainable tensors. No inherited XSC authority |
| Resources/supervision | Historical wrapper bypassed watchdog and deleted checkpoints. Successor repairs this path; minima unchanged. Training's6GiB preload floor is still not an allocation proof; code's former “qualification has no gradients” comment was false because qualification performs backward |
| Evaluator | Exact/number declared contracts and raw bounded generation exist; broad semantic scoring absent. Builder marks natural missing-information answers as answerable exact-text tasks, so its zero false-abstention field is not an epistemic answerability result |
| Leakage/shortcuts | Four exact keys pass; semantic/generator review is not machine-complete; base generator sharing with 43/64 DEV rows against the candidate pool and 25/64 against consumed rows, plus a concrete near-duplicate task. Label/position/query/metadata attacks need a proper new bank |
| Execution authority | Old reviews/envelopes apply only to old hashes/commits. This successor has **no native or training permission from them** |

Fresh audit run of the original MQ test suite: **94 passed,1 skipped out of95** in45.869s. Both random native-Qwen CPU cases ran under pinned Transformers; they are random weights, not official-source qualification. The one skip is Windows symlink-creation privilege. The raw XML is preserved. Successor validation is recorded separately in [VALIDATION.md](docs/model-quality/audit-2026-09-15/VALIDATION.md).

## Proposed QTB-001 corpus specification — not yet built

QTB-001 replaces the casual v3 pool as a **new scientific data intervention**. Do not change the old frozen bank or retrofit its approval. Freeze this bank before the objective×LR experiment; all six cells then use it unchanged.

| Training function | Exact proposed count | Provenance and verifiable target |
|---|---:|---|
| Grounded answerability |800 rows /400 matched scenarios|Fresh fictional records; one answerable and one missing-evidence variant per scenario.200scenarios use explicit UNKNOWN contract,200 use an explicitly requested short natural abstention; labels/format tracked separately |
| Arithmetic/compositional solving |600|100 each: integer arithmetic, ratios/percentages, durations/units, set counting, conditional constraints, signed state updates. Fresh operands/structures, exact integer/rational/symbolic solver; difficulty bounded prospectively |
| Capability preservation and grounded dialogue |400|200evidence-grounded concise dialogue tasks generated with explicit state;200independently verified concise/preservation rows from owned v3/behavior TRAIN sources after actual generator-family exclusion. Source answers alone are not gold verification |
| Output contracts |200|50each: answer-only integer/token, fixed-key JSON, exact list/field transformation, explanation-requested short verified intermediate state. Keep explanation-requested and answer-only tasks distinct |

**Pinned source materials:** ChatBubble v3 and behavior corpora/generator code at`8c3ed88a821b83e531ab682b3201879ea4fc928b`; symmetric eligibility-before-query practice from SupaExperiment depth3 replacement at`09e033d36f93853e901036c201946b4478ade4f8`, independently reproduced/audited by NRA at`8e3d467ee02cb5d749ab9fff16f5d999e6035767`. Borrow generator methods with attribution, not already evaluated FLOOR1/BSEL rows. Historical puzzle/Forge results may motivate task types, not choose “winning” eval items for training. Teacher-generated text is excluded unless the answer and any intermediate state pass an independent solver/check.

**Partition before generation:** immutable scenario graphs, underlying base generator families, template variants and lexicon pools assigned to TRAIN, openDEV and protectedCONFIRM. A new seed or a renamed family ID is insufficient. Paired variants stay in one split. Keep system instructions balanced across labels. Requested answer location, query identity, candidate count, length, metadata and missing-evidence wording must not reveal labels. Apply all eligibility restrictions to all candidates **before** selecting the queried candidate.

**Development/confirmation:** propose512openDEV rows across at least32subfamilies with balanced matched answerability and explicit arithmetic/format/retention strata; reserve a separately designed fresh confirmation bank after a prospective precision/power calculation and before recipe choice. This exceeds MQv1's64DEV cap and therefore needs a separately registered evaluation successor. More rows alone do not guarantee power. The current code repair does not change that cap. No protected outputs were opened here.

**Required audit receipts before freeze:** exact IDs/text; NFKC/case/whitespace normalized text; digit-blinded text; actual source-base-generator/template/scenario overlap; near-duplicate attacks; independently replayed gold/unique-solution checks; first/last/min/max/copy/query-choice/position attacks; label-frequency and length-only baselines; metadata-only and question-only attacks; worst-stratum rates with correct chance denominators; negative controls on known-bad generators so a passing detector is not vacuous. Every failure remains an artifact; never discard a failing prefix secretly. Missing adjudication/rights becomes a blocker, not an APPROVED string.

**Prospective acceptance and actual exposure:** require zero incorrect solver golds, zero prohibited exact/normalized/digit-blind overlaps and zero prohibited underlying-family/scenario overlap. Register each shortcut attack's worst-stratum tolerance, chance denominator, uncertainty rule and exact DEV allocation before collecting model outputs. A generic PASS receipt is insufficient.

The 2,000 counts above describe the candidate bank, not executed exposure. For the first 25k screen, propose a **200-row effective bank**: 80 grounded rows (40 intact paired scenarios), 60 solving rows (10 per listed family), 40 preservation/dialogue rows (20 each), and 20 output-contract rows (5 each). Select within frozen training-only strata by a predeclared seed, without model outputs. Compile all selected rows before freezing the effective bank; it must meet the existing 22,500–25,000 processed-token range without truncation, padding for budget, splitting pairs, silently discarding long rows or searching for an outcome-favorable subset. If it does not fit, stop and register a budget or sampler successor before any training. Do not launch a nominally balanced pool while allowing an unrestricted shuffled prefix to determine the actual training mix. Retain separate selection/order seeds, and freeze all pair-preserving update groups equally across objectives.

**Frozen manifest:** raw SHA256perfile; generator commit/config/seed; solver version; source rights/reviewer evidence; original IDs and underlying families; split scenario allocation; exact selected ordered IDs; native tokenizer/source assets; full token IDs/masks; processed and supervised counts; update groups; every exclusion/rejection with reason; DEV contracts and scoring code; budget/cost ceilings; confirmation-role declaration. Select eligible rows by registered corpus properties before model outputs, not evaluation performance. Preserve pair integrity through selection: amend the data-plan builder prospectively if row-prefix sampling would split pairs. Freeze this change across both objectives. Report actual coverage, not a nominal25k label.

## Corrections to inherited current-status claims

| Prior claim / location | Evidence-bounded current reading |
|---|---|
| Root continuation: WSD alone+7.5, pooledp=.00029 | Four recipe directions favor WSD/.95, but warmup/schedule/beta2 and repeated-evaluation dependence block that attribution/inference |
| Root continuation / CORE: Opallo lost abstention, not retrieval | Strict regression is real; posthoc containment149→150 does not prove retrieval equivalence; exactUNKNOWN13→0, semantic and presentation dimensions separate |
| Repair:17format+11abstention explains26; no new contamination surface | Counts are not a disjoint partition; reusing a bank does not validate target/scenario leakage; original thresholds unchanged |
| Root continuation: capacity ceiling; reasoning data closed; fullB28 buys nothing | Specific nulls/tradeoffs, no universal ceiling/equivalence; FLOOR1 elicitation and adapter acquisition contradict broad closure |
| MQREADME/POOL_STATUS: no corpus/native work, only RAM blocker | Candidate pool,4native envelopes and8cell results exist; six bypassed supervisor and all checkpoints absent; data/evaluator/provenance limitations also matter |
| MQ three-seed: harness established6/6; more seeds cannot help | Order consistency only, incomplete operational evidence. More order repetitions are not new test items; independent training draws can still reveal variability. No guarantee that adding seeds or families creates significance |
| MQ first-pair:12.7× fewer supervised targets means efficiency | Correct accounting at equal processed tokens; no demonstrated12.7×compute or serving advantage |
| SupaSTATUS: MLP+93at served bias;−41% quality | +93atcanonical5.15625;−41.23%declared proxy atbias0; actual3.75notcaptured. Not production risk or a general score |
| Old state tracking always below blind | ADJACENCY1 shortcut claim remains withdrawn; corrected FLOOR1 clearsd2/d3 under exemplar, with severe retrieval/abstention damage and aggregate superiority unestablished |
| Old full-width/XSC or issue4 PASS interchangeable | Distinct loaders/subjects; XSC official execution absent; issue4expanded32native cases0executed |
| Earlier4B/8B donors missing / no capability gap | Later official donor execution exists; E7 capability conclusion withdrawn. New standalone matched evaluation remains needed |
| Architecture SBR publication incomplete; Beta onlymain; RAG implementation pending | Live SBR framework exists; Beta has6heads; RAG generator runner exists but native blocked. Exact head/receipt citations in repository matrices |
| EAM/R2/hybrid results can be discarded or generalized | Preserve leaked-bank corrections, R2bounded answer improvement, HLGAnegative and hybrid-memory negative promotion; none is broad Supa efficacy |

## Completed work, blockers and permissions

**Completed in this iteration:** fresh cross-repository audit before intervention; offline MQ output/hash/mask/accounting checks; source-weight hashes; raw long-context reclassification; generator-family/near-duplicate diagnostics; historical CPU fixture tests; byte-preserving local evidence archive; isolated operational successor and current-status reconciliation. See validation for the exact completed successor tests. Historical model runs are attributed to their original records, not this task.

**Still blocked/unexecuted:** QTB-001construction/review/freeze; qualified semantic evaluator and native behavior collector; fresh successor/source/runtime GPU qualification/max-shape fit/resume; new objective×LR screen; independent-draw confirmation; actualBF16serving; clean standalone4B comparison and costs; XSCnative repair/recovery. CI failures before runner allocation are infrastructure failures, not failed model tests; Architecture EAMI/SBR current annotations name account billing/spending limitations.

**Permissions:** the user authorized inspection, new isolated branches, justified code/docs, tests, commits and pushes; those are the exercised permissions. No old research branch was modified. No model download, GPU training, cloud job/spend, source checkpoint edit, registration threshold change, new data approval, release, force-push or message to another person was performed. Historical approvals are recorded at their original scope and do not silently open the new scientific gates. Local resource availability must be measured at the actual next launch; this memo does not repeat an old memory reading as current fact.
