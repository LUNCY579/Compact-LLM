# ChatBubble model-quality evidence audit

Audit date: 2026-09-15 UTC. Read-only audit of live `LUNCY579/ChatBubble` heads, fetched once and re-read with `git show`; source worktree remained on `research/model-quality-targeted-gpu-v1`. No model was loaded, trained, converted or downloaded. No old branch was changed. The companion `chatbubble-results-audit.json` contains all 15 live branch heads, direct parents and trees, 32 committed dose artifacts, per-item outcomes, source/data hashes, update/LR traces, and a new mechanical reclassification of raw long-context outputs. MQ-001's own implementation, pool and local executions are covered by the root audit.

## Decision-relevant findings

1. **No model is established as an overall quality winner.** Opallo is the best documented published ordinary-tail candidate on the repeatedly inspected arithmetic panel, not a confirmed general improvement. Full B28 WSD scores 76/150 versus Opallo's FP32 parent 74/150, but retains 37 versus 41 of 47 source successes and trains 596.0M versus 314.6M parameters. Nonsignificance does not establish equivalence.
2. **A newly verified literal-abstention correction changes the headline.** In the committed raw 256-row long-context outputs, Supa emits full-response `UNKNOWN` on **13/64** unanswerable rows; Opallo emits it on **0/64**, not 2. Its two historical UNKNOWN hits are `UNKNOWN.`. The source counter searched a substring / normalized punctuation. Opallo has one additional explicit prose abstention, `The current evidence does not identify the cause.` Literal compliance, semantic abstention and unsupported answering must be measured separately. A blind semantic/context review has not been performed by this audit.
3. **Published BF16 scores are BF16-storage / FP32-compute scores.** The 73/150 measurement upcasts the BF16-rounded weights through the RAB reference loader. It is not native BF16 inference parity. FP32 checkpoint 74 → BF16-rounded 73, with 41 source successes retained in both, ten text differences and one correctness loss.
4. **Historical WSD evidence changes schedule and beta2 together.** Four matched training draws favor the WSD/.95 recipe by +7,+4,+13,+6 panel rows, but the comparator is cosine 3%-warmup with beta2 .999. `WSD_REPLICATED.md` incorrectly says only the schedule differs. Its pooled McNemar independence argument also incorrectly calls repeated evaluation rows disjoint because training draws differ. The direction replicates; the causal attribution and p-value strength require correction.
5. **Registered repair has not executed in the audited evidence.** Its proposed natural-language abstention treatment is not literal UNKNOWN supervision, uses 42 rows/5,302 targets against 41/5,233, and changes update count. Its pooled row-by-seed test has the same dependence issue. Preserve its registration and thresholds; do not elevate it to clean confirmatory evidence. The already registered inference screen explicitly identifies these limitations and is a better immediate measurement-repair lane once qualified.
6. **Cross-scale is still a conditional hypothesis.** XSC's 36-reader reference is 5.376B parameters with a 757M interface, official execution absent. It has an unexecuted nonpersistent-RoPE native-loading risk. The issue-4 branch qualifies a different RAB loader lineage and cannot close XSC's gap. Hlast4's 49/150 and zero losses at one dose justify a bounded sparse-reader hypothesis, not efficacy.
7. **The larger-model question remains open and deserves a clean standalone comparison.** Old official Qwen3-8B NF4 Showcase evidence is 53/64 versus the actual older cross-scale target Supa-A1.1's selected 45/64; Qwen3-4B NF4 is 43/64. The old claim that donors have no meaningful capability gap was explicitly withdrawn. These are different targets, precisions and selection histories from current Supa/Opallo, not a fresh ranking.

## Live branch matrix

Full direct-parent/tree identities are in the companion JSON. Lineage below distinguishes immediate commit parents from the scientific ancestor.

| Live branch | Current remote SHA | Scientific lineage / question | Implemented and executed | Current classification / blocker |
|---|---|---|---|---|
| `main` | `a6ee824584a9e28881826ab92a11db48e124d90c` | Historical family, distillation, long-context, training/merge libraries; archived 2026-08-30 audit | Published Supa family metadata and earlier development experiments; RAB/Opallo continuation lives on research descendants | Historical baseline/index; main headlines are not current model-quality status |
| `research/ced-rab001-hotfix001-qualification-v1` | `f53cf4a56998e76e79df19f81b6d5141633f9c7d` | From `2f65f9e96f8649a86c5d2f4369181df5b8b21aa5`; RAB mechanical→quality→retention; ordinary adaptation vs structural CED | Official Supa G0, multiple receipt ladders, failed conversions, official-weight exploratory GPU B28/H/P doses and CPU panel probes, four training draws for tail recipe, published Opallo, long-context/Puzzle evaluations; repair built not trained | Active evidence parent; registered zero-loss retention gate still FAIL for Opallo's six lost answers; no sealed quality confirmation |
| `research/model-quality-targeted-gpu-v1` | `8c3ed88a821b83e531ab682b3201879ea4fc928b` | New MQ package from `a36de96b30eb03265db3a18e464a2e2a23ee67bf`, before f53 memory-note sibling | Contains native qualification and objective-pair/order replication claims; root audit verifies actual envelopes and pool | Active successor under separate root audit; not merely synthetic pending work |
| `research/opallo-format-abstention-screen-v1` | `d59729433fae8632d0e2bbe6bf99e185fb4c2694` | Same a36 parent; zero-update B/F/G/FG prompt×UNKNOWN-gate screen | 32 stdlib synthetic tests, 192 synthetic oracle rows, create-only/tamper tests; **zero native calls, zero calibration/check outcomes** | Registered engineering screen; native adapter/runtime/assets/EOS/confidence parity/supervisor/union-overlap/scenario review pending |
| `research/ced-official-conversion-v1` | `8056e41cc2483407a90ac066099e2dc58c74e878` | Direct parent `2f65f9e...`; construction-only official CED path | Early official G0/G1/G2 lineage plus additive construction code | Historical, contained by later RAB quality/retention findings; construction PASS cannot imply useful model |
| `research/ced-native-loader-issue4-v2` | `165789576fce93b48132effedd3a515763da3fa2` | Scientific base `e7285e5f68774bc2cc340a92cae7c7963bb569f2`; expanded RAB loader qualification | Existing native repair inherited; new 20 supervisor unit tests pass, required 32 native cases **0 executed**. Remote CI attempts failed pre-execution; local environment lacked pinned Transformers | Implemented, fresh native BLOCKED; historical109 native passes not rebound to expanded suite; not an XSC repair |
| `research/ced-dual28-local-p1-preparation-v1` | `6a56cb64ec03fb729b006192fd58a7d55975e707` | D28 subject `cd6ef9bd49724da6879bfa4d945af3bfaa6d9758`, parent FWT `94fbe179...` | D28 subject94 random tests; tiny 28+28 smoke25 H/P steps. Launcher11 mocked/synthetic tests; official P1 false. Later RAB separately executes donor-weight H/P initialization/doses | Historical reference/conditional architecture; do not copy later RAB exploratory evidence into old official P1 receipt |
| `research/ced-full-width-remote-receipt-94fbe179-v1` | `1490dc05bb9bd8d363d214a51115685697851f8f` | FWT subject `94fbe17945a9d7d88ac2422dab124a95d35ee653` from `aa9bdcd1ff805066b9566230006542704f8b9321` | Subject75 tiny random tests +25-step paired smoke; no native/official. Receipt workflow source-provenance-only; 2 attempts, runner0, zero steps/artifacts | Blocked pre-runner; reference M/P implemented, not pretrained efficacy |
| `research/ced-cross-scale-e28-d36-v1` | `d82e27c48b2a75b3a9459148990127cd9b14a2e9` | Direct scientific parent D28 `cd6ef9bd...`; full Supa E28+Qwen3-4B D36 H/P memory | 132 scoped random/mechanical tests +2 native skips; tiny-width full-depth active memory/cache/causality tests;25-step random smoke; official-shape meta counts | Mechanical only. No official P1/P2, GPU fit, trained model quality. Native source buffer materialization risk unresolved |
| `research/ced-cross-scale-remote-receipt-d82e27c-v1` | `2645f8ea92d1bfdd3bee5be3b2a51e6ad6f325de` | Immutable d82 subject, workflow parent `cf5f93c045e1d12273fae704a6f53b85427631aa` | 23 stdlib receipt tests;2 remote attempts no runner/steps; native0 | Blocked pre-runner; explicitly records static RoPE risk; no PASS |
| `research/rineya-cross-scale-qwen4b-8b` | `fde553c908db39e87a15a52ed239d1ba0a68bfd3` | Lab base `27f807f039fae5c5cd20934100690606b5f9c9e7`; parameter transfer into **Supa-A1.1**, not current ungated Supa | Official4B/8B projections, DEV negatives, v3 null diagnostics, pinned donor streaming E5/E6, withdrawn E7; v4 alignment tooling; newest single-layer8-prompt ablation | Historical negative for P4/P5 transfer; alignment conditional. README donor-missing state superseded by donor reload/execution. No current training-free winner |
| `research/tgsa-rineya-pretrained-retrofit` | `927405ab1b35a42cf7a5acb16fd42d2123ded567` | Mid-v3 TGSA fork; real-model retrofit gated after competence | Historical actual A1.1 midpoint soup170→173/256 but answerable regressions: REJECT; dose170→170 REJECT; Pulse vector170→60 REJECT. TGSA candidate still deferred | Explicit HOLD. No current TGSA model-quality challenger; preserve anchor-sensitive transfer negatives |
| `research/tgsa-v5-1-unified-programme` | `b60d4703fc546581998bd06e64d67c3a0e3067c5` | v5.1 from `b09149002a7892a87a461af0b3c852ed14842905`, contains `de3bd632...` | Typed kernel/controls/telemetry/resume apparatus, TinyLM mechanisms and MTP paths; README458 tests; v3 incompetent/inconclusive and v4 regime rejected | Competence-first programme; no demonstrated real-model quality, competent confirmatory mechanism or end-to-end speed |
| `research/positional-sequential-attention` | `b234327382570c8aaf6e270d354b482bd5ab1b13` | Earlier Halo significance/position/cache frontier | Latest13-cell Showcase compression screen:10–29/64 vs full-cache30, compressed persistent25.8–48.0MB;64 rows only16 packs | Historical exploratory serving-efficiency curve; no Supa quality frontier. PASS is execution, not improvement; stability confirmation of chosen knee absent |
| `codex/exp-peer-stabilizer-matrix` | `3b2aa5f239188da92f8cfcb402e17e001ad5a86e` | PCSM from frozen prereg producer `6c825c3...`, registration`4a63345...` | Synthetic mathematical feasibility, real gradient cache and one-row/one-step resource pilot. Real-peer NLL1.854 vs shuffled1.880 is mostly overlapping gradient support | Phase-C200-cell qualification not completed in committed evidence; PhaseDSTOP, no SFT or model-quality claim |

## Ordinary-model quality cards

### Exact common provenance and update geometry

RAB's ungated Supa source is `Rineya-Supa-1.7.0-devfreeze`, 596,049,920 unique parameters, source weight SHA-256 `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`. It is itself `BF16(FP32(Nimbus)+LoRA_B@LoRA_A*26/16)`: a rank16 Stage-A checkpoint200 adapter originally trained on Halo, transferred to Nimbus at scale0.8125. Supa `CORE.json` preserves source Nimbus hash `7b66b8d0...`, Halo `38fb96fe...`, adapter `517966d7...`; the early codename Taria remains in historical dense-validation records. Its public dense Demo161/256 differs from PEFT reference163/256: 251/256 raw outputs identical, two net hits lost. This is historical selection/stability evidence, not untouched sealed confirmation.

Ordinary RAB doses use fixed `outputs/splits_large` TRAIN:10,000 selected v3 corpus rows, excluding the panel's percentage/ratio/time classes. Seed17 consumes41 rows/5,233 next-token targets. Each entire input row is forwarded including the last token, so full input tokens are **5,274** (derived targets+rows). Loss is **whole-sequence next-token CE**, including prompt tokens; the artifact's `evidence_kind=exploratory_arm_dose_response_only` is misleading. Microbatch1, no accumulation, one update per row, AdamW CPU/offloaded states, beta1 .9, epsilon1e-8, weight decay0, global norm clip1.0, TF32 off, FP32 training and CPU FP32 panel evaluation. WSD is token-indexed linear20% warmup/stable60%/1-sqrt20% decay to10% peak LR, against requested5,200-target budget. Actual LR reaches1e-5 and ends1e-6. Cosine control has3% warmup, same nominal LR/end fraction, beta2 .999.

The exact row IDs, per-step target counts, data/source/harness SHA-256 and parameter checkpoint manifests are retained in each `ARM_DOSE_*.json`; the companion audit JSON extracts them without loading weights. All row/seed outcomes below are exploratory development instruments; they do not pass the frozen zero-loss retention gate unless explicitly stated.

| Candidate | Total / trainable params | Rows / targets / updates | beta2 / schedule | Panel correct; retained/gained/lost | Stops | Recorded wall seconds |
|---|---:|---:|---|---|---:|---:|
| Frozen Supa |596,049,920 /0|0/0/0|none|47;47/0/0|148/150|not a training cost|
| B28 full, seed17 retry2 |596,049,920 /596,049,920|41/5,233/41|.999 cosine3%|73;41/32/6|150|956.39|
| B28 full WSD seed17 |596,049,920 /596,049,920|41/5,233/41|.95 WSD20/60/20|76;37/39/10|150|735.17|
| Last8 cosine seed17 |596,049,920 /125,847,552|41/5,233/41|.999 cosine3%|47;46/1/1|148|763.72|
| Last12 cosine seed17 |596,049,920 /188,771,328|41/5,233/41|.999 cosine3%|52;44/8/3|145|356.83|
| Last16 cosine seed17 |596,049,920 /251,695,104|41/5,233/41|.999 cosine3%|55;40/15/7|148|885.38|
| Last20 cosine seed17 |596,049,920 /314,618,880|41/5,233/41|.999 cosine3%|67;40/27/7|150|326.22|
| Last20 WSD seed17 FP32 |596,049,920 /314,618,880|41/5,233/41|.95 WSD20/60/20|74;41/33/6|150|308.57|
| Last20 WSD seed23 |same|40/5,284/40|same|62;37/25/10|150|318.44|
| Last20 WSD seed29 |same|39/5,259/39|same|73;42/31/5|150|317.06|
| Last20 WSD seed41 |same|39/5,260/39|same|68;39/29/8|150|313.86|
| Last20 WSD3x dose seed17 |same|118/15,734/118|same|64;38/26/9|149|643.20|
| Last20 WSD mixed30chunked |same|142/25,073/142|same|48;32/16/15|150|514.91|
| Last20 WSD forge75 |same|345/119,392/345|same|63;34/29/13|150|615.29|

Wall counters span the timed train/probe section and are influenced by CPU probes and host contention; initialization/cache/discarded crashes are not totalled. Recorded peak CUDA allocation: full B28 cosine6.702GiB, full B28 WSD7.605GiB, last20 WSD5.894GiB. Host snapshots show15.218GiB total and significant commit-over-physical pressure; this is not a measured per-model RSS maximum. Two FP32 Adam moment arrays are analytically4,768,399,360 bytes for full B28 and2,516,951,040 for last20, excluding parameters/gradients/staging. Do not label sparse/offloaded storage savings measured speedups.

### Opallo publication and downstream behavior

`Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/CORE.json`, `PANEL.json`, `FILES.json` and `FULL_PARAMETER_COMPARISON.json` bind310 tensors,1,192,135,096 safetensors bytes, weight SHA-256 `e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b`.220 tail tensors substitute the FP32 training checkpoint, hash`915af77f4ad2a1813df46a5b5a65b9dddd138b820e2e9e53ad8845bdf8d0db0d`. Frozen blocks0–7, embeddings and final norm are byte-identical to Supa. Registered publication floor>=70 correct/>=38 retained is met at73/41; this is **publication acceptance**, not a licensed efficacy gate.

FP32→BF16 storage cast moves314,613,606/314,618,880 trained values, max relative difference.00389105; arithmetic correct74→73, retained41→41, gained33→32, lost6 unchanged. The one correctness loss is `rineya_v3_sft__percentage_reasoning__reas_percentage_reasoning_s00__00223`.140/150 output texts and150/150 termination flags match. Published panel permissive73 and strict59 contrast frozen47/41. `validate_published_core.py:208` explicitly records **FP32 compute after BF16 weight upcast**.

Other instruments:

| Instrument | Supa | Opallo / FP32 parent | Qualification |
|---|---:|---:|---|
| Forge IID exact |32/200|30/200|FP32 parent, CUDA,80 overlong rows skipped before200 scored; exposed floor instrument|
| Forge transfer exact |44/200|36/200|FP32 parent, CUDA,97 overlong rows skipped; 24-token generation cap|
| Forge IID termination |source in committed artifact|186/200|Not proof of no reasoning regression|
| Forge transfer termination |184/200|178/200|Longer output; pooled p=.020 historically quoted does not survive its own4-test correction|
| Puzzle exact |102/450|96/450|BF16 storage, FP32 CPU compute; guessing-floor summary.229 is a diagnostic baseline, not a universal capacity theorem|
| Long-context RAB first-line exact |161/256|135/256|Same-session FP32 CUDA loader; all256 terminate|
| Answerable tolerant, post-hoc |149/192|150/192|Mechanistic description; no preregistered equivalence margin|
| Full literal UNKNOWN, newly recomputed |13/64|**0/64**|Case-sensitive full response after outer whitespace only|
| Substring unknown, historical counter |13/64|2/64|Opallo emits `UNKNOWN.` twice|
| False full literal UNKNOWN |0/192|0/192|Does not bound all semantic false abstention|
| Mean generated tokens |2.9453125|4.3046875|245 longer,2 shorter,9 same; stopping256/256 both|

Old claims of no lost retrieval, unchanged reasoning and a capacity ceiling are too broad. The evidence is a specific small-bank comparison and cannot establish equivalence, broad impossibility, or an absence of learnable reasoning. Long-context strict scorer is RAB's own first-line normalization and is not identical to the Demo suite normalizer. The17 post-hoc rescued outputs and11 fewer substring UNKNOWNs overlap with other gains/losses; they are not an additive partition of net26.

## Registered repair and screen status

`REPAIR_PREREG.md` fixed>=7/64 treatment-mean historical UNKNOWN and pooled paired McNemar p<.05, arithmetic mean>=62 and retained>=37, across order seeds17/23/29. Local `outputs/splits_curated_repair_a33_b5233_s17/SPLIT_MANIFEST.json` and42-row train bank exist; local output inventory contains no repair-arm run/checkpoint artifacts, and RAB f53 adds only a memory blocker note after a36. Current memory availability should be measured before any execution; the historical1.04GiB end-session observation is not today's telemetry.

The registration's claim that all controls are the shipped model is incorrect: only selection seed17's original row order reproduces that FP32 parent; reordered retraining and BF16 publication are distinct. Failed natural-language supervision would not prove literal abstention unteachable. Since these are design/endpoint weaknesses known **before** repair outcomes, a separately frozen amendment/successor can preserve historical tests while adding valid literal and blind semantic endpoints and family/order-aware inference. Do not change historical success thresholds after any eventual result.

`experiments/opallo_behavior_screen_v1/PROTOCOL.md` on d597 is already the strongest source for these corrections. B original; F fixed additional format instruction; G exact first-token UNKNOWN probability gate; FG both. Only B/F need generation, G/FG replay captures.16 calibration scenario packs/64 rows and32 check packs/128 rows, synthetic same generator, not template-held-out. FG vsB primary:>=4 answerable gains under each of RAB-first-line/full-literal,>=4 full-literal UNKNOWN gains, zero lost baseline-correct answerables, zero new false literal UNKNOWN. No p-value claim. Registered limits400 pre-retention generations,900seconds/capture,3600screen,4096context,24newtokens, zero optimizer updates. Pending native adapter/parity/assets/watchdog and union overlap/scenario-target review are genuine blockers, not completed integration.

## CED and structural evidence boundaries

Same-scale RAB dose artifacts use same seed17 41rows/5233targets/41updates, FP32 GPU train+CPU probes:

| Arm | Total / trainable parameters | Cosine/.999 panel retained/lost | WSD/.95 panel retained/gained/lost | WSD seconds |
|---|---:|---|---|---:|
| H56 |1,156,087,424 /119,571,072|16;11/36|37;23/14/24|729.13|
| P56 same-token control |same|48;34/13|52;32/20/15|679.46|
| Hlast4 |1,156,087,424 /18,880,128|not run|49;47/2/0|726.25|

Hlast4 keeps all H modules but restricts trainability to the tail; it does not establish the cost of deleting24 read sites. These are one-draw exploratory comparisons. The registered zero-loss gate accepts Hlast4 at this dose but does not establish a net quality gain or memory-content contribution. H/P must be compared to ordinary adaptation with competitive tuning, not just a frozen source. Correct/removed/shuffled/counterfactual memory efficacy controls have no confirmed result here.

Initialization H/P/D56 reproduces source47/150 because zero output paths preserve its function. Pure replacement C56/C28 scores0 and fails termination. Original width-shrunk D40 is degenerate despite mechanical parity, and pure C28/C40 fail strict CED parity. Historical500k recovery endpoints have final-update/resume mismatch and incomplete raw predictions; preserve them as qualified negative evidence, not exact matched-final claims. Latent KV rank1024 whitening is an exact function-preserving storage result; rank512 ties aggregate47 but loses8/gains8, plain rank512 scores0. None is new model quality.

XSC E28/D36 source is Supa plus `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`. Full decoder4,022,468,096 + encoder596,049,920 + interface757,169,792 =5,375,687,808. Each of36 readers20,974,208 plus shared writer2,098,304. BF16 weights10,751,375,616 bytes;65 KV banks,2,080MiB BF16 KV at8192tokens, analytic only. Original P1 requires64GiB available physical/cgroup RAM, pinned CPU runtime and clean source-bound checkout; current local15.2GiB hardware cannot meet that reference gate. Sparse native-width4–8 readers then compact-width is a reasonable separated hypothesis, conditional on standalone4B qualification and actual memory-useful task evidence. Do not inherit PASS from the RAB issue4 branch or from132+2skip synthetic receipt.

## Earlier scale transfer and alternative lanes

Old cross-scale target is **Supa-A1.1**, weight`a99f079d...`, not Opallo's Supa1.7 source. P1 truncation, embedding ridge and Procrustes yielded structurally valid incoherent models. Spectral strength.10 consensus alpha.15 gave MERGEDEV48/59 vs47 but DEV-FULL144/180 both, one gain/one loss; no sealed finalist. The historical cosine.932/sign-agreement.926 donor consensus was withdrawn when shape-matched random donors reproduced it better.

Later pinned official donor reloads did execute. Streaming-forward gate relative error2.8e-6; E5 prompt-structure CKA.82–.96 vs shuffled.03–.21; E6 on40train/19held-out prompts and9013positions gave full-map R².774–.950 and cosine.873–.982, rank16 R².07–.33. Adding donor features did not improve prediction of **Supa's own representations** over Qwen0.6 base under tested linear maps. This does not establish lack of donor capabilities. `correction_0001_e7_metric_withdrawn.md` explicitly withdraws teacher-forced answer-likelihood capability conclusions and identifies old8B Showcase headroom.

Latest fde layer-ablation screen is official-weight Supa-A1.1,28 leave-one-out skips on8 prompts,553.8sec CPU,3.10GiB peakRSS; layers0–2 sensitive, middle skips often fluent. It measures inference skip sensitivity, not trainable importance, pruning recovery or quality. The current claim of broad middle redundancy must remain bounded to8 prompts/operator.

Main's `student_distillation_v1.7.7_complete.md` is a command/run-history record with an old Stage-D oracle capability1.0. It is not an Opallo/Supa comparison on current instruments; older student architectures and oracle scores cannot serve as modern quality candidates without provenance-bound remeasurement. Likewise old tiny TGSA competence, PCSM one-row NLL and Halo cache-frontier PASS labels do not establish a superior ordinary model. The retrofit ledger's actual three A1.1 transfers all failed their no-regression promotion despite one aggregate gain.

## Documentation corrections to carry into a new authoritative ledger

- Preserve old files. Point root `CONTINUATION.md` and nested RAB `CONTINUATION.md` at one current status file. Nested file says GPU unused, recommends H/P smoke after those doses executed, and contradicts itself on test counts/digests.
- Replace literal13→2 with literal13→0 plus normalized13→2 and unreviewed semantic abstention. Preserve historical source counters under their real definitions.
- Label Opallo73 as BF16-storage/FP32-compute; native BF16 serving is unqualified by that number.
- Replace WSD-only attribution with WSD/beta2 recipe contrast; do not treat pooled repeated panel rows as independent trials. Keep4/4 direction and all per-seed outcomes.
- Remove universal capacity ceiling, all reasoning data falsified, more dose proven harmful, full training buys nothing, retrieval equivalent, all further CED pointless. Replace each with the tested recipe/instrument/outcome and uncertainty.
- Preserve old cross-scale donor-missing record while pointing current status to actual E5/E6 and E7 correction; remove inference that no useful larger model exists.
- Keep official-native/mechanical/exploratory/replicated/publication/confirmatory labels separate. No action in this audit opens optimizer, release, sealed, GPU-fit or architecture-efficacy permissions.

## Minimal next sequence suggested by this evidence

1. Finish corrected measurement/source/data/native qualifications and publish a current ledger on a new branch; retain all observed negatives.
2. Qualify the existing zero-update Opallo format/gate screen with literal, semantic and false-abstention separation, then execute only its frozen calibration→freeze→check order when hardware permits.
3. MQ root audit decides its corrected data/objective lane; do not repeat already completed cells or replace missing old checkpoints with newly trained lookalikes. If a six-cell LR screen proceeds, isolate objective and LR at fixed data/order/update groups; earlier1e-4 damage does not by itself eliminate3e-5, and3e-6 remains a plausible preservation cell.
4. A bounded standalone Qwen3-4B (and optionally8B if feasible) evaluation on corrected fresh instruments should precede spending on any cross-scale memory training; old teacher-forced negative claims do not answer that comparison.
5. Keep sparse XSC as conditional engineering/recovery, with separately qualified native loader, a strong ordinary4B control, same-tokenP and memory-content interventions. Park pruning, broad depth/schedule searches, parameter transport and TGSA/PCSM promotion until their prerequisites or deployment need warrant them.

The shortest supported path is better supervision and reliable evaluation against ordinary small/large baselines, not additional architectural search inferred from mechanical PASS labels.
