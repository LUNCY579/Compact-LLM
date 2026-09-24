# SMALLVD-001 PREREG — smaller Rineya students by vocabulary and depth reduction: structural qualification and surgery-only damage

> **STATUS: REGISTERED. NO STUDENT BUILT, NO LAYER CONFIGURATION SCORED, NO CANDIDATE VOCABULARY
> SELECTED.** G0 identity is complete (all nine recorded hashes match, `G0_TEACHER_LINEAGE.json`).
> Nothing below has been measured on a pruned model at the time of this commit.
>
> **Class:** STRUCTURAL QUALIFICATION + MECHANISM SCOUT. `quality_claim: false`,
> `release_permitted: false`. No training, no teacher-response generation for distillation, no
> budget grant requested by this document.
>
> **TIMING DISCLOSURE (recorded at 2026-09-24T20:34:27Z, before any result was read):** the vocabulary-candidate
> job (`build_vocab_candidates.py`) was LAUNCHED a few minutes BEFORE this document was committed.
> Its REJECT/FLAG thresholds (1.10 / 1.25 / 1.50) and REJECT rules were already fixed, identically,
> in this draft and in the builder's code, whose sha256 the job records in `VOCAB_CANDIDATES.json`.
> No layer configuration had been scored and no student built. The same slip MODCOEF-001's seal
> disclosed; recorded so the order of events is checkable rather than assumed.
> **Correction, same day:** that launch was REFUSED AT STARTUP by its own host-RAM gate (0.97 GiB
> free against 1.2 required; commit 9.6 GiB over physical from a peer lane's CPU jobs) before it read
> a single corpus row. No vocabulary result existed at registration. The job runs after this
> commit, so the ordering slip had no consequence -- recorded anyway, because the disclosure above
> was written before the refusal was seen.
>
> **Records namespace:** `rineya-dataset-retention-records/smallvd001-20260924/`, claimed with a
> create-only `mkdir` before this document was written (the DEPTHPROF-001 §6 rule).

---

## 0. Operator direction, and the park decisions it supersedes

The operator's brief of 2026-09-24 opens this track. Three standing decisions said to park it,
and they are recorded here so the reversal is explicit rather than silent:

- `docs/model-quality/audit-2026-09-15/chatbubble-results-audit.md:140` — *"Park pruning, broad
  depth/schedule searches … until their prerequisites or deployment need warrant them."*
- `MODEL_QUALITY_STATUS.md:98-103` — park *"pruning before a stronger model warrants it"*.
- `architecture-beta-audit.md:91` — *"Park: compression/depth joint intervention behind ordinary
  quality lane."*

The brief is the deployment need. It does **not** answer the second decision's premise (no
stronger model exists yet). That premise is carried into §1 as the reason teacher selection
stays open.

## 1. Teacher: compression subject pinned, distillation teacher UNRESOLVED

**No record in the programme designates a qualified or strongest ~596M teacher** (teacher-lineage
scout; `MODEL_QUALITY_STATUS.md:40-42`: *"No candidate dominates all measured dimensions"*).

| endpoint (recorded, digest a329eb7b) | Supa 1.7 | Titan-Exp | Titan-Sparse | Opallo |
|---|---:|---:|---:|---:|
| acquisition /384 | 14 | **234** | 232 | 8 |
| arithmetic panel /150 | 47 | 69 | 72 | **73** |
| long context /256 (spent, report-only) | 161 | 162 | **164** | 135 |
| RCI-001 abstention /128 (**CONFIRMED**) | **40** | — | 25 | — |
| HSKILL2 answerable / boundary | **40 / 71** | 35 / 68 | 34 / 70 | — |
| alias /360 | **270** | 224 | 234 | — |
| status | showcase_selected_stability_validated, provisional false | experimental, release_permitted false | experimental, release_permitted false | exploratory, not gated |

**Decision (separating selection from compression, as the brief requires):**

- **Compression subject = Rineya-Supa-1.7.0-devfreeze**, weights sha256 `962abc5b…dd24b`,
  data-section sha256 `0432480f…b240`. It is the only non-provisional, non-experimental dense
  core, it holds the one CONFIRMED advantage (abstention), and it is the anchor of every
  instrument. This is the model whose vocabulary and blocks are measured and cut.
- **Distillation teacher = UNRESOLVED, and it is the operator's decision.** Every Supa-line
  candidate has byte-identical `config.json` and tokenizer files and the same tensor inventory,
  so the surgery, the id map and the layer-study code transfer unchanged to any of them. The
  choice becomes binding only at Phase 11, which this document does not open.

## 2. Exact parameter accounting (measured, not estimated)

`param_accounting.py`: three counts agree on the teacher (header 596,049,920 = analytic =
torch-by-storage), and the safetensors layout predictor reproduces the teacher's header
**byte-for-byte** before any predicted size is printed. P = 1024·V + 15,730,944·L + 1024.

**The iso-parameter frontier near the brief's candidate** — each row trades one block
(15,730,944) for ~15,362 vocabulary rows:

| config | rows | blocks | unique params | vs teacher |
|---|---:|---:|---:|---:|
| V1/16 | 98,304 | 16 | 352,359,424 | 0.591 |
| **V2/18 (brief's candidate)** | **65,536** | **18** | **350,266,880** | **0.588** |
| V3/19 | 49,152 | 19 | 349,220,608 | 0.586 |
| V4/20 | 32,768 | 20 | 348,174,336 | 0.584 |
| V0/12 | 151,936 | 12 | 344,354,816 | 0.578 |

The brief's question "is there a clearly safer nearby configuration" is registered as a
comparison **along this frontier** (§5), not as a free search.

## 3. Tokenizer candidates

**Construction** (`vocab_prune.py`): closure-respecting greedy removal over BPE defining merges,
by mixture-weighted utility on the calibration FIT split, with the protected set of
`protected_tokens.py` (structural, output-contract, lexical, evidence-based Unicode). Monotone
old→new id map; the 26 added tokens follow the retained regular tokens in their original order.

**Candidates by exact total embedding rows:** V1 98,304 · **V2 65,536 (primary)** · V3 49,152 ·
V4 32,768. All multiples of 128; V2 fits token ids in uint16.

**Category weights** (declared before any count): `token_utility.CATEGORY_WEIGHTS`.

**Split:** each calibration row goes to FIT or HELD-OUT by `sha256(row_id) mod 10 < 7`. Utility
is counted on FIT only; every reported tokenizer number is on HELD-OUT.

**REJECT (hard) a candidate if any of these holds on HELD-OUT:**
1. any row fails exact round-trip (`decode(encode(s)) == NFC(s)`);
2. any special/added token fails to map to its remapped id or to encode to exactly that id;
3. the rendered chat template differs from the teacher's, or tokenizes differently where every
   teacher token is retained;
4. any output-contract token (`UNKNOWN` 51378, ` UNKNOWN` 61395) is missing.

**FLAG (reported, not rejected):** HELD-OUT overall token ratio new/old > 1.10, or any category
ratio > 1.25, or P99 per-row ratio > 1.50.

**Author's expectations, stated so they can be wrong:** V2 overall token ratio ≤ 1.05 and V4 ≤
1.12 on the English-centric HELD-OUT split; all four candidates pass every REJECT criterion.

## 4. Layer study (Phases 7-8)

**Panels.** Calibration rows drawn from sources disjoint BY ID AND BY CONTENT from every
evaluation instrument (sealed RCI-001/RCI-002 rows, SELECT-VIEWS views, HSKILL2 rows, the
retention probe references, the long-context suite, the arithmetic panel). Chat rows are
pre-rendered with the real chat template under transformers 4.51.3 with `enable_thinking=False`.
Each row goes to SELECT or HOLDOUT by `sha256(row_id) mod 2`. *(Sources: §4a, filled in from the
corpus scout before the first model call; any change after the first layer is scored requires an
amendment.)*

### 4a. Panel and corpus sources (built by `build_calibration.py`; hashes in `calibration/CALIBRATION_MANIFEST.json`)

| panel category | kind | source | rows (select / holdout by sha256 parity) | scorer |
|---|---|---|---|---|
| concise_reasoning | chat | v3 by_class: causal_reasoning, conditional_rules, logical_implication, sequence_pattern, spatial_relations, schedule_ordering, estimation_sanity, comparison_ranking, constraint_selection, physical_commonsense, contradiction_detection | 64 | exact (normalised) |
| grounded_extraction | chat | v3: evidence_grounded_dialogue, evidence_inference, multi_turn_context_update, acknowledgment_update | 64 | exact |
| instruction_writing | chat | v3: communication_rewrite, conversation_closure, natural_follow_up, practical_next_step, plan_revision, informal_texting | 64 | none (flip rate / KL only) |
| conversation | chat | v3: empathy_validation, deescalation, tactful_disagreement, apology_repair, negotiation_compromise, preference_opinion, comparative_recommendation, decision_explanation, roleplay_reasoning, troubleshooting_diagnosis | 64 | none |
| abstention_unknown | chat | RTG-1 dev (32 UNKNOWN + 32 determinate; 0-3/UNKNOWN contract) | 64 | exact |
| long_context | chat | cc0 long-context corpus, **validation** split, single-question packets ≤ 11,000 chars (~2.7K tokens) | 64 | exact |
| text_english_prose / text_python / text_javascript / text_markdown | text | StoryForge (public-domain; CRLF normalised) / site-packages + stdlib Python / npm + @nestjs JS/TS / dist-info METADATA | 32 each | — |

One row per v3 template family. **Excluded wholesale:** v3 classes percentage_reasoning,
ratio_proportion, time_duration, boundary_setting, clarification_request,
insufficient_information, counting_sets, unit_conversion and arithmetic_chain (they carry the
150-row panel/retention probe, HSKILL1/2 rows or their template siblings; HSKILL-CONFIRM-002 is
excluded by its source classes and **never opened**); the v1/v2 corpora; the v3 preservation slice;
every DPO file; Forge RL v2; the records repo; the Demo and Supa-A1 evidence copies; the
Invisible-Limit lab; cc0 `sealed_eval`. **Disjointness is checked, not assumed:** exact
normalised last-user-turn match and shared ≥ 80-char sentences against the eight open instrument
files, with a positive control on EACH path (exact and sentence) and a negative control on fresh
text. The build refuses if any control fails.

Caveat, recorded rather than hidden: Supa's long-context LoRA was trained on the cc0 **train**
2k/4k strata. The panel uses the cc0 **validation** split, but it is the teacher's own training
distribution. For a calibration panel that is appropriate; it would not be for an evaluation.

**Continuation for teacher forcing:** the teacher's OWN greedy output plus `<|im_end|>` — the
behaviour a distilled student should reproduce. Plain-text rows use every position.

**Registered objective (search):** category-mean, over categories, of position-mean
KL(teacher ‖ student) at the scored positions of the SELECT split. Lower is better.

**Search:** greedy backward elimination from 28 to 12 blocks, every remaining block re-tried at
every step (Σ = 328 evaluations). Plus four comparators: `evenly_spaced`, `single_ablation`,
`block_influence` (ShortGPT), `angular_consecutive` (Gromov), and at 19 blocks the named
hypothesis `lineage_donor` (drop {6,7,8,9,10,14,18,23,24}, the blocks the 19-layer donor dropped;
claimed in `lineage.md`, not weight-verified).

**Evaluation of every candidate set:** HOLDOUT split only — the same KL objective, top-1/top-5
agreement, per-row tail (P95, max), and greedy generation (flip rate vs the teacher's own output,
reference accuracy by the row's scorer, abstention rate, length, cap-hit rate), all by category.

**Registered selection rule for the A2/A3 layer set at 18 blocks:** the candidate set with the
lowest HOLDOUT objective. The SELECT objective chooses nothing.

**Protected-skill flag:** any category whose HOLDOUT generation flip rate is ≥ 0.90 at the
selected set is reported as CATASTROPHIC for that skill, whatever the average says.

**Author's expectations:** blocks 0–2 are catastrophic individually (four prior instruments
agree); greedy beats `evenly_spaced` on HOLDOUT at 18 blocks; the 18-block surgery-only student is
heavily damaged (HOLDOUT flip rate > 0.5), i.e. recovery training is required, not optional;
and the single-ablation ranking and the greedy set diverge after ~4 removals (interaction).

## 5. The iso-parameter frontier (the "safer nearby configuration" question)

For each configuration in the §2 frontier table: the candidate vocabulary (§3) with the best
HOLDOUT layer set at that depth (§4). Measured on HOLDOUT by **tokenizer-independent** quantities:
answer bits-per-byte of the reference answers (answer tokens + `<|im_end|>` over answer UTF-8
bytes) and generation metrics.

**Registered rule.** V2/18 remains the recommended first student unless another frontier
configuration has HOLDOUT answer bpb at least **1 % lower** and no additional CATASTROPHIC
category. In that case that configuration is reported as the evidence-backed safer nearby
alternative. **This is a surgery-only comparison.** It measures immediate structural damage, not
recoverability, and the recommendation says so.

## 6. Factorial arms and build gates

| arm | vocabulary | blocks | unique params |
|---|---|---|---:|
| A0 | 151,936 (teacher) | 28 | 596,049,920 |
| A1 | V2 65,536 | 28 | 507,576,320 |
| A2 | 151,936 | 18 (§4 rule) | 438,740,480 |
| A3 | V2 65,536 | 18 (same set as A2, never re-selected) | 350,266,880 |

- **G1 vocabulary surgery:** `tests/test_tokenizer.py` + negative controls pass on the real V1–V4
  directories.
- **G2 layer surgery / G3 structural student:** `surgery.verify_built` + `tests/test_surgery.py`
  pass on the built A1/A2/A3: exact row and block copies, shapes and file size equal to the
  accounting prediction, unique parameter count exact, tie survives reload by storage identity,
  save/reload parity, BITWISE-equal final hidden states against teacher-with-bypass, and
  generation references no deleted id. A negative control (wrong layer map) must fail.
- **Disk:** peak one arm (≤ 1.02 GB). Each arm is built, verified, hashed and DELETED; it
  rebuilds exactly from the teacher, the committed builder and the committed id map. The records
  tree is not grown by weights, because free disk sits ~3 GiB above a peer lane's 23.03 GiB
  training floor.

## 7. G4: surgery-only damage, measured separately for A1, A2 and A3

On the HOLDOUT panels, through virtual students (bitwise-equivalent per G3): A1 and A3 by answer
bpb and generation; A2 additionally by KL. Additivity (brief Q4) is read as
Δ(A3) vs Δ(A1) + Δ(A2) on answer bpb, reported descriptively with per-row bootstrap intervals —
no significance claim is registered.

The full Phase 10 evaluation on development instruments is a **plan** here
(`PREDISTILLATION_EVAL_PLAN.md`). The qualified probes hard-lock the teacher geometry, so it needs
an evaluation shim that is itself qualified against A0 before it may score a student.

## 8. What this does NOT license

- No quality, release or promotion claim about any student.
- No statement about recoverability: every number here is surgery-only.
- No teacher selection. §1 pins a compression subject, not a distillation teacher.
- No training. Recovery and distillation need a qualified lane for non-B28 geometry: the native
  lane hard-requires 28 blocks, 151,936 rows and 596,049,920 parameters, and 95 of 100 grants
  exclude `architectural_change`.

## 9. Envelope

One GPU job at a time under the card-measured lease (`gpu_lease.py`, vendored verbatim), held for
builds as well as scoring. Host RAM `require(3.6)` before load and `guard(1.5)` between units; the
teacher loads with `device_map="cuda"`, so the host never holds an fp32 copy. Thresholds are never
lowered to force a launch.

## 10. GO / HOLD for recovery (brief deliverable 17), registered in advance

**GO for recovery/distillation requires ALL of:** G0–G4 PASS; G5 (dataset, tokenization and
training smoke on the student geometry) PASS; a qualified training lane for the student geometry;
a registered plan with a matched ordinary-training control (S1) and verified-teacher-target arm
(S2); the operator's teacher decision (§1); and an operator grant whose `excludes` no longer
contains `architectural_change`. **Anything less is HOLD, with the missing items named.**
