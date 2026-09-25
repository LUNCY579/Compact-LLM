# SMALLVD-001 PREREG — smaller Rineya students by vocabulary and depth reduction: structural qualification and surgery-only damage

> **STATUS: REGISTERED, WITH AMENDMENTS 1-4 (before any result) AND 6 (after the vocabulary result, disclosed); end of document.
> NO STUDENT BUILT, NO LAYER CONFIGURATION SCORED, NO CANDIDATE VOCABULARY SELECTED.** G0 identity is complete (all nine recorded hashes match, `G0_TEACHER_LINEAGE.json`).
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
torch-by-storage) *[Amendment 1, A1.8: only header = analytic was measured at registration; the
torch count on A0 is now measured before any arm is built]*, and the safetensors layout predictor reproduces the teacher's header
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
lowest HOLDOUT objective. The SELECT objective chooses nothing. *[Amendment 1, A1.1: the lowest
**SELECT2** objective; every reported figure comes from REPORT.]*

**Protected-skill flag:** any category whose HOLDOUT generation flip rate is ≥ 0.90 at the
selected set is reported as CATASTROPHIC for that skill, whatever the average says. *[Amendment 1:
on REPORT; a §10 HOLD condition for A1 (V-SAFE), a design obligation for A2/A3 (A1.3).]*

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
category. *[Amendment 1, A1.1: decided on SELECT2 bpb and SELECT2 CATASTROPHIC; reported on
REPORT.]* In that case that configuration is reported as the evidence-backed safer nearby
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

*[Amendment 1, A1.2: the pass condition is the brief's, stated there; "G4 (calibration panels)" is
what this run can deliver, and full G4 stays PENDING until an instrument shim is qualified.]*

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

**GO for recovery/distillation requires ALL of:** G0–G4 PASS *[Amendment 1: full G4, A1.2]*;
**V-SAFE — A1 has no CATASTROPHIC category on REPORT** *[Amendment 1, A1.3]*; **D-COLLAPSE — the
recommended student beats knowing nothing in every chat category on REPORT answer bpb** *[Amendment 3]*;
the S1/S2 plan's protected-skill categories a SUPERSET of `LAYER_REPORT.json → G4.recovery_protected_categories`
*[A1.3, fixed by A3.2]*; if the operator names a compression subject other
than Supa 1.7.0, §4 re-run on it *[A1.5]*; G5 (dataset, tokenization and
training smoke on the student geometry) PASS; a qualified training lane for the student geometry;
a registered plan with a matched ordinary-training control (S1) and verified-teacher-target arm
(S2); the operator's teacher decision (§1); and an operator grant whose `excludes` no longer
contains `architectural_change`. **Anything less is HOLD, with the missing items named.**

---

## Amendment 1 — 2026-09-24, filed BEFORE any layer was scored, any vocabulary candidate built, or G0b run

**Registration status at filing:** no layer configuration has been scored on the real teacher, no
vocabulary candidate exists, G0b has not run, no student has been built. Everything run so far on a
model is the tiny random 9.9M-parameter plumbing model in scratch. This is an addition to an unscored
design, not an amendment to a result. Sources: the Rineya Delta independent cold review
(`smallvd001-20260924/INDEPENDENT_COLD_REVIEW.md`, records commit `399723a`) and three defects that
the tiny-model dry run of the amended code caught (A1.8).

### A1.1 HOLDOUT is split into SELECT2 and REPORT (review Q2)

§4 chose the A2/A3 set as the lowest-HOLDOUT candidate among five strategies (+ `lineage_donor`),
and §5 decided the frontier on HOLDOUT too. HOLDOUT was therefore a selection instrument, and the
winner's HOLDOUT figure a best-of-N estimate. **Fix:** `layer_select.holdout_subsplit`, a
deterministic split of the HOLDOUT rows that depends on ids and reference labels only. Within each
stratum (category × "reference is UNKNOWN"), rows are ordered by `sha256(salt + id)` with salt
`smallvd001-amendment1-report-v1`, then alternate SELECT2, REPORT, SELECT2, ...

- On the committed panel file: HOLDOUT 257 → **SELECT2 133 / REPORT 124** (chat 96 / 90). Per chat
  category in REPORT: abstention_unknown 14 (8 UNKNOWN + 6 determinate), concise_reasoning 16,
  conversation 18, grounded_extraction 13, instruction_writing 14, long_context 15. Split sha256
  (canonical sorted `id → part` JSON) `b156cb15…4914`.
- The answer-type stratum was added after a first, category-only version put 11 UNKNOWN and only 3
  determinate abstention rows in REPORT, too few to see over-abstention. Both versions were
  computed from the panel file alone, before any model output existed.
- **What chooses what.**

  | decision | instrument |
  |---|---|
  | greedy search (unchanged) | SELECT |
  | the A2/A3 set at 18 blocks (§4) | lowest **SELECT2** objective |
  | each frontier depth's layer set (S6) | lowest SELECT2 objective |
  | the §5 frontier decision | SELECT2 answer bpb and SELECT2 CATASTROPHIC categories |
  | every figure reported for a selected configuration, G4 and V-SAFE | **REPORT**, which chooses nothing |

  The single-block table (S3) uses all HOLDOUT rows, because it selects nothing.
- **Guards.** Every stage file records the split sha256 it was computed under. A resume under a
  different split refuses, and so does the report generator. The guard's positive control: the
  report refused a dry-run study computed under the category-only split.
- **Cost, stated.** REPORT has 13–18 chat rows per category, so per-category rates carry wide
  intervals. A CATASTROPHIC flag (flip ≥ 0.90) on 14 rows means ≥ 13 flips.

### A1.2 G4 is the brief's gate, stated here rather than only in the evaluation plan (review Q4)

The brief defines it: *"Gate G4 — Calibration. PASS only if compression damage is measured
separately for vocabulary-only, layer-only and combined arms."* §7 described how damage is measured
but never quoted this pass condition, and `PREDISTILLATION_EVAL_PLAN.md` §8 held the only
definition. **G4 (calibration panels) PASSES iff** A1 = V2/28, A2 = V0/selected-18 and
A3 = V2/selected-18 are each measured on REPORT with:

- finite answer bpb, overall and in every category;
- generation metrics in every chat category;
- for A2 also the REPORT KL objective;
- per-row records stored.

The S6 layer sets of A2/A3 must equal the registered selection. G4 cannot pass before G3, because
G3's bitwise virtual-vs-built equivalence is what licenses measuring through virtual students.

**The plan's instrument clause was vacuous.** *"… AND on every instrument whose shim replay of A0 is
exact"* is satisfied by zero instruments, and no shim exists yet. So this run can deliver
**G4 (calibration panels)** only. **Full G4 stays PENDING** until at least one instrument shim is
qualified; it is never PASS merely because no instrument qualified. `report_layer_study.py` prints
exactly these two statuses.

### A1.3 The CATASTROPHIC flag in §10: wired where it can both pass and fail

The review proposed making CATASTROPHIC a HOLD condition in §10. **Adopted for A1; not for A2/A3,**
for this reason.

- **A1 (vocabulary only) has a sharp prediction.**
  - By the invariance theorem, a row whose prompt and teacher continuation use only retained ids
    is reproduced by the virtual A1 greedy decode: identical input ids, and restricting the logits
    cannot move an argmax that is already retained. The only slack is fp32 batching numerics.
  - A CATASTROPHIC category for A1 means the pruning removed tokens the teacher actually emits
    there. **New §10 condition V-SAFE:** A1 has no CATASTROPHIC category on REPORT, otherwise
    HOLD.
  - It can fail: the dry run on a random tiny model fired it (HOLD). It can pass: an
    English-centric 65,536-row candidate is predicted to flip only rows where the teacher emits a
    pruned token.
- **A2/A3 (depth) do not.**
  - Flip rate is a normalised full-string mismatch over a 64-token continuation. In the two
    open-ended categories (conversation, instruction_writing; no scorer), one divergent token flips
    the row.
  - **Registered prediction, so it can be wrong:** A2 and A3 have REPORT flip ≥ 0.90 on
    conversation AND instruction_writing.
  - A HOLD condition predicted to fire whatever the recoverability is a gate that cannot pass on
    achievable data. §8 already forbids reading surgery-only damage as recoverability.
  - For A2/A3 the flag is therefore a **design obligation** instead: every REPORT-CATASTROPHIC
    category of the recommended student must be a named protected-skill category, with its own
    recovery endpoint, in the S1/S2 plan that §10 requires.

### A1.4 Invariance-test coverage (review Q1)

The line-level test floored a COUNT (≥ 10 pieces), not coverage. Changes:

- **(a) A word-level check.** BPE never merges across a pre-tokenizer chunk, so the theorem holds
  per chunk.
  - `tokenizer_validation.word_invariance` checks every distinct pre-tokenized HELD-OUT word whose
    teacher encoding is all-retained, and records the qualifying fraction (distinct and
    occurrence-weighted) in `VOCAB_CANDIDATES.json`.
  - **New REJECT 5:** any violation.
- **(b) A fixture test that records and floors the fraction.**
  - `test_retained_only_text_encodes_identically` records the line and word fractions (printed
    into the G1 log via `-rP`).
  - It asserts word coverage ≥ `0.75 + 0.20·(rows − 32768)/(151936 − 32768)`.
  - The floor was set from the corpus-free construction, measured by the test's own function
    before any real candidate existed: 309 distinct words, qualifying 0.835 / 0.880 / 0.913 /
    0.974 at 32K / 49K / 65K / 98K, 0 violations. Margins are 0.085–0.114.
- **(c) The negative control now fails at word level too.** With reversed merges on the fixtures
  alone: all 285 distinct words qualify under the identity map, and **187** re-segment.

### A1.5 The layer set is Supa-specific (review Q4b)

The A2/A3 set is chosen on Supa 1.7.0's KL against Supa's own greedy continuations, so it is a
property of those weights.

- **Commitment:** if the operator's teacher decision (§1) names a model other than Supa 1.7.0 as
  the model to be COMPRESSED, §4 is re-run with that model as `--teacher` before Phase 11. Panels,
  splits, rules and code stay the same. This run's set is not transferred.
- If Supa stays the compression subject and only the distillation-target source changes, the set
  stands, with a caveat: the targets would then come from a different model than the continuations
  the set was chosen on.

### A1.6 The 80-character sentence gate (review Q3)

It catches shared framing, not shared arithmetic: 0 % of probe rows are under 80 characters, but
the operands sit in short sentences (e.g. "It is reduced by 20%." at 21 characters). It is not
load-bearing here, because class exclusion removes the probe's entire source classes. It must not
be reused, unexamined, on a corpus where it would be.

### A1.7 Unchanged

The greedy search and its SELECT objective, the depth grid, the frontier table, the 1 % margin,
the 0.90 flip threshold, the REJECT/FLAG thresholds of §3, and every build gate of §6.

### A1.8 Three defects the tiny-model dry run caught before any real build

1. **G3 would have failed on every real arm.** `surgery.unique_params_torch` walked
   `module.buffers()` as well as parameters, so it counted the rotary `inv_freq` (head_dim / 2
   floats, recomputed from config and never saved) as weights.
   - Effect: +8 on the tiny model (A1: by-storage 4,416,520 vs torch and analytic 4,416,512);
     +64 on the real one.
   - Fixed: parameters only, with buffers reported separately.
2. **§2's "three counts agree on the teacher (header = analytic = torch-by-storage)" was not
   measured.** The torch count had never been run on the teacher; no record of it exists, and the
   function as written would have returned 596,049,984.
   - The claim registered was HEADER = ANALYTIC.
   - `run_surgery_arms.py` now measures A0 on the arms' own code path before building anything,
     and refuses unless header = analytic = torch = by-storage. On the dry run all four are
     9,946,112.
3. **The wrong-layer-map negative control hardcoded 28 blocks.** On the 6-block dry teacher it
   picked the full-depth arm and "dropped" a block that does not exist, so it passed for the wrong
   reason. It now uses the teacher's own depth, and the fresh dry chain runs it on A2 and A3 only.
   It would have been correct on the real 28-block teacher; it was vacuous on the only model it
   had run on.

**Also:** every queue launch now uses fresh dry-run directories. A dry run resumed from an earlier
launch verifies nothing about the current code.

---

## Amendment 2 — 2026-09-24: an exploratory arm, H-LC (training-free linear compensation of the dropped blocks)

**Filed before any layer was scored**, so before the registered 18-block set is known. The same
status as Amendment 1: nothing scored, no candidate built, G0b not run.

### A2.1 Why

§4 already predicts that the 18-block surgery-only student is heavily damaged, and A1.3 predicts
REPORT flip ≥ 0.90 on the open-ended categories. Plain pruning replaces a dropped run by the
identity. A cheaper recovery starts from a smaller surgery-only loss, and could move which depth is
feasible. The question: **can a closed-form, gradient-free correction that preserves the geometry
recover part of that loss?** No record in the programme has tried it: 0 hits for
compensation/ReplaceMe/fold across the records tree and the repo, with a positive control that hit.

### A2.2 Mechanism (`linear_compensation.py`)

For each maximal run [s..e] of dropped blocks whose predecessor p = s − 1 is kept:

- **Fit.** A d × d matrix T minimising ‖(y_p + m_p T) − h_e(teacher)‖² + λ‖T‖², where m_p is block
  p's MLP output and y_p is the residual it is added to.
- **Fold.** Into p's `down_proj`: W_down ← Tᵀ W_down. That is an exact identity (row-vector
  convention), so the geometry, the parameter count and the file layout are unchanged, and T = I is
  plain pruning.
- **Order.** Runs are fitted first to last, each on student activations with earlier folds
  applied, against the teacher's residual.
- A run starting at block 0 has no predecessor and stays uncompensated (recorded).

**Registered, fixed before any fit:**

| choice | value |
|---|---|
| fit set | calibration **SELECT** only (never SELECT2 or REPORT) |
| positions | every real position except position 0 (attention-sink massive activations would dominate an L2 fit) |
| λ | 1e-3 · tr(XᵀX) / d |
| tuning | none |

### A2.3 Arms, all on layer sets fixed by the registered pipeline (no new selection)

- **LC@selected:** the registered 18-block set (A2's).
- **LC@angular:** the 18-block `angular_consecutive` set, a single contiguous run, the shape the
  method was designed for.

Each is scored on SELECT2 and REPORT: KL objective, greedy generation, and answer bpb under V0
(A2-LC) and under V2 (A3-LC). The PLAIN student of the same set is recomputed in the same process
as the paired comparator.

### A2.4 Controls (the run refuses if either fails)

- An **identity fold** must reproduce the plain student's hidden states **bitwise** (dry run: yes,
  both arms).
- The plain student's REPORT objective is cross-checked against S5's, as a reproducibility canary
  (dry run: exactly equal, 0.006688232547289644 both).

Unit tests (`tests/test_linear_compensation.py`, 6 pass):

- identity fold bitwise inert, and the fold restored exactly on exit;
- fold equals the direct computation y_p + m_p T;
- refuses inside `keep_layers`, where indices are renumbered;
- **positive control:** a dropped block made exactly the identity must fit to T ≈ I with residual
  < 1e-4;
- the fit is never worse than identity on its own objective.

### A2.5 Verdict (written into `run_lc_arms.py` before any fit)

The contrast is LC@selected vs plain@selected on REPORT:

- r = 1 − obj(LC)/obj(plain), obj = the §4 category-mean answer KL;
- 95 % interval by a category-stratified paired bootstrap over REPORT rows (2000 resamples, seed 17).

| verdict | condition |
|---|---|
| **SUPPORTED** | r ≥ 0.20 and lower bound > 0 |
| **REAL_BUT_SMALL** | 0 < r < 0.20 and lower bound > 0 |
| **HARMFUL** | upper bound < 0 |
| **INCONCLUSIVE** | otherwise |
| **UNDERPOWERED** | any category has fewer than 5 REPORT rows |

The UNDERPOWERED rule was added after the tiny dry run, where one row per category gave a
zero-width interval. The real REPORT has 7–18 rows per category.

Everything else is descriptive: flips, accuracy, bpb, and LC@angular.

### A2.6 Author's expectation, stated so it can be wrong

- r in **[0.05, 0.40]** at 18 blocks.
- Gains should be larger on the contiguous `angular` set than on a scattered greedy set.
- It could be **HARMFUL**. The fit minimises L2 on the residual stream, not KL on the logits, and a
  few massive dimensions can dominate L2 even with position 0 removed.

### A2.7 What it licenses

**Nothing about A2/A3, §5 or §10:** the registered arms and decisions are unchanged. If SUPPORTED,
the recommended first student (deliverable 16) names the LC variant as the candidate *starting
point* for recovery, and needs its own registration. The choice between LC@selected and LC@angular
is made on **SELECT2**, and its REPORT figure is the one reported.

**Cost:** about 30 min of GPU after the surgery arms, in the same queue under the same lease and
RAM gates.

---

## Amendment 3 — 2026-09-24: a non-saturating gate for the depth arms (D-COLLAPSE), and the protected list fixed now

**Filed before any layer was scored.** Source: the reviewer's supplement,
`smallvd001-20260924/COLD_REVIEW_SUPPLEMENT.md` (records `e09687a`). It found A1.3's decline sound,
but identified two defects in it.

1. **The prediction was registered without the gate it would vindicate, and the branches were
   asymmetric.** If flip ≥ 0.90 comes in, nothing was gated. If it comes in below 0.90, adding a
   gate afterwards is indistinguishable from moving the goalposts.
2. **"A design obligation on the S1/S2 plan" relocated the defect** into a document that does not
   exist yet.

### A3.1 The gate: D-COLLAPSE, on answer bpb rather than flip or KL

**Correction to the supplement's premise.** Answer bpb *is* available where flip saturates. The
open-ended categories carry reference answers; only their scorer is absent. On the committed panel,
conversation has 64 / 64 rows with a non-empty reference and instruction_writing 64 / 64, and S6
already scores answer bpb on them. KL would not do: it is defined only within one vocabulary, so it
cannot score A3.

**Definition** (`report_layer_study.retained_fraction`). Per chat category, on REPORT, the
recommended student keeps this share of the teacher's advantage over knowing nothing:

    retained = (u − s) / (u − t)

- s = the student's answer bpb;
- t = the teacher's (A0) answer bpb on the same rows and bytes;
- u = uniform over the student's own vocabulary under its own tokenisation (tokens · log₂V / bytes).

It is 1 at teacher level and 0 at no better than uniform. No new measurement is needed, and the
scale is anchored by each student's own vocabulary.

**New §10 HOLD condition, D-COLLAPSE:** the recommended student (A3, or the §5 alternative) has
retained > 0 in **every** chat category on REPORT. It is also HOLD if any category's share is
undefined (u ≤ t).

- **It can fail.** A collapsed student can exceed the uniform bpb, because answer NLL is unbounded.
  The tiny random-weight dry run fired it, since there even the "teacher" is worse than uniform.
- **It can pass.** It only asks that the student beat knowing nothing, the minimal premise of
  deriving a student from a teacher at all.
- **What it does not claim.** It does NOT claim recoverability (§8). It detects collapse: a state
  in which derivation from the teacher carries no measurable information advantage on a skill.

### A3.2 The protected list is computed now, by a registered rule, not in a future plan

`LAYER_REPORT.json → G4.recovery_protected_categories` = the recommended student's
REPORT-CATASTROPHIC categories ∪ the categories with retained < **0.5** (or undefined).

**New §10 condition:** the S1/S2 plan's protected-skill categories must be a **superset** of this
list. The content is fixed by this run; the future plan is checked against it, and cannot redefine
it.

### A3.3 Both branches of A1.3's prediction are now pre-committed

- A1.3's flip prediction stays **descriptive**. Neither outcome adds or removes a gate.
- The depth arms are gated by D-COLLAPSE, registered here, in either branch.
- Nothing is retrofitted after the layer study.

### A3.4 Author's expectation, so it can be wrong

- D-COLLAPSE **PASSES**.
- The recommended student's retained share lies in **[0.2, 0.8]** in most categories.
- At least one category is below 0.5, i.e. `recovery_protected_categories` is non-empty beyond the
  CATASTROPHIC ones.

### A3.5 Unchanged

Everything in Amendments 1–2, the verdict rules of H-LC, and every threshold of §3–§6.

---

## Amendment 4 — 2026-09-24: H-LC corrections, filed before any fit

**Source:** `smallvd001-20260924/COLD_REVIEW_HLC.md` (records `481b87b`). **Status:** unchanged;
nothing scored, nothing fitted.

### A4.1 The ridge shrank toward ZERO: now toward I, scaled by the median eigenvalue

The review flagged that λ = 1e-3·tr(XᵀX)/d is 1e-3 × the *mean* eigenvalue, so a heavy-tailed
activation spectrum makes it a strong prior. The review read that prior as "T ≈ I". **The code was
worse than that.** The textbook penalty λ‖T‖² shrinks toward **T = 0**, which would pull the fit
toward *deleting* block p's MLP output, not toward plain pruning.

The positive-control test did not catch it. It passed with `atol=0.05`, which tolerated
I − λ(XᵀX)⁻¹.

**Now** (`linear_compensation.py`):

- the fit is min ‖R − XT‖² + λ‖T − I‖², so it shrinks toward T = I, plain pruning, the free null;
- λ = 1e-3 × the **median** eigenvalue of XᵀX;
- every fit records the spectrum (min / p10 / median / mean / max), λ/median, λ/max and the
  participation ratio tr(XᵀX)² / ‖XᵀX‖²_F.

**New tests:**

- an identity-optimal fit recovers I at `atol=1e-6`;
- with λ → 10¹² × median, the fit collapses to **I** (the old rule collapsed to 0).

With the penalty toward I, the fit is also never worse than identity on its own residual. That
holds exactly, where before it held only approximately.

### A4.2 Is L2 a proxy for KL? Measured, not argued

For every compensated run, `run_lc_arms.py` records:

- the fit-residual drop;
- the SELECT2 objective change when that run's fold is added to the earlier ones.

It also records the Spearman correlation across all runs of both arms (negative = a good proxy).
This is descriptive, and SELECT2 only.

### A4.3 What LC@angular licenses (the disposal rule the review asked for)

**Nothing about the student.** Its REPORT figures are descriptive.

- If LC@angular's reduction clearly exceeds LC@selected's, that is recorded as a **mechanism
  finding about applicability**: the method works on contiguous runs, and the registered set is
  not one. It may motivate a new, separately registered experiment.
- It never stands in for H-LC's verdict.
- The A2.7 follow-up naming, including the SELECT2 choice between the arms, applies **only if
  H-LC is SUPPORTED**.

### A4.4 The identity control's claim is narrowed

The identity fold does take the live path: clone, matmul, copy and restore all run. But at T = I
the transform is exact, so bitwise equality detects **state corruption only**. Tᵀ vs T, sign and
convention errors are invisible to it.

The algebra is covered by two other tests:

- the fold-equals-direct test with a random T;
- a new **bitwise T = 2I** test. A power-of-two scale is exact in floating point, so block p's
  output must equal y_p + 2·m_p bitwise.

### A4.5 fp32 is asserted, not defaulted

In bf16 (8 mantissa bits) a small T − I can round away entirely, and H-LC would then measure
exactly as plain for a reason unconnected to the hypothesis.

- `folded()` refuses non-fp32 weights (tested).
- `run_lc_arms.py` refuses a non-fp32 model.

### A4.6 Power, simulated before any fit, with the committed verdict

**Setup** (`design/hlc_power_sim.py` → `design/HLC_POWER_SIM.json`):

- the real REPORT category sizes: 6 chat categories of 13–18 rows, 4 text categories of 7–10;
- positions per row: chat U[10, 65], text U[100, 511];
- per-row plain KL lognormal with log-sd s_kl ∈ {0.5, 1.0, 1.5};
- per-row LC/plain ratio lognormal with mean 1 − r and log-sd s_ratio ∈ {0.1, 0.3, 0.6, 1.0};
- 40 simulations per cell, 2000 resamples each.

**Results:**

- **At true r = 0.30, P(SUPPORTED) ≥ 0.88** whenever s_ratio ≤ 0.6, at every s_kl. At
  s_ratio = 1.0 it is 0.75–0.80.
- **At true r = 0.40, P(SUPPORTED) ≥ 0.78** in every cell.
- The minimum r with P(lower bound > 0) ≥ 0.8 is about 0.10 when the per-row benefit is
  consistent, 0.2–0.3 at s_ratio 0.6, and 0.3–0.4+ at s_ratio 1.0.
- At r = 0.20, the bar itself, P(SUPPORTED) is 0.35–0.70, as expected at a threshold.

**Consequences:**

- **The 0.20 bar is kept: SUPPORTED is reachable.** The review's "effective n ≈ 6" does not bind,
  because the bootstrap resamples rows within fixed categories. That also means the interval is
  conditional on these ten categories; generalisation to others is not claimed.
- **One regime the review did not raise:** at s_ratio = 1.0, a true r = 0.10 still reaches
  SUPPORTED 25–28 % of the time. The effect is real (lower bound > 0), but the "≥ 0.20" size is
  noise. So a SUPPORTED verdict now also carries `size_precise` = (lower bound ≥ 0.10), plus the
  interval width. The simulation grid is the key for reading the realised width.
- **UNDERPOWERED** is a guard for subsampled dry runs; the real REPORT never reaches it.
- **The SUPPORTED/HARMFUL asymmetry is deliberate.** Plain pruning is the free null, so any
  reliable harm disqualifies the method, while a gain must clear a usefulness bar to justify a new
  mechanism in the student.

**Provenance:** `HLC_POWER_SIM.json` was produced by the scratch copy of the simulator. The
committed copy differs only in its `sys.path` line, its output filename and its docstring; same
seed, same grid. The verdict moved to `lc_verdict.py` (pure Python) with its logic unchanged, and
its objective is tested equal to the study's.

### A4.7 Unchanged

The fit set, the position-0 exclusion, both arms, the 0.20 bar, the bootstrap, and the fact that
nothing here touches A2/A3, §5 or §10.

---

## Amendment 6 — 2026-09-25, filed AFTER the vocabulary result: REJECT rule 1 rejected the identity tokenizer

**This amendment follows a result and says so.** The vocabulary candidates were built at
2026-09-25T01:41Z (`vocab/VOCAB_CANDIDATES.json`). That record is untouched, and it stands as
written. It is the reason this amendment exists.

### A6.1 What was registered, and what it returned

Registered REJECT rule 1 (§3): *"any row fails exact round-trip (decode(encode(s)) == NFC(s))"*.

**All four candidates were REJECTED by it, on exactly ONE of 9,838 HELD-OUT rows** —
`py:97f3297b4e9e2c4d`, the same row in all four. It is a Python regex character class made of
Arabic and Syriac combining marks. No other REJECT rule fired, and no FLAG fired.

### A6.2 The missing positive control: the rule rejects the unpruned teacher

The teacher's own tokenizer fails rule 1 on that same row (`vocab/VOCAB_RULE1_TEACHER_EQUIVALENCE.json`,
`teacher_fails_registered_rule1: ["py:97f3297b4e9e2c4d"]`). So rule 1 rejects the **identity**
tokenizer. What it measures on this row is a normaliser property of the text, not pruning.

The likely cause is canonical-ordering differences for these marks between the Rust normaliser in
tokenizers 0.21.4 and Python's `unicodedata` 14.0.0. That is **not established**, and nothing here
depends on it.

The rule was never run on the identity candidate before registration. The fixture test did compare
against both the teacher and NFC, but the fixtures contain no such marks. It is the same gap as
every other gate in this programme that was never shown able to pass.

### A6.3 The corrected rule 1′ is stricter in the relevant sense, not looser

**Rule 1′:** a candidate's `decode(encode(s))` must equal the **teacher's** `decode(encode(s))`,
byte for byte, on every HELD-OUT row.

- **Result:** 0 differing rows for V1, V2, V3 and V4, out of 9,838.
- **So:** pruning introduced no round-trip loss at all. All four candidates PASS rule 1′. With no
  other REJECT and no FLAG, all four are **PASS**, and V2 stays the registered A1/A3 vocabulary.
- **The NFC comparison stays reported:** each candidate fails it on exactly the one row the teacher
  fails.

### A6.4 Why this is not moving the goalposts

1. The correction is forced by the **teacher's** behaviour, a positive control independent of how
   any candidate performed.
2. Rule 1′ is equality with the teacher, the strongest round-trip property a pruned tokenizer can
   have. It is not a tolerance, and not a looser threshold.
3. Both verdicts are recorded side by side: **REJECT under rule 1 as registered, PASS under 1′.** A
   reader can apply either.

Asked of the independent reviewer before any use of V2 is reported: is this correction admissible,
or should the vocabulary arms be reported as REJECT-by-registration with 1′ as a sensitivity
analysis?

### A6.5 Measured, alongside (all registered expectations met)

| candidate | HELD-OUT token ratio | P99 per-row ratio | author's expectation |
|---|---:|---:|---|
| V1 | 1.0005 | 1.002 | — |
| V2 | 1.0027 | 1.017 | V2 ≤ 1.05 |
| V3 | 1.0070 | 1.039 | — |
| V4 | 1.0196 | 1.094 | V4 ≤ 1.12 |

- **Word-level invariance (REJECT 5):** 0 violations over 68,670 / 65,955 / 61,575 / 50,362
  qualifying distinct words (qualifying fraction 0.994 / 0.955 / 0.891 / 0.729).
- **G1:** 16/16 on the real candidates. Fixture word coverage is 0.987 / 0.955 / 0.919 / 0.893
  against floors of 0.860 / 0.805 / 0.777 / 0.750.
