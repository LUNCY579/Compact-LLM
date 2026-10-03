# Data card — Rineya LCR full candidate

Version **1.1.0-full-candidate**, generated September 16, 2026 (America/Chicago). English, entirely original synthetic records. **6,000 actual rows; complete content, not full acceptance.**

## 1. Purpose and authority

The supplied brief commissions single-turn rehearsal data to preserve long-context retrieval and abstention while learning a new skill. This dataset is not an evaluation instrument. The source brief is preserved verbatim in `provenance/ATTACHED_SPEC.md`.

The user's attachment is the authority for this delivery. Its v1 names (`temporal_correlation`, `latest_verified`, `multi_condition`, `count_with_exclusion`) are intentionally preserved. This is **not** a claim of compliance with the repository's later v2 taxonomy, rendered-prompt bands or additional gates. No evaluation-suite rows, answers or templates were opened to design this corpus.

## 2. Actual composition

| Task | Rows |
|---|---:|
| `alias_resolution` | 1200 |
| `causal_disambiguation` | 1000 |
| `count_with_exclusion` | 800 |
| `latest_verified` | 400 |
| `multi_condition` | 400 |
| `object_state_tracking` | 500 |
| `rule_exception` | 1000 |
| `temporal_correlation` | 700 |

Train/dev/holdout: **4,800/600/600**. Intended short/medium/long bands: **2,000 each**. Early/middle/late/distributed position allocations: **2,400/1,200/1,200/1,200**. Direct/UNKNOWN/counterfactual/paraphrase allocations: **2,700/1,800/900/600**. Hard/medium allocations: **3,600/2,400**. All four UNKNOWN mechanisms contribute **450** rows each.

Every row has a unique scenario group. Combined and messages-only exports are views, not extra independent examples. The split assignment, task/variant/position/difficulty allocation, entity spaces and split case-layout profiles were fixed before rendering. Balanced color/location question selectors were also assigned before the final whole-corpus generation; see the committed-in-package registries.

All 15 global stratum-pair tables are measured, with maximum relative error **0.5%**. The 30 conditional within-band pair tables have maximum relative error **5.2632%** under the audit's actual-marginal convention. Thus a literal ±3% requirement for every small within-band cell is **not met**. Integer discreteness is disclosed, not silently treated as a pass.

## 3. Semantic construction

Records contain two complete task accounts, two additional near-miss accounts and many unrelated background facts. A query selects an account by a conjunction of descriptive color and location; hard queries use synonyms, while medium queries use literal descriptors. Entity-absent cases have no account matching that description, rather than using a visibly absent proper name as a cue.

Alias tasks use reference chains ending at a custodian. Causal and multi-condition tasks require conjunctions over candidate traits. Rule tasks apply the highest-priority matching exception over a default. Count tasks sum disjoint batches after exclusions. Temporal tasks find the immediate successor of a reference event in the stated ordering. State tasks replay observations to a checkpoint. Latest-verified tasks select the most recent verified edition, not merely the last textual mention.

The independent checker consumes only model-visible text. It parses the prose and uses separately implemented solving procedures; for example, count verification explicitly builds item sets rather than reusing the generator's arithmetic sum. Gold answers, metadata, stored worlds and the assistant completion do not enter the solve. They are compared only afterward.

**Independence limit:** generator and checker have separate implementations but share authorship and a designed grammar. This is not independent staffing, blind human certification or proof against every shared conceptual error.

## 4. Measured semantic and evidence results

All **6,000/6,000** answers and unanswerability reasons rederived correctly; **85,141** task facts parsed; **0** answer disagreements; **0** reason disagreements. Every answer is one whitespace-delimited token (invented name, integer, or literal `UNKNOWN`). All messages use exactly system/user/assistant roles.

On the **4,200 answerable rows**, positive evidence passed its declared placement test: early **1,680**, middle **840**, late **840**, distributed **840**. Early means all positive-span ends are within the first quarter; late starts are in the final quarter; middle lies within the central half. All 840 answerable distributed rows passed the implemented test requiring at least two evidence-fragment deletions to change the derived answer. Exact Unicode character offsets are retained in every row.

All **1,800 UNKNOWN rows have empty positive-needle spans**. Related, retraction, contradiction or absence-neighborhood spans are separate. Their position quotas are not credited as positive answer evidence. There are 1,080 hard UNKNOWN rows for which positive-evidence lexical non-overlap is inapplicable, not a fabricated pass.

Implemented hard-row checks find at least eight counted non-evidence names and two near-miss account candidates, with no literal question/positive-evidence word overlap on applicable hard rows. The metadata's distractor count is the set of named entities outside positive evidence, **not an independently adjudicated count of irrelevant concepts**; for UNKNOWN rows the positive set is empty. Explicit separate background entities are also present. Near misses differ in one query-binding criterion, not necessarily in one deep causal reasoning step. The hard label is therefore an operational curriculum label, not a validated model-difficulty rank.

## 5. Length qualification — unresolved

The intended source context bands are short 300–600, medium 600–1,100 and long 1,100–1,600 **native tokens**. These are not measured values. All native `context_tokens`, `total_tokens`, rendered token counts and capacity fields remain **null** because the exact four hash-locked tokenizer files were unavailable. No other tokenizer or estimated conversion is substituted.

Actual context **word** counts, provided only as lexical measurements:

| Intended band | Rows | Minimum words | Median words | Maximum words |
|---|---:|---:|---:|---:|
| short | 2,000 | 235 | 274 | 453 |
| medium | 2,000 | 480 | 517 | 555 |
| long | 2,000 | 815 | 860 | 904 |

Complete factual sentences were added where evidence placement required it. Nothing was truncated. **Neither the 2,048-token full-chat cap, the 128-token reserve nor actual band membership is certified.** The native census attempts are preserved as BLOCKED with zero measured rows, not zero failures. Some intended short rows may move outside their native band; no prediction is made without tokenization.

## 6. Shortcut audit — measured controls and a known limitation

The requested exact-answer baselines are implemented and measured before any target-model call: majority, task majority, always UNKNOWN, recent/first/last entity, first/last task-valued candidate, question-only and context-only TF-IDF nearest neighbor. An additional majority baseline restricted to answerable training rows is included. TRAIN nearest-neighbor scores use fivefold out-of-fold prediction; dev and holdout use TRAIN fitting only.

The largest overall exact-answer score on answerable rows is **1.333333%**, and the largest per-task score is **5.535714%**. Both are below 25%. The 30% overall always-UNKNOWN rate is reported separately from its 0% accuracy on answerable rows.

Question-only answerability classification (unigram TF-IDF plus question length, fitted TRAIN only) achieves balanced accuracy **50.6796% TRAIN**, **47.2619% DEV**, **48.4127% holdout**. A TRAIN-selected single-threshold surface rule scores **49.6032% DEV** and **49.0079% holdout**. Exhaustive threshold enumeration covers 21 finite visible-text features and both directions, not every possible feature; the optimistic all-cohort best is **51.9246%**.

**Known stronger rule:** bind the requested account and read explicit validity/withdrawal/nonissuance notices. This cheap task-aware rule gets **87.5% balanced accuracy**. It is stronger than the question-only or raw surface controls and shows that much of the abstention remains easy after account binding. The corpus is **not certified shortcut-free or universally nontrivially unanswerable**.

An earlier build had question-only DEV balanced accuracy 56.3889%. The design was changed and **all rows regenerated**, using balanced selector allocation and one stable question stem per task. No bad rows were discarded. This reduces accidental answerability cues but also restricts question-language diversity. Data dev/holdout were examined during this iteration and **must not be treated as fresh sealed benchmarks**.

## 7. Leakage and structural separation

Across the delivered splits, full contexts and questions have **zero** exact, NFKC-whitespace and digit-blinded matches. Actual name pools contain **1,024 names per split**, with zero intersections. Scenario IDs and registered typed-degree case-layout signatures are also disjoint.

This is **not strict structural disjointness**. Split-specific domains/depths coexist with shared operator primitives, local sentence patterns and reduced solution motifs. The audit reports **352/392/467** shared name-blind sentence forms for train–holdout/train–dev/dev–holdout respectively, and **136/136/124** shared exact context 8-grams. Shared integers and `UNKNOWN` answer strings are counted rather than hidden.

No complete external evaluation bank was accessible to the generation process, and no frozen bank was opened. External text/entity/template/graph overlap remains **unmeasured (null)**, not zero. Internal separation cannot substitute for a blinded external exclusion receipt.

## 8. Other limits and provenance

The counterfactual label means a provisional forecast in the record is contradicted by the authoritative account. It does not measure contradiction of a pretrained model's prior knowledge. Paraphrase is a curriculum allocation with synonym-based query wording, not a separately randomized causal intervention. Common templates, a finite invented-name vocabulary, explicit account completeness and regular prose limit the linguistic and world diversity.

The 235 passing tests are mechanics and data-checker tests, not model quality evidence. No pretrained model weights were read, no inference or training was performed, no retention effect was measured, and no existing experimental budget or branch was changed. Original synthetic data are offered with CC0-1.0 dedication intent and original code under MIT; the source brief is preserved as supplied and is not relicensed by this package.

## 9. Disposition

**Full corpus content delivered; strict acceptance remains unclosed.** `reports/acceptance.json` is the machine-readable authority. Do not treat the presence of five receipt filenames as a five-gate pass. Native token fit, complete exclusion, strong structural separation, nontrivial abstention and the strict conditional balancing interpretation all require resolution before this can be called an accepted corpus under the attached brief.
