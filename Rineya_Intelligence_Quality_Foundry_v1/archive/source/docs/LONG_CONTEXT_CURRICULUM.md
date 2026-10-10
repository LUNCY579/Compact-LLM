# Rineya long-context understanding and reasoning curriculum

Status: PROPOSED future dataset design, grounded in read-only local evidence and primary-source research reviewed on 30 September 2026, America/Chicago. This document does not amend the running centralized experiment, authorize training, establish native fit, or guarantee intelligence gains or retention.

## Current evidence and scope

The centralized report identifies the primary student as Rineya-Titan-Experimental-1.0.0-devfreeze: 596,049,920 unique parameters, 28 layers, hidden size 1024, 16 query heads and 8 KV heads. Its declared 40,960-token context is not a qualified operating envelope. The complete unpadded CE512 mechanics trial passed its eight checks and four disposable updates; sustained training and capability effects remain separate questions.

The source-to-array TRAIN contract preserves 5,120 rows, 680,532 input positions and 63,546 supervised positions, with maximum total sequence length 398. This is short-context compatibility evidence, not long-context acquisition. The 512-token runtime uses one sequence per microbatch, no packing, no padding, no custom positions/cache and completion-only supervision. Prior assistant turns within a conversation remain masked; only the final assistant reply and EOS are supervised.

The existing long-context DEV sentinel has 32 semantic worlds and 256 related rows, with exact complete prompt lengths of 2,024 and 4,072 tokens. Its two mechanisms are numeric range joins and authorized revision selection. Eight rows per parent vary length, answerability and neutral/adversarial distractors. Native long-context preservation remains unqualified. Existing CHECK/FINAL remain closed and are not inputs to this foundry.

These observations come from the sibling repository's EVALUATION_PROTOCOL.md, TRAIN_SCOPE_CONTRACT.md, RESEARCH_PROTOCOL.md, AMENDMENT_004.md, README.md and CENTRALIZED_EXPERIMENT_REPORT.md. They describe a read-only snapshot; another agent may subsequently advance native work. Check current receipts before any integration. The older Rineya-Dataset README describes MaxRL Composable Skills v2, a synthetic short-answer corpus with exposed evaluations; it is not evidence of native long-context or natural-language preservation.

## Objective and evidence standard

Teach transferable reasoning with unfamiliar evidence and skills while keeping instruction-following outputs reliable. Distinguish at least five axes: logical depth, evidence distance, number of relevant records, distractor difficulty and output constraints. Increasing them together makes a failure hard to interpret.

A long prompt with one answer-bearing fact demonstrates retrieval under distraction. It does not demonstrate understanding of the whole document. A prompt with three easily recognized records remains a useful sentinel, but broad competence needs aggregation, state changes, grounded synthesis, contextual skill learning and realistic source structures. RULER distinguishes retrieval, tracing and aggregation; BABILong studies reasoning over distributed facts. These papers motivate task coverage, not a predicted Titan effect. [RULER](https://arxiv.org/abs/2404.06654), [BABILong](https://arxiv.org/abs/2406.10149).

Use three separate labels: DATA_QUALIFIED for reviewed canonical examples, TOKEN_QUALIFIED for an exact tokenizer/template measurement, and TRAINING_READY only after a matching registered runtime and release contract qualify. Quality checks, agreement, training loss and dataset size do not substitute for before/after model comparisons.

## Task families and independent validation

| Family | Target behavior | Required proof or review |
|---|---|---|
| Distributed evidence joins | Resolve references across 2-5 separated records and infer a result | Independently parse and solve visible text; remove necessary blocks; change each active join stage |
| Cross-document aggregation | Count, sum, compare, deduplicate and identify exceptions across many relevant records | Independent enumeration; record-order invariance; altered-record counterfactuals; support/distractor counts |
| Authoritative revisions | Apply authority, effective time, supersession and revocation rules | Newer unauthorized records leave the answer unchanged; authorized changes can change it; explicit tie policy |
| Dialogue state and preferences | Track corrections, commitments, preferences, unresolved questions and references over turns | Replayable event ledger; final-state solver; pronoun review; superseded-preference controls |
| Rules and exceptions | Apply scopes, exceptions, precedence and applicability conditions | Separate rule engine; rule ablation; precedence perturbation; underdetermined twins |
| Causal and temporal reasoning | Infer ordering, preconditions and outcomes from distributed events | Executable transitions or finite stipulated causal model; interventions; explicit uncertainty |
| Structured data reasoning | Join tables, filter, calculate and handle duplicates or missing data | Independent relational implementation; row permutations; duplicate/null/tie controls |
| Repository comprehension | Infer dependency behavior across small files and describe or repair a concrete issue | Executable fixtures/tests; file evidence ledger; held-out repository/template clusters |
| Contextual skill learning | Learn unfamiliar DSLs, label maps, vocabularies or rule systems and apply them to new instances | Independent interpreter; held-out combinations; label permutations; ambiguous demonstration detection |
| Grounded synthesis | Integrate complementary and conflicting sources in a concise response | Claim-to-source support ledger; coverage and contradiction review; allowable answer criteria |
| Precise output instructions | Solve a substantive problem while satisfying JSON, list, length, language or ordering constraints | Semantic verifier plus independent format verifier; malformed and wrong-answer negatives |
| Instruction hierarchy and answerability | Follow actual privileged instructions and distinguish solvable, ambiguous and unsupported questions | Simple-content and benign controls; role/conflict controls; missing-evidence or impossibility proof |

LongBench v2 motivates coverage of documents, multi-document evidence, dialogue, repositories, structured data and contextual learning. Its public questions are evaluation material, not foundry training templates. [LongBench v2](https://arxiv.org/abs/2412.15204).

Low-overlap variants should ask about an association requiring a contextual definition or relation, rather than directly repeat the answer-bearing text. Avoid making success depend on unknown trivia: supply the necessary association in the context when the target is contextual reasoning. NoLiMa motivates separating latent association from literal matching. Record query/evidence lexical overlap as a diagnostic; low overlap alone does not establish difficulty or quality. [NoLiMa](https://arxiv.org/abs/2502.05167).

For contextual skill learning, vary demonstration count, rule count, label-space size, novel composition and position independently. Support definitions and examples must make the test mapping identifiable. Label relabeling must update all examples and targets consistently. Include unfamiliar inputs that require application rather than verbatim copying. LongICLBench reports difficulty with long demonstrations and large label spaces; Many-Shot ICL shows benefits in studied larger models. Neither proves these benefits for Titan. [LongICLBench](https://arxiv.org/abs/2404.02060), [Many-Shot In-Context Learning](https://arxiv.org/abs/2404.11018).

## Minimal sufficient evidence and proof traces

Every deterministic task needs a public problem and a private typed semantic specification. The private specification must not be included in policy input. Store a solver-independent evidence ledger with public record identifiers, operative rules, intermediate values and the final certificate.

Minimal sufficient evidence means the chosen support set suffices to determine the answer and each claimed necessary block has a tested dependency. It does not mean every scalar in a block is necessary, that real documents contain no redundancy, or that deleting an arbitrary fact should always destroy answerability. For natural sources, allow redundant support and record alternative sufficient evidence sets when they exist.

The generator's answer function cannot serve as the independent verifier. At least one verifier must reconstruct the answer from the final visible messages. Two programs reading the same hidden answer field do not constitute independent solving. A rendered task can diverge from its intended specification through wording errors, omitted rules or inadvertent answer leakage; visible-text solving catches these gaps.

For each deterministic row:

1. Validate its source ancestry, role, schema, target envelope and visible evidence.
2. Independently solve the final public prompt and compare the resulting valid certificate to the target.
3. Remove each declared necessary evidence block. Require the registered changed result or underdetermination, with alternate valid worlds where appropriate.
4. Change each active operation/rule/value while holding irrelevant details fixed. Recover a different valid answer where the intervention is intended to matter.
5. Change irrelevant details and permute allowable presentation order. Preserve the answer.
6. Reject malformed, wrong, duplicate-key, extra-field and type-confused responses as applicable.
7. Check for shortcuts from query-only text, aliases, winner position, answer frequency and formatting artifacts.

Do not confuse a local counterfactual with global shortcut exclusion. Record what was tested and what remains unknown.

For natural-language answers, inspect every factual claim against identified source spans. Unsupported precision, outside knowledge disguised as supplied evidence, missed qualifications and unresolved contradictions block acceptance. Multiple acceptable phrasings need explicit semantic criteria; exact string matching cannot validate prose synthesis. High-stakes source documents require domain-qualified review before use.

## Reasoning and output are separate objectives

Keep a typed proof trace for verification without forcing every user-visible response to reproduce it. Training targets should match the requested output: final answer only, concise explanation with evidence references, or a specified structured response. Never append a rationale to an answer-only or exact-format target.

For teaching explanations, derive concise auditable steps from a verified specification and independently check their claims. A teacher's fluent chain of thought is not evidence of correctness. Longer rationale supervision is a separate intervention and needs its own target/token budget and comparison. Preserve response-format diversity without creating inconsistent instructions for the same row.

IFEval motivates executable output constraints, but content correctness needs a separate verifier. A perfectly formatted wrong answer fails. IH-Challenge motivates simple, objectively gradable hierarchy tasks paired with benign controls so always refusing is ineffective. Add content complexity and length only after the basic hierarchy behavior is measurable. [IFEval](https://arxiv.org/abs/2311.07911), [IH-Challenge](https://arxiv.org/abs/2603.10521).

Role priority must be explicit in the dataset's policy contract and supported by the qualified tokenizer/template. Text that says SYSTEM inside a quoted document is data, not an actual system message. Do not invent unsupported role tokens. Document revision authority and instruction-message privilege are different mechanisms and need different controls.

## Progressive curriculum and token-length ladder

All choices below are PROPOSED. Preserve the active experiment's historical rows, seeds, loss, resource limits and scope. New foundry releases are separate prospective interventions.

| Stage | Content progression | Length progression and prerequisite |
|---|---|---|
| A | Clean atomic rules, joins, updates, simple arithmetic and formats | Fit a complete sequence within 512; establish short-task competence |
| B | Same semantics with evidence moved and separated | Separate 1024 and 2048 runtime qualifications; paired short/long comparisons |
| C | More relevant records, aggregation and state tracking | 4096 only after qualified capture/runtime and the shorter-stage controls |
| D | Difficult distractors, natural paraphrases and multi-document joins | 8192 conditional on measured fit and retained shorter-context performance |
| E | Contextual skill learning, realistic dialogue and source-grounded synthesis | 16384 conditional on competence and registered compute/resource scope |
| F | Unseen compositions and realistic long sources | 32768 conditional; keep bounded short-task retention checks |
| G | Explore the declared maximum only if scientifically justified | A separate scope approaching 40960 with complete response/EOS budget inside the ceiling |

Length alone is not a promotion criterion. Advancement requires task competence, source-correct preservation, qualified capture and independently registered resource/runtime evidence. Proposed long canonical examples can be prepared before native qualification, but must remain marked ineligible for current training.

At a given length, counterbalance evidence early, middle, late and distributed. Lost in the Middle motivates this positional control. Measure evidence density, relevant-record count, dependency span, distractor type/count, rule depth and instruction count separately from total tokens. [Lost in the Middle](https://arxiv.org/abs/2307.03172).

Keep a minimal short rendition of each TRAIN world for diagnosis and its longer variants in the same role. A short/long comparison separates inability to solve the underlying task from inability to use distant evidence. Increasing logical complexity is a distinct curriculum factor. Lengthened siblings do not increase the number of independent worlds.

LongAlign supports studying instruction data of comparable length to the intended use. Its packing and batching methods are not part of this plan's currently qualified no-packing CE512 interface. No longer training length follows automatically from a tokenizer's successful encoding. [LongAlign](https://arxiv.org/abs/2401.18058).

## Native tokenizer profiling and no truncation

Freeze the exact local tokenizer files, chat-template bytes, library versions and template arguments. The centralized report records tokenizer SHA256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 and template SHA256 a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8. Read the actual receipts at implementation time; do not infer defaults from these identities.

Measure the final complete rendering with the intended native tokenizer. Per row record full templated prompt tokens, complete target tokens including EOS, total sequence length, reserved answer budget, prompt/target token boundary, supervised count/mask, masked prior assistant turns, token-ID hash, evidence spans, and tokenizer/template identities. Store output caps separately from measured target lengths.

Token counts are not additive across arbitrary text fragments. Re-encode after edits and measure spans after final templating. Character/word counts and a different model's tokenizer are diagnostics only. They cannot support TOKEN_QUALIFIED or a native training-ready claim.

For a total context ceiling B and reserved output budget R, the occupied prompt must fit B-R. Include EOS and any required assistant-template suffix in the actual accounting. Existing 2024/4072 prompts with 24-token capture budgets illustrate this distinction. A 40960-token declaration does not permit a 40960-token prompt plus an additional answer.

Never truncate a verified target or necessary evidence. Reject or create a new recorded revision if a row does not fit. Length fitting may insert complete semantically controlled text; it cannot hide the answer in filler or use PAD/EOS as artificial content. Validate actual token spans and the final visible problem after fitting. Unsupported long rows remain canonical/reviewed, not train-ready.

## Split ancestry, transfer slices and benchmark firewall

Assign canonical parents and roles before variants, solving, teacher calls, difficulty estimates or model outcomes. All renamed, translated, paraphrased, lengthened, answerability and counterfactual derivatives inherit the same parent role. Normalize superficial alias changes while preserving substantive semantic relationships.

For natural sources, group entire documents and revisions, related document collections, conversation threads, code repositories and derivative summaries. Paragraph-level splitting from one source across roles is not independent evaluation. Document the intended independent unit and count parents separately from views.

Maintain separate transfer slices for new instances, reserved wording, unfamiliar sources, held-out rule compositions, unseen code templates, low-overlap associations and changed evidence placement. These evaluate different hypotheses; aggregate them only alongside per-slice reporting.

TRAIN_SELECTION can support registered difficulty/selection decisions. DEV supports development and continuation decisions. CHECK compares only frozen endpoints. FINAL requires its own frozen confirmation and multiplicity plan. A bank already exposed cannot become sealed by renaming or re-splitting. Existing protected banks are never opened by generation/review agents.

Public benchmarks are external evaluation sources. This document uses research mechanisms as design inspiration; it does not import benchmark rows, answers, close paraphrases or hidden solutions. Any later source import requires provenance, revision, rights and an explicit exclusion inventory. Pretraining contamination cannot be proven absent by local exact hashes.

Check byte, normalized-text and token collisions, source ancestry and near duplicates. Semantic/source checks supplement exact matching. An evaluation failure taxonomy may motivate a fresh TRAIN generator, but the evaluated prompt and its close derivatives cannot become training examples. No judge should receive CHECK/FINAL examples while reviewing training data.

Separate policy messages from private verifier specifications, oracle answers, proof traces, quality diagnostics and model outcomes. The actor receives only the intended policy view. Metadata that encodes role, winner or difficulty must not leak through filenames, template wording or visible annotations.

## Every-row review and repeated improvement

Use independent roles: designer/generator, visible-text verifier, semantic reviewer, shortcut red team, and adjudicator. Reviewers see exact row/version hashes; disagreement records include issue spans, evidence identifiers and a typed disposition. Deterministic failures block release even if reviewers agree.

Each repair creates an immutable revision and reruns relevant controls. Keep rejected originals and unsuccessful attempts. Acceptance must not be forced by a quota. An unresolved ambiguity is a deferred or rejected example, not a low-confidence training target.

For each release cycle: classify defects, propose a generator change, review that change, generate a small fresh TRAIN-only pilot, review every row, run dependency/adversarial controls, freeze identities, and prepare a separately registered student comparison. Scaling follows quality and coverage evidence. A changed generator may require new parent identity and contamination checks.

The planned 72-row pilot must be independently blind-solved row by row from its public messages. Verify the expected answer only after recording the blind answer, evidence and output-contract interpretation. Any references supplied to reviewers should remain hidden until that first-pass judgment is complete. Report reviewed-row counts, disagreements, repairs and unresolved rows; do not call all rows reviewed merely because a script processed them.

The pilot's synthetic templates are a mechanics and review demonstration with limited structural, linguistic and domain diversity. It cannot establish broad natural-language understanding, robust hierarchy behavior, long-context competence, clinical competence or retention. Native token measurement may still be pending. Its descendants need richer independent templates and later licensed/author-created natural sources with complete support review.

Same-model agents share potential biases. Different role prompts and blind first passes improve error discovery but do not constitute independent model-family or human validation. Qualified human/reference review remains necessary for difficult prose; managed-agent consensus is not calibrated language grading or student efficacy.

## Student evaluation and limitations

Titan's current capture, sustained E1 fit, acquisition, retention and native long-context preservation remain open in the reviewed snapshot. A well-reviewed dataset cannot promise maximal intelligence or no forgetting. A 596M student may need smaller rule systems, shorter demonstrations and staged compositions; these are hypotheses to test, not established capacity limits.

Before/after evaluation must preserve tokenizer, decoding, runtime, prompts and scoring identities. Report malformed/EOS failures, source-correct losses and source-incorrect repairs, per family and transfer slice. A zero source-correct denominator is undefined, not perfect preservation. Resample entire parents with all their views and seeds together; sibling rows are not independent observations.

Retention guards and rehearsal rows must be disjoint. Acquisition gains cannot cancel a represented family regression. Missing panels remain UNMEASURED. More examples, lower loss and confident explanations cannot promote a recipe without frozen behavioral gates.

The research sources below support design mechanisms and expose known benchmark weaknesses. Most retrieved evidence is the primary papers' abstracts plus the official IH-Challenge research page, not a full reproduction or audit of their released datasets. Reported large-model effects cannot be transferred quantitatively to Titan.

See RESEARCH_SOURCES.json for reviewed URLs, scope and source-specific caveats.
