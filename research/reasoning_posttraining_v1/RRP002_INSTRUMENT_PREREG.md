# RRP-002 measurement instrument — pre-registration

Status: **DESIGN, NOTHING BUILT OR RUN.** No model, no GPU and no budget were used to produce
this document, and none is needed to build the banks it specifies.

Purpose: RRP-001 qualified the native lane and met an engineering microfit criterion on 32 TRAIN
worlds. It measured nothing about reasoning. This registers the instrument that must exist
before any ladder arm can support a claim, and records what the available data **cannot**
support, before any arm is run.

Everything numeric below was measured this session, from the byte-verified releases, with the
pinned tokenizer. Sources: `evidence/inspection-20260915/FORGE_V1_TOKEN_LENGTH_SURVEY.json`,
`FORGE_V2_TOKEN_LENGTH_SURVEY.json`, and the inventory reported in `RRP002_INVENTORY.json`.

## 1. The blunt verdict first: no independent confirmation bank is achievable

The programme asks for a fresh confirmation bank, "ideally independently authored/rendered tasks
beyond the Forge grammar, not merely different seeds or renamed entities." **That is not
achievable from anything on this machine.** Recorded plainly, before any result depends on it:

- **Nothing is label-sealed.** Every gold for every split of both releases ships in plaintext
  `metadata/*.jsonl.gz`. Forge v1's own DATA_CARD says its holdouts are not cryptographically
  sealed or independently administered benchmarks.
- **Authorship is shared.** Forge v2's `oracle.py` header concedes that the shared task grammar
  and shared authorship remain common-mode risks. Forge v1's DATA_CARD says its two
  implementations are not equivalent to independently audited formal proofs, because parser and
  generator authorship is shared.
- **A new seed changes instances, not the grammar.** Regeneration at a new seed changes ids and
  case content but not grammar, renderer, solver, author, family set, or record syntax. Forge
  v1's README says outright that a new seed makes another candidate corpus, not a new
  independent benchmark. And the labels would be ours too.

So the strongest honest label available for the reserved bank is:

> **`FRESH_INSTANCE_WITHIN_GRAMMAR_SAME_AUTHOR_NOT_INDEPENDENT`**

It satisfies "reserved before adaptive selection" and "kept out of the optimizing loop". It
**fails** "independently authored/rendered". **Do not let the word "fresh" in a filename imply
independence.** Any eventual report must state this limitation in the same breath as the number.

Partial mitigation, and the best available: treat **Forge v1 and Forge v2 as two separately
rendered composition domains** (different generators, different family sets, different record
syntax, different answer encодings), and the non-Forge v3 ordinary-language corpus as a third,
genuinely different domain. Cross-domain agreement is weaker than independence but much stronger
than one grammar alone.

## 2. Measured token lengths change the plan

Both releases ship `token_count: null` and `tokenizer_status: "NOT_MEASURED"`, so every length
claim in them is whitespace **words**. Measured with the pinned tokenizer, words are not a
constant multiple of tokens: the tokens-per-word ratio ranges **0.355 to 28.4** (p50 ≈ 3.2).

Prompt tokens, and how many rows fit a window with the 128-token completion reserve:

| release / split | rows | p50 | max | fits 2048 | fits 4096 | fits 8192 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 train | 8,000 | 1,172 | 8,831 | 5,147 (64%) | 6,912 (86%) | 7,883 (99%) |
| v1 validation | 800 | 1,164 | 8,787 | 522 (65%) | 692 (87%) | 788 (99%) |
| v1 test_iid | 800 | 1,157 | 8,750 | 523 (65%) | 693 | 788 |
| v1 test_transfer | 320 | 1,287 | 8,716 | 204 (64%) | 273 | 314 |
| **v1 test_length** | 80 | **15,918** | **35,875** | **0** | **0** | **0** |
| v2 train | 16,000 | 3,325 | 15,223 | 5,297 (33%) | 10,880 (68%) | 14,080 (88%) |
| v2 validation | 1,600 | 3,325 | 15,158 | 532 (33%) | 1,088 (68%) | 1,408 (88%) |
| v2 test_iid | 1,600 | 3,326 | 15,129 | 526 (33%) | 1,088 | 1,408 |
| v2 test_transfer | 640 | 3,253 | 7,730 | 206 (32%) | 528 (83%) | **640 (100%)** |
| **v2 test_length** | 160 | **42,600** | **61,231** | **0** | **0** | **0** |
| v2 probes (384 views) | 384 | 13,906 | 15,119 | only the 64 `short` | — | — |

Consequences, all forced by measurement rather than preference:

1. **The currently qualified window is 2048 tokens** (`qualify-001` qualified
   `qualified_training_shape_tokens: 2048`). At 2048, v1 gives 65% coverage but v2 only 33%.
2. **Selecting "the rows that fit" is a biased subsample.** It systematically drops the long
   cases, which are the object of study. Any 2048-window bank must be described as a
   short-context bank, not as the release's distribution.
3. **Both `test_length` splits are permanently out of reach on this host** — 11k–61k tokens.
   `attn_implementation` is pinned to `eager`, so attention scores are materialized: at 16 heads
   and fp32, one layer's score matrix is ~1.07 GiB at 4,096 tokens, ~4.3 GiB at 8,192 and
   ~17 GiB at 16,384, against ~7.2 GiB of free VRAM. **4096 is the realistic next qualified
   window; 8192 is marginal; 16K and the length splits are out.** This must be stated as a host
   limit, not a finding about the model.
4. Answer targets are tiny — ANSWER ≤ 44 tokens, IMPOSSIBLE ≤ 8, UNKNOWN ≤ 5 — so the 128
   completion reserve is generous and could safely drop to 64, buying a little prompt room.
5. **Evidence position is already length-controlled.** In every v2 split, the
   early/middle/late/distributed strata all sit at p50 ≈ 3,325. Position varies while length is
   held fixed, which is exactly the required separation, shipped.

## 3. Strata: five categories, not three, and mostly confounded with family in v1

v1 encodes `reference_answer` as a plain **string**; an unanswerable world carries a literal
sentinel as its answer. Counted directly in v1 validation (800 rows):

| sentinel | count | families carrying it |
| --- | --- | --- |
| UNKNOWN | 79 | 9 of 16 families |
| IMPOSSIBLE | 10 | `temporal_order` only |
| UNREACHABLE | 10 | `weighted_routes` only |
| INCONSISTENT | 5 | `truth_tellers` only |
| BOTH | 12 | `deduction` — this is an **ANSWER**, a real logical verdict, not an abstention |

Two rules follow, and both must be enforced in code:

- **Do not collapse UNKNOWN / IMPOSSIBLE / UNREACHABLE / INCONSISTENT into one bucket.** They
  mean different things: missing information, no valid ordering, no path exists, premises
  contradict. `BOTH` must never be counted as abstention.
- **In v1, every non-UNKNOWN stratum is confounded with a single family.** "IMPOSSIBLE handling"
  cannot be compared across v1 families, because the stratum *is* the family. Only UNKNOWN
  spans families and supports a cross-family abstention comparison.

v2 is better here: it carries a structured `reference.status`, and both UNKNOWN and IMPOSSIBLE
span families. v2 validation: ANSWER 1,218 / IMPOSSIBLE 244 / UNKNOWN 138.

**Therefore: use v2 for the ANSWER/UNKNOWN/IMPOSSIBLE stratified abstention analysis, and v1's
UNKNOWN-only stratum as the cross-grammar replication of it.** This is the reverse of the
window-coverage preference in §2, and the tension is real: v2 has the better strata and the
worse length fit. It is resolved by qualifying 4096 before the abstention arm, not by pretending
2048 suffices.

## 4. Banks to build

All CPU-only. Sizes are registered here and are not to be adjusted after seeing any result.

| bank | source | size | disjointness axis |
| --- | --- | --- | --- |
| `RRP2-DEV-V1` | v1 `data/validation` + `metadata/validation` | 384 exact worlds, 48 from each of the 8 microfit families | v1 validation has **zero prior exposure**: a scan of every `.json` under `experiments/` found 521 forge_v1 ids, **none** from validation. Release policy marks it `development_only`. Disjoint from v1 train on id / `prompt_sha256` / `case_sha256`. |
| `RRP2-DEV-V1-PROSE` | same, `reference_with_slots` rows | 100, diagnostic only | Excluded from accuracy: the release scorer refuses to grade prose. CE and raw text only. |
| `RRP2-DEV-V2` | v2 `data/rl/validation` + `metadata/validation` | 1,600, stratified ANSWER/UNKNOWN/IMPOSSIBLE | Release verifiers assert disjointness from train on id, `group_id` and normalized `case_sha256`; I confirmed zero overlap on all ten pairwise split comparisons. |
| `RRP2-PRESERVE` | `residual_capability_v23` direct_reasoning dev (4,860) + procedural_rag dev (1,080) + v3 preservation slice `answer_or_abstain` stratum (125 answer / 125 abstain) | registered subset, TBD before use | Ordinary-language and grounded-answerability, non-Forge. **DEV only** — `residual_v23` dev/ood_dev were already consumed for dose calibration in the completed v2.3 run, so they cannot serve as confirmation. |
| `RRP2-RESERVED` | regenerated v1+v2 at an unused seed | registered before any arm runs | `FRESH_INSTANCE_WITHIN_GRAMMAR_SAME_AUTHOR_NOT_INDEPENDENT` per §1. Reserved before adaptive selection, answers and per-item errors withheld from the optimizing loop. |

**Mandatory build gate for `RRP2-RESERVED`.** Regeneration at a new seed is *not* automatically
fresh. Measured: sampling 480 new-seed (20260916) validation-index cases, **7 collided exactly
on `case_sha256`** with the original 10,000 — 6 of 30 in `hanoi` (20%) and 1 in `communication`.
The generator's duplicate check is within-build only and cannot see the previous release. So the
build must diff every new case against the original case/prompt hashes and **drop collisions**,
reporting how many were dropped. A regenerated bank published without that diff is contaminated.

Paired variants cluster correctly for free: `group_id == parent_id == canonical world id` in
both releases, and v1's probe builder sets `group_id = parent_id` specifically so paired analysis
groups on one key.

## 5. Same-world length × position × depth variants — verified constructible

The inventory established by execution, not inference, that this is buildable:

- **Deterministic replay is byte-exact.** From `(family, split, index, seed, generation_attempt)`
  alone, `forge.rng_for(...)` plus `generation_attempt+1` builder calls plus
  `forge.render(case, r, requested_user_words, position_mode)` reproduced the released user
  prompt **exactly for 12/12** sampled `pilot/validation_128` rows.
- **Depth and length separate orthogonally.** Re-rendering one replayed case at
  `target_words ∈ {256, 512, 1024, 2048, 4096}` × `position_mode ∈ {early, middle, late,
  distributed}` produced 20 variants whose answer was invariant under `verify.solve_prompt` in
  **20/20**, with `core_record_count` held constant at 12 while `added_distractor_blocks` scaled
  1 → 4 → 11 → 24 → 51. Distractor length varies; operator depth does not.
- **Longer required state histories are reachable** via the `hard=True` flag the release itself
  uses only for `test_transfer`: measured depth increases on all 8 microfit families, e.g.
  `signed_ledger` entries 7–16 → 25–35, `program_trace` steps 7–15 → 28, `relational_join` hops
  2–5 → 7–9.
- Note **the shipped release cannot** cross depth × length × position, because depth is
  confounded with split there. A regenerated bank can. This is the main reason to regenerate.

Register: any regenerated variant must be re-verified by the independent solver, and any variant
whose answer changes under re-rendering is a **build failure**, not a harder item.

## 6. Shortcut baselines and positive control

All are pure functions of the model-visible prompt. None may read a gold label.

| baseline | definition |
| --- | --- |
| `constant_unknown` | always emit `UNKNOWN`. Scores exactly the UNKNOWN base rate — the floor any abstention claim must clear. |
| `constant_majority` | always emit the most frequent TRAIN answer. Fitted on TRAIN only. |
| `cue_copy` | emit the surface identifier most textually similar to the question's cue. |
| `last_mentioned` | emit the last candidate entity/value appearing in the records. |
| `first_mentioned` | the symmetric control for `last_mentioned`. |
| `last_delta_only` | `signed_ledger`/`ledger_replay`: opening balance plus only the final delta. Retained from the RRP-001 B-1 investigation. |
| `sibling_gold_collision` | not a predictor: for each error, whether it exactly equals another world's gold **in the same family**. Tests B-1 as reinterpreted. Requires an explicit chance-level baseline. |

**Positive control:** `scripts/verify.py:238 solve_prompt(task, prompt)` in Forge v1 is already
an exact solver over the visible prompt text, and is sha256-pinned in `DATA_CONTRACT.json` as
`forge_v1_visible_text_solver`. It must score **100%** on any bank before that bank is used. A
bank where the solver fails is a broken bank, and a broken instrument is not a model failure.
Forge v2's `forge_rl/oracle.py` is the v2 equivalent.

## 7. A correction to an inherited assumption

`config.Design.max_dev_rows = 64` does **not** bind RRP-002. It is enforced only in
`data.compile_plan` and `data.validate_plan`, and no RRP path calls either — `prepare.py`,
`preflight.py`, `controller.py` and `engine.py` import `Design` for the trainable-tail and token
constants only. So DEV may be registered at any size without amending MQ-001 or touching
`validate_native_design`. The 64-row cap was an MQ-001 pool constraint, not a programme one.

Relatedly, MQ-001 supplies **no statistics layer at all**: `evaluation.classify` supports only
`exact` (list membership) and `number` (Decimal equality) and refuses everything else, and
`analyze.summarize` skips unscoreable rows. World-paired comparison with cluster dependence,
and the uncertainty method, must be built for RRP-002. They do not exist to reuse.

## 8. Ordered next steps, none needing a GPU

1. Build the scorer: five-way stratum classification, strict vs contract-tolerant parsing as
   separately observable quantities, and a refusal for anything it cannot grade.
2. Wire the v1 and v2 positive-control solvers and require 100% on every bank before use.
3. Implement the shortcut baselines and report every one alongside any model number.
4. Build `RRP2-DEV-V1` and `RRP2-DEV-V2` with a full overlap audit against v1/v2 train and
   against the 32 already-consumed microfit worlds.
5. Build `RRP2-PRESERVE`.
6. Implement world-paired comparison with cluster-robust uncertainty, then predeclare primary
   contrasts, family weights, cluster units, practical margins and the allowed number of
   model-selection reads.
7. Only then: register the 4096-window native qualification, and request an envelope for the
   ladder. RRP-001's grant is scoped to `RRP-001-MICROFIT` and its adaptive round is spent.

## 9. Explicitly not claimed

- Nothing here is built or run. This is a design with measured inputs.
- No independent confirmation bank exists or can be built here (§1).
- Both length-stress splits are unusable on this host (§2), which is a host limit and says
  nothing about the model or about sub-billion models generally.
- Human data review remains NOT PERFORMED. This agent will not self-certify it.
