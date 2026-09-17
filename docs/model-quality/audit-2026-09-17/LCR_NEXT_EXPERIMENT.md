# LCR successor: build usable rehearsal and fresh measurement first

**Proposed design, not a registration, generated corpus, human approval or model result.** RRP-015 remains unchanged. This review read the public commissions, exclusion manifest and existing scaffold; it did not read protected banks or run models.

## Reuse the substrate; repair the contract before generating

At inspection, `../rineya-dataset-retention-records/lcr-v2-corpus` contains only `lcr/common.py`, `__init__.py` and bytecode. Its `data/` and `reports/` directories are empty. Reuse its deterministic RNG, invented entity pools, prose assembler, measured span placement and `compile_row` bridge in a **versioned successor**. Task generators, allocator, independent answer checker and receipts still need implementation.

Repair these explicit mismatches first:

- Use the v2 addendum's **rendered-prompt** bands `[600,1000)`, `[1000,1400)`, `[1400,1920]`; the scaffold currently bins context tokens at 300/600/1100/1600. Measure complete rendered rows, reserve 128 completion tokens, and require total ≤2048. Never truncate evidence.
- Use the eight task names below and a new corpus version; emit the required measured fields and an actual checker/version instead of `verified_by: null`.
- `WorldSpec` describes venues and entity pools. Twenty venues and disjoint names do **not** prove disjoint reasoning graphs. Add explicit graph/template-subfamily ownership and fingerprints before allocation.
- Resolve a commission conflict by a recorded amendment: the exclusion manifest lists all eight required capability names under `forbidden_task_type_names`, while addendum C4 requires them. Proposed interpretation: reuse these **metadata taxonomy names**, while forbidding copied scenarios, templates, domains, entities and text. Do not silently waive the other exclusions.
- Distinguish the original commission's “never evaluation” TRAIN purpose from its 80/10/10 partition instruction. Declare TRAIN, open DEV and protected confirmation roles at creation, in separate files with separate access rules.

## Concrete corpus and verification contract

Preserve the 6,000-row commission as **4,800 TRAIN / 600 open DEV / 600 protected confirmation-candidate rows**. Reserve their graph families, templates and lexicons before generation; all descendants of one scenario remain together. The protected bank is never used for debugging, recipe choice or thresholds. A separate checker may validate it without revealing rows or outputs to the training agent.

| Task | TRAIN / DEV / protected | Generator answer; separately implemented check |
|---|---:|---|
| `alias_resolution` | 960 / 120 / 120 | Compose alias edges; independent graph reachability with uniqueness check |
| `causal_disambiguation` | 800 / 100 / 100 | Apply explicitly stated fictional causal criteria; enumerate eligible causes |
| `rule_exception` | 800 / 100 / 100 | Rule/exception precedence; independently evaluate all applicable rules |
| `count_with_exclusions` | 640 / 80 / 80 | Set difference with duplicate identity; enumerate distinct eligible objects |
| `temporal_correction` | 560 / 70 / 70 | Apply valid corrections at query time; independently select authorized versions |
| `object_state_tracking` | 400 / 50 / 50 | Replay changes to the query cutoff; separate event interpreter |
| `latest_verified_record` | 320 / 40 / 40 | Select latest admissible signed record; independent filter and ordering |
| `multi_condition_identity` | 320 / 40 / 40 | Intersect predicates; independently enumerate candidates and uniqueness |

Each split has equal length-band counts, 40/20/20/20% needle positions, 45/30/15/10% direct/unanswerable/counterfactual/paraphrase variants, and 60/40% hard/medium difficulty. Balance four missing-evidence mechanisms within unanswerables. Allocate integer counts prospectively with largest remainders and publish pairwise cross-tabs; do not claim a fully populated Cartesian product from only 600 evaluation rows. Target at least four independently defined structural subfamilies per task in each evaluation partition. Renamed venues or different seeds do not count as new structures.

Gold must follow from the **rendered visible record**, not hidden generator metadata. The checker parses that record and independently solves it. Fault-injection tests must detect deleted decisive facts, altered quantities, reversed precedence, duplicate identities, late corrections and wrong answers. Causal tasks concern stated fictional rules, not inferred real-world causation. Human review remains `NOT_PERFORMED` until performed. A second agent/implementation within the same project provides implementation separation, **not independent external validation**.

## Admission tests, completed before model calls

1. Verify 100% of answers and spans; zero disagreements. Verify full tokenization, answer+EOS masks, no target leakage, ≤1920 prompt tokens and ≤2048 total. Publish every rejection and final source/code/tokenizer hash. Do not obtain balance by silently discarding difficult rows.
2. Apply the supplied exclusion fingerprints without reopening `ANY_READ` banks. Check IDs, scenario graphs, entities, templates, exact/NFKC/digit-blinded user text and eight-word spans. Keep answer-string overlap descriptive: common integers and `UNKNOWN` are not themselves contamination. The manifest alone is not a semantic-disjointness certificate.
3. Match questions across answerability variants; put gap/retraction language on relevant and irrelevant entities in both classes. Run question-only bag-of-words/length classification and surface-feature thresholds with **scenario-grouped** validation. Balanced accuracy >0.55 fails the build. Labels may not be reconstructed from question phrasing, lengths or missing-record markers.
4. Publish majority, always-UNKNOWN, first/last-candidate, last-entity, question-only and context-only floors, overall and by task. Apply the commission's 25% shortcut limit to **answerable rows**; applying it to the whole bank would contradict 30% unanswerables. Check prose-density requirements from the addendum over the full corpus.
5. Freeze generation/scoring contracts separately. TRAIN can explicitly request `UNKNOWN`; the protected sentinel-absent evaluation contract must not silently become the TRAIN prompt. Keep strict answer text, canonical content, abstention and false abstention separate. Compute scenario/family-aware power before calling 600 protected rows sufficient: the smallest task has only 40 rows, and variants are correlated. Expand the protected allocation prospectively if the required precision is unattainable; never relax margins after outcomes.

## Bounded experiment and hardware path

CPU preparation comes first: stream generation and tokenizer checks, at most two workers, target ≤2 GiB combined RSS and reserve 1 GiB for candidate data/receipts. Start with **96 disposable TRAIN-only engineering rows** covering eight tasks × three bands × four positions. They cannot become DEV or confirmation. Verify the whole pipeline before the 6,000-row build; these resource numbers are preparation limits, not measured guarantees.

On the 8 GiB GPU / 16 GiB host, qualify batch-one forward/backward at representative **and maximum 2048-token** shapes under the actual source, FP32 tail-training recipe, CPU optimizer and watchdog. Existing 162-token RRP-015 rows do not qualify this shape. Keep resource guards unchanged. If it does not fit, stop and register a new measured implementation; do not silently shorten contexts or claim 4096 support.

After data acceptance, preregister an exploratory paired comparison from exact frozen Supa: the same 384 clean transition parents plus either the clean short abstention pool or the new LCR TRAIN pool, each receiving **54,172 processed rehearsal tokens**, with at most one final-row overshoot. One exposure and selection seed 17 initially; three independent paired draws before a replicated claim. At 600–1920 prompt tokens this budget buys roughly 28–90 long rows, so publish actual distinct scenarios, task coverage, supervised tokens and updates. This tests the complete source/length/example-count intervention, not an isolated effect of length or a universal optimal replay ratio.

Use fresh DEV to measure acquisition and long-context/abstention tradeoffs. Keep old spent suites report-only and keep protected outputs closed until the recipe and analysis are frozen. A shortfall can motivate a separately registered dose study; it does not prove that long-context rehearsal is ineffective. No release claim follows from corpus acceptance or this pilot.

Sources: [original commission](../../LCR_GENERATION_PROMPT.md), [binding v2 addendum](../../LCR_GENERATION_PROMPT_V2_ADDENDUM.md), [dataset audit](DATASET_AUDIT.md), [confirmation gap](../../../CONFIRMATION_INSTRUMENT_GAP.md). External scaffold and exclusion manifest remain unchanged.
