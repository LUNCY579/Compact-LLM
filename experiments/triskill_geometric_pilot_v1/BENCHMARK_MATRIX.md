# TRISKILL geometric pilot: staged benchmark matrix

This instrument tests **transition-conditioned answerability** from parent T
(single-event transition tracking) and parent H (answerability/abstention). It
does not add a trained third delta. Four chain prompts, including three
dependent transfer endpoints and one fully known state control, ask for deciding
whether separated initial-state evidence still matters after an unfamiliar
rule chain. Missing or conflicting initial state can become answerable after
a reset. Without such an erasure, a surviving ambiguity requires `UNKNOWN`.
The optional conjunction control calculates a known state and independently
abstains on an unrelated missing fact; success there alone is not transfer.

There are no training examples, optimizer steps, or model imports here.
DEVELOPMENT fixtures are for testing the instrument and native mechanics.
Their gold must not become a third-skill training objective. Freeze construction
and all coefficients before SCREEN; never revise them using SCREEN outcomes.
No old Stage B file, protected bank, training bank, or spent bank is opened by
this generator. Local split signatures establish local disjointness only.
**Global held-out membership remains unverified** until a separately approved
hashed exclusion registry is checked without exposing protected examples.
SCREEN is generated exploratory evaluation, not protected confirmation.

## Frozen tiny instrument

`build_development()` contains eight deterministic fixtures with different
world, state, and rule signatures from `build_screen()`. State signatures
include the named register and its evidence; absence itself is a shared task
structure. Rule signatures for transfer encode the entire coefficient chain.
Neither nominal signature disjointness nor a different payload proves global
training exclusion or a difference in difficulty.

`build_screen()` contains exactly eight rows:

| Rows | Contract and provenance | n | Gold / property |
|---|---|---:|---|
| T ledger | Fresh complete ledger payload; original `fresh_dual_skill_instrument_v1/generate.py` SYSTEM verbatim; local stdlib oracle | 1 | `{"A":-39,"B":30,"C":46,"D":-24}` |
| T register | Fresh complete register payload; same original register SYSTEM and modulo 97 contract | 1 | `{"X":55,"Y":23,"Z":44}` |
| H answerable | Generated complete natural-language notes; one requested code is stated | 1 | `9` |
| H missing | Generated complete natural-language notes; only the other object's code is stated | 1 | `UNKNOWN` |
| Transfer missing erased | All 97 initial values; scale, reset, then affine update | 1 | `93` |
| Known-state control | Two agreeing initial records separated by the rule log; three nonzero affine updates require the stated value; not a transfer endpoint | 1 | `25` |
| Transfer missing survives | No initial value; invertible affine updates preserve 97 possible outcomes | 1 | `UNKNOWN` |
| Transfer conflict survives | Two conflicting initial records; invertible updates preserve two outcomes | 1 | `UNKNOWN` |

The T pair reuses original task contracts, not original payloads. Historical T
signal does not prove this two-row sample has useful headroom. H uses a changed
integer/`UNKNOWN` output contract and needs fresh qualification. Neither parent
is qualified for unfamiliar multi-step affine answerability by historical
scores alone. Run the four parent rows first. A prerequisite to interpreting
transfer is a paired correct gain by T over base on a T row and by H over base
on an H row, with no source-correct damage in either diagnostic family. A base
ceiling, absent gain, or parent regression means **STOP_PARENT_UNQUALIFIED**;
native mechanics may still be reported. Two rows per parent are a screening
gate, not a statistically confirmed parent advantage.

The four chain prompts distinguish reset-erased missing state, a known initial
state control requiring three nonzero updates, surviving missing state and a
surviving contradiction. The known-state control answers `25` without a reset,
and changing its initial value changes the answer. It is marked
`known_state_control`, `new_task_structure=False`, `transfer_endpoint=False`;
gains on this row cannot qualify transfer. The other three rows are the primary
dependent endpoints, and can themselves be solved by the reset-presence
shortcut. DEVELOPMENT also retains a conflict-erased reset fixture. This is
still a narrow **transition-conditioned answerability** screen. It does not
qualify general multi-step
state reasoning, broad separated-evidence integration, or genuinely new
generalization. Stronger integration requires a later authorized panel where
known initial values, conflicting values, irrelevant missing fields and
evidence precedence independently affect the answer. A positive screen remains
a reason to test that panel, not a claim to have passed it.

The primary score is strict contract correctness. JSON whitespace/key order
are permitted, but wrong types, missing/extra keys, duplicate keys and NaN are
rejected. Transfer/H accept a bare canonical integer or exactly `UNKNOWN`.
A separate semantic diagnostic accepts only an unambiguous supported wrapper
(whole-answer fence, complete labeled answer, or a single-key answer JSON).
Prefix prose before a final-answer label is rejected conservatively; it could
contain a competing answer. Canonical integers use ASCII digits. JSON format
validity requires every expected key and exact JSON value types, independently
of whether the values are correct.
Report strict format validity as well as strict and semantic correctness.

## Arms, gates and interpretation

All measured families use unchanged base, T alone, H alone, ordinary T+H
addition, construction, identity/zero intervention, matched random
projection/rotation, and prompt-only composition when those arms fit the
prospectively frozen budget. Addition is a comparator. Random controls must
match actual rank, norm and additional compute, not just share a method name.
Identity must reproduce its declared source under the same inference settings;
a mismatch invalidates mechanics. Prompt-only control uses the same task and
output contract, and counts all additional prompt tokens. If it cannot fit,
mark it unmeasured. Do not imply that an omitted arm was passed.

Stage 0 is CPU fixture validation only. Stage 1 is this tiny native screen,
with parent qualification before transfer and no coefficient search. Native
calls/forwards/tokens, peak memory, wall time, arm allocation, device and seed
must be frozen in the pilot registration before loading a model. The first
native stage has a hard 600-second wall ceiling and the existing conservative
memory reserves. Check live research jobs, leases, receipts and fresh resources
before each native stage; GPU model ownership is exclusive. No job is stopped
by this instrument. This document authorizes no extra native calls or training.

CPU-only token counting with the local Supa tokenizer and original Qwen3
thinking-disabled render found a maximum of 239 input tokens over DEVELOPMENT
and SCREEN, including the prompt-only prefix, below the frozen 256 input slots.
The longest reference output is 21 tokens, below 24 output slots. This checks
budget feasibility, not model accuracy or memory qualification. No model,
torch or transformers was imported during this count.

For every paired comparator report row IDs, correct gains, correct-to-wrong
damage, damage divided by source-correct rows, and results by task kind. The
helper reports per-family, transition-subfamily and answerability groups and
false abstentions separately. Absent model-call keys remain unmeasured and are
excluded from paired gains/losses; explicit empty text is a measured failure.
Coverage counts must accompany every score, and an unavailable comparator
cannot satisfy the expansion gate. The
helper supplies descriptive Wilson 95% intervals and an exact two-sided
McNemar test; do not treat a nonsignificant result as equivalence. On the three
dependent endpoints even three gains and zero losses have two-sided p = 0.25.
Stage 1 cannot confirm a new capability. An encouraging screen requires at
least one strict gain on the three dependent endpoints over unchanged base,
T alone, H alone, ordinary
addition, identity/zero, matched random and prompt-only comparators, zero
transfer losses relative to those comparators, no new source-correct parent
damage, and passing identity mechanics. Every required comparison needs full
paired coverage; omitted arms make this gate unmeasured rather than passed.
No third-capability inference is permitted if the best parent is equal or
better. A gain that only restores T-alone performance is interference repair,
not new transfer. Known-state control gains are reported separately and cannot
satisfy this gate. Failure to clear
this bar is a useful falsifier of this specific construction; it is not a
global impossibility result about geometry, nonlinear interaction or skill
recombination.

Stage 2 is a proposed exploratory expansion requiring separate authorization.
It uses newly generated, prospectively frozen rows and no original Stage B
reads. Family pass/failure criteria below are planned before any expansion.
Stage 3 confirmation requires new authorization, verified hashed held-out
membership, a prospective power analysis and sample size (at least 48 rows per
measured retention panel is a floor, not evidence of adequate gain power).
Third-task paired gain over all controls must have a lower 95% confidence bound
above zero after a preregistered multiple-comparison correction. Per-family
strict retention loss must be <= 5 percentage points, with a 95% upper bound
on source-correct damage <= 10%. Score these family limits separately: an
average cannot hide ledger/sign-flip damage or answerable-to-abstention damage.

## Full requested family matrix

The same native base/parent/control arms are the baseline wherever measured.
Current n counts overlapping coverage of the eight rows, not independent
datasets. Future n is a proposal, not a launched suite. Every family needs
prospectively frozen IDs and an exclusion receipt before protected claims.

| Family | Provenance / held-out membership | Scoring | Stage 1 n / proposed Stage 2 n | Pass / failure criterion | Current status |
|---|---|---|---|---|---|
| Instruction following | Generated T/H/transfer output contracts; local split only | Strict schema, complete keys, canonical atom; semantic separate | 8 / 12 generated constraint variants | Screen: zero new source-correct format damage. Expansion: >= base compliance, <= 5 pp loss; format failures are failures even with correct semantics | Available diagnostic |
| Roleplaying | New short persona/continuity scripts, no imported protected examples; membership pending | Blinded independent calibrated rubric: persona adherence, continuity, scenario consistency, requested format, each 0-2; report dimensions separately | 0 / 12 scripts | >= base per dimension; <= 5 pp strict constraint loss. Calibration and agreement gate below must pass first | Unmeasured |
| Math | Transfer affine calculations modulo 97; local enumeration oracle | Exact final value and correct `UNKNOWN`; answerable calculation reported separately | 4 / 24 balanced arithmetic/affine rows | Screen: no comparator-correct loss and one gain to justify expansion. Expansion: paired gain and <= 5 pp answerable retention loss | Available narrow affine diagnostic |
| Language | New English meaning-preserving rewrites and controlled responses; membership pending; no multilingual claim | Independent calibrated meaning preservation, fluency and naturalness rubric, each 0-2; exact factual constraints separate | 0 / 12 prompts | >= base per rubric dimension; <= 5 pp factual/constraint loss. Calibration gate first | Unmeasured |
| Long-context understanding | Newly generated separated-evidence documents; memory-qualified lengths only; membership pending | Exact retrieval plus evidence integration/contradiction, reported by token length | 0 / 12 rows after 8-row memory qualification | Must first fit all declared context/output lengths under reserves and wall budget; then >= base at each length and <= 5 pp loss. OOM/truncation is gate failure, not accuracy | Unmeasured; separate memory gate |
| Answerability / abstention | H2 + 4 chain prompts (3 dependent endpoints + 1 known-state control); both answerable and unanswerable, local split only | Strict exact, answerable accuracy and unanswerable `UNKNOWN` recall; false abstention and unsupported answer rates | 6 / 24 balanced rows | Parent signal gate above. Zero screen source-correct damage; expansion <= 5 pp loss separately in answerable and unanswerable groups | Available format-modified H diagnostic |
| State tracking | Original-contract T2 + transfer4; local split only | Exact full T JSON or final transfer value; ledger and register separate | 6 / 24 balanced ledger/register/prerequisite/chain rows | Fresh T gain gate; zero screen source-correct damage. Expansion <= 5 pp loss in each transition subfamily | Available; prerequisite panel unmeasured |
| Unfamiliar rule application | Transfer4 uses explicit new named affine rules and chains; local split only | Enumerated finite-domain oracle, answerable/unknown separate | 4 / 24 newly named/rule-structured rows | Screen gain beyond controls without their correct-to-wrong loss; expansion must generalize across new rule chains, not reused signatures | Available narrow affine diagnostic |
| Separated-evidence integration | Transfer4 has separated layout and one numerical known-state chain; stronger generated dependent-evidence panel needed, membership pending | Exact dependence on relevant evidence, independence from irrelevant missing fields, evidence precedence and reset erasure | 0 qualified (4 narrow layout diagnostics) / 24 shuffled/dependent-evidence rows | All evidence must matter in the designed contrasts; expansion gain must persist after evidence-order changes with <= 5 pp base retention loss | Unmeasured broadly; tiny layout screen is insufficient |
| Contradiction detection | One conflict-survives SCREEN row; conflict-erased DEVELOPMENT fixture is not native screen evidence; local split only | `UNKNOWN` after surviving conflict; future contrast must answer value after erased conflict | 1 / 24 balanced erased/surviving contradictions | Surviving conflict correct to clear screen; expansion must pass both types with <= 5 pp loss separately by type | Available single-case diagnostic; erased-conflict native unmeasured |
| Verifiable error correction | Proposed generated erroneous intermediate traces; truth from same independent oracle; membership pending | Identify first wrong step and exact corrected final state; also retain correct traces without unnecessary edits | 0 / 24 balanced corrupted/correct traces | Exact repair gain over base with <= 5 pp correct-trace/source-correct damage; finding any error is insufficient | Unmeasured |
| Skill composition | 3 dependent endpoints; known-state control separate; 2 DEVELOPMENT conjunction controls outside native8; local split only | Strict transition-conditioned answerability oracle; known-state/conjunction accuracy separate; each parent and ordinary combination measured | 3 narrow endpoints, 1 known-state control, 0 native conjunction / 24 stronger transfer + 12 conjunction | Endpoint gain beyond base, both parents, addition, identity/zero, matched random and prompt controls; known-state/conjunction-only gain does not pass. No parent regression | Available narrow third-task screen; broader generalization and conjunction native unmeasured |

Roleplaying and language require two independent graders blinded to arm, with
written 0/1/2 anchors and at least ten calibration anchors before target
scoring. Require >= 0.8 weighted agreement with adjudicated calibration gold
and >= 0.8 inter-grader weighted agreement; adjudicate target disagreements
without revealing arm. If calibration is absent, scores are unavailable.
Use humans or a separately authorized local grader with independent prompts;
no paid grader calls. A construction model grading itself alone is insufficient.
This repository artifact supplies the rubric plan, not calibrated graders.

Long-context memory qualification is separate from behavior: proposed eight
fixtures, four each at 2,048 and 4,096 total input tokens, with output allowance
frozen before use. Those lengths may be reduced prospectively to fit the actual
model and hardware; do not claim an untested longer context. It needs its own
authorized budget and fresh ownership checks. The current short fixtures do
not qualify memory or long-context understanding.

CPU oracle/scorer passes, native intervention mechanics, behavioral improvement
and independent confirmation are four distinct outcomes. All unavailable
panels remain unmeasured. No protected-set edit, checkpoint overwrite, paid
compute, global model setting change or full suite is part of this benchmark.
