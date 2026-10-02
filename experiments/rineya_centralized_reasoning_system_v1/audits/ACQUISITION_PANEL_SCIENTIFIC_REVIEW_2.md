# Acquisition-panel scientific review 2

2026-10-02; reviewer `/root/acq_science`, fresh non-builder scrutiny of candidate2.
**The earlier-selection repair is realized, but the acquisition evaluation
population remains rejected at this freeze: a terminal+first+penultimate public
control solves 32/32 while skipping 84 other candidate guard evaluations.** Accept
only the narrow supplied source-feasibility finding, separately from evaluation-
panel acceptance. Do not substitute this finite positional recipe for the
original acquisition endpoint. CRS-LA-H01/LA-I1-002 remain open.

This is round2 of the acquisition-data milestone, whose maximum is three
review/repair rounds. I changed only this report. I ran no suite, core, oracle,
constructor, hand counterexample, bank generation, tokenizer, framework, model,
native operation, host database, protected/confirmation bank or quarantine
repair. All review calculations used the retained public/private/raw records.
Shared supplied semantics, model and workspace limit independence; CPU agreement
does not establish independent world truth or discovered procedures.

## Exact freeze and raw evidence

All three physical file hashes matched the assigned freeze:

| Artifact | SHA256 |
| --- | --- |
| acquisition_data.py | 90f0cf990e736f4d455ceba07b93ed06957ef68759b3b0e46fe0fef4af6b4943 |
| tests/test_acquisition_data.py | 1329e3ae43e76d6062ab124a1005778aeb4556a08d2d437977db286ca7ac32b0 |
| contracts/ACQUISITION_PANEL_V1.json | 7fd4246c00583dd4e8363022ff04002768d5779af27fe72bf2689693e17c5968 |

I read source/tests, the exact contract, and retained evidence in
`outputs/acquisition_second_freeze_root_validation_001/`. REPORT.json physically
matched c99ad46b2a01f9625246d7fa3c2d045b03db3fbc40a4094e2cdb2cf3f32a90f6.
All seven raw refs matched their physical hashes, including PRIVATE_BUNDLE.json
c9acf12fb4d9d1ac7d1f6bb2e7f4c1b07a00da9f41ee248188a9e4bae75e52e3,
PRIVATE_CORE_OUTPUTS.json 5aae8078cd4b058bb3785e91e53d0ed6eb507087a77f6a21cdb6e10abb62aa91,
and CPU_RECEIPT.json eaada430d9888ad760e22baefe9ce240da83a257e7bf3f6617cb6f71b45f8331.
PLAN, ATTEMPT, test stdout and validation-wrapper hashes also matched. All five
retained bundle sections recomputed to the manifest hashes:

| Section | SHA256 |
| --- | --- |
| public | f8d60b951de2825b3e103ca5bb6eb7be25d299b304305eb5f63b3f84e80f3b41 |
| gold | d27a8b97d4d746a67645e58e0b5ecf6e467aff2b5a2b3db5a9df6eb8901565fa |
| latent | 9f769b7383488112b67c74535e0435652feb48a7cfc11468e5d26c3b5c58ee91 |
| lineage | 25702f6b392c53bd9c7c0acf291e542774e32e695503b49c6aa9e4bc278ea02e |
| assessments | c6218e9265bb9335d1f2fc44e821227c62a2c37fe68bf1223addf4260b4cd2c2 |

Root's one retained observation ran 15 class tests, one 32-core comparison batch
and 32 canonical/oracle rename controls; all passed. The builder's seven targeted
tests were not repeated in that observation. Every retained raw core answer
matched its exact scheduled gold status/value, with no core error. All 32 planned
rows remain; eight of each status and two per family/status remain balanced.
Source sizes are 706–3010 UTF8 bytes, and maximum retained canonical search is 1025
states. These are bounded CPU/source findings, not token fit or learning.

The validation wall is 3.5864788 seconds. Root reports whole-command wall 4.15628
seconds separately; nested core rows sum 0.047 seconds and 192 operations, 1060
extracted events, 56153 source bytes scanned and 110354 verification bytes scanned,
with peak serialized state 13803 bytes. Nested core time must not be added to
validation or command wall. Cumulative named main/rename/diagnostic accounting is
160 main plus 160 rename plus one diagnostic, hence 161 core calls and 321 combined
invocations. Prior builder raw rows remain unavailable; this fresh observation
does not reconstruct them. Coordinator/OPS adjudicates cumulative authorization.

My read-only analysis made zero additional oracle/core/scorer/bank/hand-world
calls. It scored 32 public three-position predictions against retained gold,
checked 32 raw answers, 32 old/new ancestry associations, 32 kinship-code pairs and
five bank hashes. The main audit command returned in 0.4032833 seconds, including
Python startup, file loading, hashing and parsing; that is not exclusive shortcut
or CPU-engine time. Metadata/shape reads are ancillary review work, not endpoint
invocations. No full validation batch was repeated.

## What candidate2 actually repairs

All eight supported answers now select an earlier owner, two in each family.
Candidate count/selected position cells are:

| Family | Count and selected zero-based positions |
| --- | --- |
| empty_joint_gates | N=2,pos0; N=5,pos0 |
| canceled_prefix_joint_gates | N=6,pos4; N=8,pos0 |
| alias_selector_joint_gates | N=3,pos1; N=4,pos0 |
| conjunctive_joint_gates | N=7,pos5; N=9,pos0 |

SUPPORTED and REFUTED cases share an excluded terminal with the same normalized
terminal-local guard/history pattern within each family: confirmed modality,
supported then refuted terminal requirement history, refuted common exception,
and the same alias/conjunction decoration. My public parse found one such local
profile per family across both statuses; source inspection confirms no changed
terminal recipe encodes which earlier case is supported. The different outcome
comes from earlier applicability. This is a genuine repair of round1's terminal-
only counterexample, not merely a new root ID or a claimed selection position.

Individual public controls score terminal 24/32, first 13/32, penultimate 12/32 and
fixed-position 1 11/32 semantic exact. However, `selection_recipe` at lines 322–330
always puts the earlier winner at first or penultimate. In the empty family its
N=2 first-positive penultimate is also first, so even that family's two supported
positions are identical. No scheduled winner is outside the union of first,
penultimate and terminal. The truthful registration says this is not depth=N;
that disclosure prevents overclaiming but does not cure the population shortcut.

## Reproducer and complete three-position negative

I implemented an independent read-only parser for the exact rendered source;
it did not call the supplied public-position helper, latent oracle or core.
Candidate membership uses public query aliases, confirmed modality, query cutoff
and active public cancellation. It sorts those candidates by public event time.
For each chosen position it evaluates only that candidate's REQUIRE/EXCEPT facts
at its binding cutoff, retaining latest facts and ties. Model inputs are
source/query/rules only; gold is used solely after prediction to score it.

The complete decision rule is:

```text
N = number of public uncanceled confirmed query-entity candidates
positions = unique {0, N-2, N-1}, in decreasing chronological order
evaluate guards/history only at those positions
if terminal is UNKNOWN or CONTRADICTORY:
    return terminal's status and null value
otherwise if any inspected candidate is SUPPORTED:
    return the latest such candidate's resolved owner
otherwise:
    return REFUTED with null value under the complete EMPTY fallback
```

Observed semantic status/value accuracy is 32/32. There are 176 scheduled guarded
candidates in total, but only 92 distinct chosen-position evaluations across the
32 cases: 84 are omitted. The parser still reads the source and collects candidate
records;84 is a skipped guard-evaluation count, not unread source events, saved
tokens, measured runtime savings or full-context independence. The check did not
choose a mapping from private labels or use family/draw/root IDs to predict.

The underlying task still requires chronology, typed alias resolution and gate
evaluation. This negative is therefore a supplied-recipe simplification, not an
assertion that all semantic reasoning is absent. It shows why the individual
controls and earlier-selection count are insufficient to qualify the claimed
broader selection population. Intermediates outside those three positions remain
potentially relevant under allowed interventions, but do not defeat the shortcut
in any scheduled case. The larger N therefore cannot supply evidence of realized
N-stage selection burden.

Other residuals remain visible: length-bucket fitted majority 17/32, identifier
5/32, last literal 9/32, family and FACT-count majorities 8/32, draw/order 8/32 and
last-mentioned-owner 2/32. Label-fitted majorities use the same open-TRAIN gold,
not held-out predictions. Neither incomplete controls nor this one newly exposed
shortcut establish general shortcut absence.

## Related revision, not fresh independent parents

Every v2 ancestry row correctly points to its corresponding retained rejected-v1
family/slot root and retains `[prior_root, revised_root]`. All 32 v2 kinship codes
equal the corresponding v1 codes. Versioned source and root identity do not make
these unrelated parents; the explicit `RECIPE_REVISION_OF_UNQUALIFIED_PANEL`
operation is appropriate. They are not an additional 32 independent root samples.

The evidence assignment itself also changed. `assigned_evidence` seeds its
shuffle with VERSION, so the version bump reassigns 28 of 32 family/slot statuses,
although draw 307/order 401, total balance and family balance remain unchanged.
Exactly one slot for each old status retains that status ; 28 change. This is a
registered related assignment revision, not replay of v1 labels, and cannot be
hidden as a cosmetic rename. The frozen v2 contract/plan and retained raw v2
answers expose it prospectively; no model outcomes were used. Keep this complete
relationship and both rejected populations in the handoff.

The canonicalizer still uses the reviewed full/potential-cone/kinship graph
structure and fail-unavailable budget; typed equality, order/ties, query cutoffs
and finite kinship boundaries retain their reviewed meaning. The revised code
adds bounded scalar validation before serialization/census and exact code/SHA
association checks. OPS owns operational boundary acceptance. Direct replay,
known optimizer deny-set and source pins provide cooperative source provenance,
not proof against undisclosed edits, pretraining overlap or statistical
independence. Model-facing source/query/rules and scorer-only gold remain separate;
the optimizer target API remains unavailable for this panel. Constructor,
renderer, latent oracle and core still share supplied assumptions.

## Bounded prospective round3 criterion

**Yes: varying predeclared nonterminal winning positions, including interior
positions, can close this specific three-position flaw within the remaining 32
planned parents.** It need not enlarge or replace the panel. Freeze v2's exact
family/slot status assignment explicitly instead of calling a VERSION-dependent
reshuffle. Retain eight supported cases, all family/status balance, the 32-slot
denominator, draw/order and all v1/v2 ancestry links; a later version is a further
related revision, not a fresh parent census.

Before any new observation, prospectively name at least one supported winner in
each family at an interior position1..N-3, outside `{0,N-2,N-1}`. This is feasible
in the existing v2 supported cells: empty N=5; canceled-prefix N=6 or 8; alias N=4;
conjunction N=7 or9. N=2 and3 cannot provide such an interior, and should stay
visible rather than be replaced. Other statuses retain the exact blocked-terminal
counterexample coverage. Do not select positions by trying model outcomes or
resample slots. The three-position control would then miss at least those four
supported owner answers, so its exact score must be at most 28/32 under the declared
unchanged balance.

Review the frozen recipe and later separately scoped retained evidence for actual
selection positions, unchanged v2 statuses, opposing local terminal evidence,
full/cone/kinship controls and complete source/latent/core reconciliation. Score
the complete three-position rule, not just its individual components. Do not
change failed cases, denominator or interpretation after the observation.

This is a prospective finite-task criterion for the last permitted review round,
not authorization to build or run it, not an automatic acceptance commitment,
and not proof of shortcut absence. A broader positional control may still exploit
the eventual finite recipe. Neither passing this criterion nor CPU source/data
qualification establishes depth=N, the optimizer's original matched template,
32 statistically independent procedures, model-only acquisition, retention or
original H01/LA-I1-002 closure. The separate population/contrast gate must preserve
those distinctions. No new native/training, tokenizer or owner-exception authority
follows; optimizer 837bd/37f, consumed token study 9a72 and all quarantines remain
unchanged.
