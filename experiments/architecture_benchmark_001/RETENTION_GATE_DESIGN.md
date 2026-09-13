# RAB-001 training retention gate: design checkpoint v1

Status: **proposed design, before implementation**. This document fixes proposed numerical
rules for review; it is not an activated protocol amendment or a qualification receipt.
No candidate was evaluated in this session. Historical arm scores were already known, so
applying this policy to them would be retrospective validation, never preregistration.
G3-G5 remain closed.

## Verified starting state and unresolved prerequisites

- Branch: `research/ced-rab001-hotfix001-qualification-v1`, starting HEAD `e7285e5`.
- Implementation: `8b47b39595dcdd7ee82512de16e4ea30abee4642063de7f5119be76ed6d6621f`,
  31 manifest files. Exact repository function bodies were evaluated with their stdlib
  dependencies via AST extraction to avoid importing torch under host-memory pressure.
- All 14 inspected live/archive receipt JSON hashes match. Declared present artifacts match
  their hashes. The live PASS receipt DAGs under `outputs/` validate against their recorded
  contexts. Both CED receipts retain their three strict-parity failures; D40 quality retains
  its KL, termination and distinct-output failures.
- The copies under `evidence/` omit model binaries and preserve parent paths from the old
  directory layout. They are records of qualification, not standalone reusable DAGs. Use the
  original live paths under `outputs/` for authorization.
- **Fresh suite pending, explicitly accepted by the user.** `hostbudget.require(4.0, ...)`
  refused before importing torch or pytest: 0.06 GiB free, 99% memory load. Later readings
  were 0.40-1.04 GiB. The documented 191 passed / 0 skipped is not a new result. Installed
  package metadata matches the pinned versions; native/runtime execution remains unverified
  in this session. No model job ran.
- The legacy scorer's existing 11 cases pass when explicitly called. Four additional
  cheap counterexamples demonstrate that this is insufficient: AM/PM, wrong units and
  negation can false-pass, while equivalent duration units can false-fail.

Machine-readable checks: `evidence/retention-design-v1/BASELINE_AUDIT.json` and
`evidence/retention-design-v1/RECONCILIATION.json`. Neither is a model-qualification receipt.

## What disagrees with the handoff

1. **The registered primary endpoint is already capability-shaped.**
   `protocol.json:G3_G5.primary_endpoint` is
   `compositional_reasoning_exact_or_executable_verification`. The recovery-plan example
   leaves its endpoint null. No binding pair of exclusively CE/KL endpoints was found.
   The real gap is that no executable training retention criterion exists, while G3 budgets,
   margins, manifests and execution remain unset/disabled. The measurements support refusing
   CE or KL as a sole certificate; they do not prove every likelihood threshold mathematically
   incapable of separating two unequal likelihood scores.
2. **G1_QUALITY detects but does not interlock optimization.** In `gates.py`, it is a leaf;
   neither G2 nor G2_CACHED depends on it. Preserve the historical engineering stages, but
   require quality and retention parents in the new supported training path.
3. **The 500k JSONs are not exact matched final-state evidence.** They are committed and
   byte-identical to the live copies. C28 reports 468 optimizer updates but its last evaluation
   records update 467. `long_budget.py:395-415` evaluates the milestone before the final
   seven-row partial-window update and suppresses reevaluation using token count alone.
   Its scores describe pre-final-update weights; the effect of the final update is unmeasured.
4. **Resume omits accumulated gradients and the microbatch counter.**
   `long_budget.py:360-371` saves cursor/model/optimizer but resumes with zero gradients and
   `micro=0`. The first recorded update-boundary mismatch is update 117: C28 125,834 tokens,
   control 126,070, with different learning rates. Control records 3,743 consumed rows but
   only 467 full eight-row updates, consistent with seven discarded microbatches. Identical
   row counters and an identical schedule function do not establish identical applied work.
   C28's published JSON also lacks a TRAIN fingerprint; the resume guard accepts its absence.
5. **Full paired answers are missing.** Only eight truncated final samples are retained;
   checkpoints omit all sample vectors. The recorded 2/60, 26/60 and source 20/60 cannot be
   independently rescored or decomposed into lost and gained source successes. The recorded
   CE-gap arithmetic reproduces (98.4157% closure, 0.115811 overall, 0.494772 reply, 4.27223x
   concentration), with the checkpoint/matching caveats above. These remain exploratory
   observations, not certification or proof that the exact matched endpoint was reached.
6. **Costs and budget wording are wrong in the proposed continuation edit.** Recorded 500k
   times are 5.14 h for C28 and 6.50 h for control; 10.3 h is C28's extrapolated 1M cost.
   The timer excludes initialization/cache creation and lost work since the last saved state.
   Also, 1,000,198 pair tokens exceed a 1,000,000 pair ceiling by 198; the historical claim
   that this fits both cap interpretations is false.
7. **Scoring and label review need work before certification.**
   `scoring.py:30-39` accepts `10:25 PM` for `10:25 AM`, `5 hours` for `5 minutes`, and `not 5`
   for `5`; it rejects `60 minutes` for `1 hour`. The source-control sample `54 sensor tags`
   matches the question, whose stored reference incorrectly says `54 spare fuses`. Merely
   tightening unit matching would create another wrong verdict. Self-tests are called by
   consumers or under `__main__`, not automatically at import as the docs claim.
8. **Some scientific/process conclusions are overstated or stale.** The 0.5-nat cost is an
   initialization observation for tested sharing layouts, not a theorem about all topologies.
   Later small-displacement training and latent-KV results qualify the original blanket
   conclusions. Pooling the same 60 task families across seeds does not create 180 independent
   task families. Commit `c8b9e28` records a completed 28-agent internal review; independent
   external review is still not established.
9. **Holdout has a specific scope.** Current large TRAIN has 10,000 rows/families, rather
   than the original 800. TRAIN/DEV/TASK are mutually family/ID-disjoint, but TRAIN overlaps
   calibration on 12 families and one source row; DEV and TASK each overlap two calibration
   families. The existing TASK has been repeatedly inspected. It can be an open monitor,
   not an untouched confirmatory panel.

Do not repair or relabel the historical JSONs. Append this reconciliation to their reading
context. No 500k run should be repeated just to produce this gate design.

## Proposed gate and fixed rules

The gate has two distinct outputs: an engineering permission for the next bounded training
block, and an optional terminal statistical certificate. Neither is broad model capability
certification. All thresholds below are policy choices, not estimates optimized against
historical arm outcomes. A changed policy requires a new version and prospective registration.

| Item | Proposed v1 value |
|---|---|
| Probe classes | `percentage_reasoning`, `ratio_proportion`, `time_duration` only |
| Open monitor | All 150 existing TASK rows, 50 per class; reviewed labels in a separately versioned manifest |
| Monitor reference | Union of frozen-source-correct and candidate-initialization-correct items |
| Allowed losses | **0 correct-to-incorrect transitions in each class**, relative to that immutable reference |
| Source informativeness for the finite monitor | At least 1 correctly answered, terminated source item in every class; otherwise INSUFFICIENT |
| Generation floor | At least **80% terminated in each class** at every monitored boundary |
| Check cadence | Step 0, then the first completed optimizer update crossing each **5,000 applied-target-token** boundary, plus the terminal update |
| Candidate selection for confirmation | One prospectively frozen candidate recipe, seeds **17, 29, 43**, one terminal look per seed |
| Fresh confirmation panel | **300 families per class**, one hash-selected row per family; no candidate-informed replacement or top-up |
| Confirmation retention floor | At least **90% of frozen-source successes retained in each class and every seed** |
| Confidence rule | One-sided exact binomial upper bound on conditional forgetting **<= 0.10**, alpha **0.05/9** per seed/class |
| Minimum source successes in confirmation | **50 per class**; below this, INSUFFICIENT even if no candidate losses are observed |
| CE, reply CE, KL, top-1 | Diagnostics only during training; no offsetting capability failure |
| Sealed TEST | No access |

An item is correct only if generation terminates and the frozen typed evaluator accepts its
answer. Missing rows, duplicates, nonfinite computation, unknown answer types, changed
identities or incomplete checks never pass. A gain on one question cannot compensate for
forgetting another. Report retained/lost/gained/neither counts and ordinary paired accuracy
changes as well, without making those gains an escape from the retention rule.

At step zero, source losses block the retention lane. A broken initialization is a recovery
problem; passing a later recovery demonstration cannot retroactively claim uninterrupted
retention. The present CED strict-parity failures remain independent blockers.

The monitor says exactly what happened on its known finite panel. Repeated monitor checks do
not create repeated population confidence claims. A passing check permits the next bounded
block only; it cannot promise that capability never dipped between observations. A failure
halts subsequent optimizer updates and preserves the failed checkpoint and all evidence.

### Confirmation scope and power

For each class, let m be the number of source-correct families in the fixed fresh panel,
and l the number of those families the terminal candidate fails. Infer the conditional
forgetting probability using an exact binomial upper confidence bound. With zero losses,
the upper bound is `1 - (0.05/9) ** (1/m)`; at m=50 this is 0.098648. This certifies a narrow
90% retention claim under the registered family-sampling assumptions, not equality to source
or retention of untested skills. Require every seed/class result to pass; never pool repeated
families across seeds as independent observations.

The alpha allocation supplies simultaneous coverage for the nine reported bounds. It does
not require independent seeds. Binomial inference does require an adequately justified family
sampling model; distinct family IDs alone do not establish semantic independence. If the
inventory/exposure review cannot justify it, issue finite-panel measurements only.

The 300-family panel is a **bounded feasibility design, not a guaranteed-power design**.
At m=90, at most two losses pass; a hypothetical true forgetting rate of 2% passes this one
component with probability 0.731175. At m=200, nine losses pass and that probability is
0.992521. These are prospective binomial calculations, not candidate measurements. Low
power is a reason to register a larger panel before a campaign, never to enlarge it after
seeing a failure. Insufficient evidence is not evidence of inferiority, and cannot open a gate.

Method references: [NIST exact binomial intervals](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm)
and [NIST simultaneous coverage via Bonferroni](https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm).
The policy thresholds and family design above are this proposal, not recommendations from NIST.

The confirmation panel must be family-disjoint from all TRAIN, DEV, calibration, pilot,
monitor and previously scored/tuned candidate panels. Record exposure, including uncertainty
from historical baseline scans lacking IDs. No claim that the corpus was globally unread:
earlier builders loaded it. Sample by a fixed seeded hash ordering after metadata-only
exclusions; audit label correctness before candidate evaluation; freeze every selected row and
every exclusion. If 300 suitable families per class cannot be supplied, do not silently use
the old panel or reduce the count. Human review/attestation remains pending until performed.

Fresh source baselines use the new typed scorer; historical 65%/50%/30% cannot be carried over
without measurement. The fixed sample is chosen independently of which questions source or
candidate answers correctly. The source-correct subset is the predeclared conditional estimand,
not a license to replace source misses with easier questions.

The nine-bound allocation covers one registered recipe. Additional candidates, search trials,
or certificate claims require a prospectively frozen multiplicity/exposure plan. The matched
ordinary continuation is a comparison control, not a substitute for the immutable source.
An operational campaign must freeze its recipe, token cap, search budget, endpoint and full
cost ledger separately; this design does not fill the existing null G3 fields by implication.

## Implementation shape to build after this checkpoint

1. Put the authoritative typed scorer and pure retention-check calculation under the hashed
   package. Numeric values use exact decimal/rational representations; clocks preserve AM/PM;
   durations normalize reviewed units; typed counts reject contradictory units and negation.
   Save typed parse results alongside full raw text. Establish a deliberately bounded answer
   grammar and reviewed aliases before source measurement. Do not duplicate the legacy regex
   or silently reinterpret historical scores.
2. Add explicit baseline, monitor and confirmation receipt types with mandatory named checks
   and complete prediction artifacts. Bind source weights, initialization and evaluated
   checkpoint hashes, protocol/policy/scorer implementation, tokenizer/template/EOS settings,
   CPU FP32 runtime, data/family/exposure manifests, optimizer recipe, seed and counters.
   Receipts remain content-bound evidence, not signed external attestations. Recompute checks
   from prediction artifacts rather than trusting supplied booleans.
3. Add a training guard used by supported entry points before optimizer work and before the
   next bounded block. Require current mechanical and initialization-quality receipts plus
   step-zero retention. Keep G2's historical 25-step engineering scope intact; do not relabel
   it as retained capability. Keep G3-G5 disabled until their other prerequisites are met.
   Arbitrary exploratory Python outside the supported runner cannot be made impossible by
   this gate; label it unauthorised for certification rather than claiming universal enforcement.
4. Evaluate only at completed optimizer-update boundaries. Persist/hash the exact evaluated
   weights and optimizer step; flush the final partial window before final evaluation.
   Distinguish spent compute tokens from tokens actually applied to optimizer updates. Reject
   an exceeded campaign cap; prospective data/window handling must make the cap attainable.
5. Save resumable states only at zero-pending-gradient boundaries in v1. Defer periodic saves
   to such boundaries. Bind complete state and original references; reject missing identity
   fields and attempts to reset the reference on resume. Record crashes and discarded work;
   preserve attempt history and use create-only checkpoint/receipt directories. Read PID/process
   state before any stale-lock handling.
6. Run only one model job on CPU. Use `hostbudget.require()` at startup and `guard()` between
   update/evaluation units, including native qualification jobs. Source inference and candidate
   inference are sequential; no concurrent teacher model. Memory aborts produce incomplete
   records and cannot become success. UTF-8-safe logging must not destroy completed compute.

## Validation and landing sequence

First complete the pending baseline suite in the pinned environment when memory permits;
reconcile any disagreement before code changes. Then freeze the reviewed policy/amendment
and evaluator cases before obtaining any new candidate score.

Cheap controls must cover: punctuation versus value, decimals, AM/PM, equivalent and wrong
units, negation, malformed answers and reference-label drift; exact identity positive;
one lost source success despite new gains; a class collapse hidden by aggregate improvement;
CE improves while answers fail; no source successes; missing/duplicate/tampered predictions;
wrong checkpoint/seed/policy; and confidence-bound equality/insufficient-sample edges.

Use a tiny deterministic training fixture to compare uninterrupted versus resumed updates
at every accumulation position. Verify optimizer state, data cursor, applied/spent counters
and final prediction hashes, including the seven-row partial-window case that the historical
harness misses. Check step-zero equality across seeds and that each seed actually changes
TRAIN order. A test must show the next optimizer step cannot execute after a failed monitor.

After the package implementation is frozen: run the full suite with zero native skips,
matrix and status; regenerate the necessary qualification ladder and controls in a new root;
verify unchanged measurements to the element. Preserve all failures. Verify `ced_last1`
and local group 1 as bitwise identities using cheap source controls under the one-job rule.
Only then measure the newly scored source and eligible gate controls. Historical negative
arms can demonstrate detection retrospectively, but cannot validate prospective calibration.

The two read-only subagent reviews in this session completed and identified concrete issues.
They are internal adversarial checks, not independent external review. The design has not
yet been implemented, requalified, or tested on a fresh confirmatory panel.
