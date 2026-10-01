# Reviewed successor: CPU verified, native behavior unmeasured

The knowledge-gap follow-up is implemented on
`codex/triskill-geometric-gapfix-20261001`, worktree
`C:/Users/clclc/Documents/Codex/2026-10-01/task-3/ChatBubble-triskill-gapfix`.
Implementation commit: `ef1fcbef18d92195375be0fdbb3ff1f05f786e9e`.
Parent: `09a0514e1cdf8e7b1dc7960fc4a8aa2aea94cbbe`.
Actual tripleskill source: `codex/triskill-stage-a`,
`01f47694988a85bdedab7831d1652f0b9bf07e9f`, verified through git metadata in
`LUNCY579/ChatBubble`. No replacement research line was guessed.

**67 CPU tests passed in 1.798 seconds**, with zero failures or errors.
Production source-binding validation passed. The original renderer reproduced
384 DEV prompts byte-exactly and rejected its missing-think-block negative
control. Tokenizer-only checks found a maximum 239 input tokens including the
prompt control, and 21 reference output tokens, within 256/24 slots. No torch or
transformers module was imported in these verification runs.

Actual model loads: **0**. Forward calls: **0**. Native generations: **0**.
Training examples used for construction: **0**. Paid grader calls: **0**.
Checkpoint-weight reads in this follow-up: **0**. CPU fixtures and generated
gold labels exercised scoring only; they did not become training data or select
the geometric direction. All behavioral benchmark families remain unmeasured.
These results establish CPU mechanics, not native mechanics or a third skill.

## What changed

- `test_geometry.py` now checks compensated coordinate invariance separately
  from fixed-downstream behavior change, commuting-projector nulls, parent-swap
  inversion, parent-basis invariance, inverse composition and a counterexample
  to task-level behavioral inversion. The construction and `geometry.py` are
  unchanged.
- `worker.py` adds REVERSE using R.T on TH, while retaining 64 total screen
  generations. All nine arms cover the three dependent endpoints and separate
  known-state control. IDENTITY/REVERSE parent retention is unmeasured. Existing
  fresh parent qualification and fail-closed containment are retained.
- `benchmark.py` reports distinct strict/semantic paired 2x2 tables, gains,
  source-correct losses, format repair, net-rate uncertainty and coverage. It
  adds 24 DEVELOPMENT-only format variants of six semantic cases. Screen
  membership and golds remain unchanged; no native format panel was run.
- `dependency_binding.py` pins worker, geometry, benchmark, binding, the exact
  sibling renderer and its exact DEV fixture. Native functions execute from
  verified source snapshots, including a fresh worker. Missing/tampered files,
  path overrides, links/reparse points and unverified cached dependencies fail
  closed. The fixture snapshot does not read any other bank. Source binding
  does not authenticate the interpreter or replace containment; the supervisor
  must pin bootstrap bytes before execution.
- `future_power.py` supplies a prospective stdlib-only paired-test calculation.
  [FUTURE_PROTOCOL.md](FUTURE_PROTOCOL.md) explicitly requires separate
  authorization, fresh membership, adequate per-family retention sampling and
  a concrete compute budget before execution.

Independent reviews covered algebra/allocation/power, scoring and binding.
The scoring reviewer checked 2,601 paired-count tables and 84 confidence
intervals against independent references. The math reviewer checked exact
rational rejection thresholds for discordance counts 0-512 and independently
summed joint multinomial power. Binding review found a stale live-worker gap;
the verified-worker snapshot repair passed 30 focused worker/binding tests and
rereview. The combined 67-test suite then passed. Minor documentation/lease
precision issues were corrected before freezing.

## Interpretation and uncertainty

T is familiar state transition; H is boundary/abstention. C is narrowly
transition-conditioned answering/abstention: determine whether explicit rules
erase or preserve uncertainty about the final state. This requires a dependent
decision in the task specification, but does not prove causal use of both
learned skills. The current endpoints admit a reset-presence shortcut. Known
arithmetic and independent conjunction gains do not qualify transfer.

A compensated change of basis preserves the implemented map. This intervention
keeps downstream computation fixed, so it can change outputs even though it
preserves hidden-state norm. The parent-subspace commutator is an algebraic
interaction, not a semantic objective. It cannot identify useful C direction
without further evidence. Useful steering, latent-ability recombination, a
natural causal mechanism and genuinely new generalization remain separate
claims. Dormant-pathway steering is possible; no mechanism claim is made.
The four primary-source checks are in [SOURCES_VERIFIED.md](SOURCES_VERIFIED.md).

The three-wins/zero-losses example is a **synthetic scorer canary**, not observed
model behavior. Its exact two-sided p is .25, with conservative descriptive
paired net-rate bounds approximately [-.536, 1]. Two parent examples cannot
establish retention. Generic scoring intervals assume iid cases; format variants
must be analysed as repeated measures within source worlds.

At proposed N=512, assumed gain/loss probabilities .15/.05 give .992281
per-comparator paired rejection power and an all-eight lower bound .938251.
That lower bound does not assume independence among comparator tests. It covers
paired rejection only, not the added effect-interval and retention gates. Weaker
assumptions have much lower power. No confirmation or automatic expansion was
launched. The original [twelve-family matrix](BENCHMARK_MATRIX.md) remains
historical; [FOLLOWUP_PREREG.md](FOLLOWUP_PREREG.md) supplies this successor's
control/semantic-scoring amendment. Roleplaying/language calibration and
long-context memory qualification remain unavailable.

## Freeze, preserved scope and next decision

[FREEZE_GAPFIX.json](FREEZE_GAPFIX.json) SHA-256:
`2b65dd960a0e48f02f2c3c078b1cc98875b309442b9a7c02c8af510d1892cb7a`.
It binds 30 local artifacts plus six dependency roles. The original renderer
SHA-256 is `30b1791e1c30afb9f1cb40b0e57b3c8c62d6c39c23a8e3f20af2e17cfc1c937b`;
the exact DEV fixture SHA-256 is
`46d95c672b7805872ebdc79dd45a688fbd59e4038eeea4ea6216b5ad82c4435f`.
The unchanged geometry NPZ SHA-256 is
`52563d5ba4b230f01053cef0c2330645861e20de6ce66be71336ced235c78f7c`.
The freeze excludes its own bytes and later validation/report receipts to avoid
circular pins; the evidence commit records those receipts. The successor
rejects the legacy incomplete freeze; the historical branch and freeze remain
unchanged and interpretable under their original implementation.

The read-only resource snapshot reports 11.36 GiB host RAM free and 6.67 GiB GPU
free, but GPU attribution is incomplete and foreign compute-holder records are
present. These records include desktop graphics and do not identify model loads.
Exclusive model ownership is unproved. No jobs were stopped. The qualified
owner-verifier handoff is absent; an actual entry-negative test stopped at
`STOP_CONTAINMENT_INTEGRATION_UNQUALIFIED`, before model imports or lease/output
creation. Current authorization is CPU-only independently of resource capacity.
Reference UTC and host receipt time are both retained because their wall clocks
differ; no OS clock changes were made. Monotonic budget logic is unchanged.

A future initial screen proposal remains <=600 seconds, one model load, 64
screen generations, <=1,600 forwards and <=20,000 token positions. The planned
allocation needs 1,552 forwards and at most 18,892 positions. It requires fresh
resources/ownership and the separately qualified containment/source handoff;
the fresh parent gate may stop after only 12 screen generations. Nothing in
this report grants native permission. A positive tiny screen would justify
considering a separately powered study, not an intelligence claim.

[SCOPE_GAPFIX.json](SCOPE_GAPFIX.json) records preserved concurrent worktrees.
The original pilot stays clean at 09a0514e; actual tripleskill's 50 untracked
items and main's four existing dirty/untracked entries remain present. All
historical geometry, dataset, registration and CPU receipts hash unchanged.
The five unpublished fixes, exact-prefix/Windows owner lane, protected sets,
spent banks and Stage B were not modified. No push, merge, deployment, checkpoint
overwrite, global-setting change, paid compute or unrelated app work occurred.

Evidence: [CPU test log](CPU_GAPFIX_TESTS.log),
[production validation](CPU_GAPFIX_VALIDATION.json),
[resource snapshot](RESOURCE_GAPFIX_PREFLIGHT.json),
[prospective power table](POWER_PROTOCOL.json), and
[development format fixtures](format_development.json).
