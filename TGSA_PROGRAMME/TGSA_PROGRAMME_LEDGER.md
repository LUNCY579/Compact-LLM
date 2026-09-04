# TGSA programme ledger

Single index of what the Trajectory-Guided Sequential Attention programme has
established, where each record lives, and what blocks the next step.

Machine-readable source: [`tgsa_programme_ledger.json`](tgsa_programme_ledger.json)

Generated: 2026-09-04.

---

## 0. What this branch is, and is not

**Is:** a read-only aggregation of *records* — results, decisions, provenance
pointers, and verbatim copies of result documents from branches that are not in
the main chain.

**Is not:** a merge. No implementation file from any stage is copied here. No
evidence is pooled. Nothing is relabelled under a generic "TGSA+MTP" identity.

**This is an orphan branch.** It shares no history with any implementation
branch and its tree contains nothing but `TGSA_PROGRAMME/` and a `.gitignore`.
That is deliberate.
An earlier version of this branch descended from the Phase 2A branch and so
carried a full copy of the lab, which made it look like somewhere you could
work — and someone reasonably concluded it was the most advanced branch to
experiment on. It is not. It has no code, and it must never acquire any.

**To run TGSA experiments, use `research/tgsa-v5-1-unified-programme`.**
Nothing here is a substitute for it, and this branch is deliberately missing the
Phase 2B and retrofit code entirely — those live only on their own branches and
cannot be combined.

This is deliberate, and it is not a stylistic choice.
`TGSA_MTP_CANONICAL_PATH.md` (read with `git show research/tgsa-v5-1-unified-programme:Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/TGSA_MTP_CANONICAL_PATH.md`)
governs the programme and states that the two MTP histories must not be merged
into one implementation namespace, that outputs from the two Phase 2 stages
cannot be pooled or described as replications, and that both branch histories
must be preserved. Phase 2A conditions a future horizon on the previous
future-token embedding and the trajectory coordinate; Phase 2B predicts offsets
`2..K` directly from the same causal hidden state and never feeds one predicted
token to the next. The causal graphs differ, so a merged namespace would make
every downstream claim ambiguous about which intervention produced it.

A ledger gets you one place to look without paying that cost.

---

## 1. Programme status in one line

**No TGSA mechanism hypothesis has been resolved.** H1a, H1b and H1c are all
open. There is no confirmed TGSA result, positive or negative.

The blocking gate is a **new competence preregistration** with learnable,
non-saturated primary families. The v4 band is rejected and permits no further
measurement, so the next step is not more Phase 2 implementation — that code is
already written and gated. It is a regime that can actually carry a contrast.

**v5.1 is that preregistration, and it is now frozen** on
`research/tgsa-v5-1-unified-programme`. It has *not* been executed. Stage 0
apparatus qualification is prepared and costed, but it cannot start until the
human operator supplies `--attest-no-prior-outcome-inspection`. That flag
records a *human* attestation that no nondevelopment outcome was inspected
before the prospective lock; the contract must be frozen **before** any
nondevelopment run, because running first and locking after destroys the
prospective registration irrecoverably. Every run to date is
`development_only=true`.

---

## 2. Which branch do I use?

**Work on `research/tgsa-v5-1-unified-programme`.** It is the canonical branch
as of 2026-09-03 and contains the entire Phase 1 chain plus Phase 2A (C1–C6)
and the whole v5 chain (C9–C10). Everything else is either archived, a separate
stage that must stay separate, or records-only.

| Use | Branch |
|---|---|
| **Active work** | `research/tgsa-v5-1-unified-programme` |
| Separate active stage — never merge into Phase 2A | `research/tgsa-mtp-rineya-complementarity-v1` (Phase 2B) |
| Parked, third in sequence | `research/tgsa-rineya-pretrained-retrofit` |
| Records only, no code | `research/tgsa-programme-ledger` |
| Frozen evidence — tag only, no branch | `archive/tgsa-phase1-v4-regime-rejected` |
| **Archived — tag only, no branch** | `archive/tgsa-phase1-launch-audit`, `archive/tgsa-phase1-v2-launch-ready`, `archive/tgsa-phase1-v3-inconclusive`, `archive/tgsa-phase2a-superseded-2026-08-31`, `archive/tgsa-v5-competence-execution-superseded-2026-09-03` |
| **Tagged, retirement proposed — refs still on origin** | `research/tgsa-phase2a-ci-and-competence`, `research/tgsa-v5-competence-and-mechanism` |

GitHub currently shows six TGSA branches. Four remain once the two proposed
retirements are approved; the other stages are reachable only through annotated
tags.

Retiring a ref to a pushed tag is preservation, not deletion — the commit, tree
and message are unchanged and reachable. Each tag was verified to point at the
exact commit its branch pointed at and confirmed contained in the active branch
before any ref was removed. For the two refs proposed on 2026-09-03, containment
was verified twice over: `merge-base` returns the branch tip itself, and
`git rev-list --count <tip> ^fd6e20f` returns 0.

For the frozen v4 boundary this is also an enforcement mechanism. The contract
says it must never receive Phase-2 code merges; a branch that does not exist on
origin cannot be pushed to. A read-only local checkout remains at
`Chat-Bubble-v4` for convenience — it is not a work surface. The same now
applies to `Chat-Bubble-phase2a-next`, which still has the superseded Phase 2A
branch checked out.

**Tags are safe, but they are not free.** `tgsa/v5_1_programme.py` resolves
lineage with `cat-file -e <sha>` and `merge-base`, never by branch name, so a
retired branch breaks nothing — provided the object is present locally. A v5.1
CI run already failed on exactly this after v4 was retired. Always fetch with
`--tags` before running any verifier.

```bash
git show archive/tgsa-phase1-v3-inconclusive:Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/PHASE1_V3_RESULT.md
```

```bash
git branch research/tgsa-phase1-identifiability-v3-powered-confirmatory archive/tgsa-phase1-v3-inconclusive
```

## 3. Lineage

```text
bef44ce  C1  phase1-model-identifiability-launch-audit    ARCHIVED  archive/tgsa-phase1-launch-audit
   |
e0063f0  C2  phase1-identifiability-v2-launch-ready       ARCHIVED  archive/tgsa-phase1-v2-launch-ready
   |
04d3c02  (mid-v3; C8 forks here)
   |\
   | `- ef5194b  C8  rineya-pretrained-retrofit           PARKED    parked/tgsa-rineya-retrofit-2026-08-31 (covers eea4f64 only)
   |
6c97ac1  C3  phase1-v3-powered-confirmatory              ARCHIVED  archive/tgsa-phase1-v3-inconclusive
   |
8b5737e  C4  phase1-v4-competence-first                  FROZEN    archive/tgsa-phase1-v4-regime-rejected
   |\
   | `- 49aa961  C7  mtp-rineya-complementarity-v1        ACTIVE (Phase 2B)
   |
d71b558  C5  phase2-mtp-complementarity                  ARCHIVED  archive/tgsa-phase2a-superseded-2026-08-31
   |
de3bd63  C6  phase2a-ci-and-competence                   SUPERSEDED  archive/tgsa-phase2a-ci-and-competence-superseded-2026-09-03
         C9  v5-competence-execution                     SUPERSEDED  archive/tgsa-v5-competence-execution-superseded-2026-09-03
             (C6 and C9 are the same commit -- two branch names, one object)
   |
b091490  C10 v5-competence-and-mechanism                 SUPERSEDED  archive/tgsa-v5-competence-and-mechanism-superseded-2026-09-03
   |
fd6e20f  C11 v5-1-unified-programme                      <-- ACTIVE, work here
```

The v5 chain is a strict linear extension of the spine, not a set of divergent
forks: `de3bd63` → `b091490` (+5 commits) → `fd6e20f` (+1). **C1 through C6,
C9 and C10 are all contained in C11** — verified, not assumed. Only C7
(7 commits) and C8 (15 commits) hold work outside it, and both are deliberately
separate stages.

`fd6e20f` is one plain commit on `b091490`, and it is what origin now holds,
with CI green (run `33774170876`). It replaced a **four**-commit bootstrap
(`26c563b`, `b6a42a6`, `6c94adb`, `b7b2cbe`) whose entire diff was ten binary
parts, a materialization workflow and a documentation whitespace repair, so
discarding it dropped no real content. origin's tip immediately before the plain
materialization was `b7b2cbe`, not `6c94adb`.

---

## 4. Stage table

| ID | Branch | Role | MTP semantics | Status | In C11? |
|---|---|---|---|---|---|
| C1 | `…phase1-model-identifiability-launch-audit` | Phase 1 launch audit | — | archived, ref deleted | yes |
| C2 | `…phase1-identifiability-v2-launch-ready` | Phase 1 v2 | — | archived, ref deleted | yes |
| C3 | `…phase1-identifiability-v3-powered-confirmatory` | Phase 1 v3 confirmatory | — | **INCONCLUSIVE**, ref deleted | yes |
| C4 | `…phase1-v4-competence-first` | competence / regime boundary | — | **V4_REGIME_FAMILY_REJECTED**, frozen, tag only | yes |
| C5 | `…phase2-mtp-complementarity` | Phase 2A predecessor | `sequential_teacher_forced_recursive_v1` | archived 2026-08-31, ref deleted | yes |
| C6 | `…phase2a-ci-and-competence` | Phase 2A; v5 execution base | `sequential_teacher_forced_recursive_v1` | **SUPERSEDED** 2026-09-03, retirement proposed | yes |
| C7 | `…mtp-rineya-complementarity-v1` | Phase 2B system factorial | `parallel_independent_offsets_v1` | active, no result yet | no |
| C8 | `…rineya-pretrained-retrofit` | real Qwen3/Rineya retrofit | — | **PARKED**, sequenced third | no |
| C9 | `…v5-competence-execution` | v5 competence execution | — | **SUPERSEDED** 2026-09-03, ref already gone | yes |
| C10 | `…v5-competence-and-mechanism` | v5 competence + mechanism; v5.1 base | — | **SUPERSEDED** 2026-09-03, retirement proposed | yes |
| **C11** | **`…v5-1-unified-programme`** | **Phase 1 v5.1 unified programme, canonical** | none at this stage | **ACTIVE**, frozen, no result yet | — |

C6 and C9 are the same commit (`de3bd63`) under two branch names. Worktree
paths, exact commits and per-stage results are in the JSON.

### Test counts across the chain

| stage | commit | collected | README claimed |
|---|---|---:|---:|
| C5 | `d71b558` | 350 | 349 |
| C6 / C9 | `de3bd63` | 396 | 396 |
| C10 | `b091490` | 436 | 396 |
| C11 | `fd6e20f` | 444 | 444 |

All four measured, not asserted. C11's 444 = 436 at C10, plus 4
programme-contract regressions, plus 4 auto-parametrized
`test_seed_determinism` cases — one per newly registered source file. The
`440` that v5.1's own draft carried counted only the first four, which is the
count drift the materialization corrected.

### Consolidation of 2026-08-31

Two branches had been claiming the Phase 2A role. C6 had grown as an undeclared
continuation of C5 — same stage, same semantics, same lineage — but held 396
tests against 350 and was the sole holder of the v5 preregistration, the
within-study qualification gate and per-family null scoring. It was *not* the
sole holder of the programme contract: C5 carried all four
`phase2_program_contract` files byte-identically, inherited from `1c14521`
before the C5/C6 merge-base `81755bf`. Meanwhile C5's README named itself Phase 2A and
`TGSA_MTP_CANONICAL_PATH.md` did not mention C6 at all.

Resolved toward C6: `ce4f134` merged C5 in so nothing was stranded, and
`188ed21` moved the designation across all six references (canonical path stages
table and branch lifecycle, README chain, runbook lineage, programme contract
stage field, CI trigger). A within-stage rename — no evidence pooled, the
Phase 2A/2B separation untouched, no registered estimand changed. Two branches
claiming one stage was itself an evidence-identity risk.

### Consolidation of 2026-09-03

Three branches were claiming the v5 lineage, and the canonical tip was not
reviewable content. `research/tgsa-v5-competence-execution` was identical to
C6; `research/tgsa-v5-competence-and-mechanism` was its strict descendant; and
`research/tgsa-v5-1-unified-programme` held a *bootstrap* rather than the
programme — ten binary parts under `.v51-bootstrap/` plus a `contents:write`
workflow that reassembled a tarball, verified it, and force-pushed a
materialized commit.

That workflow failed **four** times — runs `33699105352`, `33699971132`,
`33700347537`, `33772927517` — and no failure was scientific.

Run `33700347537`, the **third**, is the well-known one: all 444 tests, contract
verification, schema validation and the probe-plus-qualification smoke chain
passed, and it printed *"bootstrap validation passed for the exact final v5.1
tree"* — then it died on `git diff --check`, because
`V5_1_UNIFIED_PROGRAMME.md` lines 3 and 4 end in two trailing spaces (Markdown
hard breaks).

The **fourth and last**, run `33772927517` on `b7b2cbe`, repaired exactly those
two lines, cleared `git diff --check`, again passed all 444 tests and printed the
same validation line — and then had its push rejected outright:

```text
! [remote rejected] HEAD -> research/tgsa-v5-1-unified-programme
  (refusing to allow a GitHub App to create or update workflow
   `.github/workflows/tgsa-v5.yml` without `workflows` permission)
```

That permission boundary, not the whitespace guard, is what finally ended the
bootstrap route. A `contents: write` token cannot rewrite a workflow file, and
the package contains one.

It was replaced rather than repaired. The opaque blob made the content
unreviewable in the diff and had already accreted an inline CI string-patch
hack. The package was reassembled from both the git blobs and the working tree,
both matching sha256
`b03f5032a4ffe315653e35307bd1256017bb669b31ae814adac6fdc742d5217a`, and the 15
real files were committed plainly on `b091490` as `fd6e20f`. The 16th packaged
file, `.v51-bootstrap/materialize.sh`, was bootstrap machinery and was dropped.
Two corrections were applied fail-closed on exactly one occurrence each
(`440 tests` → `444 tests`; `records 440 tests` → `records 444 tests`), and the
two trailing-space line endings were stripped after confirming no other line in
the package carried trailing whitespace.

Verified before hand-off: 444 tests passed; the contract schema validated;
`verify_v5_1_programme_contract.py` passed with all ten checks green; the
verification report validated against its schema; the development-only probe and
qualification smoke chain both returned `status=complete` and exit `0`; and
`git diff --check b091490..HEAD` is clean.

`fd6e20f` fixes two latent defects, but **only one came from the package**:
`tgsa-v5.yml` resolved `${{ runner.temp }}` in a job-level `env:` block before
runner allocation, leaving `ARTIFACT_ROOT` and `PROGRAMME_REPORT` empty. For
that one, **reassembling a package to its exact sha256 proves integrity, not
correctness.**

The other defect was *not* in the package. The plan schema listed
`famili_roles` in its `required` array — so `family_roles` was never actually
enforced — but `evidence/phase1_v5_plan.schema.json` is not among the package's
16 entries, and the extracted tarball contains zero occurrences of the byte
string `famili`. That typo was pre-existing repository content at the audited
base `b091490`, introduced by `9602d3d`. Three scripts also needed the exec bit
`materialize.sh` applied with `chmod +x`.

Three annotated tags were created and the two surviving superseded refs were
tagged for retirement, deferred until CI is confirmed green on the materialized
tree. **Ordering matters:** `verify_programme_ledger.py` resolves every stage
commit locally, so this ledger refresh must not reach origin before `fd6e20f` —
an ordering now satisfied, since v5.1 landed first.

### Apparatus review of 2026-09-04

Three adversarial passes were run over the v5 apparatus and then over the fixes
to it. **Four defects were confirmed, three of them high severity, and no
existing verifier would have caught any of them.**

| defect | consequence |
|---|---|
| per-run floor left at v4's `0.070` | a cell could be certified `GO` with one of three seeds **below** the family null of `0.1318` |
| interrupt between `optimizer.step()` and its bookkeeping | resume re-applied that step; measured divergence `0b48dd27…` vs `a9d1dadf…` |
| `select_qualified_exposure` checked no report provenance | the mechanism-study authorization could be issued from foreign or development-only evidence |
| `CUBLAS_WORKSPACE_CONFIG` in no fingerprint | a resume could silently change CUDA arithmetic |

All four are fixed on `research/tgsa-v5-1-gate-hardening` (`1af0b5a`, five
commits on `fd6e20f`), which is on origin and **not merged**. C11 itself is
untouched, so the frozen registration is unchanged pending a merge decision.

**Two things about this are worth carrying forward.**

*Development mode hid two of the four.* `_validate_frozen_band` returns early
when `development_only` is true, so no smoke run could ever reach the missing
`--frozen-band`; and the smoke plan's null is low enough that the
below-chance-seed hole never opened. The escape hatch that makes the apparatus
testable is the same thing that concealed the defects that only matter in a real
run.

*The first review reported a clean result for work it had not examined.* It
returned `confirmed: []` with every candidate counted as refuted. Four refuters
had in fact died and returned null verdicts, and the tally counted null as
refuted; two finder dimensions had not run at all. Re-run with unadjudicated
findings tracked separately, the same review returned the four confirmed defects
above. A verification harness that cannot distinguish *refuted* from *never
checked* will report success for whatever it skipped.

---

---

## 5. The two results that exist

### Phase 1 v3 — `INCONCLUSIVE`

Study `tgsa-phase1-identifiability-v3-2026-08-29`. 3 arms × 20 seeds = 60 cells,
2,500 updates each, 60/60 completed. Evidence chain: 2,001 checks, all passed.

| contrast | estimate | 95% CI | P(>0) | negative seeds |
|---|---:|---|---:|---:|
| H1a delta − history | +0.0014 | [−0.0021, +0.0048] | 0.790 | 9/20 |
| H1b delta − cascade | +0.0004 | [−0.0025, +0.0034] | 0.619 | 8/20 |
| H1c aligned − replay-deranged | +0.0019 | [−0.0003, +0.0040] | 0.959 | 7/20 |

The decision was `inconclusive` because **no arm was competent**, not because the
contrast was small. All four sensitivity estimators agree on H1a, so the
conclusion is not estimator-dependent.

Evidence: `…Lab_v1/evidence/v3_result/`.

### Phase 1 v4 — `V4_REGIME_FAMILY_REJECTED`

Both registered competence measurements returned `NO_GO`; the escalation ladder
answered `NO FURTHER MEASUREMENT PERMITTED`. No confirmatory matrix was
dispatched. Equal compute in both regimes: 1,280,000 sample presentations.

| regime | distinct examples | effective epochs | verdict | failed checks |
|---|---:|---:|---|---|
| `easy_50k_20k` | 49,998 | 25.6 | NO_GO | reliability, memorization, saturation |
| `easy_200k_20k` | 199,998 | 6.4 | NO_GO | reliability |

The deciding check was `no_primary_family_saturated`. Only `state_update` ever
moved (floor → saturation, +0.947 and +0.890); `direct_lookup` and
`pointer_chase` were inert and responded to no lever. The "phase transition" the
programme had tracked since the pilot was one family learning arithmetic.

Evidence: `…Lab_v1/evidence/v4_competence/`.

> Neither v3 nor v4 is evidence against TGSA. A contrast over non-learning,
> inert or saturated task families cannot identify the mechanism in either
> direction. Both are apparatus results.

### Retrofit candidate intake — no candidate is ship-positive

C8's [`CANDIDATE_LEDGER.md`](imported/retrofit/CANDIDATE_LEDGER.md) is already a
cross-programme audit: it scans every ref and tag and classifies ten mechanisms.
Six rejected (A1.1/Baya soup, adapter dose 1.0625, Pulse negative vector, 4B/8B
spectral consensus, normalized significance/count KV retention, Supa abstention
threshold copied to Baya), three deferred pending evidence (TGSA v5, TGSA+MTP
Phase 2A, PSA/SGSA), one already integrated (Supa-A1.1 FP16 + abstention).
No promotion is authorized.

---

## 6. Facts that span the branches

- Every implementation branch descends from `bef44ce` and shares the Phase-1
  chain and the PART 11 reporting standard with `OBSERVED` / `SUPPORTED` /
  `HYPOTHESIS` marking. This ledger is the one exception — an orphan with no
  shared history.
- **The v5 chain introduced no new MTP semantics.** v5 and v5.1 qualify the
  apparatus first; `mtp.enabled` is `false` in the v5 plans, on the stated
  ground that no auxiliary future-prediction objective may shape the
  trajectory before the apparatus is qualified.
- **C7 and C8 share a near-identical retrofit codebase.** Of 27 co-modified
  files, only three differ (`pyproject.toml`, `rineya_tgsa/__init__.py`,
  `tests/test_retrofit_correctness.py`). C8 is authoritative: 60 retrofit files
  against C7's 37. This makes promotion-order step 7 a small job.
- **The retrofit tree and the lab tree are near-disjoint.** The spine carries
  zero retrofit files; C8 carries a stale 84-file lab against the C11 spine's
  **175** — so it is 91 files behind, not the 63 the 2026-08-31 snapshot recorded
  against the old `de3bd63` spine — missing the v3 final result, the v4
  rejection, all Phase 2 work and the whole v5/v5.1 programme.
- **The programme contract is stage-neutral.** It defines both `phase2a` and
  `phase2b`, and the verifier routes on `study_kind`, so C6 and C7 carry copies
  that differ only in the phase2a stage's `branch` field — one line — rather than
  divergent forks. They are **not** byte-identical: `188ed21` moved the Phase 2A
  designation across six references, and that field is one of them. C11 adds a
  *second*,
  separate contract (`v5_1_programme_contract.json`) that pins six SHAs and
  keeps the two Phase 2 causal identities apart by construction.
- Checkpoints are excluded programme-wide. `*.pt` / `*.pth` are gitignored and
  the repository tracks zero of them.

---

## 7. Open items

1. **No competence-qualified regime exists.** This is the single blocking gate
   for the whole programme. No Phase 2 confirmatory run is authorized until a new
   preregistration produces a `GO` on learnable, non-saturated primary families.
2. **Stage 0 apparatus qualification is prepared and costed but not run.** It
   cannot start until the human operator supplies
   `--attest-no-prior-outcome-inspection`. Running first and locking after would
   destroy the prospective registration irrecoverably.
3. **`research/tgsa-v5-1-gate-hardening` (`1af0b5a`) awaits a merge decision
   into C11.** It fixes the four apparatus defects above and registers one new
   gate in `PHASE1_V5_PREREGISTRATION.md` §7. **Stage 0 should not run before it
   lands:** on `fd6e20f` the qualification gate can certify a cell `GO` with a
   seed below the family null, and an interrupt can silently diverge a resumed
   cell.
4. **Two remote refs are tagged and proposed for retirement but still exist on
   origin:** `research/tgsa-phase2a-ci-and-competence` and
   `research/tgsa-v5-competence-and-mechanism`. Deferred by decision until CI is
   confirmed green on the materialized tree.
5. **Promotion-order step 7 is not started** — the port manifest reconciling
   C7's retrofit snapshot against C8's authoritative copy. Small job: the two
   differ in three files.
6. **Two C8 documents name branches that are going away.** Its
   `candidate_ledger.json` names `research/tgsa-phase2a-ci-and-competence` five
   times, three of them `origin/<branch>:<path>` evidence pointers that stop
   resolving if that ref is retired; `PRODUCTION_RETROFIT_AUDIT.md`'s ref-scan
   table still names both the *old* Phase 2A branch and the retired v4 branch;
   and `PHASE2A_PREREQUISITE_HOLD.md` — repointed on 2026-08-31 — names the
   branch now being retired. `CANDIDATE_LEDGER.md` itself cites commits
   (`d16ea1c`), not branches. C8 is parked, so all of this is stale-on-purpose;
   refresh as part of lifting the hold, not before.

Closed on 2026-09-03: the three-way v5 branch ambiguity; the unreviewable
bootstrap tip; and the C6 and C8 tip drift this ledger had carried since
2026-08-31.

Closed on 2026-08-31: the C5/C6 designation conflict; C5's divergence from the
Phase 2A lineage; the unpushed backlog; and the ledger's own ambiguity, which is
now resolved structurally by making this an orphan branch.

Also resolved, and worth recording because it was a live risk rather than
untidiness: **PR #2 was open and `MERGEABLE` against the frozen v4 branch.**
`TGSA_MTP_CANONICAL_PATH.md` forbids exactly that in two places, so the frozen
evidence boundary — the contract's `required_phase1_commit` — sat one click from
receiving 5,183 lines of Phase-2 code. It is closed, and v4 no longer exists as
a branch on origin, so the path is closed twice over. PR #3 was closed as
superseded; its head is the active branch. No pull requests remain open.

---

## 8. Imported records

Verbatim copies, byte-for-byte, with source branch, source commit and SHA-256
recorded per file in the JSON. They are copies for convenience of reading, not
new artifacts, and they are not evidence in their own right — cite the source
branch and commit, never this path.

- `imported/phase2b/` — 3 documents from C7 at `49aa961`
- `imported/retrofit/` — 9 documents plus `candidate_ledger.json` and its
  schema from C8, at `eea4f64` except `PHASE2A_PREREQUISITE_HOLD.md`, which
  was **re-imported at `ef5194b`** on 2026-09-03 (sha256 `44008d5822d6…`)
  after C8's tip moved and changed it. The superseded `eea4f64` copy hashed
  `a186a17114…`.

**Not imported:** any source code; the `evidence/phase2_mtp_smoke_local/`
rehearsal output on C5 (regenerated scratch — CPU, one seed, three updates,
`confirmatory=false`); any checkpoint.

---

## 9. Maintaining this

The ledger is a snapshot, and a stale ledger is worse than none. Refresh it when
a stage reaches a decision, a branch tip moves in a way that changes a recorded
result, or an open item closes. Update the JSON first — it holds the hashes and
commits — then reconcile this document to it.

`verify_programme_ledger.py` resolves every stage commit in the local
repository, so never push a ledger refresh ahead of the commits it records —
a stage commit that has not reached origin will not resolve in a fresh clone,
and the provenance check fails closed.

Do not add implementation code to this branch. If that ever becomes tempting,
the thing that is actually needed is a port manifest on the target branch.
