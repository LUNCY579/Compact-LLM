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
| Review surface, same commit as canonical | `research/tgsa-v5-1-gate-hardening` (`b60d470`) |
| **Archived — tag only, no branch** | `archive/tgsa-phase1-launch-audit`, `archive/tgsa-phase1-v2-launch-ready`, `archive/tgsa-phase1-v3-inconclusive`, `archive/tgsa-phase2a-superseded-2026-08-31`, `archive/tgsa-v5-competence-execution-superseded-2026-09-03`, `archive/tgsa-phase2a-ci-and-competence-superseded-2026-09-03`, `archive/tgsa-v5-competence-and-mechanism-superseded-2026-09-03` |

GitHub now shows five TGSA branches. The two proposed retirements were carried out
on 2026-09-04, so every other stage is reachable only through annotated tags.
`research/tgsa-v5-1-gate-hardening` is the fifth and points at the same commit as
the canonical branch; it is kept deliberately as a review surface and a third
independent path to `b60d470`, not as a work surface.

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
   | `- 927405a  C8  rineya-pretrained-retrofit           PARKED    parked/tgsa-rineya-retrofit-2026-08-31 (covers eea4f64 only)
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
b60d470  C11 v5-1-unified-programme                      <-- ACTIVE, work here
         fd6e20f held this branch 2026-09-03 to 2026-09-04; preserved by
         archive/tgsa-v5-1-pre-gate-hardening-2026-09-04. Do NOT run
         Stage 0 from that commit -- see section 4.
```

The v5 chain is a strict linear extension of the spine, not a set of divergent
forks: `de3bd63` → `b091490` (+5) → `fd6e20f` (+1) → `b60d470` (+6). **C1 through C6,
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
| C6 | `…phase2a-ci-and-competence` | Phase 2A; v5 execution base | `sequential_teacher_forced_recursive_v1` | **SUPERSEDED** 2026-09-03, ref deleted 2026-09-04 | yes |
| C7 | `…mtp-rineya-complementarity-v1` | Phase 2B system factorial | `parallel_independent_offsets_v1` | active, no result yet | no |
| C8 | `…rineya-pretrained-retrofit` | real Qwen3/Rineya retrofit | — | **PARKED**, sequenced third | no |
| C9 | `…v5-competence-execution` | v5 competence execution | — | **SUPERSEDED** 2026-09-03, ref already gone | yes |
| C10 | `…v5-competence-and-mechanism` | v5 competence + mechanism; v5.1 base | — | **SUPERSEDED** 2026-09-03, ref deleted 2026-09-04 | yes |
| **C11** | **`…v5-1-unified-programme`** | **Phase 1 v5.1 unified programme, canonical** | none at this stage | **ACTIVE**, frozen, no result yet | — |

C6 and C9 are the same commit (`de3bd63`) under two branch names. Worktree
paths, exact commits and per-stage results are in the JSON.

### Test counts across the chain

| stage | commit | collected | README claimed |
|---|---|---:|---:|
| C5 | `d71b558` | 350 | 349 |
| C6 / C9 | `de3bd63` | 396 | 396 |
| C10 | `b091490` | 436 | 396 |
| C11 | `b60d470` | 458 | 458 |

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
tree. That condition was met by CI run `33889037837` on `b60d470` and both
refs were deleted on 2026-09-04 — see *Ref retirement and backups of
2026-09-04* below.

**Ordering matters:** `verify_programme_ledger.py` resolves every stage
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

All four are fixed and **merged**. C11 was fast-forwarded `fd6e20f` → `b60d470`
on 2026-09-04: six commits, 839 insertions and 9 deletions across 11 files, no
merge commit, history still linear. 458 tests pass, the contract verifier is
green on all ten checks, `programme_sha256` is unchanged at `d52d89a3…`, the
locked plan blob `0538faf4…` is byte-unchanged, and **no file under `configs/`
or `evidence/` was touched** — the registration itself is unaltered.

Three backups were taken before the merge, all verified:

| backup | where |
|---|---|
| the pre-merge frozen tip | tag `archive/tgsa-v5-1-pre-gate-hardening-2026-09-04` → `fd6e20f`, on origin |
| the change as a reviewable branch | `research/tgsa-v5-1-gate-hardening` → `b60d470`, on origin |
| every ref, off GitHub | `BusinessProject/tgsa-backups/tgsa-all-refs-2026-09-04.bundle`, verified as recording a complete history |

A fourth review pass returned 13 candidates, **0 confirmed and 0
unadjudicated** across 16 agents. That last number is the one that matters: the
first pass reported clean while four of its refuters had died, and the fix
distinguishing *refuted* from *never checked* is what makes this readable. It
was launched against `1af0b5a` and the branch moved to `b60d470` beneath it, so
the final two commits — the diagnostic-totality fix and the line-ending
restoration — were not its declared target.

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

### Ref retirement and backups of 2026-09-04

The gate-hardening work was merged into the registered branch as a **pure
fast-forward**, `fd6e20f` -> `b60d470`, so the history stayed linear and no
merge commit exists. Measured on the merged tip: **458 tests pass**, the
contract verifier exits `0` on all ten checks with `programme_sha256` unchanged
at `d52d89a3020972dde7935120094d55dfe50dc8d3e8b6fe288860daa14e8142e9`, the
locked plan blob `0538faf4d2ad142dbce9c408910f9dcbcf1fd064` is byte-unchanged,
and **zero files under `configs/` or `evidence/` were touched**. The
registration itself is unaltered; only the code that enforces it changed. CI run
`33889037837` succeeded on `headSha b60d470` exactly.

**The two superseded refs were then deleted from origin.** The condition the
operator set — CI green on the materialized tree — was met on the right commit
rather than an approximate one.

| deleted from origin | commit | still reachable via |
|---|---|---|
| `research/tgsa-phase2a-ci-and-competence` | `de3bd63` | ancestry of `b60d470`; two archive tags; three bundles |
| `research/tgsa-v5-competence-and-mechanism` | `b091490` | ancestry of `b60d470`; one archive tag; three bundles |

Both commits turned out to be **ancestors of the canonical tip**, so they would
survive with no tag at all — `merge-base` returns each tip itself. Only the
*remote* refs were removed; the local branch and the `Chat-Bubble-phase2a-next`
worktree were left untouched, as were the other eleven worktrees.

*This was measured, not inferred.* `tgsa/v5_1_programme.py` resolves the pinned
lineage with `cat-file -e` and `merge-base`, never by branch name, so the
question is whether the **objects** reach a fresh clone, not whether the branch
names exist. A clone was taken from origin **after** both deletions: HEAD
`b60d470`, worktree clean, all five pinned commits (`de3bd63`, `b091490`,
`49aa961`, `ef5194b`, `8b5737e`) resolved, the contract verifier exited `0` with
an identical `programme_sha256`, and the full suite passed 458 tests inside the
clone. A reviewer needs nothing but origin.

**Backups.** *Three* verified bundles are kept off GitHub under
`C:/Users/clclc/BusinessProject/tgsa-backups/`:
`tgsa-all-refs-2026-09-04.bundle` (pre-merge: unified-programme at `fd6e20f`,
ledger at `2e0e3fe`), `tgsa-all-refs-2026-09-04-post-merge.bundle`
(post-merge, taken *before* the deletions, so it holds `b60d470`, ledger
`4581fc0`, and both retired refs) and `tgsa-all-refs-2026-09-04-final.bundle`
(taken *after* the deletions at 15:21: `b60d470`, ledger `26b1d0d`, with the two
retired refs surviving as local branches). The first is retained deliberately: it
predates the ledger commit recording the merge, and is the only off-GitHub copy
of the pre-merge registered tip as it stood when the merge decision was taken.
The third was created on 2026-09-04 but went **unrecorded here until the
2026-09-05 audit found it on disk** — this document said "two bundles" in three
separate places while three existed.
Recovery is
`git branch <name> archive/tgsa-v5-1-pre-gate-hardening-2026-09-04`; that tag's
annotation carries a **DO NOT RUN STAGE 0 FROM THIS COMMIT** warning and the
reason, so the pre-hardening apparatus cannot be restored and then used by
mistake.

**The Stage 0 execution plan is now a record**, at
`TGSA_PROGRAMME/STAGE0_EXECUTION_PLAN.md`. It had existed only in a
session-scoped OS temp directory, which is not a backup. Its probe thresholds
were re-checked field by field against the locked plan and match. It remains
**PREPARED, NOT RUN**.

**What the remaining blocker actually says.** The flag
`--attest-no-prior-outcome-inspection` records one narrow sentence:

> No nondevelopment outcome produced under the locked v5 plan was inspected
> before the prospective v5.1 lock.

It concerns outcomes produced *under the locked v5 plan* — not any prior TGSA
result. v4's rejected-regime evidence was produced under a different plan and is
frozen historical evidence, so having inspected it does not bear on this
attestation. The supporting facts were checked on 2026-09-04: **no
`probe_report.json`, `qualification_report.json`, frozen band or
`v5_1_registration_verification.json` has ever been committed on any ref or tag
in this repository**, no `runs/`, `results/` or `outputs/` directory exists in
the v5.1 worktree, the worktree is clean, and every run made during preparation
used a `development_only=true` plan. Those facts are recorded so the operator's
attestation can be *informed*; **only the human operator may assert it**, and it
was not asserted by the assistant.

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
2. **Stage 0 is prepared, costed, and no longer blocked on the apparatus** —
   the four defects are merged as of `b60d470`. It still cannot start until the
   human operator supplies
   `--attest-no-prior-outcome-inspection`. Running first and locking after would
   destroy the prospective registration irrecoverably.
3. **Promotion-order step 7 is not started** — the port manifest reconciling
   C7's retrofit snapshot against C8's authoritative copy. Small job: the two
   differ in three files.
4. **The Stage 0 pre-flight audit is incomplete and must be finished before the
   operator attests.** Three of seven dimensions never ran — `gate-arithmetic`,
   `data-flow-chain` and `plan-contract-schema`, the last being the dimension the
   `famili_roles` typo lived in — the completeness critic never ran, and seven
   candidate findings were never adjudicated because their refuters died on a
   usage limit. One confirmed defect was found and fixed. **Treat the remainder as
   unexamined, not clean.**

Closed on 2026-09-04: the apparatus-defect block on Stage 0, merged at
`b60d470`; the retirement of the last two superseded refs, verified
non-destructive by a fresh clone taken after the deletions; and **the C8
branch-name evidence pointers**, repaired at `927405a`.

That last item is worth its own note, because the count this ledger carried was
wrong in both directions. `candidate_ledger.json` held **four** broken
`origin/<branch>:<path>` pointers, not three: a fourth named
`research/tgsa-phase1-v4-competence-first`, retired on 2026-08-31 and broken the
same way, which no open item had recorded. And **three** of the ref-scan table's
six inspected branches are retired, not two —
`research/tgsa-phase2-mtp-complementarity` is gone from origin as well. All four
pointers now name commits: `de3bd63`, and `8b5737e` for the v4 one, which is the
frozen v4 boundary the hold note already used as its anchor and whose blob is
bit-identical at both commits (`467da3d7`). That is the form
`tgsa/v5_1_programme.py` resolves by, so these cannot break again. The two
`source_branch` fields still name the retired branch deliberately: they are
history, not pointers, and the `source_commit` beside each one resolves.
`CANDIDATE_LEDGER.md` needed no change — it already cited `d16ea1c`. C8 remains
**PARKED**: no lab file moved, nothing was merged, and
`verify_candidate_ledger.py` still returns `promotable []`.

**Audited 2026-09-05, and the repair holds — but two of its claims had been
asserted rather than measured.** A repoint is only safe if the new target names
the same content the old pointer resolved to, and that was never checked: if
origin's ref had been *ahead* of the local branch at deletion, rewriting to
`de3bd63` would have silently changed the evidence. The remote reflog died with
the ref, but `tgsa-all-refs-2026-09-04-post-merge.bundle` was taken *before* the
deletions and still holds
`refs/remotes/origin/research/tgsa-phase2a-ci-and-competence` → `de3bd63` and
`refs/heads/research/tgsa-phase1-v4-competence-first` → `8b5737e` — the exact
two commits the pointers now name. Origin was not ahead; both repoints are
content-preserving, proven against an artifact origin cannot retro-edit.
Durability likewise: `git ls-remote` puts origin's tip at `b60d470` and both
targets are ancestors of it, so both reach any fresh clone. The second gap was a
skipped regression check — `candidate_ledger.json` is consumed by
`rineya_tgsa/promotion.py`, not just the standalone verifier, so the suite was
the check that mattered. Run since: **71 passed**.

**Pushed and proven from origin alone, 2026-09-05.** C8 went first
(`ef5194b..927405a`), then this branch (`26b1d0d..ceadce6`) — that order is
forced by §9: a ledger recording a stage commit origin cannot resolve fails
closed. A clone was then taken from origin with nothing local available.
`verify_programme_ledger.py` returned **53 checks PASSED** inside it, C8's own
`verify_candidate_ledger.py` returned `PASS` with `promotable []`, and all four
repointed pointers resolved under `git show` with real content. The repair now
meets the same standard the retired refs were held to.

> **Gotcha for anyone repeating that test on Windows.** Run from a deep
> directory, the verifier reports a *false* `FAILED` with three
> `unreadable source: 49aa961:…` errors against the `imported/phase2b` records.
> Nothing is wrong with the ledger. `git show <rev>:<path>` falls back to
> `stat()`ing the whole argument as a filename once it exceeds `MAX_PATH`, and
> reports `Filename too long`. The same clone at a 17-character path passes 53/53.
> The `phase2b` paths are the longest recorded, so they fail first and are the
> canary.

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

### Stage 0 pre-flight audit of 2026-09-05

Stage 0 is a *prospective* registration: running it before the lock destroys the
registration irrecoverably. So the last cheap moment to catch an apparatus or
runbook defect is **before the operator attests**. The 2026-09-04 round had
already caught one defect of the only class that really threatens this — a
command defect invisible to every permitted test, because the code it would trip
sits behind `if not development_only`. This audit went hunting for more of that
class: static, read-only, no pipeline execution, no attestation flag.

**One confirmed defect, and it is the same shape as the last one.**
`STAGE0_EXECUTION_PLAN.md` §4.0 told the operator to *"point `--report` at
whichever rung passed."* That is correct only when `C1` is the rung that passes.
`freeze_band_after_probe` walks the registered ladder `C1 -> C2 -> C3` from the
**first** rung and raises `probe report for rung C1 is missing; rung selection
must follow the registered order` the moment a rung has no report — a *failing*
rung's report is still required, because it is the evidence that the climb was
registered. `ADVANCE_CAPACITY_RUNG` is deliberately not one of the loop's break
conditions, so the loop advances and demands the next rung's report too.
Symmetrically, a report *after* the first pass is refused. The document stated
neither constraint.

Concretely: probe `C1` returns `NO_GO` after ~0.79 h, probe `C2` returns `GO`
after ~1.60 h, the operator follows §4.0 as written, and the selector dies
naming rung `C1` — no band written, the whole 36-cell qualification blocked, and
the doc pointing away from the fix. It fails closed, so nothing is destroyed and
a re-run recovers it; the cost is the operator stopped mid-climb by their own
runbook.

Why nothing caught it: for the registered plan `freeze_band_after_probe` raises
`development-only probe reports cannot freeze a real band`
(`tgsa/phase1_v5.py:1075`) *before* the ladder loop is entered, so no
development-mode run reaches the check — and every `freeze_band_after_probe`
test uses `C1` as the passing rung, so the "`C1` failed, a later rung passed"
path has **no coverage at all**. Fixed in the document; the code is correct and
was not touched. `--report` is `action="append"` on both subcommands, so
repetition was always supported — this was a documentation defect, not a missing
capability. The uncovered ladder path remains an open recommendation, deliberately
not acted on because adding that test would touch C11, the frozen tree.

> **This audit cleared nothing, and must not be read as if it did.** 26 of its 32
> agents died on a usage limit. Three dimensions never ran at all, including
> `plan-contract-schema` — where the `famili_roles` typo lived — and
> `gate-arithmetic`, where a threshold compared in the wrong direction would live.
> Seven candidate findings were never adjudicated.

**And the accounting bug recurred — inside the audit harness itself.** The
survivor rule was `confirmed >= 2 || (confirmed >= 1 && refuted == 0)`, so a
finding whose three refuters *all died* scored `confirmed = 0` and fell into the
**refuted** bucket having never been examined. Seven findings were mislabelled
that way. The run was only readable because the harness *also* tracked
unadjudicated separately and emitted an incomplete-adjudication list — the only
reason the mislabelling was visible instead of silent. The lesson, restated
because restating it on 2026-09-04 did not prevent it: a three-way verdict
— confirmed / refuted / **never-checked** — has to be carried into the *final
bucketing*, not just into the tally. Defaulting an unexamined finding to
"refuted" turns a dead verifier into a clean bill of health.

---

## 8. Imported records

Verbatim copies, byte-for-byte, with source branch, source commit and SHA-256
recorded per file in the JSON. They are copies for convenience of reading, not
new artifacts, and they are not evidence in their own right — cite the source
branch and commit, never this path.

- `imported/phase2b/` — 3 documents from C7 at `49aa961`
- `imported/retrofit/` — 9 documents plus `candidate_ledger.json` and its
  schema from C8, at `eea4f64` except three **re-imported at `927405a`** on
  2026-09-04, after that commit repaired C8's broken evidence pointers:
  `candidate_ledger.json` (`74a1ff79c530…`), `PRODUCTION_RETROFIT_AUDIT.md`
  (`0d32f8026822…`) and `PHASE2A_PREREQUISITE_HOLD.md` (`632d26748d95…`). The
  last of those had already been re-imported once before, at `ef5194b` on
  2026-09-03 (`44008d5822d6…`), superseding an `eea4f64` copy that hashed
  `a186a17114…`. Each copy is byte-identical to its source, so the two LF files
  and the one CRLF file keep the line endings they have on C8.

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
