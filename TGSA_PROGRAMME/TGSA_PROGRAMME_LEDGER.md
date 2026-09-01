# TGSA programme ledger

Single index of what the Trajectory-Guided Sequential Attention programme has
established, where each record lives, and what blocks the next step.

Machine-readable source: [`tgsa_programme_ledger.json`](tgsa_programme_ledger.json)

Generated: 2026-08-31.

---

## 0. What this branch is, and is not

**Is:** a read-only aggregation of *records* — results, decisions, provenance
pointers, and verbatim copies of result documents from branches that are not in
the main chain.

**Is not:** a merge. No implementation file from any stage is copied here. No
evidence is pooled. Nothing is relabelled under a generic "TGSA+MTP" identity.

**This is an orphan branch.** It shares no history with any implementation
branch and its tree contains nothing but `TGSA_PROGRAMME/`. That is deliberate.
An earlier version of this branch descended from the Phase 2A branch and so
carried a full copy of the lab, which made it look like somewhere you could
work — and someone reasonably concluded it was the most advanced branch to
experiment on. It is not. It has no code, and it must never acquire any.

**To run TGSA experiments, use `research/tgsa-phase2a-ci-and-competence`.**
Nothing here is a substitute for it, and this branch is deliberately missing the
Phase 2B and retrofit code entirely — those live only on their own branches and
cannot be combined.

This is deliberate, and it is not a stylistic choice.
`TGSA_MTP_CANONICAL_PATH.md` (read with `git show research/tgsa-phase2a-ci-and-competence:Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/TGSA_MTP_CANONICAL_PATH.md`)
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

---

## 2. Which branch do I use?

**Work on `research/tgsa-phase2a-ci-and-competence`.** It is the canonical
Phase 2A branch as of 2026-08-31 and contains the entire Phase 1 chain (C1–C5).
Everything else is either archived, a separate stage that must stay separate, or
records-only.

| Use | Branch |
|---|---|
| **Active work** | `research/tgsa-phase2a-ci-and-competence` |
| Separate active stage — never merge into Phase 2A | `research/tgsa-mtp-rineya-complementarity-v1` (Phase 2B) |
| Parked, third in sequence | `research/tgsa-rineya-pretrained-retrofit` |
| Records only, no code | `research/tgsa-programme-ledger` |
| Frozen evidence, no commits | `research/tgsa-phase1-v4-competence-first` |
| **Archived — branch refs deleted, read through the tag** | `archive/tgsa-phase1-launch-audit`, `archive/tgsa-phase1-v2-launch-ready`, `archive/tgsa-phase1-v3-inconclusive`, `archive/tgsa-phase2a-superseded-2026-08-31` |

The four archived branch refs were deleted on 2026-08-31 after their annotated
tags were confirmed on origin. Nothing was lost: each tag points at the exact
commit the branch pointed at, each is on the remote, and all four are contained
in the active branch. To bring one back:

```bash
git branch research/tgsa-phase1-identifiability-v3-powered-confirmatory archive/tgsa-phase1-v3-inconclusive
```

## 3. Lineage

```text
bef44ce  C1  phase1-model-identifiability-launch-audit    ARCHIVED  archive/tgsa-phase1-launch-audit
   └─ e0063f0  C2  phase1-identifiability-v2-launch-ready ARCHIVED  archive/tgsa-phase1-v2-launch-ready
        ├─ 04d3c02 ─ C8 rineya-pretrained-retrofit eea4f64  PARKED  parked/tgsa-rineya-retrofit-2026-08-31
        └─ 6c97ac1  C3  phase1-v3-powered-confirmatory     ARCHIVED  archive/tgsa-phase1-v3-inconclusive
             └─ 8b5737e  C4  phase1-v4-competence-first     FROZEN   archive/tgsa-phase1-v4-regime-rejected
                  ├─ d71b558  C5  phase2-mtp-complementarity  ARCHIVED  archive/tgsa-phase2a-superseded-2026-08-31
                  │    └─ 188ed21  C6  phase2a-ci-and-competence   <-- ACTIVE, work here
                  └─ 49aa961  C7  mtp-rineya-complementarity-v1    ACTIVE (Phase 2B)
```

Six of the eight branches are a single linear chain. **C1 through C5 are all
contained in C6** — verified, not assumed. Only C7 (6 commits) and C8 (13
commits) hold work outside it, and both are deliberately separate stages.

Five branch names remain: the active Phase 2A branch, Phase 2B, the retrofit,
this ledger, and the frozen v4 evidence branch. The other four exist only as
tags. Four worktrees remain, one per live branch, plus the frozen v4 checkout.

---

## 4. Stage table

| ID | Branch | Role | MTP semantics | Status | In C6? |
|---|---|---|---|---|---|
| C1 | `…phase1-model-identifiability-launch-audit` | Phase 1 launch audit | — | archived, ref deleted | yes |
| C2 | `…phase1-identifiability-v2-launch-ready` | Phase 1 v2 | — | archived, ref deleted | yes |
| C3 | `…phase1-identifiability-v3-powered-confirmatory` | Phase 1 v3 confirmatory | — | **INCONCLUSIVE**, ref deleted | yes |
| C4 | `…phase1-v4-competence-first` | competence / regime boundary | — | **V4_REGIME_FAMILY_REJECTED**, frozen, branch kept | yes |
| C5 | `…phase2-mtp-complementarity` | Phase 2A predecessor | `sequential_teacher_forced_recursive_v1` | archived 2026-08-31, ref deleted | yes |
| **C6** | **`…phase2a-ci-and-competence`** | **Phase 2A, canonical** | `sequential_teacher_forced_recursive_v1` | **ACTIVE**, no result yet | — |
| C7 | `…mtp-rineya-complementarity-v1` | Phase 2B system factorial | `parallel_independent_offsets_v1` | active, no result yet | no |
| C8 | `…rineya-pretrained-retrofit` | real Qwen3/Rineya retrofit | — | **PARKED**, sequenced third | no |

Worktree paths, exact commits and per-stage results are in the JSON.

### Consolidation of 2026-08-31

Two branches had been claiming the Phase 2A role. C6 had grown as an undeclared
continuation of C5 — same stage, same semantics, same lineage — but held 396
tests against 349 and was the sole holder of the v5 preregistration, the
within-study qualification gate, per-family null scoring and the programme
contract. Meanwhile C5's README named itself Phase 2A and
`TGSA_MTP_CANONICAL_PATH.md` did not mention C6 at all.

Resolved toward C6: `ce4f134` merged C5 in so nothing was stranded, and
`188ed21` moved the designation across all six references (canonical path stages
table and branch lifecycle, README chain, runbook lineage, programme contract
stage field, CI trigger). A within-stage rename — no evidence pooled, the
Phase 2A/2B separation untouched, no registered estimand changed. Two branches
claiming one stage was itself an evidence-identity risk.

---

## 5. The two results that exist

### Phase 1 v3 — `INCONCLUSIVE`

Study `tgsa-phase1-identifiability-v3-2026-08-29`. 3 arms × 20 seeds = 60 cells,
2,500 updates each, 60/60 completed. Evidence chain: 2,001 checks, all passed.

| contrast | estimate | 95% CI | P(>0) | negative seeds |
|---|---:|---|---:|---:|
| H1a delta − history | +0.0014 | [−0.0021, +0.0048] | 0.790 | 9/20 |
| H1b delta − cascade | +0.0004 | [−0.0025, +0.0034] | 0.619 | 8/20 |
| H1c aligned − replay-deranged | +0.0019 | [−0.0003, +0.0040] | 0.959 | — |

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

- All eight descend from `bef44ce` and share the Phase-1 chain and the PART 11
  reporting standard with `OBSERVED` / `SUPPORTED` / `HYPOTHESIS` marking.
- **C7 and C8 share a near-identical retrofit codebase.** Of 27 co-modified
  files, only three differ (`pyproject.toml`, `rineya_tgsa/__init__.py`,
  `tests/test_retrofit_correctness.py`). C8 is authoritative: 59 retrofit files
  against C7's 37. This makes promotion-order step 7 a small job.
- **The retrofit tree and the lab tree are near-disjoint.** The spine carries
  zero retrofit files; C8 carries a stale 84-file lab against the spine's 147,
  missing the v3 final result, the v4 rejection, and all Phase 2 work.
- **The programme contract is stage-neutral.** It defines both `phase2a` and
  `phase2b`, and the verifier routes on `study_kind`, so C6 and C7 carry
  byte-identical copies rather than divergent forks.
- Checkpoints are excluded programme-wide. `*.pt` / `*.pth` are gitignored and
  the repository tracks zero of them.

---

## 7. Open items

1. **No competence-qualified regime exists.** This is the single blocking gate
   for the whole programme. No Phase 2 confirmatory run is authorized until a new
   preregistration produces a `GO` on learnable, non-saturated primary families.
2. **Promotion-order step 7 is not started** — the port manifest reconciling
   C7's retrofit snapshot against C8's authoritative copy. Small job: the two
   differ in three files.
3. **C8's `CANDIDATE_LEDGER.md` routing table still names the old Phase 2A
   branch.** C8 is parked, so this is stale-on-purpose until the hold lifts;
   refresh it as part of lifting the hold, not before.

Closed on 2026-08-31: the C5/C6 designation conflict; C5's divergence from the
Phase 2A lineage; and the unpushed backlog — all five branches and three new
tags are on origin, every push a clean fast-forward with no force.

---

## 8. Imported records

Verbatim copies, byte-for-byte, with source branch, source commit and SHA-256
recorded per file in the JSON. They are copies for convenience of reading, not
new artifacts, and they are not evidence in their own right — cite the source
branch and commit, never this path.

- `imported/phase2b/` — 3 documents from C7 at `49aa961`
- `imported/retrofit/` — 11 documents plus `candidate_ledger.json` and its
  schema from C8 at `eea4f64`

**Not imported:** any source code; the `evidence/phase2_mtp_smoke_local/`
rehearsal output on C5 (regenerated scratch — CPU, one seed, three updates,
`confirmatory=false`); any checkpoint.

---

## 9. Maintaining this

The ledger is a snapshot, and a stale ledger is worse than none. Refresh it when
a stage reaches a decision, a branch tip moves in a way that changes a recorded
result, or an open item closes. Update the JSON first — it holds the hashes and
commits — then reconcile this document to it.

Do not add implementation code to this branch. If that ever becomes tempting,
the thing that is actually needed is a port manifest on the target branch.
