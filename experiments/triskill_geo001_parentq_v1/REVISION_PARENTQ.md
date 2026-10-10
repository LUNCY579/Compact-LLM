# Revised parent qualification packet — unfrozen, 2026-10-05

**No native parent qualification has occurred.** This revision reproduces mechanical and
calibration defects, repairs routine ownership/accounting paths, and prepares decisions.
It does not freeze or authorize a model run. The review's CPU bootstrap reaching preflight
was source/bootstrap evidence only. Geometric composition remains deferred.

This is the single packet entry point. Its components are the preserved
[review](REVIEW_PARENTQ.md), [historical preregistration](PREREG_PARENTQ.md),
[prospective options](PROSPECTIVE_DESIGN_OPTIONS.md), [readiness guard](READINESS_PARENTQ.json)
and [exact native-run review candidate](NATIVE_RUN_PROPOSAL.md). The proposal remains blocked;
its exact current membership is unsuitable for a fresh representative qualification claim.

## Identity, ownership and preserved evidence

Workspace: `C:/Users/clclc/BusinessProject/ChatBubble-triskill-geo001-sectionh`.
Branch: `research/triskill-geo001-parentq`; HEAD remains
`3e07fd8740e8ff4debd94f9c17ca3e06d056a0ed`, derived from section-H `9b44e1ed`.
Origin metadata identifies `LUNCY579/ChatBubble`. No branch switch, reset, commit, push,
merge, serving-policy change, promotion or other owner's process termination was performed.
No applicable AGENT.md/AGENTS.md was found in the workspace or its ancestor directories.
The root README, protocol, package scripts/status, pilot reports/amendments/future protocol,
latest local CPU receipts, and launch runbook were read before implementation.
The runbook's other-experiment authorizations were not imported.

The initial local change was only untracked REVIEW_PARENTQ.md, SHA256
`8e4ce29d56585197d5f5421154ec32330f2046132d510adde97a3ca99f452178`.
It remains untracked and unchanged. The original parent bank and canary remain unchanged,
including their failed design evidence. Every tracked GEO artifact remains unchanged;
the frozen section-H digest remains
`f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687`.
The reviewed CPU receipt `4c8a88bc…41ca97` and all prior reports/freeze files are retained.
No new outputs, environments or caches are placed under source or .codex.

No PARENTQ/GEO native worker was found. Other independent preparation chats and a
SYNTH-CRS worker were active; they were not interrupted. Small verification workers were
admitted serially only while holding the shared hardware-worker.lock OS-exclusive handle.
A surviving hardware lock file is not treated as a busy handle. CPU verification acquired
no GPU lease. Central and CRS ownership schemes are inspected/fail-closed for future native
admission; this packet does not assert they are currently idle or reserve a native slot.

## Reproduced findings

The [calibration receipt](C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/calibration-002/CPU_CALIBRATION_REPRODUCTION.json)
uses original decision/generator bytes from HEAD, SHA-verified frozen GEO code, synthetic
credited outputs and fabricated canary strings. It constructs only the eight T/H parent
rows; no J/K oracle or model output is used. It confirms:

- Perfect constructed data: P_T=.967422485352, P_H=.904303075397,
  P=.874843128711; plug-in P=1. The four-cell prior supplies uniform arm marginals.
- A declared independent toy with true gate-pass probability .817152 gives QUALIFIED
  27.495%, MARGINAL 72.18%, NOT_QUALIFIED .325% in 20,000 simulated banks, seed 20261005.
  These probabilities concern the toy, not Rineya parents.
- Equal independent BASE/H Bernoulli null settings pass the four-row H gate from
  .117329302127 at p=.08 to .3477504 at p=.6. This is illustrative, not a universal null bound.
- All six conditional-bank values are positive; the screen conditional uses -6 on its
  true branch. Constructor identity does not justify exchangeability across these classes.
- T14 repeats screen T2's 19*3=57 arithmetic. The added operative-number scan also finds
  bank T3's arithmetic in exposed T_BANK_V3X. Full-payload disjointness did not remove these.
- Blanket H_OFF abstention can earn the legacy prompt label while missing all twelve
  answerable rows. NOT_SPECIFIC can occur without a BASE prompt effect. A BASE ceiling
  cannot establish missing H transport. Natural abstentions remain outside GEO's credit rule.

Corrected illustrative probabilities (the precise synthetic policies and paired cells are
in the receipt, and each number is tested):

| Original synthetic fixture | P | Legacy verdict |
|---|---:|---|
| Perfect parents, BASE T/missing incorrect and H answerable correct | .874843129 | QUALIFIED |
| Shuffled-index 2/3 fixture | .667757598 | MARGINAL |
| Shuffled-index 1/2 fixture | .365668832 | NOT_QUALIFIED:H |
| H never abstains | .075695851 | NOT_QUALIFIED:H |
| H always abstains, BASE H correct | .013088516 | NOT_QUALIFIED:H |
| T and BASE T both wrong | .049578262 | NOT_QUALIFIED:T |

These fixtures do not estimate capability or select a new prior, threshold or population.

## Routine changes and what was verified

LOAD_BEGUN_PARENTQ.json is now create-only before from_pretrained, with load attempts
charged even if loading crashes or the hard wall prevents a terminal receipt. Historical
incomplete directories still count. A canonical external runtime ledger is pinned across
freezes, so a different output directory cannot reset attempts or replay a grant.

The supervisor disables gpu_lease's reclaim path, refuses foreign temporary/central leases,
dual-leases the central lock token, holds the hardware OS handle, and checks CRS SQLite
state read-only. Missing, unknown, active or incompatible CRS state refuses. Every unit
rechecks tokens and the supervisor's PID/create time. Cleanup checks PID identity, catches
taskkill errors, waits after kills, handles errors after Popen, and retains leases if worker
identity/death cannot be established. Argument failures and launch failures leave receipts;
launch stamps include microseconds and PID. Grants bind experiment, attempt, freeze and
hardware slot, and already-used grant text is refused. SQLite handles close explicitly.

Worker bootstrap/preload refusals persist blocked receipts. Freeze bytes are hashed and
parsed from one snapshot; passing receipt contents and counts are checked, not merely a
passing envelope. GPU-lease pins must equal GEO's. Progress events include time. Canary
misses record first token divergence, top-two margin and chosen-minus-reference logit where
available. Source token traces are unavailable: reference tokens are explicitly reconstructed
from stored text, so this diagnostic cannot prove native-source parity or rescue a failure.

The 6.5 GiB host preload floor is a **provisional conservative guard**, replacing PARENTQ's
5 GiB floor; it is not measured lane qualification. Existing active reserves and GPU floors
are preserved. Failed overlap audits and actual token/text disagreement cannot automatically
qualify parents; nondeterminism holds retain the diagnostic verdict without selecting an
attempt. A decision exception becomes a receipt rather than escaping finally.

Validation now pins both exposed T banks; enforces ten contract rows, 56 A/B exact
abstentions, 58 pairwise-different canary rows before filtering and 56 after; and records
operative collisions. The original bank is not rewritten. The freeze builder requires all
declared dependencies committed, complete resolved decisions, independent review, a source-bound
supervised full CPU-suite receipt and full registration validation. It writes outside source.
Historical derivation remains auditable, while its generator cannot overwrite the safety fixes.

The credited scorer, prior, .70/.40 thresholds, 7/8 canary threshold, original rows and
geometric operator are preserved. Additional Q2 ceiling/blanket/specificity flags and H_ON
lenient answer counts are descriptive; legacy labels cannot authorize redesign.

## Verification, retained failures and limitations

Evidence root: `C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/`.
The [verification index](C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/VERIFICATION_INDEX.json)
binds final source hashes, preservation checks, every retained stage receipt and the
independent offline review disposition. Latest verified runs:

| Stage | Result | Supervised wall / peak worker-tree RSS | Evidence |
|---|---|---|---|
| Small Windows containment | PASS: exclusive handle, admitted child/grandchild and bounded tree kill | 3.156 s / 63,234,048 B | [containment-002 receipt](C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/containment-002/OFFLINE_SUPERVISOR_RECEIPT.json) |
| CPU fixtures | 75 tests passed, including 17 worker-mechanics tests | 7.453 s / 130,158,592 B | [test log](C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/fixtures-003/worker.log), [test counts](C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/fixtures-003/CPU_FIXTURES.json) |
| Lightweight registration | Mechanical checks pass; full passed=false, design_ready=false | 1.359 s / 155,787,264 B | [validation-003](C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/validation-003/CPU_VALIDATION_PARENTQ.json) |
| Synthetic calibration | PASS; legacy findings reproduced under declared toy assumptions | 1.110 s / 56,074,240 B | [calibration-002](C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005/calibration-002/CPU_CALIBRATION_REPRODUCTION.json) |

All fixtures run with CUDA hidden, one CPU affinity, framework imports blocked, 512 MiB
worker-tree RSS/private-commit cap, 900 s wall cap, 50 MiB output cap, 2 GiB host admission
and 1.5 GiB active reserve. A Windows Job is assigned before fixture imports and kills
descendants on close; the tiny containment fixture verifies exclusive-handle refusal,
Job membership and whole-tree wall termination. These are CPU process-mechanics fixtures.

The first validation controller saw a normal-exit process snapshot race after its child
successfully wrote validation; that failure receipt is retained. The first combined suite
found a SQLite handle leak and a missing module in the runner's list. Both failures and
their logs remain; corrected runs are linked in the final verification index. No failed
receipt was rewritten as successful. Full Torch/tiny-model tests were adapted but not run
under this lightweight slot. Native CUDA determinism, padding, canary, memory, WMI survival
and model-loading containment are unqualified; no native bootstrap is repeated here.

The lightweight registration receipt deliberately has mechanical_checks_passed=true,
passed=false, validation_profile=LIGHTWEIGHT_NO_TORCH and design_ready=false. Full
AutoTokenizer equivalence was not re-run in this profile. It cannot be used to freeze or
load a model. No new environment or dependency installation was needed.

Current native model loads/forwards/generations, training updates and paid calls are zero.
The only large asset used was the local tokenizer for token counts; checkpoint weights were
not read or rehashed. Source weight digests are inherited manifest identities, not a fresh
weight-custody or backup attestation. Sealed/confirmation banks were not opened.
No fixture receipt is Rineya capability evidence.

An OS hardware handle cannot be read externally while exclusive; unit checks bind it to
the same live supervisor and its START token/path. CRS is polled read-only, without taking
a CRS-owned lease; an uncooperative owner can race those polls. A genuine exclusive slot
and native review must resolve that limitation. If the supervisor exits unexpectedly,
Windows releases that handle; retained central/temp records still block native reuse.
PID/create-time checks reduce accidental PID reuse but do not constitute a native
containment qualification. Native failure paths remain subject to the full reviewed slot.

## Original hypothesis and GEO boundary

Preserve the parent definitions, layer-14 output subspaces, rank four, eta=.1 and fixed
downstream network. The target remains transition-conditioned answering/abstention.
Independent conjunction, easier arithmetic, prompt-format repair or a trained delta_C
cannot silently substitute for that third-skill question. Original weak/failed evidence,
random-update inertia, unresolved ARMPERP control and failed TRISKILL coefficient retention
remain limitations; no positive claim replaces them.

Section-H construction, rows and decision rules are unchanged. The review found GEO's
one-shot loaded-attempt accounting/ownership gaps. Correcting those requires an owned GEO
amendment and new freeze, followed by PARENTQ re-pinning. This revision does not edit frozen
GEO files or execute its composition screen. J's conditional dependency (sorrel/tansy and
the yarrow control) remains an instrument-design fact and must be linked prospectively to
the T conditional class; it is not a basis for selecting rows from J outputs.

Registration corrections retained explicitly: 58 canaries differ pairwise, then 56 fit the
length rule; H training gold as well as prompts were read; the historical 224/224 parity
claim covers BASE/SPARSE only; the historical decode fixture uses no forced EOS and
zero-based positions 0, 3, 17 and 31 over six sampled prompts, not every arbitrary EOS
position (this larger Torch fixture is unrun here); recorded throughput is .039–.043 s/step;
the .08–.38 construction-entry estimate was for first-implementation rows and not recomputed
after section H. Scale training only used factors {+-1,+-2}; factors 3/4 are not demonstrated
training support. T_BANK_V5 exclusion remains construction-based and unverified.

## Decisions required before a native parent screen

All ten decision keys in READINESS_PARENTQ.json remain PENDING:

1. Exact fixed-parent gate feasibility versus calibrated new-world predictive readiness.
2. Retain the weak exploratory GEO gate versus amend it for parent reliability/retention.
3. Target strata, operative-collision policy, versioned replacement membership and seed.
4. Canary threshold, numerical diagnostics and a canary-failure route.
5. Padding failure versus incomplete mechanics routes and reviewed repair/re-freeze.
6. Exposed-row policy, including exact-parent reuse and partial attempts.
7. Q2 selectivity, answer retention, BASE ceilings and secondary descriptive readouts.
8. Cumulative retries, terminal holds, overlap completeness and nondeterminism policy.
9. GEO supervision amendment, owned review, new freeze and PARENTQ re-pin ordering.
10. TT backup/custody evidence, full CPU/native containment qualification and named hardware slot.

**Next concrete decision:** choose the estimand together with exact-parent exposure eligibility.
Directly measuring the fixed eight parents is attractive for deterministic feasibility;
it still needs an explicit reuse exception or retirement policy, and does not establish
reliable parent capability. Until that choice, no new bank, prompt contract or outcome-sensitive
rule is selected, the packet stays unfrozen, and geometric composition stays deferred.
