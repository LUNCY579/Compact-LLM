# SCP boundary lineage decision — review only

**2026-09-19 | Proposed canonical lineage: B | NATIVE_REVIEW_LOCKED**

This is an administrative, documentation-only proposal. It creates no native or
training authority, does not certify either adapter, and does not merge the two
implementations. Review this decision before treating it as accepted ownership.
`BOUNDARY_REVIEW.md` remains the current execution-boundary description; the older
README integration snapshot is historical, not a current launch recipe.

## Exact comparison

Repository: `LUNCY579/ChatBubble`.

| Name | Branch | Inspected commit |
| --- | --- | --- |
| A: standalone offline boundary | `research/scp-native-qualification-boundary-v1` | `7486a1490517e9bd0ed8e29631a601d713ec20e8` |
| B: locked integration successor | `research/sequential-complementary-native-boundary-v1` | `9551d7ba50784a5acd19e8512faaba6d0efc6aca` |
| Frozen SCP publication / merge base | `research/sequential-complementary-skills-v1` | `3ad2769a9f8343a9697e1720fcdc9efa18c3845a` |
| B's integration-review parent | `research/sequential-complementary-native-v1` | `b4f010619290a562c54cdaf3bb6edd4eb9112a91` |

GitHub's A-to-B comparison returned `diverged`, B ahead by 2 and behind by 1.
A is a direct child of the merge base. B follows the integration parent above.
Neither qualification branch contains the other. The compare file list is not a
proof that A's independently added package was copied into B.

```text
3ad2769a
|-- 7486a149  A: experiments/scp_native_qualification_v1
`-- b4f01061  integration review
    `-- 9551d7ba  B: experiments/sequential_complementary_native_v1
        `-- documentation-only review successor (single parent B; NOT a merge)
```

## What each branch actually implements

| Surface | A at 7486a149 | B at 9551d7ba |
| --- | --- | --- |
| Source change from merge base | 13 added files, confined to its new package; original SCP native refusal untouched. | 19 added files and a narrow change to SCP `reasoning.py`; incorporates the integration parent and then locks native entry points. |
| Subject loading | Stdlib streaming hashes and bounded safetensors headers; no live tensor loader. | Retains dormant exact-FP32 tensor materialization, sparse-delta reconstruction, tied-storage and immutable-state checks. Code exists; native execution remains unqualified. |
| Host adapter | Literal `ScriptedTransport` and `FixtureTokenizer` only; rejects arbitrary/live substitutes even through `native=False`. | Retains typed `GenerationBackend`, fixed REP/MR policy and zero-update student probes behind the new unconditional lock. |
| Supervision/accounting | Reservation callback bridge and supplied resource-value checks; no live MQ/RRP attachment. | Retains MQ lease/claimed-child/watchdog/resource integration and RRP Journal code, but public launch/worker paths refuse before using them. The new first-host draft is not connected to this old controller. |
| Generation and evidence | Fresh per-request caches, reserve-before-step, finite full fixture logits, raw EOS tokens, terminal failure behavior; independent create-only fixture bundle/event verifier. | Complete generation-config EOS validation and decoder wiring; necessary-only student-result screening plus dormant terminal/claim/journal verification. |
| Current authorization surface | Prospective request validator accepts an approval-shaped decision but still requires `native_execution_permitted=False`; `launch` always refuses. | Pending-only first-host request with every permission false; CLI `qualify`, `_worker`, `propose`, loader, live authorization and generation paths are code-locked. |
| Test evidence | Recorded 114 scoped standalone-package CPU/stdlib checks; parent 85 not rerun. | Recorded 121 partial-source stdlib/boundary checks; historical 112 and full-checkout regression not rerun. Some tests execute extracted functions or inspect source structure. |

Both pin the same three weight-file hashes and runtime versions. This does not make
their request schemas, observer capabilities, protocols or receipts interchangeable.
Neither published count is native qualification, and neither is a current-session
rerun. Do not add 114, 121, 112 or 85 into a combined regression total.

## Do not union the envelopes

All columns below are proposals or dormant ceilings, not approvals.

| Limit | A prospective combined smoke | B current first-host draft | B preserved legacy controller |
| --- | ---: | ---: | ---: |
| Worker launches | 1 | 1 | 2 |
| Generation requests | 12 | 0 | 10 |
| Generated-token allowance | 928 | 0 | 576 |
| Forward allowance/ceiling | 934, including six simulated qualification reservations | 64 actual forwards maximum | 1,024 actual forwards maximum |
| Owned-work seconds | 900 | 900 | 3,600 |
| Worker-time budget including overhead | 620 | 620 | 2,600 |
| New-evidence ceiling | 64 MiB | 64 MiB | 128 MiB |

A's fixture executes a schedule, not real O/T/REP/MR native qualification or six
window forwards. B's first-host proposal is student-only loader/tokenizer/cache
mechanics at 2,048 tokens, batch 1, FP32 eager, with zero generation. Keep that
narrow proposal; do not import A's broader allowances or B's dormant two-stage grant.

`boundary.BRANCH` binds B, while inherited `contracts.BRANCH` still binds
`research/sequential-complementary-native-v1`. This intentional incompatibility is
an unfinished integration gate, not a typo to repair during de-duplication.
Neither guard is changed here. The new review branch therefore cannot use `draft`
as though it were B. No branch renaming, guard bypass, permit translation, or JSON
approval may be used to make a launch path work.

## Decision and selective combination

**Recommend B as the sole implementation lineage, not as a qualified release.**
It retains the model-specific loader, student probes and supervisor/journal work
already published at b4f01061, adds earlier refusal points, and provides the smaller
first-host proposal. Selecting A instead would strand that integration work or
require a larger port. The seven-check difference is not the reason to choose B.

**Retain A at its exact commit as a non-contained reference.** It is not redundant,
fully superseded, or safe to delete on the present evidence. Preserve its useful
requirements without copying a second launcher, request schema or receipt namespace:

| A requirement to retain in review | B status and safe next treatment |
| --- | --- |
| Fixture transport cannot impersonate a live host | B has typed/live checks and the new lock; reconcile old positive fixtures using explicit fixture-only isolation, never by relaxing production refusal. |
| Independent event-order, reservation/token/EOS accounting and tamper rejection | B emits raw tokens and has receipt checks, but equivalent coverage to A's complete fixture verifier is not established. Review/port negative cases into B with fixture-only labels; no new native PASS producer. |
| Failed attempts retain reservations and cannot retry | Inspect inherited B generation/supervision tests under the lock; retain failures and prove parity before claiming coverage. |
| Asset checks are explicitly weaker than loaded-model qualification | Preserve that claim boundary in B. Do not treat headers, hashes or host-shaped dictionaries as observed native execution. |
| Resource observation freshness and explicit RSS limits | A validates supplied timestamps/values, not an independent observer. Qualify B's actual observer separately; do not accept supplied JSON as its replacement. |

Do not copy A's asset policy wholesale: A requires the subject config to be the
source config with `torch_dtype=float32`; B also admits unchanged source-sidecar
bytes with an explicit FP32 loader. B additionally allowlists asset-directory
entries and requires BF16 source storage. These policies need a reviewed, pinned
resolution, not an assertion that the implementations are equivalent.

For each retained A requirement, a later review must record `already covered`,
`ported and independently tested`, or `intentionally deferred/out of scope`, with
source-bound evidence. All such dispositions remain pending here.

## Smallest safe change and acceptance checks

This successor changes exactly two documentation paths: a current-routing preface
in `README.md` and this decision. The entire former README follows the preface
unchanged. No Python, test, workflow, model, dataset, source pin, historical receipt,
permit, grant validator, numeric limit, or experiment outcome is edited. No A code
is cherry-picked and A is not made a second commit parent.

Before accepting publication, independently verify the new branch ref, its single
parent B, the exact two-path diff, and both changed Git blob identities. Verify
that A, B, b4f01061 and the frozen SCP branch refs have not moved. Publication checks
are separate from this pre-publication document; it is not its own push receipt.

The documentation session does not rerun either scoped suite, the historical 112,
the parent 85, full-checkout regression, or native qualification. No model stack is
imported, no permit is consumed, and no workflow is manually dispatched. Future
full-checkout failures must be retained rather than hidden by weakening the lock.

The next engineering gate is B's full-checkout regression and fixture reconciliation,
then reviewed student-only adapter/observer/receipt integration on an isolated
successor. Exact-host observations and a separately approved one-launch envelope
come only after that review. This document supplies no execution command or approval.
Training, backward/optimizer work, REP/MR generation, teacher/cloud calls, protected
bank reads, scoring, confirmation, release and model-quality claims remain closed.

## Branch disposition: no deletion in this change

Keep both original qualification refs untouched for review. This is ownership
convergence, not physical branch-count reduction or an ancestry merge. The temporary
documentation branch is a review vehicle, not a third experimental implementation.
After review, continue only the chosen lineage unless separately directed otherwise.

Retiring A requires separate explicit approval, an exact-tip archive tag verified
at the expected SHA, preserved artifacts and a review of open PRs, workflows,
scripts, worktrees and evidence consumers. Those retirement/consumer checks are not
completed by this source comparison; absence of an observation is not a clean bill.
The eventual reason would be `alternative closed; continuation selected`, not
`fully contained`, unless later ancestry/content evidence actually proves that.
No branch is renamed, merged, archived or deleted by this proposal.

## Pinned evidence map

- A: [commit](https://github.com/LUNCY579/ChatBubble/commit/7486a1490517e9bd0ed8e29631a601d713ec20e8), with `experiments/scp_native_qualification_v1/{README.md,contract.py,assets.py,adapter.py,evidence.py,LOCAL_TEST_REPORT.json}`.
- B: [commit](https://github.com/LUNCY579/ChatBubble/commit/9551d7ba50784a5acd19e8512faaba6d0efc6aca), with this package's `BOUNDARY_REVIEW.md`, `boundary.py`, `contracts.py`, `native.py`, `run.py`, `receipt_review.py`, `__main__.py`, `tests_boundary/test_boundary.py`, and `evidence/boundary_v1/LOCAL_TEST_REPORT.json`.
- B's unchanged `tests/test_native_contracts.py` still includes positive `test_native_rep_mr_use_frozen_policy_and_fresh_caches` and `test_generation_cannot_switch_prompt_after_authorization` paths. These conflict with the new unconditional lock if run as originally written; this is a source-inspection finding, not a claimed fresh test-run failure count.
