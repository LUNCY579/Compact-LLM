# CRS-001 native operational revision 2: targeted scientific review

Recorded 2026-10-02, America/Chicago, under the updated client environment.
Reviewer `/root/scientific_audit`; builder `/root/runtime_audit`.
Disposition: **SCIENTIFICALLY_APPROVED_FOR_BOUNDED_DIAGNOSTIC**, conditional on
separate operations acceptance, clean committed/review bindings, current
asset/runtime/resource admission, fenced lease and cumulative budget gates.
This records scientific scope acceptance, not new owner permission or evidence
that the successor has run a Rineya model.

Managed worktree:
`C:/Users/clclc/.codex/worktrees/crs-native-desktop-v2/ChatBubble-rineya-centralized-reasoning-system-v1`.
Approved prior source: `9fd345d9bbcd0cdc6b300dcb07f6b68dbe8914ec`.
The original scientific review and primary CPU lifecycle/instrument records are
preserved. This reviewer did not build the repair. Reviewers share model lineage,
workspace/host privileges and source evidence; this is a separate artifact pass,
not independent ground truth or an OS-secret evaluator. No model/native launch,
protected gold, confirmation access or unrelated-process termination occurred.

## Frozen reviewed identities

Paths below resolve within `experiments/rineya_centralized_reasoning_system_v1/`
in this worktree.

| Artifact | SHA256 |
| --- | --- |
| `native.py` | `b94b4dc2ce9e6c041efecaff84a885e6b4fa6e6986d85832d6aa019128d7a852` |
| `tests/test_native.py` | `800eb02ffa7ea4889496ef9d478199ab205fe08f9709d6e84ec8caba33087d5b` |
| `contracts/NATIVE_DIAGNOSTIC_V1.json` | `291b7803e745ade2ccfcc5a6e9a8eafc31a9c174541ccd5436f730bde57a8707` |
| `NATIVE_OPERATIONAL_AMENDMENT_002.md` | `112f71e35090d75e4e8906bbe514306ba66ffd33e973a245ad13f8006f2dca94` |
| `contracts/history/NATIVE_DIAGNOSTIC_V1_a57a7fa0.json` | `a57a7fa091aeda2a63e8b1452fdeed53474f627246a3f4429b62af924a45babe` |

A committed successor tip is pending the coordinator's freeze. The eventual
attestation must bind that exact tip and these bytes; previous launch approval
does not silently authorize a changed source/contract.

## Raw failure and repair premise

Read original failed-run supervisor result, stderr and complete-tree record,
plus raw NULL-versus-exact-owned-Job probe. Fresh byte bindings:

| Evidence | SHA256 |
| --- | --- |
| `results/d2_9fd345d9_epoch001_statechange2/SUPERVISOR_RESULT.json` | `cd5d726e26771f152602377ddf135128a6e143efcfc48d9e9dcae2db10c3da3b` |
| Same run's `stderr.log` | `ff3f7abf0b7d8076343ed19b553f300546484f19d2059c0d6a34f7dd5dae7bb5` |
| `outputs/job_identity_probe_v1/NULL_VS_EXACT_OWN_JOB.json` | `a2627233b14144547954f818d08aa50f7a2fd58b05eee5da9d08847599af9cb7` |

The original attempt is FAILED, elapsed qualification worker time
1.1090000000003783 seconds, with an observed empty owned tree afterward. Stderr
places rejection at the pre-framework Job witness before model construction,
checkpoint load or forward. Thus it provides no negative model-quality result.
Its cost and failure remain retained; this repair does not reset them.

The actual bounded stdlib child probe reports membership in both queried Jobs,
NULL flags `0x3000` and explicit supervisor-owned flags `0x2000`. The explicit
query handle was closed afterward. This establishes the observed identity/limit
difference, not a complete hierarchy or a claim that NULL necessarily names an
outer Job. Microsoft documents that NULL queries the calling process's associated
Job and, when nested, its immediate Job. [QueryInformationJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject)

## Targeted scientific invariance checks

Independently compared the repaired source against `git show 9fd345d9:...`,
rather than only the builder's narrative. AST comparisons excluding source
positions are identical for:

- `causal_plan`, `numerical_decision`, `causal_mask_reference_bytes`;
- reader token-boundary validation, closed reader/training interfaces;
- mask, rotary, final-normalization hooks and resource guard.

The complete worker tail beginning `snapshot = committed_snapshot()` is
AST-identical. The worker body from CallAccounting initialization through
framework/model math, protected comparisons, source/base checks, result/cost
finalization is also byte-identical. Therefore the change introduces no different
model forward, dtype, attention/cache mode, scoring rule or numerical threshold.

The contract's changed existing top-level fields are exactly `status`, the native
source digest within `source_sha256`, CPU test metadata and test digest. Other
dependency source pins are unchanged. Added fields record operational revision,
parent contract, amendment, old failure/probes and explicit owned-Job identity.
The original a57a7fa0 contract copy is byte-identical to its committed predecessor.
In particular, these scientific/resource conditions remain unchanged:

| Condition | Retained scope |
| --- | --- |
| Subject/assets/runtime | Exact existing Titan, config/tokenizer and installed runtime pins |
| Mathematical path | BF16, eager, eval/frozen base, cache off, passive hooks |
| Dose | Four 131-token forwards, 524 input positions, 316 logical model/load/copy calls |
| Interventions | Mutate starts 128/129/130; protect rows 127, 127–128, 127–129 |
| Causal criterion | Actual protected raw-byte identity and finite zero numeric differences for all six comparisons |
| Baseline | Same original failed bulk logits/hidden fingerprints and target logprobs |
| Historical failed capture | FAILED_AND_RETAINED; maximum 0.03125 and mean 0.01 limits unchanged |
| Resource envelope | 600 seconds, 2 GiB RSS, 1.25 GiB system reserve, 3.3125 GiB pre-worker budget, 4/0.5 GiB GPU, 4 GiB disk, 100 ms watchdog |
| GPU process policy | Existing frozen registry unchanged; relocated-game registry amendment cancelled |
| Prohibited inference | No capture, reader576, training, capability/retention or protected-bank qualification |

No original capture scope or tolerance is weakened. Passing this operational
repair cannot resolve the prefix-versus-bulk discrepancy or identify a GEMM,
attention or kernel mechanism.

## Repair boundary and bounded controls

The sender duplicates its own Job handle with desired access 4
(`JOB_OBJECT_QUERY`), inheritance false and no SAME_ACCESS option, into the
uniquely observed pinned CPython worker after exact Job membership/creation-time
checks. These are explicit access/inheritance parameters of DuplicateHandle;
the resulting handle is valid in the target process. [DuplicateHandle](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle)

GO binds the recipient PID and creation time while retaining contract,
owner/fence and supervised-ancestry checks. Before framework import, the worker
queries membership/limits through that explicit handle and enforces the same
kill-on-close/no-breakaway rule. NULL flags are retained as observations; no
existing Job flags are changed. The temporary child handle closes on acceptance
and PID/time rejection, as well as unconsumed GO rejection, before model work.
This prevents that query reference from keeping the owned Job alive after the
parent closes its last handle. Full Windows/access/handle proof is separately
owned by the operations reviewer.

Read and hash-verified the three actual no-model repair probe receipts against
the revised contract. Valid recipient: EXACT_OWN_JOB_VERIFIED, query handle
unusable afterward, four observed owned launcher/CPython members, complete tree
terminated on parent last-handle close. Wrong PID and wrong creation time:
REJECTED_BEFORE_FRAMEWORK, handle closed/unusable, two observed members terminated
on parent close. Every probe reports zero framework imports, model loads and
GPU calls.

Independent registered-interpreter suite invocation in this worktree:

```text
python -B -m unittest experiments.rineya_centralized_reasoning_system_v1.tests.test_native -v
```

Observed **27/27 passed**, reported interval 2.625 seconds. This includes actual
owned stdlib Windows query/closure/tree controls and existing rejection/accounting
fixtures, not a Rineya/CUDA model. Additional pure CPU rule replay accepts
`0x2000` with actual membership and still rejects `0x3000`, `0x2800` or absent
membership. Correcting the queried object did not relax the rule itself.

## Disposition and interpretation limits

No consequential scientific objection remains for this narrowly changed
pre-framework operational boundary. Conditional diagnostic-only disposition is
bound to the hashes above, after independent operations acceptance and fresh
committed launch attestation. The original failure is preserved and charged;
the successor is a new registered operational version, not an unchanged retry.

Any eventual model run still tests only the three recorded same-shape future-
suffix interventions on one fixed trace. Successful protected-row invariance
does not establish universal causality, prefix/full likelihood equivalence,
native attention at 576 tokens, reasoning gain, Rineya discovery, weight learning,
retention or calibration. Native/source/model checks and current physical
admission remain necessary. CHECK/FINAL and all reader/training/efficacy gates
remain closed. This review launched no native model and altered no primary CPU
or prior native evidence.
