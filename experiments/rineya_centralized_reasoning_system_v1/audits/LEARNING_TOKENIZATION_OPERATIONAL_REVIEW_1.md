# Learning tokenization operational instrument review 1

Reviewer `/root/runtime_audit`; builder `/root/data_audit`, 2026-10-02.
**REPAIR_REQUIRED â€” ACTUAL TOKEN MEASUREMENT REMAINS CLOSED.** This is the first
source review of the separate CPU token/mask instrument. Existing native/reader/
probe quarantines and owner-exception boundaries remain unchanged.

I read the complete frozen source, tests, contract, both tokenization scope reviews,
Root's process fixture/annotation and causal positional reference/review. I did
not build this instrument. I authored its preceding operations scope assessment;
shared design involvement, model/workspace privileges and earlier pipeline work
limit independence. This is fresh code/interface scrutiny, not a new tokenizer,
process/API, model or semantic measurement. Only this audit report is changed.

## Frozen subjects and permitted validation

| Artifact | SHA256 |
| --- | --- |
| learning_tokenization.py | 2d7ad4a0d752b7ca205530a8672986b3cc3c3152857e0f54715ad3d17c11b3b0 |
| tests/test_learning_tokenization.py | bdb9ad308d5447f44a8a8bd7b44d95f5f634c7d2d9cf3d93c90cb9dbacbb37ae |
| contracts/LEARNING_TOKENIZATION_V1.json | d17a60a72d00ff4f58e47d418b3a1fed85be10ce8cdd3dd4c93f793b9a688abb |

Pins were checked before and after review. All 21 supplied synthetic tests passed
independently in 0.618 seconds, with no skips/errors. Additional controls exercised
production functions with explicitly synthetic process/observer/backend/time
adapters and temporary JSON. No actual tokenizers/Jinja/psutil import, subprocess,
framework/model/tokenizer, current host check, canonical host database, holdout
or protected bank was executed. The tests' temporary files are protocol fixtures,
not native or actual token evidence; none is retained as eligible measurement.

The actual entry flag is False. checked_contract, worker, supervisor and deferred
backend refuse before real imports or Popen. This closure was replayed. The future
backend is implemented: exact local Tokenizer.from_file, pinned Jinja sandbox,
known special IDs, no padding/truncation, renderer parity and CPU measurement
logic exist. The prospective process supervisor/worker also exist. They are not
mere documentation stubs, but the defects below prevent source acceptance and
qualification of that future path.

## Sound source/data and artifact mechanics inspected

load_pairs binds the accepted learning-data receipt and its four fixed bank hashes,
then invokes the accepted immutable registered TRAIN32 builder/validator and exact
ordered input/target pair hash. Public user content contains only the fixed
instruction and source/query/rules; IDs, family, split, private target/oracle and
feedback are excluded. TRAIN gold is intentional response supervision. No new
DEV/CONFIRM/CHECK/FINAL generator or runtime/ledger/model import occurs.

The manual/Jinja subset is frozen; full template newline bytes and declared
training exclusion are distinct. Prefix/whole-sequence ID association, EOS151645,
prompt/header ignore labels, shifted logit ownership and full N versus served
N-1 positions have meaningful synthetic positive/negative controls. Nonfinite,
bool, extra EOS/PAD/start, sentinel, boundary, unsupported size and mask mutations
reject. No truncation/resampling or model/training eligibility is inferred.

Every scheduled parent remains in measurement output, including control failure,
partial encode failure and unsupported fit. Attempt counts include failures;
unknown/unmeasured totals are not fabricated successes. Worker reference hashes
are computed from actual file bytes written by this instrument's canonical UTF8+
newline writer, avoiding the earlier reader's parsed-object checksum assumption.
The data/approval/source/caps and installed asset pins are checked before the
prospective actual path. Review roles are exact non-builder Root science and
Runtime operations. These are cooperative artifact/provenance protections,
not authentication against same-user code/receipt forgery or backend truth.

## Prior Root process fixture is limited evidence

Root's process fixture receipt72c28c0841fdd1c35ff771b75d294d84d786a295a0435365b802bf50823a230d
reports an actual direct base CPython helper, four RSS samples, sampled no descendants,
exit 0 and 0.4433259 seconds. The annotation correctly distinguishes requested 10 ms
sleep from achieved gaps and sampled RSS from absolute peak. That helper neither
exercised this new supervisor deadline/failure implementation nor encoded tokens.
Its separate peak_wset availability observation is not a final-live witness for
this instrument. The synthetic causal reference11a1afda correctly maps prompt 3,
response 2+EOS, full 6 and served 5 positions; it proves positional arithmetic only.

## LT-OPS-001 â€” P1: final-live peak witness precedes result validation

supervise samples process resources at the loop top, then reads/validates the
worker result and writes EXIT_ALLOWED using samples[-1]. No fresh sample follows
that potentially delayed validation. A worker can finish result publication after
the loop-top sample, leaving serialization/post-sample allocation peaks unobserved
when exit is authorized. The source calls this a final live kernel peak witness,
but it is an earlier sample.

Independent production-supervisor synthetic reproducer: begin with the supplied
valid fixture and observer peak 200; wrap only validate_measurement so that after
validation it sets peak_wset 536870913 (one byte above 512 MiB). The worker/process
adapters remain synthetic. Observed return 0, status CPU_MEASUREMENT_RETURNED,
failure None, CPU_outcome_unknown False, savedpeak 200 and EXIT_ALLOWED True. No second
sample sees the breach. This directly challenges the production ordering, not a
rewritten predicate or actual host measurement.

Verification: take and enforce a fresh exact-PID/creation live RSS/kernel-peak and
host-reserve observation after validating immutable worker bytes and immediately
before authorizing exit. Bind that fresh sample/timestamp to EXIT_ALLOWED/result;
reject any post-validation breach/unavailable identity. Preserve actual gaps and
sampled coverage limits. Add the above late-peak/resource-change control.

## LT-OPS-002 â€” P1: a remaining descendant is reported as known FAILED

process_sample rejects unexpected descendants, but discards their identities by
raising a generic error. Supervisor cleanup kills/reaps only the root; unknown
classification checks only absent root/exit code. An observed child may remain,
yet CPU_outcome_unknown False and statusFAILED imply a known ended outcome.

Independent synthetic reproducer: observer.children=[777], with the existing
supplied root adapter. Production supervisor returns 2/FAILED, root is killed,
CPU_outcome_unknown False, no descendant identity/exit proof retained and the
synthetic child list remains[777]. No real child or process is launched.

Verification: durably retain observed descendant PID/creation/identity or explicit
unavailable evidence, and keep CPU outcome unresolved/unknown until all observed
owned work is reconciled. Do not infer child exit from root exit or terminate an
unrelated process. No new containment framework or quarantined helper copy/edit
is required; sampled unexpected descendants must fail honestly and block a repeat.
Add surviving/vanished/reused descendant and cleanup/publication failure controls.

## LT-OPS-003 â€” P1: lost launch outcome does not block another attempt

The supervisor correctly marks an injected lost Popen responseUNKNOWN, with no
observed PID or fabricated zero cost. But output-path nonexistence is the only
repeat guard. There is no durable registered CPU attempt or reconciliation check
outside the chosen new output directory.

Independent synthetic reproducer: inject Popen raising OSError after launch intent;
run again with a fresh output under the same registration. Both returnUNKNOWN and
the patched Popen is invoked twice. This uses no actual process/host database.
Changing the output path can therefore duplicate an unresolved CPU worker and
violate the one-disposable-worker/resource assumptions.

Verification: bind a single registered CPU attempt/idempotency key to exact source,
contract, data and output; retain unknown creation/orphan state independently of
caller-selected directory. A lost response or observed surviving descendant permits
reconciliation only, not another launch. Scope a lightweight CPU-specific durable
attempt guard; do not use/reset the canonical model ledger or expand the campaign.
Replay the two-fresh-output reproducer and prove only one Popen call before trusted
reconciliation. Preserve every failed/unknown case denominator and actual CPU cost.

## LT-OPS-004 â€” P2: immutable actual activation transition is absent

The source-only contract is historically frozen with registration_open=False, and
a supplied closure test requires that False. All actual functions call the one
fixed CONTRACT path and require its flagTrue. Mutating the same closed contract
to open it changes the historical bytes and contradicts its closure test; supplying
a new registration hash alone still reads the old fixed path. The installed backend
body exists, but there is no implemented reviewed immutable activation selection/
transition preserving this closed contract.

Verification: define and test an explicit new immutable actual registration bound
to the frozen instrument/data/asset/dependency/caps/source/review identities while
retaining this source-only record/closure evidence. Require exact activation refs
before imports/launch and reject caller-only dictionaries or unrelated registrations.
Do not silently flip the historic source contract or rename an unapproved path.
No actual measurement is allowed until that transition and source reviews qualify.

## LT-OPS-005 â€” P1: prospective worker entry lacks supervisor admission proof

worker checks an opened contract, parent/root PID+creation fields and source reviews,
but does not bind GO to CPU_INTENT/command/output/registration attempt, restrict its
output to the admitted child, or verify a meaningful EXIT_ALLOWED identity/sample.
EXIT_ALLOWED is accepted solely by existence. Source approval and a parent identity
do not prove that the resource observer/timer actually admitted this worker.

Independent future-path simulation: temporarily substitute the unopened contract
check and actual library/process providers with the supplied synthetic adapters;
retain production worker and actual checked_reviews over temporary source approvals.
A fabricated matching GO plus an arbitrary preexisting EXIT_ALLOWED in an output
outside PACKAGE/outputs returns 0 and writes 32-case worker output. There is no
CPU_INTENT or CPU_SUPERVISOR_RESULT and no resource observer was invoked. All backend
and process identities here are explicitly synthetic, not actual token evidence.
The present False flag prevented any real dependency call throughout review.

Verification: couple child entry and final authorization to the exact durable CPU
attempt/immutable intent+command, admitted output, parent and runtime identities,
source/activation/review references and fresh resource witness. An arbitrary parent
or prewritten marker must reject before backend load. Reuse ordinary subprocess
controls in this narrow CPU scope; do not build another containment framework or
modify the quarantined paths. Test absent/swapped intent/attempt/output, wrong
marker/identity/sample, direct child invocation and legitimate supervised entry.

## LT-OPS-006 — P2: normal permitted exit is misclassified as failure

Root supplied a separate synthetic production-supervisor counterexample, preserved
at evidence/learning_tokenization_round1_cpu_001/NORMAL_EXIT_RACE_REPLAY.json,
SHA406263ab70cc1c2c9bd3f256c6638ce608fb799fb0d6d1004f678c5db4fb5f06.
I inspected and hash-verified that receipt; I did not claim its execution as my
independent control. The first harness setup lacked a default observer PID and
never reached the target; the corrected fixture is explicitly disclosed.

After validated worker output and EXIT_ALLOWED, one poll returns live and the
next observer call raises ProcessLookupError for ordinary process exit. wait then
confirms exit0, the complete worker reference is retained and root was not killed,
yet the production supervisor returns2/FAILED. No actual backend/observer/subprocess
or model ran. This is a completion race, not missing live resource telemetry.

Verification: process disappearance alone cannot prove success. After an admitted
fresh final witness, immutable validated worker result and matching permitted-exit
identity, refresh/reap the exact owned root and accept only known expected exit0/2,
no timeout, no unresolved descendant and successful cleanup. Reject unavailable
observation while still live and preserve unknowns/errors. Add the normal-exit
race control and opposing still-live/reused PID/missing witness/result cases.
This repair is ordinary narrow finalization, not another containment framework.

## Scope and disposition

The fixed TRAIN pairing, syntactic/mask fixtures, real-byte writer references and
actual closed gate are useful source progress. All 21 test passes are retained as
synthetic engineering evidence. Six consequential findings remain in the prospective
actual resource/admission/recovery path, so source acceptance is withheld. This is
one coherent first review/repair finding set; parent must scope bounded next edits
and preserve these exact bytes/test receipt before repair. Scientific review is
separate. Actual tokenizer/psutil/process measurement stays closed until both roles
accept fresh instrument/activation bytes and reliable resource observation qualifies.
No evidence here opens model/learning eligibility or any existing quarantine.
