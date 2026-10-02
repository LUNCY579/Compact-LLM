# Learning tokenization operations review 2

2026-10-02. Reviewer `/root/runtime_audit`; builder `/root/data_audit`.
Disposition: **REPAIR_REQUIRED; SOURCE LAUNCH ACCEPTANCE WITHHELD; ACTUAL CPU
TOKEN MEASUREMENT REMAINS CLOSED.** This is the second review of this separate
instrument milestone. The prior native, reader and owned-identity probe
quarantines remain unchanged. This report grants no fourth repair to those tracks.

I inspected the frozen source, tests and contract, their changes from the retained
round1 bytes, the complete first SCI/OPS reports and seven-item adjudication,
both tokenizer scope assessments, accepted TRAIN32 anchors and the root-authored
process/ownership controls. I did not build this instrument. I previously built
operations/native/reader adapters and audited this bank; shared workspace,
privileges, model family and design history limit independence. The present
review used stdlib, supplied synthetic token/observer/timer/launch adapters and
recording hooks only. No real tokenizers, Jinja, psutil, framework or model was
imported; no actual measurement subprocess, host database, tokenizer call,
protected bank, new data, holdout or training was used. No builder file was edited.

| Frozen artifact | SHA256 |
| --- | --- |
| learning_tokenization.py | 40b3a9a63b33ada176d224399fb7f24c63c2fcd00ea6ad963e5e6c1ff994f34c |
| tests/test_learning_tokenization.py | 99566718f3875648f7b262913de07302e1cbe1b2fb5846cc5bb2c7e95bdecb93 |
| contracts/LEARNING_TOKENIZATION_V1.json | 7478376b67506569fc98106bd89a836585fce0de2af3c8d62fd3fc5c92b83a34 |

## Independent checks and repaired first-round criteria

All31 supplied synthetic tests passed independently, with zero failures, errors
or skips, in4.410 seconds under the pinned direct CPython3.11.15 with
`-I -S -B -X utf8`. An earlier independent run of these same bytes passed31 in
4.285 seconds. The later run also exercised the recording-hook reproducer below.
These are source/fixture observations; neither run used the installed tokenizer,
Jinja or psutil. The synthetic reversible chunk encoder is not Qwen BPE.

```text
...............................
----------------------------------------------------------------------
Ran 31 tests in 4.410s

OK
```

The current source substantially closes the seven adjudicated criteria under
those synthetic production interfaces:

| Criterion | Source and independent synthetic result |
| --- | --- |
| Full returned encoding association | Existing returned calls now retain bounded exact integer vectors; validation replays length/SHA, manual/Jinja full-vector parity and joined-training plus recorded EOS/newline suffix. Corrupt counts/hashes, missing/bool/oversized vectors and contradictory association reject without new encodes. |
| Fresh final resource witness | The supervisor validates the worker and then calls `process_sample` again immediately before publishing bound `EXIT_ALLOWED`; a validator-induced peak above512MiB refuses exit and prevents repeat launch. The final sample is the one retained in the receipt and EXIT marker. |
| Unexpected descendants/unavailable identities | Captures listed/fresh PID and creation or explicit unavailable/reused identity. Surviving, vanished and reused synthetic descendants remain unresolved/UNKNOWN after root-only cleanup; no descendant termination or absence claim is invented. |
| One durable CPU attempt | A fixed create-only guard is written before attempted Popen, separately from fixed output. Lost launch responses, cleanup errors and missing final publication block repeat; alternate output is rejected. Same known terminal receipt replays without a second launch or rewrite. |
| Immutable activation preserving source closure | The historical contract remains `actual_measurement_registration_open=False`; its closure API remains closed. Prospective activation is a separate exact-path, byte-bound payload containing fixed key/output/caps, clean-source binding and two exact non-builder roles. No actual activation was created. |
| Coupled child/exit authority | Guard, intent, GO, command, admission, source/activation/reviews, parent/runtime PID and creation, direct executable and initial sample are coupled. Wrong command, parent, intent/guard, stale sample and prewritten EXIT reject. Final EXIT binds actual worker bytes and expected typed exit0/2. This is cooperative artifact association, not adversarial OS authentication. |
| Permitted normal exit versus live errors | Normal disappearance after a fresh authorized final witness accepts only a complete verified worker, expected exact-root exit/reap and no timeout/unresolved/cleanup error. Still-live missing process, other live observer error, reused identity, corrupt worker, unavailable peak and bad terminal metadata reject. |

The production writer's newline is retained in actual file SHA references, and
readers compare exact resolved paths plus physical bytes. The supplied production
supervisor fixtures use this actual writer rather than a canonical-only writer.
Final publication failure leaves the permanent guard unresolved; it does not
create a completed artifact or allow a new output launch. The no-framework and
closed-entry tests continue to refuse before a real dependency or Popen.

The fixed32 accepted TRAIN parents and input/target order remain source-bound.
User messages contain only instruction plus source/query/rules; private status/
value are intentional TRAIN targets, not public prompts. The source retains all
32 failure/unmeasured cases and failed encode attempts. No native/training
eligibility is inferred. Ceilings remain32 cases,184 encodes,one tokenizer load,
60s,512MiB,1.25GiB live reserve/1.75GiB prelaunch,8MiB output and512/64/576 geometry.
A working-set kernel peak and measured polling gaps are disclosed; polling is
not absolute process-tree lifetime coverage. Selected dependencies do not attest
all transitive imports or establish actual backend fit.

## LTK-OPS2-01: worker admission itself spawns forbidden descendants

**Evidence:** `source_snapshot` at source lines360-362 executes two
`subprocess.check_output` calls: Git `rev-parse HEAD` and `status --porcelain`.
`admission` calls it at line463. CLI `main` invokes admission at line822 before
selecting `--worker`, and `worker` invokes admission again at line663, before its
GO/parent checks and backend entry. Thus the prospective actual worker deliberately
creates Git child processes twice. `process_sample` at lines547-562 rejects every
observed descendant and permanently treats that observation as unresolved.

All supplied supervisor/admission fixtures patch `source_snapshot` to a clean
in-memory dictionary (for example tests line331). They therefore bypass the
worker's actual child-process-producing integration path. Passing31 fixtures
cannot qualify the advertised child-free actual entry.

**Executed stdlib production-call reproducer:** use the supplied temporary
synthetic registration helper, mock only `source_contract` to the loaded frozen
configuration, and replace `subprocess.check_output` with a recording hook. Keep
production `source_snapshot`, `admission` and worker admission intact. After the
first admission, call worker with a deliberately wrong `sys.executable` so it
stops immediately after its second admission and before real psutil/backend
imports. Temporary activation/review files are synthetic and outside canonical
outputs; no actual registration was created.

```python
calls = []
def observed_git(command, **kwargs):
    calls.append(list(command))
    return 'synthetic\n' if command[1] == 'rev-parse' else ''

cfg = token.read_json(token.CONTRACT)
with tempfile.TemporaryDirectory() as directory:
    package = Path(directory)
    with patch.object(token, 'PACKAGE', package), \
         patch.object(token, 'source_contract', return_value=cfg):
        args = tests.TokenizationTests().fixture_registration(package, cfg)
        with patch.object(token.subprocess, 'check_output', side_effect=observed_git):
            token.admission(args)
            assert len(calls) == 2
            with patch.object(token.sys, 'executable', 'synthetic-unregistered.exe'):
                try:
                    token.worker(args, cfg)
                except token.TokenizationBlocked as error:
                    assert str(error) == 'direct pinned base CPython required'
                else:
                    raise AssertionError('must stop before real library import')
            assert len(calls) == 4
assert not {'tokenizers', 'jinja2', 'psutil', 'torch', 'transformers'}.intersection(sys.modules)
```

Observed recording output:

```json
{"actual_backends":0,"actual_measurement_processes":0,"recorded_synthetic_check_output_calls":[["git","rev-parse","HEAD"],["git","status","--porcelain"],["git","rev-parse","HEAD"],["git","status","--porcelain"]],"worker_stopped_before_real_import":"direct pinned base CPython required"}
```

**Impact:** an authored Git descendant violates the no-descendants admission
policy. If sampled, it blocks measurement and consumes the single attempt as
UNKNOWN; if transiently missed, the path still spawned a descendant and cannot
support the child-free claim. This review does not claim a real process was
observed, a real attempt failed, or a Git child would always outlive a poll. The
executed counterexample establishes the actual call graph, which suffices to
withhold source launch acceptance before actual CPU measurement.

**Narrow verification criterion:** retain the supervisor's clean-commit/dirty
observation and bind that observation into the immutable activation/admission/
intent/GO. The actual child entry must independently verify its immutable code,
data, contract, activation, reviews, command and parent/resource proof using
bounded file reads, without Git or any other child-side subprocess. Preserve the
strict descendant rejection, no new containment framework or executable registry,
one attempt, all caps and all denominator/unknown/publication gates. Add a
production child-entry integration fixture that keeps its actual source-check
path and makes every child-side subprocess call fatal; do not hide that path by
patching `source_snapshot` to a clean dictionary. Retain meaningful parent dirty/
commit and child code/data/activation drift negative controls. No additional token
encode, asset, bank, model, host ledger or authority is required.

## Disposition and continuation

This is the complete consequential OPS2 finding set from this review. The known
seven first-round failures are substantially repaired under synthetic controls,
but LTK-OPS2-01 withholds exact-byte source launch acceptance. The separately
registered future actual path is implemented rather than only a permanently
closed stub, but its current descendant-producing worker entry cannot yet be
qualified. Permanent source closure and absence of an actual activation continue
to block all real measurement.

Parent adjudication may authorize one coherent final third source candidate within
this milestone's existing cap, after preserving these exact bytes and this
reproducer. I have not changed builder files, widened authority or independently
approved that future candidate. A fresh exact-source SCI/OPS review and immutable
activation remain required before any actual CPU artifact creation. CPU token/mask
measurement, if subsequently admitted, would establish only this fixed TRAIN32
byte/ID/ownership/fit/cost prerequisite; it would not reopen native, reader, probe
or training qualification, measure learning, or resolve historical bulk numerics.
