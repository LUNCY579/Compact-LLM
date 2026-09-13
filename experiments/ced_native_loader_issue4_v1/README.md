# RAB-001 issue #4: isolated native-loader qualification v2

**Scope: native tiny-random loader mechanics only. No official-source or improvement claim.**

Task branch: `research/ced-native-loader-issue4-v2`.
Frozen parent: `research/ced-rab001-hotfix001-qualification-v1` at
`e7285e5f68774bc2cc340a92cae7c7963bb569f2`.
Issue: https://github.com/LUNCY579/ChatBubble/issues/4

## Reconcile the issue with the actual remote

Issue #4 describes `f680b5dc62fac36f4b4347793f034215c3580a87`.
The current remote CED descendant is 51 commits ahead of that commit and already
contains `run._materialize_native_buffers`, `run._require_materialized`, their
actual `_load` call sites, and 15 native-loader tests. The original CED branch name
and historical loader-task branch name were absent in the inspected branch list.
No missing branch is recreated, no existing ref is moved, and no evidence is erased.

The inherited `evidence/NATIVE_LOADER_QUALIFICATION.md` and
`evidence/NATIVE_LOADER_MECHANICAL.json` report an earlier Windows run with
109 passes and zero skips. They are retained as historical evidence, **not**
reattributed to this new commit, this Linux runtime, or the expanded 32-case suite.
Other inherited campaign results are not executed or endorsed by this task.

## Repair retained, not reinvented

The inherited real loader constructs native Qwen3 on meta, assigns state tensors
strictly, and ties embeddings. Before `.to(cpu, float32)`, it invokes the native
RoPE initializer to reconstruct nonpersistent `inv_freq`, restores
`attention_scaling` and the `original_inv_freq` alias, and subsequently verifies
both buffers and parameters. Unknown unmaterialized buffers fail closed.

This branch changes **no inherited RAB-001 file**, no forward numerics, tolerances,
attention topology, tokenizer, source identity, protocol, or dependency pin. The
small repair is already present in the selected parent. The isolated contribution
is stronger real-loader regression coverage and a reproducible no-skip supervisor.

## Exactly what runs

The 180-second child process selects only these modules, with pytest plugin
autoload and conftest loading disabled:

| Module | Required cases |
|---|---:|
| Inherited `tests/test_native.py` | 4 |
| Inherited `tests/test_native_loader.py` | 15 |
| New `test_native_issue4.py` | 13 |
| Total required native cases | 32 |

New cases reproduce the original meta-transfer exception by temporarily disabling
only reconstruction inside the actual loader; restore the repair and check the
real return path; exercise FP32/BF16-input save/reload; compare full logits on
left/right-padded batches and one/two-token cached continuation; reject missing,
extra and wrong-shape source tensors; confirm the real official shape validator
still rejects tiny sources; reject dependency drift; and check assigned source
tensors remain unchanged after load and forward.

Fixtures use real safetensors files and a real locally saved tokenizer. The only
shape substitution is `run.source_config` inside explicit tests. There is no
fixture fallback or permissive loader in production. The BF16-input round trip
saves the loader's FP32 result; it does not claim byte-identical BF16 serialization.
Padding comparisons score valid positions only; this is not packed-segment or
arbitrary-mask qualification. Native/reference tolerances are inherited unchanged.

The 20 independent standard-library supervisor tests exercise rejection logic on
synthetic JUnit documents. Their success is **not native-model execution evidence**.

## Receipt integrity and boundaries

`qualify.py` requires the exact requested HEAD, a clean checkout, ancestry from the
frozen parent, and byte-unchanged inherited RAB-001 files. It records Git identity,
tracked-file hashes, installed package versions and native Transformers source
hashes. A post-run identity check, successful pytest exit, exact test-name inventory
and zero skips/errors/failures are required for PASS. Output directories are new,
write-once and outside the checkout. Blocked, failed, timed-out and interrupted
attempts remain distinct from PASS. The wrapper accepts no official source path.

Every receipt leaves official G0/G1, optimizer execution, recovery, DEV, sealed
evaluation, efficacy, GPU fit, deployment and paid-execution permissions false.
The inherited G3-G5 switch remains false. No optimizer or G0/G1 entry point runs.
The qualified object would be only the actual native loader on random CPU fixtures,
not full CED integration, official Supa qualification, model quality, or production.

The dedicated workflow runs only on pushes of this task's Python/workflow files to
this new branch. It uses pinned action commits, read-only contents permission,
CPU-only PyTorch, the inherited exact native requirements and a ten-minute job cap.
It checks out no published model directory and uploads no checkpoint fixtures.
It does not alter billing or use paid/larger runners. A workflow with no runner or
steps is infrastructure-blocked, not an executed suite. Do not automatically retry
such a run or relax pins to obtain a green badge.

## Reproduction in a clean checkout

Use an existing authorized CPU environment, or install the exact inherited
`requirements-native.txt` after installing CPU PyTorch 2.10.0. Then run from the
repository root:

```bash
python -m unittest experiments.ced_native_loader_issue4_v1.test_supervisor -v
python -m experiments.ced_native_loader_issue4_v1.qualify --preflight-only
python -m experiments.ced_native_loader_issue4_v1.qualify \
  --expected-head "$(git rev-parse HEAD)" --out /absolute/new/path/outside/checkout
```

Preflight does not run a model. A missing Transformers installation must report
BLOCKED; it must not be turned into a successful skipped suite. Run receipts apply
only to their recorded HEAD/runtime. A documentation-only follow-up commit does
not change an earlier receipt's tested HEAD.

## Handoff

Do not close issue #4 on the basis of synthetic supervisor tests, historical results,
or a workflow submitted without execution. A fresh 32/32 no-skip native receipt is
the next acceptance boundary. Further CED work remains separately gated and must
coordinate ordinary D40 controls with RineyaArchitecture's canonical native track.
