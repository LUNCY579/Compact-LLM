# SCP-NQ-001: locked first-host qualification boundary

**Current state: native execution is locked in code. Training and confirmation remain closed.**

This is an isolated review successor, not an execution approval:

- Frozen SCP publication: `research/sequential-complementary-skills-v1@3ad2769a9f8343a9697e1720fcdc9efa18c3845a`.
- Existing integration-review parent: `research/sequential-complementary-native-v1@b4f010619290a562c54cdaf3bb6edd4eb9112a91`.
- This review branch: `research/sequential-complementary-native-boundary-v1`.

Neither existing branch is moved, renamed, deleted, or relabeled. The original
112-test artifacts remain historical evidence; they are not this change's test
results. Read this document for the current gate; the earlier README describes
the preserved integration snapshot, not the currently available CLI.

## What changed

`boundary.py` implements an unconditional review lock, a strict **pending-only**
exact-host request format, separate request/forward/token limits, bounded JSON
reading, clean-checkout binding and create-only external output. There is no
unlock flag, approval command, approved example, environment-variable bypass, or
permit consumer. A host-shaped dictionary and its digest are not verified host
observations. A draft cannot assert asset verification or independent host review.

`run.launch` and `run.worker` refuse as their first operation, before importing
supervisor/resource components, reading a grant, creating a lease/journal, claiming
a worker, or doing model work. The package CLI refuses historical `qualify`,
`_worker`, and `propose` commands before importing MQ/RRP or the model stack.
The loader, native backend entry, authorization and generation paths also refuse;
a live-worker forward refuses before charging or calling the model. Tiny CPU
helper construction is not authority to execute a pretrained job or issue a receipt.

The existing exact-Supa-to-Titan tensor materializer is retained. Its dormant
adapter now verifies the **complete generation-config EOS set** against the bound
source sidecar. The decoder stops on every declared EOS ID, not only
`tokenizer.eos_token_id`, and records those IDs with raw generations. It rejects
missing, duplicate, boolean, out-of-vocabulary, or tokenizer-inconsistent EOS
configurations rather than falling back. This is code/fixture evidence, not a
pretrained decoder qualification.

`receipt_review.py` provides a necessary-only screen for the first student result:
three distinct assets, 310 full tensors/220 update tensors, exact state identity,
2,048-token mechanical scope, zero gradients/backward/optimizer/scoring, no
generation, matching source/runtime and bounded actual forwards. Even a consistent
candidate returns `native_pass: false` and explicitly requires the terminal,
owned-worker, journal, host/approval and asset checks. It never creates a PASS or
satisfies the broader SCP training gate.

MQ/RRP implementations, losses, optimizers, checkpoints, source locks, F/P/T
preparation, historical grants, prior evidence, catalogs and model weights are
unchanged. The previous `contracts.py` branch guard and two-stage registration are
also unchanged; **do not rename a branch or edit an old grant to fit this request**.

## Smallest first envelope: student mechanics only

The earlier discussion proposed a combined O/T/REP/MR smoke. Inspection of the
published adapter shows that its real public phases are `student` and `mr`, not a
combined four-path launch. The first-host proposal is therefore deliberately
narrower: **one student loader/tokenizer/cache qualification, with zero generated
tokens**. This avoids inventing a combined runner or silently spending a second
launch. REP/MR, O/T generation and training are not covered by this envelope.

The implemented student probes compare the native prefix with unchanged RRP
serialization, hidden/head logits, short cached/uncached logits, causal isolation,
2,048-token last-logit cache parity, FP32, no gradients, and unchanged state.
The short fixture has at most 64 tokens; the current probe structure requires at
most `2 + ceil(64/3) + 1 + 3 = 28` forwards. The proposed 64-forward ceiling is an
upper spending bound, not a predicted runtime or permission for adaptive probes.

| Proposed first-host limit | Value |
| --- | ---: |
| Worker launches / retries | 1 / 0 |
| Watchdog | 600 seconds |
| Journal worker-time capacity, including inherited overhead | 620 seconds |
| Total owned work | 900 seconds |
| Actual forward calls, including failed attempts | 64 maximum |
| Generation requests / generated tokens | 0 / 0 |
| Training tokens, backward, optimizer, teacher/cloud spend, adaptive rounds | 0 |
| New evidence bytes | 64 MiB |
| Process RSS ceiling | 8 GiB |
| Free disk floor | 14 GiB |
| Qualified sequence shape | 2,048 tokens, batch 1, FP32 eager |

These values are **pending proposals**, not a populated or approved grant. Retain
all unchanged MQ RAM/GPU floors; passing a floor does not prove the maximum shape
will fit. The actual host, assets and pinned native runtime have not been inspected
or executed in this implementation session.

## Safe commands available now

From a checkout of this review branch:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m experiments.sequential_complementary_native_v1 status
PYTHONDONTWRITEBYTECODE=1 python -m experiments.sequential_complementary_native_v1 host-template
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  experiments/sequential_complementary_native_v1/tests_boundary -q \
  --junitxml=../scp-boundary-review-new-attempt.xml
```

Use a new evidence filename on every attempt. `host-template` prints an incomplete
metadata template; it cannot be passed to `draft` while fields remain null. After
collecting actual host metadata, an operator can produce a pending request:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m experiments.sequential_complementary_native_v1 draft \
  --host ../actual-host-observations.json --out ../scp-first-host-request-new.json
```

The checkout must be clean, on the exact successor branch, and descended from the
pinned integration parent. Hidden index flags and in-checkout/symlink output paths
are rejected. Runtime pins are inherited (`torch 2.10.0`, `transformers 4.57.1`,
`safetensors 0.7.0`); the metadata draft does not import or qualify those libraries.
It does not hash model assets, register a budget, or launch a worker. There is no
valid native launch command on this commit.

## Required before the separately approved launch

First complete a fresh **full-checkout** regression and independent adapter review.
The new suite tests cold refusal paths and stdlib contracts; it is not a replacement
for the historical native-helper or SCP/RRP suites. Positive fixture paths that
previously impersonated a live backend must be reconciled explicitly with the new
lock, not counted as native authority or made green by weakening the production
gate. Retain failures and document fixture-only isolation.

Then bind actual source, trained-partial, incumbent, config, tokenizer, template,
generation sidecar and full tensor inventories. Preserve the FP32 config exception
and exact per-tensor `source + M_0.25(trained - source)` construction. Bind the actual
Python executable, physical GPU UUID, driver/CUDA runtime and library builds to the
same clean source tree that will execute.

A separately reviewed source change must connect this **one-launch, student-only**
request to the unchanged Journal/supervisor, enforce the exact phase and physical
host again in the child, and verify settled receipts independently. The current
pending request is intentionally not compatible with the old approval validator.
Only after those changes and their negative tests are reviewed should the user
approve the exact numeric envelope. This publication does not remove the lock,
make that approval, register a native run, or consume a permit.

## Stops and receipt meaning

Before loading, refuse missing approval, wrong branch/source/asset/runtime/host,
unsafe evidence paths, resource refusal or an unsettled earlier reservation. During
an approved future attempt, stop on nonfinite values, parity/cache/causality failure,
state mutation, unexpected precision, any generation, scoring or training activity,
resource/budget violation, watchdog expiry, or evidence-write failure. Retain the
failed attempt and its charges; no automatic retry, fallback precision, timeout
extension, bank substitution, or cleanup of another job's lock is permitted.

A software fixture result is not a native result. A process exit of zero is not a
native PASS. A `screen_student_result` candidate is still unverified. A future
properly verified student PASS would cover only the recorded exact-host inference
mechanics; it would not authorize REP/MR, teacher replication, F/P/T training,
protected/spent-bank reads, independent confirmation, release or quality claims.
No universal preservation, independent skill composition, or efficacy claim follows.

## Evidence scope

`evidence/boundary_v1/LOCAL_TEST_REPORT.json` binds the changed source/test bytes and
fresh scoped test artifacts. The two edited integration modules were first copied
and matched to their GitHub Git-blob hashes. Tests ran in a **partial source tree**
with Python/pytest, not a full repository clone or the pinned native runtime.
Some tests execute extracted source functions without their decorators/dependencies
to prove that refusal is the first operation; these are ordering tests, not complete
module/runtime integration. EOS policy tests execute the actual stdlib helpers;
loader/decoder wiring is inspected structurally, not exercised on pretrained weights.
The unchanged package initializer was separately matched to its remote blob.

No pretrained model was loaded, no native worker was launched, no training or
confirmation was performed, and no GitHub workflow was requested. Remote branch,
commit parent/tree and all changed blob identities must be checked after publication;
there is no self-referential "pushed" assertion in the local test report.
