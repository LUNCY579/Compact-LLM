# SCP-001-NQ-001 — Native qualification boundary

**Status: OFFLINE_BOUNDARY_IMPLEMENTED; NATIVE_INTEGRATION_BLOCKED.**

This isolated successor implements the safely resolvable qualification boundary,
not an authorized native experiment. It preserves SCP-001 at
`3ad2769a9f8343a9697e1720fcdc9efa18c3845a` and adds only this directory on
`research/scp-native-qualification-boundary-v1`. No parent file, source lock,
training grant, checkpoint, model catalog, frozen bank or historical receipt changes.

## What actually runs

| Surface | Implemented behavior | What it does not establish |
|---|---|---|
| `contract.py` | Strict prospective request, pinned weights/runtime, zero closed-lane budgets, fixed unscored probe | Human authority, actual host identity, or permission to launch |
| `assets.py` | Streaming SHA-256, bounded safetensors header parsing, exact 310/220 tensor names/shapes, FP32 subject/partial, tokenizer/generation sidecars and config exception | Finite materialized tensors, algebra recomputation, tied storage or loaded-model parity |
| `adapter.py` | Executable fixture host port; independent caches, reserve-before-step, all-logit finiteness, raw terminal tokens, no truncation and terminal failure behavior | A live PyTorch/Hugging Face transport or a qualified native tokenizer |
| `Reservations` | Explicit unit bridge shaped for unchanged RRP `Journal.reserve` | Creation/consumption of a real grant or actual MQ/RRP integration |
| `resource_gate` | Strict validation of supplied RAM/GPU/RSS/time values | An independent host observer or memory-fit certificate |
| `preflight.py` | Clean exact-branch/ancestor/additive-diff checks and optional real local asset-file audit, without importing native libraries | Native host readiness or permit spending |
| `evidence.py` | Create-only external fixture bundles; independent hash, event-order, token, termination and accounting verification | Signed attestation or any native PASS |
| CLI `launch` | Unconditionally refuses before reading a request or creating a worker | No environment switch, JSON permission or fixture PASS can unlock it |

**The live subject loader, MQ supervised worker attachment, native window/serialization
checks and native receipt acceptance remain unimplemented/unqualified here.** This
is deliberate staging, not a completed native adapter disguised as a fixture PASS.
Do not pass a real backend through `native=False`: `HostAdapter` admits only the
literal `ScriptedTransport` and `FixtureTokenizer` types. The original
`sequential_complementary_v1.reasoning.run_case(native=True)` refusal is untouched.

## Review and reproduce

From the repository root, with Python 3.10+ and an existing pytest installation:

```bash
python -m pytest -q -p no:cacheprovider experiments/scp_native_qualification_v1/tests
python -m experiments.scp_native_qualification_v1 status
python -m experiments.scp_native_qualification_v1 draft
python -m experiments.scp_native_qualification_v1 fixture --out ../scp-nq-fixture-001
python -m experiments.scp_native_qualification_v1 verify-fixture --directory ../scp-nq-fixture-001
```

Use a fresh external directory with an existing parent. Fixture mode executes
literal four-entry logit tuples, not even a random neural network. Its twelve
requests exercise the **allowance schedule** only; they do not qualify the inherited
O/T/REP/MR prompts, candidate selection, a real tokenizer or six window-probe forwards.
The 934 forward allowance is an accounting simulation; actual fixture decoder calls
and pretrained calls are reported separately. No model is downloaded or imported.

On a clean checkout of the published successor, optional explicit local-file preflight:

```bash
python -m experiments.scp_native_qualification_v1 preflight \
  --source /absolute/path/to/Supa \
  --partial /absolute/path/to/trained-partial.safetensors \
  --subject /absolute/path/to/Titan-Sparse \
  --out /absolute/path/outside/checkout/scp-nq-preflight-001
```

This reads only the explicitly named model assets/sidecars and Git metadata. It does
not load tensors or read any dataset. It reads the large weight files twice to catch
ordinary in-audit changes. Hash/header validation is not adversarial file-locking;
the eventual worker must reverify immutable assets immediately before and after use.
The report remains `ASSET_PREFLIGHT_ONLY`, even when all file checks pass.
Missing assets or an unrecognized config difference cause refusal, not a fallback.

`draft` leaves the successor commit, protocol/asset/host hashes, RSS ceiling, external
evidence path and approval fields unresolved. Populate these only from the actual
published checkout and independently observed host. `validate-request --request ...`
checks structure, not authenticity. Even `APPROVED_INFERENCE_ONLY` cannot launch this
release. The helper never invokes the inherited training-grant validator and never
sets `training_permission=True` to satisfy it.

## Next native implementation gate

Follow `PROTOCOL.md`. Add and test the live exact-Titan subject transport and
supervisor attachment in a reviewed continuation before changing either native
refusal. Reuse MQ `bounded_child`, `claim_worker`, `lease`, and `guard`, and the RRP
`Journal` unchanged. Do not reuse the historical branch-restricted controller or
`native_runtime.qualify`: that qualifier includes backward passes. No old permit is
consumed. A separately approved, exact-host, one-launch inference envelope follows
that review; missing native integration is not cured by approval alone.

No training, backward pass, optimizer, teacher/cloud call, scoring, fresh confirmation,
protected-bank access, quality claim, publication of a model or branch deletion is
performed. `LOCAL_TEST_REPORT.json` concerns only this package, not a rerun of the
parent's 85 checks and not the user's host. No remote CI result is implied.
