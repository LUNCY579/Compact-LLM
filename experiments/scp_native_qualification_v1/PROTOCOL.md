# Prospective protocol: SCP-001-NQ-001

## Scientific scope and authority

Question: can one precisely identified Titan-Sparse FP32 incumbent be loaded and
used through the specified supervised, immutable, inference-only boundary?
This is an engineering qualification, not a skill acquisition/retention experiment.
The present release cannot launch or issue `PASS_NATIVE_WIRING_ONLY`.
A request to implement/push this package does not approve future pretrained work.

The three weight-file SHA-256 identities are in `contract.WEIGHTS`, copied from
SCP-001 `SOURCE_PINS.json`. Source is promoted to FP32; the incumbent is
`source + M_0.25(trained_partial_fp32 - source_fp32)` over exactly the 220 tensors
in blocks 8 through 27. `M_0.25` retains floor(n/4) coordinates per tensor by descending
absolute magnitude, with ascending flat-index tie breaks. It is NOT scalar scaling.
No ranks, new checkpoints, teacher, auxiliary model or alternate precision are allowed.

The source and full subject inventory is exactly 310 tensors / 596,049,920 parameters.
The partial inventory is exactly 220 tensors / 314,618,880 parameters. A header alone
does not prove loaded values, finite materialization or tied embedding/head storage.
Before native acceptance, independently check masked-delta reconstruction, unchanged
outside-scope tensors, loaded parameters against the serialized subject, all gradients
disabled, native precision and before/after parameter and asset digests. Do not invoke
`select_tail` to obtain the allowlist: it enables gradients. `assets.inventory(True)`
constructs the names without touching a model.

The only admitted config exception is the source config plus `torch_dtype=float32`.
Any additional on-disk difference blocks review; do not silently loosen this rule.
Tokenizers, generation config and geometry stay source-bound. Bind the actual subject
config file hash independently rather than pretending it is Supa's config.

## First approved native run, once integration is complete

1. Controller verifies the exact committed additive source tree, protocol, assets,
   unscored probe, runtime, independently observed host, fresh one-use authority and
   external create-only evidence location. Acquire unchanged MQ lease. Start a new
   RRP journal reservation. No automatic wait, retry or stale-lock removal.
2. Unchanged MQ `bounded_child` starts one foreground worker; `claim_worker` validates
   its live parent/argv/context before imports or outputs. Recheck exact source and
   host bindings. Use original resource guards with explicit `phase="qualify"`.
3. Load exactly one local-only Titan FP32/eager subject with remote code disabled;
   never rewrite MQ's Supa lock. Disable gradients and verify inventory, precision,
   finite materialization and tied storage. Verify native direct-mode serialization
   against unchanged RRP `serialization.compile_row`, including the closed thinking
   stub and EOS/post-EOS masks. No backward, optimizer or training token exposure.
4. Reuse the inherited forward-only `instrument.qualify_inference_window` at 2048,
   keep=8, with its frozen tolerances and explicit six-forward reservations. This
   step must be inspected and integration-tested before activation; it is not run
   or imported by this release. Do not substitute `native_runtime.qualify`.
5. Execute the frozen probe once in each inherited O, T, REP, MR path. Keep the parent
   prompts/selector frozen; actual candidate-built selector prompts must fit. One
   independent cache per request. No label or scoring interface; no post-outcome
   prompt repair, candidate shortening or answer-dependent retries.
6. Check all logits for finiteness at every decode step; count all EOS IDs from the
   bound generation config; retain raw generated IDs including terminal tokens,
   unstripped nonterminal control tokens, actual input/output/forward counts and
   termination reasons. A malformed selector can be a recorded behavioral outcome,
   never a hidden oracle repair. Finish with unchanged parameters/assets and a
   terminal supervisor receipt. Independently verify evidence before settling PASS.

## Proposed exact envelope — not permission and not a runtime estimate

| Unit | Bound |
|---|---:|
| Worker launches / retries | 1 / 0 |
| Watchdog seconds | 600 |
| Journal worker-time capacity | 620 |
| Total owned-work capacity, seconds | 900 |
| Generation requests | 12 |
| Generated-token allowance, terminal tokens included | 928 |
| Window qualification forwards | 6 |
| Total forward allowance | 934 |
| New evidence bytes | 67,108,864 |
| Context / batch size | 2048 / 1 |
| Training tokens, backward, optimizer, adaptive rounds, RL, teacher/cloud | 0 |

The unchanged Journal start reserves watchdog+20 worker seconds, watchdog+140 owned
seconds and 4 MiB disk. That explains the non-identical watchdog/journal bounds.
Actual observed overrun must be recorded, not clamped. The future controller must
account for its entire owned interval, not merely worker duration.
The 934 forward ceiling assumes at most one model forward per generated token plus
six qualification forwards; any additional native model probe requires a prospective
protocol/envelope amendment, never uncounted work. Serialization is tokenizer-only.

O cap=64; T cap=288; REP and MR each reserve [64,64,64,64,32]. T here is extended direct
inference, not the teacher training presentation. Reserve maximum forwards/output
before each generation, then record actuals separately; no refunds of failed
requests. The unit bridge maps `evaluation_and_qualification_model_calls` explicitly
to forwards in THIS successor, not to generation requests. No existing lane changes.

Preserve preload floors: host 4.6 GiB / GPU 3.0 GiB; runtime floors: host 1.5 GiB /
GPU 0.5 GiB. Preserve the 14 GiB disk prerequisite. A finite process-RSS ceiling and
exact host/GPU/runtime observation must be supplied before approval. Current
`resource_gate` checks supplied values only and does not observe CUDA, cgroups or
disk. The future observer must qualify all of those; floors are not fit guarantees.
No CPU/BF16 fallback, resource-floor reduction or unowned-process termination.

## Stop and receipt rules

Pre-model missing identity/authority/assets/host/runtime/resources -> BLOCKED, with
actual model work stated. No native worker should start from this release.
Wrong precision, value/tying mismatch, nonfinite full-vocabulary logits, parity or
causality failure, asset/parameter mutation, context overflow, budget exhaustion,
watchdog, resource refusal or failed evidence sink -> stop the owned attempt,
retain raw failures and reservations, no automatic retry. A failed adapter is terminal.
Unverified worker exit leaves the journal unsettled. An exit code alone is not PASS.

Receipt types must remain separate:

- `VERIFIED_SOFTWARE_FIXTURE_ONLY`: verified synthetic mechanics and event accounting.
- `ASSET_PREFLIGHT_ONLY`: exact asset bytes/headers observed without model loading.
- `BLOCKED_NATIVE_PREREQUISITE` or `FAIL_NATIVE_QUALIFICATION`: missing prerequisite or
  failed actual attempt, with spending and uncertainty preserved.
- Future `PASS_NATIVE_WIRING_ONLY`: only after a new native verifier binds source,
  approval, host, assets, tokenizer/template/EOS, raw events, journal and observed
  terminal supervisor output, parity and zero-update proofs. NOT issuable here.

Hashes bind bytes, not honesty or authenticated human approval. No receipt from this
package can unlock training/student qualification, teacher replication, scoring,
confirmation, release or efficacy. The spent 256-row long-context suite and embedded
64 unanswerables remain closed and are not accessed for fingerprints or convenience.

## Inherited integration references, unchanged at the pinned parent

- `experiments/sequential_complementary_v1/{README.md,protocol.json,SOURCE_PINS.json,reasoning.py}`
- `experiments/model_quality_gpu_v1/{source_lock.json,native.py,resources.py,supervisor.py}`
- `experiments/reasoning_posttraining_v1/{AGENTS.md,budget.py,serialization.py,native_runtime.py,instrument.py}`

Native worker attachment, live loader/tokenizer execution, exact-window probes and
native receipt-verifier tests are outstanding gates, not hidden successful steps.
