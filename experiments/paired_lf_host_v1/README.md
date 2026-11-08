# PLF-HOST-001: paired B28 and E24/D4 frozen feature capture

**Native execution unmeasured; this package adds no backbone optimizer.**

Intended new branch: `research/ced-looped-flow-paired-host-v1`.
Pinned parent: `research/ced-rab001-hotfix001-qualification-v1@a36de96b30eb03265db3a18e464a2e2a23ee67bf`.
No inherited file, original qualification gate, checkpoint, model registration or
research branch is changed. The companion RineyaArchitecture package
`experiments/paired_lf_v1` supplies the same reader/optimizer/scoring to both hosts.

## Exact host construction

Use only the official ungated Supa 1.7.0-devfreeze assets accepted by inherited
`source.audit_source`. Its model SHA-256 is
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
The native loader is the pinned `run._load`, including its nonpersistent RoPE-buffer
repair. It is not replaced with from_pretrained fallback, a custom tokenizer, or
an unverified second loader. All inherited Python/contract files are checked against
the exact parent Git object before import; untracked inherited Python is refused.

B28 is the unchanged ordinary reference. CED_LAST4 is created through inherited
`to_ced` with encoder_layers=24, decoder_layers=4 and donor_layer=24. Residual width,
FFN width, GQA/QK/RoPE dimensions, tied embeddings and tokenizer remain unchanged.
No FFN growth, width pruning, memory augmentation, D40 donor or cross-scale stack
is introduced. Both models start from the same source, not separately tuned checkpoints.
The source-to-E24/D4 conversion is not function-preserving. Its measured baseline
must be reported, not concealed by selecting a favorable checkpoint.

This small conversion was chosen to avoid combining the harsh 14+14 recovery problem
with the first reader test. It is not a result on the legacy C28 arm. Later variants
need explicit registrations and fresh paired captures.

## What gets captured

At each problem-only prompt: a memory tensor after encoder cut 24, the host's own
final normalized query, and its native logits restricted to the seven A-G tokens.
A-G must each encode/decode as exactly one native token. Identical input IDs and
answer-code IDs are required across hosts. Labels never enter tokenizer/model calls.
The data generator/oracles are shared with the paired framework.

Only full-prefix FP32, CPU, batch-one, no-padding/no-truncation, at most 512-token
inputs are supported. No KV cache is created or reused. On two calibration prompts,
ordinary reference versus native Qwen logits must pass finite elementwise
1e-4+1e-4*abs(reference) tolerance and exact argmax. This is a bounded loader check,
NOT a replacement for the original comprehensive G0/G1/P1 qualification.

Encoder memory may be byte-identical between B28 and CED_LAST4 because the encoder
prefix is unchanged. The final query and prior give the actual host distinction.
The consumer explicitly checks this and refuses wholly indistinguishable host pairs.

Host weights are frozen and hashed before and after every capture arm. Native source
assets and parent code are reverified after capture. Per-row safetensors, input-ID
hashes, candidate mapping, host configuration, conversion lineage and weight digests
are retained. Partial captures lack CAPTURE_COMPLETE and cannot authorize a screen.

## Runbook

Use new clean worktrees and a fresh environment. Add BOTH repository roots to
PYTHONPATH; no pip install of the research source is required. On POSIX shells:

```sh
export PYTHONPATH="/absolute/RineyaArchitecture:/absolute/ChatBubble"
```

On PowerShell use a semicolon-separated `$env:PYTHONPATH`. Native collection requires
the inherited Transformers 4.51.3 pin and its requirements, plus the companion
framework requirements. Do not change a live historical training environment.

Commands below may be run from the ChatBubble root; every destination must be new
and outside source/model trees:

```sh
python -m experiments.paired_lf_host_v1 preflight --source /exact/official/core --out ../plf-runs/preflight
python -m experiments.paired_lf_host_v1 permit-template --source /exact/official/core --out ../plf-permits/capture
```

Preflight performs no weight load. Missing assets/runtime or parent files produce
an honest BLOCKED record. A successful preflight is only metadata readiness.
The permit is deliberately unapproved. Only after real separate authorization,
fill approved/approved_by/authorization_reference; leave every bound field unchanged.

```sh
python -m experiments.paired_lf_host_v1 capture --source /exact/official/core \
  --permit ../plf-permits/capture/permit.json --out ../plf-runs/capture
```

Capture requires 10 GiB free host RAM and 1 GiB free disk. A supervising process
enforces an 1,800-second wall cap and observes a 10-GiB process-tree RSS ceiling.
Sampled RSS can miss transient peaks; this is not an OS reservation or a fit guarantee.
CPU is explicit: there is no CUDA fallback, paid service, scheduler or workflow.

The result lives in `capture/bank`; logs and the receipt live outside that bank.
The RineyaArchitecture screen accepts only a complete validated paired bank and
its own explicit permit/qualification. No original RAB gate, sealed evaluation,
full decoder training or general-language improvement is licensed by these artifacts.

## Tests and delivery boundary

The new host-adapter tests use an independent shape fixture to check hooks, restricted
logit extraction, immutability, tokenization boundaries and failure handling. They
are NOT native random-Qwen tests or executions of the inherited CED constructor.
The inherited full suite and native captures must be run in an appropriate checkout.

Remote publication was unavailable during this delivery: installed GitHub actions
were read-only and direct Git DNS resolution failed. The companion bundle includes
additive source patches and a guarded apply-on-new-branch script. Do not label those
patches as already-pushed branches. Historical evidence and existing refs are preserved.
