# Official CED construction v1: G1 mechanics only

This additive entry point closes the **missing command** milestone recorded at
`ChatBubble@2f65f9e96f8649a86c5d2f4369181df5b8b21aa5`. It does **not** close the
missing official-weight CED execution milestone. No C28/C40 official PASS receipt
was produced while implementing this change. The existing README and
`evidence/OFFICIAL_GATE_LADDER.md` describe the parent implementation; their
historical receipts and limitations remain intact.

Task branch: `research/ced-official-conversion-v1`, based on
`research/ced-native-loader-qualification-v1` at the commit above. No existing
runner, tensor primitive, source lock, protocol, receipt, workflow or tag is changed.

## Fixed routes and the C40 boundary

| Arm | Immediate ordinary donor | Target | Shared-memory donor (zero-based) | FFN growth | Parameters |
|---|---|---|---:|---:|---:|
| C28 | Official, unmodified B28 after fresh G0 | 14 encoder + 14 decoder; width 1024; Q/KV 16/8 | 14 | 3072 to 3392 | 596,311,424 |
| C40 | Exact saved, unrecovered D40 from fresh G1_WIDTH | 20 encoder + 20 decoder; width 896; Q/KV 14/7 | 20 | 2496 to 2752 | 594,279,808 |

Both preserve the registered head dimension 128, vocabulary IDs, tied embedding
readout, dense GQA and causal shared-memory design. The existing `to_ced` primitive
is reused unchanged. No arbitrary donor, donor search, alternate shape, G2 checkpoint,
stock Qwen fallback, or runtime/source override flag is admitted.

**C40 here is an unrecovered mechanical initialization, not the scientific recovery
candidate.** Its purpose is to qualify construction, lineage, serialization and
causal/cache behavior without needing to open G3 to build an artifact. The future
scientific C40 study still requires an interpretable recovered ordinary donor and
matched D40 continuation. This implementation does not accept a recovered checkpoint
or waive that requirement. A later reviewed runner and preregistration must establish
those controls; this artifact cannot be quietly promoted into that study.

## Fresh prerequisites, not historical-receipt migration

The package's existing `implementation_digest()` includes every package Python file,
including tests. These additions therefore change it. Run fresh G0 on the completed
checkout in the intended pinned runtime; do not edit old receipt contexts or hashes.
For C40, also rerun G1_IDENTITY and G1_WIDTH with the same checkout, runtime, source,
pilot and declared calibration data. The complete parent chain is validated, including
required named checks, required artifacts and parent digests, before any donor loads.

The checked-in G1_WIDTH receipt does not include its large weight artifact. It cannot
substitute for the **complete local run directory**. Every bound artifact must exist
and hash correctly. G2 is neither needed nor accepted as a CED prerequisite.
Historical Windows-keyed implementation digests and platform/runtime bindings are not
normalized or grandfathered by this change. Hashes detect drift, not malicious forgery.

From the repository root, in the existing pinned native environment with the exact
registered official checkpoint already available locally:

```bash
S=Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
P=experiments/architecture_benchmark_001
O=/path/to/new/ced-conversion-runs

python -m experiments.architecture_benchmark_001 audit --source "$S"
python -m experiments.architecture_benchmark_001 qualify-source \
  --source "$S" --device cpu --out "$O/g0"

# C28 construction needs only the fresh G0.
python -m experiments.architecture_benchmark_001.ced_conversion \
  --arm C28 --source "$S" --device cpu \
  --g0 "$O/g0/G0.json" --out "$O/c28"

# C40 needs the complete fresh ordinary conversion chain, not G2.
python -m experiments.architecture_benchmark_001 qualify-identity \
  --source "$S" --device cpu --g0 "$O/g0/G0.json" --out "$O/g1-identity"
python -m experiments.architecture_benchmark_001 calibrate-d40 \
  --source "$S" --device cpu --calibration "$P/data/calibration.jsonl" \
  --g1 "$O/g1-identity/G1_IDENTITY.json" --out "$O/g1-width"
python -m experiments.architecture_benchmark_001.ced_conversion \
  --arm C40 --source "$S" --device cpu \
  --g0 "$O/g0/G0.json" --g1-identity "$O/g1-identity/G1_IDENTITY.json" \
  --g1-width "$O/g1-width/G1_WIDTH.json" \
  --calibration "$P/data/calibration.jsonl" --out "$O/c40"

python -m experiments.architecture_benchmark_001 status
```

Every stage requires a new output directory. Use the same `--pilot` across stages
when overriding the default. Existing source and prerequisite directories are protected.
There are no downloads, device fallbacks or overwrite switches. Changing runtime or
code after G0 requires fresh qualification. The supplied calibration remains synthetic,
human-review-pending engineering data; this construction step does not upgrade its
scientific status. There is no 8-GB fit promise; full-size FP32 copying/hashing still
requires substantial RAM/VRAM even though donors are released before reload.

## What a new G1_CED receipt actually checks

The runner verifies each output tensor against its immediate ordinary donor, including
shape, dtype, device, finiteness, byte hash and storage independence. It records every
source/target tensor name, tensor hash and shape. FFN gate/up extensions use explicitly
recorded cyclic row indices; added down-projection columns must be exactly zero.
Memory K/V, K normalization and input normalization use the fixed first decoder donor.
The source state is hashed before and after conversion; actual parameters must match
the registered count. FFN growth preserves that FFN's initial function, **not** the
changed full-model memory topology.

Answer-free probes use the fixed eight token IDs `(1, 7, 11, 3, 9, 5, 13, 2)`.
They check finite logits, the actual tied-embedding readout, future-token isolation,
physical causality with repeated rotary positions, left-padding and packed-document
isolation, full/chunk/token cache parity, tail-only prefill and continuation, foreign
and stale cache rejection, absence of decoder-private KV, and actual shared-bank
shapes and persistent KV bytes. Parity uses fixed `atol=rtol=1e-4`; save/reload state,
configuration and probe logits must be exact. These are bounded mechanical probes,
not exhaustive causal verification, native CED parity or long-context qualification.

No CED answers are decoded or graded. There is no optimizer, backward pass, teacher
loss, training dataset consumption, DEV/TEST access or quality endpoint in the converter.
The existing G0 still has its previously registered open engineering pilot.

## Artifacts and permissions

A successful output contains:

- `config.json` and `model.safetensors`, using the existing custom research format;
- `lineage.json`, with full checked tensor mapping, FFN indices, donor/target state
  hashes and explicit `function_preserving: false`, `recovery_required: true`;
- `provenance.json`, with the exact audited official assets, input file hashes,
  validated parent receipt snapshots, native-to-reference keys and immediate donor hash;
- for C40, a byte-exact `ordinary_lineage.json` snapshot preserving original residual,
  complete GQA-group, per-layer FFN and identity-insertion selections;
- `G1_CED_C28.json` or `G1_CED_C40.json`, binding all output artifacts and its parent.

Parent snapshots are provenance, not independently usable upstream receipts. The
original parent artifact directories remain necessary for future revalidation.
Source assets, all input files, full parent artifacts and implementation/runtime
context are rechecked before issuing PASS. Failures after creating a fresh run
directory retain `failure.json`; failed check predicates also retain a FAIL stage
receipt. Preflight rejection does not create an output directory. CLI exits 0 on
PASS and 2 on blocked/failed execution.

Every CED receipt explicitly sets `licenses_CED_G2`, `licenses_registered_training`,
`licenses_quality_evaluation` and `licenses_efficacy_claim` to **false**. The existing
ordinary `smoke` command rejects a CED receipt. G3/G4/G5 remain unconditionally closed.
The saved artifact is not a deployable HF/GGUF model and carries no quality, recovery,
reasoning, latency, total-memory or performance claim.

## Evidence for this implementation

Local CPU execution: **87 passed, 0 failed, 0 skipped** for the explicit selection:

```bash
python -m pytest -q \
  experiments/architecture_benchmark_001/tests/test_reference.py \
  experiments/architecture_benchmark_001/tests/test_ced_conversion.py
```

This is 30 existing reference-model tests plus 57 new conversion tests. The new
end-to-end tests substitute the official-source/native-runtime boundaries and model
sizes with tiny fixtures; conversion, strict receipt validation, parent chaining,
serialization, probes and negative cases execute production code. Full registered
C28/C40 parameter counts are additionally checked on metadata-only models.

The selected run is **not** the full historical 109-test package suite. Transformers
is absent here, so neither native test module was collected; zero skips in the
selected tests must not be read as native qualification. No official checkpoint was
loaded, no full-size CED was constructed, and no new official G0/G1/G2/CED receipt or
remote CI result is claimed. See `evidence/CED_CONVERSION_MECHANICAL.json` and its
bound JUnit artifact for exact runtime, source-file identities and collection scope.
Before official construction, run the complete package suite in the pinned native
environment and then execute the fresh source-bound ladder above.
