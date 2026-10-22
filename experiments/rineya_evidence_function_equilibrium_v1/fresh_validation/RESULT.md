# EFE-FUNC-001 first-stage preparation — HOLD

September 29, 2026, America/Chicago. Working branch:
`codex/efe-func001-fresh-validation`. Origin was fetched from the original checkout;
the EFE ref remained `c934589026f7456352d62a83bb102c453e15519a`. The requested separate
worktree already existed at that commit and was reused. Original checkout changes and
all sibling worktrees were preserved. No root/ancestor/target AGENTS existed; the isolated
MQ package's continuation rules were read when its loader was reviewed.

## Decision and exact blockers

**HOLD. Neither preservation nor acquisition was tested on fresh model outcomes.**

1. The original T endpoint SHA-256 `07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d`
   and H endpoint `a1103060591a91ee7d9ceb2ad7450621be352b087cd19188f2c383a0dbbc7b90`
   and their source training checkpoints are absent. September 26 deletion receipts
   classify both as Tier A, **NOT rebuildable**; recovery requires original vault bytes.
   A bounded records/HF-cache search found no local object. A later live check verified
   that the pinned vault is accessible and both LFS hashes/sizes match. A T restoration
   attempt was stopped on host RAM pressure; no complete endpoint was recovered.
2. No fresh canonical-parent-disjoint bank, complete semantic/template exposure audit,
   or qualified row semantic scorer is registered. Saved endpoint training IDs exist,
   but they do not establish all training/selection/exposure exclusions. Protected raw
   payloads were not reopened. Old selection banks and RCI confirmations remain spent.
3. Native reconstruction/loaded-parameter parity, tied weights, autonomous generation,
   cached/full parity and the maximum resource envelope are **NOT_RUN**. The new adapter
   public entry refuses even a caller-flipped launch flag. A future bounded collector
   and process supervisor are still needed; package metadata and CPU fixtures cannot
   substitute for target-host qualification.
4. Fresh resource readings reached **147,001,344 bytes (0.137 GiB) available RAM**,
   below the 4.6 GiB admission floor, with both GPU locks absent and about 6.64 GiB GPU
   memory free. After stopping only the owned restoration process tree, OS free physical
   memory was about 4.07 GiB, still below the loader floor. No native admission was attempted.
5. The fixed statistically sensitive design must fit a measured bounded native budget.
   Expected inference cost remains unestimated because the missing endpoints cannot be
   reconstructed locally. Do not reduce sample size/margins or precision to create a pass.

The original anchor `962abc5b...` and all six configuration/tokenizer sidecars were
hash-verified; its stored header has 310 BF16 tensors/596,049,920 elements. The current
Python 3.11.15 environment has Torch 2.10.0+cu126, Transformers 4.57.1, safetensors 0.7.0,
tokenizers 0.22.2, NumPy 2.4.6 and psutil 7.2.2. These are compatibility metadata;
native qualification remains unearned.

## What was completed

- [ASSET_AUDIT.md](ASSET_AUDIT.md) and `asset_audit.json`: exact source/sidecar/endpoint
  lineage, deletion/reconstruction evidence, bounded local search, original arithmetic,
  complete header/key/shape inventory and provenance hashes.
- `vault_availability.json`: current pinned private-vault revision/LFS metadata, corrected
  interpretation of the replacement-weight restriction, and the one aborted original
  restoration. The saved metadata points to an accessible recovery route; it does not
  replace hashing restored file bytes. The incomplete payload is retained externally.
- [DATA_AUDIT.md](DATA_AUDIT.md) and `data_audit.json`: data roles, spent/quarantine
  restrictions, exact training draws, semantic/format confounds and endpoint sensitivity.
- [NATIVE_REVIEW.md](NATIVE_REVIEW.md): explicit source-blob review of qualified loader
  patterns and unqualified historical generation; no runtime sibling imports.
- [PROTOCOL.md](PROTOCOL.md) and `protocol.json`: frozen five-arm design, zero composite
  selection budget, separate semantic/format/termination, 5pp acquisition/3pp preservation
  margins, exact paired confidence methods, sensitivity, exclusions, bounded costs and
  proceed/hold/reject rules. Bank reservation has no payload/admissibility claim.
- `native_adapter.py`: local-only exact asset/runtime binding, streamed original INDEP
  arithmetic helpers and private single-model CPU integration, full-KV greedy mechanics,
  fixed EOS and overflow handling. All native/CUDA launch remains disabled.
- `resources.py`, `statistics.py`, `preflight.py`: both lock conventions, owned-token
  cleanup, fail-closed memory/process inventories, reproducible exact paired bounds and
  three-way decisions, and create-only external model-free receipts.

The data audit's initial prospective recommendation allocates .05 to a one-direction
PROCEED family. The final protocol strengthens this to **.025 each for false PROCEED and
false REJECT**, total .05 over ten gates in each direction; final sensitivity is in the
external statistical receipt. The final five arms distinguish full H from H025 as the
matched second-component comparator. These design decisions were made before any fresh
candidate outcomes, and no bank has been read adaptively.

## Verification and reproducibility

Twenty-two focused CPU tests passed under `.venv-mq001`: arithmetic-order mutation,
rank/vector semantics, endpoint precision/key/shape/nonfinite rejection, frozen recipe
binding, lazy/closed native entry, full-KV/EOS/overflow mechanics using a fake model,
resource refusal/lock rollback/foreign ownership, exact statistical tails, paired damage,
and unresolved/missing-limb decision behavior. No real model was constructed and no CUDA
model worker was launched. Native resource/precision/generation qualification is absent.

From this worktree root:

```powershell
$efePython = 'C:\Users\clclc\BusinessProject\.venv-mq001\Scripts\python.exe'
& $efePython -m unittest experiments.rineya_evidence_function_equilibrium_v1.fresh_validation.test_preparation experiments.rineya_evidence_function_equilibrium_v1.fresh_validation.test_native_adapter -v
& $efePython -m experiments.rineya_evidence_function_equilibrium_v1.fresh_validation.preflight --out 'C:\Users\clclc\BusinessProject\rineya-efe-func001-records\preparation-20260929-001'
```

The preflight is create-only; a rerun needs a new output directory and a meaningful
reason. Expected exit code **2** reports HOLD, not a test crash. It only reads assets,
metadata and hardware inventories; it imports no Torch/Transformers and launches no model.
No unchanged execution retry is needed.

External receipt directory:
`C:/Users/clclc/BusinessProject/rineya-efe-func001-records/preparation-20260929-001`.
It records current RAM/VRAM/disk and both lock states, protocol/code/audit hashes and
code commit, exact statistical sensitivity, zero paired results, zero native calls,
zero generated tokens and zero paid cost. One original restoration attempt left a
670,673,638-byte incomplete logical payload; actual network bytes were not instrumented,
and no replacement weights were transferred. CPU test receipt 001 sampled only the small
Windows Python launcher; receipt 002 corrects accounting to include the interpreter tree.
Test output/elapsed-time/RSS accounting is retained externally. Guard prerequisites do not establish representative-
shape memory fit. All inference throughput, prefill/decode, KV cache, native peak-memory
and quality measurements are **NOT_RUN**, not zeros presented as measured model behavior.

No independent confirmation, model promotion, third skill, training, JVP fitting,
KV compression, DEQ, or CLARE experiment is justified or started. Feasible preparation
is finished; actual validation holds until the exact prerequisites are resolved.
