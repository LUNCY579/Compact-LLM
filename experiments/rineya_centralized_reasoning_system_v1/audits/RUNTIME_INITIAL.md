# CRS-001 initial runtime and asset audit

Owner request: pasted-text-1.txt, 2026-10-01. Auditor: Codex engineering agent
`/root/runtime_audit`. This audit made **zero model loads, forwards, generated
tokens or training updates**. It read no protected CHECK/FINAL bank. Historical
instructions and model claims are evidence, not additional authority. The new
owner request permits qualified local work; the founding state remains
`DESIGN_RECORDED_EXECUTION_NOT_AUTHORIZED`.

The machine-readable [asset inventory](model_asset_inventory.json) records full
SHA256 pins for weights, configs, tokenizer files, chat templates, custom code,
CORE provenance, amendments and top-level publication sidecars. We read checkpoint
headers using stdlib only. All six present candidates have 310 serialized tensors
and 596,049,920 parameters. Header metadata does not prove finite tensor contents
or native behavior. Rehash and inspect the exact opened assets before launch.

## Actual assets and candidate distinctions

Read-only external root:
`C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores`.
The worktree contains no weights for these candidates. Its Titan directories are
absent, while the external root contains all three Titan variants. Git publication
metadata and a worktree checkout do not establish local model availability.

| Candidate | Actual weight SHA256 | Storage | Current CRS native qualification |
|---|---|---|---|
| Titan Experimental | `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7` | BF16, 1,192,135,064 bytes | Not performed; historical capture gate failed |
| Titan Sparse | `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284` | FP32, 2,384,234,992 bytes | Not performed; FP32 must be retained |
| Titan Transition | `a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567` | FP32, 2,384,235,008 bytes | Not performed; FP32 must be retained |
| Supa 1.7 | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` | BF16, 1,192,135,096 bytes | Not performed |
| Opallo | Missing at both named roots; historical pin `e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b` | Metadata only | Missing asset, not a negative model result |
| Supa A1 | Same dense weights as Supa 1.7 | BF16 plus custom abstention code | Different serving system; not an independent dense checkpoint |
| Supa A1.1 | `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026` | FP16 plus custom abstention code | Different dtype/system; not performed |

Every present weight hash matches its own CORE record. A1/A1.1 use
`Qwen3AbstainForCausalLM`, custom Python code and an abstention logit bias near
5.15625; plain Qwen3 would change their identity. A1.1's external and worktree
`FILES.json` differ, while their common model/config/tokenizer/code sidecars match.
That difference is retained rather than silently normalized. Routed systems and
unmerged adapters were not selected as dense subjects.

All present candidates share tokenizer JSON SHA256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
and chat-template SHA256
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
Dense Titan and Supa config SHA256 is
`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
Sparse/Transition configs declare FP32 and have separate hashes. The configured
40,960-position ceiling is not a qualified context envelope. CRS has qualified no
native context length for any candidate.

## Historical evidence and its limits

The audit read the charter/package, MODEL_QUALITY_STATUS, published core registry,
central RESEARCH_PROTOCOL, RESOURCE_POLICY, retention requirements, Amendments
001–021, current NEXT_EXPERIMENT, relevant raw qualification receipts and external
CORE amendments. Protected content and historical scored bank payloads stayed closed.

Titan derives from exact Supa 1.7 via the single RRP-015 seed-17 tail-block update.
Historical SELECT-VIEWS acquisition is 234/384 versus Supa 14/384. Arithmetic
69/150 coexists with 8 source-correct losses; the legacy retention monitor fails.
The 162/256 long-context report uses a SPENT suite and includes 11 losses and 12
gains. Titan's CORE amendment attributes much of its apparent long-context
preservation to BF16 export rounding: unrounded FP32 yielded 157/256. These
historical measurements do not establish current BF16 serving qualification.

Sparse retains the largest-magnitude 25% of the same update in FP32. Its historical
232/384 acquisition, 72/150 arithmetic, and 164/256 SPENT long-context report include
6 arithmetic source-correct losses and 7 long-context losses. Its later REP-001
amendment records directionally favorable pruning in three draws but inconsistent
effect magnitude and no fresh long-context confirmation. Casting the sparse bytes
to BF16 would erase part of the defining update.

Transition combines Sparse with 0.25 times a rank-16 boundary delta. Its historical
434/864 transition, 250/336 boundary, and 109/109 answerable-attempt counts coexist
with failed boundary preservation lower bound 0.718 versus registered 0.75. Broader
Supa retention is unmeasured on fresh material. No general candidate dominance is
inferred from these different instruments. Opallo's metadata records strict
long-context regression and posthoc formatting analyses; MODEL_QUALITY_STATUS
corrects claims of intact retrieval and literal abstention. Missing weights prevent
current enrollment.

The central short CE512 result at
`results/headless_unpadded_ce512_v2_20260930_0910/E0_RESULT.json` is historical
`PASS_E0_UNPADDED_CE512_SCOPED`, source `15901919825dc1855020212edcaf6830ee7caef1`:
32 bitwise zero-adapter comparisons, finite confined backward, four disposable
updates and within-stage bitwise resume. It is not sustained accumulation-20
training qualification, capture qualification, broad retention or an E1 permit.
Historical 1024 backward breached the unchanged GPU reserve (0.161270 GiB versus
0.5 GiB). Padding likelihood also failed under its original contract.

The historical short capture receipt at
`results/headless_capture_qual_v1_20260930_1140/CAPTURE_RESULT.json` is
`BLOCKED_CAPTURE_QUALIFICATION`, source `3ed5777167817edd7032f8ed3125afd2e6daa3ad`.
The first fixture emitted three tokens then EOS; selected model log-probability
max/mean errors were 0.0414324105/0.0262993514, failing fixed 0.03125/0.01 limits.
Fixture 384 was unattempted. Geometry localized the discrepancy upstream of the
head without establishing a mechanism. Equal-shape D2 causality and a separately
registered exact-prefix scoring successor remain the next numerical path.
The owned two-update backend qualifier timed out before launching any worker.
Its source-ready injected-Torch backend has no native physical pass.

## Current host and environment

[host_initial.json](host_initial.json) was sampled on **2026-10-01 about 22:53
America/Chicago** (UTC timestamps retained). Physical memory is 31.218 GiB;
available memory in three immediate observations was 5.905754, 5.879616 and
5.855553 GiB. C: free disk was 65,596,243,968 bytes, about 61.09 GiB. These exceed
the historical 3.3125 GiB pre-worker and 4 GiB disk floors at this time; previous
low-RAM readings are not a current block.

GPU: RTX 4070 Laptop, 8,188 MiB total, 1,605 MiB used, 6,344 MiB reported free,
driver 566.07, 33–40% activity. User graphics/game applications are active.
WDDM per-process GPU memory is N/A and some process identities report insufficient
permissions. No named Rineya model worker was observed; this is not proof of
host-wide idleness. Do not terminate user jobs. Reconcile unknown processes,
host-wide lease and actual GPU activity before admission. Current sampling does
not establish sustained fit or process-tree peaks.

Pinned interpreter exists at
`C:\Users\clclc\BusinessProject\.venv-mq001\Scripts\python.exe`:
Python 3.11.15, Torch 2.10.0+cu126, Transformers 4.57.1, safetensors 0.7.0,
tokenizers 0.22.2, psutil 7.2.2. Inventory used package metadata only; it imported
no framework and initialized no CUDA context. GPU driver CUDA advertisement 12.7
is distinct from the installed Torch CUDA 12.6 build.

Native loader sources already exist under the central track:
`scripts/stream_checkpoint.py`, `qualify_native.py`, `qualify_capture.py`,
`qualify_native_backend.py`, `native_training_backend.py`, and Windows watchdog
`supervise_native.py`. The strict fixed-buffer loader supports BF16 into CUDA,
verifies all tensor copies and does not allocate a full CPU model. It cannot
silently be reused for the FP32 variants. Source hashes are in host_initial.json.
Reuse needs a CRS-scoped adapter and a new exact source binding; historical launch
configs retain their central identity and must not be edited to open CRS work.

## Concrete next checkpoint and prospective envelope

Choose exact dense BF16 Titan first for the bounded numerical diagnostic because
the relevant loader and failure reproducer exist. This is an operational selection,
not a claim that Titan dominates Supa. Keep exact Supa 1.7 as separately registered
BF16 control and Sparse/Transition as later FP32 studies with their own envelopes.

After CPU qualification and independent review, register a CRS-owned, non-evaluation
equal-length causal diagnostic for the failed 131-token trace: four forwards,
524 input positions, zero generation/updates, 600-second ceiling, one worker.
Bind current clean committed source, exact external asset hashes, installed model
and masking code, prompts, fixed bitwise causality rule and preserved bulk failure.
Retain 2 GiB process-tree RSS, 1.25 GiB available-system reserve, GPU 4/0.5 GiB,
disk 4 GiB, 64 MiB supervisor allowance, 100 ms external process-tree watchdog,
Windows kill-on-close containment and fresh strict admission. Before dispatch
require two consecutive >=3.3125 GiB observations; after imports require
1.25 + max(0, 2-currentRSS) GiB, with bounded 60-second resource-only wait.
Count all work/failures against owner epoch 2-hour and rolling 24-hour 4-hour caps.
No automatic retry follows timeout or lost response.

If equal-shape causality passes, freeze and independently review an exact-prefix
capture/scoring contract; retain failed bulk equivalence as unqualified. Only a
passed exact reader/capture path can precede the 16-case development diagnostic.
Alternatively qualify Supa in a distinct experiment; it cannot resolve Titan's
failure. No native run is admitted by this audit. Missing host-wide telemetry,
containment/lease source qualification, current source binding and unresolved
Titan capture remain launch blockers. Language grading, 2048/4096 serving and
sustained training remain additional gates before broad retention or training.
