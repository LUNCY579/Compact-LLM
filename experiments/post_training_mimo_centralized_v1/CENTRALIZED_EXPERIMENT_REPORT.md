# Centralized Rineya Titan 0.6B: laptop execution revision

30 September 2026, America/Chicago. **The inherited 7.4 GiB admission guard was
not measured Titan RAM demand. Exact GPU loading and frozen inference have now
run successfully on this laptop, with observed process-tree RAM peaks below
2 GiB. The complete 512-token CE mechanics trial now passed all eight checks,
four disposable updates, bitwise save/resume and unchanged-base verification.
Its observed process-tree RAM peak was 1.7486 GiB. The earlier 1024 VRAM failure
remains preserved. The observed RSS margin is only 1.39 MiB; sustained E1 fit,
acquisition and retention remain unmeasured. No recipe is promoted.**

## Exact primary student and source

Primary is Rineya-Titan-Experimental-1.0.0-devfreeze: 596,049,920 unique parameters,
310 serialized BF16 tensors, 28 layers, hidden size 1024, MLP 3072, 16 query heads,
8 KV heads, head dimension 128 and vocabulary 151936. Native loading verified the
embedding tie and exact copied values. Declared context 40960 is not a qualified
operating envelope. The static BF16 payload is 1,192,099,840 bytes, or 1.110 GiB.

Weight SHA256: `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
Config SHA256: `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
Tokenizer SHA256: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
Template SHA256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.

The exact local source remains the published Titan devfreeze checkpoint in
Chat-Bubble/Rineya_Published_Cores. The dedicated execution branch is
research/post-training-mimo-centralized-v1, descended from verified parent
`03db218ceebc61a2bc04f2482cf7e094045e01c1`. No parent/main merge or history rewrite
occurred. Historical 344M evidence remains preserved and outside primary training.
The first two native runs used `79ad1a61ba1bb16b93ccc9f0cd27c0e22b9bb00c`.
The tested streamed implementation was committed at
`4526d705` (abbreviated; full identity is retained in its run receipt and Git).
The measured unpadded trial used full producer
`a4454e17009c9d3aadd87e2b9b633c2969f5be00`. The 512 successor native source is
`d5640306dac3ea33b862e77bb1eb932a980a9e4f`, with evidence checker at `431a9deb`.
The complete CE512 trial used producer
`15901919825dc1855020212edcaf6830ee7caef1`; its actual evidence checker passed.
Final documentation and launcher identities are provided by the delivery commit.

## Why the RAM policy changed

The user explicitly authorized changing the RAM requirement and plan for a 16 GB
laptop, with priority given to VRAM-heavy work. Source audit traced the old 7.4/6
GiB available-memory floors to PRLC project defaults for a GPU model plus frozen
CPU reference. The central loader has no full CPU reference: it constructs BF16
weights directly on CUDA and hashes through small chunks. Earlier admission stops
therefore established compliance with a policy, not a measured requirement for
7.4 GiB free RAM or a finding that Titan cannot fit.

Rank-8 q/v adapters contain 1,146,880 FP32 parameters: 4.375 MiB. Parameters,
gradients and two Adam moments total approximately 17.5 MiB, primarily on GPU.
Activation and framework peaks need measurement for each operating envelope. Quantization or CPU
optimizer offload has not been added to this first route.

AMENDMENT_009 registered a bounded pilot with 2 GiB system reserve and 2.5 GiB
worker RSS cap. Admission reserves all remaining permitted worker growth rather
than a fixed 7.4 GiB buffer. After actual loading succeeded, AMENDMENT_010 separated
backward/resume memory measurements from the unresolved padded-forward/sampler
qualification. The numerical tolerances remain frozen.

AMENDMENT_011 replaces mapped CPU tensor access with a fixed 2 MiB read buffer.
It validates the BF16 header, offsets, shapes and complete payload coverage, copies
synchronously to GPU and checks every loaded tensor hash. Its named profile keeps
1.25 GiB available system RAM and caps process-tree RSS at 1.75 GiB. Admission is
1.25 + max(0, 1.75 - current RSS) GiB: a 3 GiB before-worker budget. These are
prospective trial bounds. A later admitted trial copied all 310 exact GPU tensors
and completed 256/512 backward and optimizer fixtures within the RAM cap. The
complete 1024 scope failed its VRAM floor. A separately registered CE512 trial
subsequently passed complete save/resume and frozen-base verification.
External 100 ms monitoring, Windows containment, GPU 4/0.5 GiB floors and disk
4 GiB floor remain active. No user application was terminated.

## Native results and remaining limits

| Attempt | Measured outcome | Peak process RAM | Interpretation |
|---|---|---|---|
| Load pilot | Exact load and finite 256-token forward; 10.406 s worker /11.219 s supervisor | 1.928 GiB | Load/inference scope only; CUDA allocation peak 1.400 GiB |
| Full laptop v1 | Exact load and 32 zero-adapter cases, max difference 0; padding check failed at 15.656 s /16.546 s | 1.951 GiB | All recorded resource checks passed; no backward/update/generation |
| Instrumented full v2 | Preload RAM 2.515 GiB versus dynamic 3.831 GiB; 7.891 s /8.719 s | 0.673 GiB | No model work; numerical error magnitude not remeasured |
| Streamed load v1 | Preload RAM 2.110 GiB versus dynamic 2.331 GiB; 5.078 s /5.875 s | 0.673 GiB | New loader not reached; no model work |
| Streamed unpadded v1 | Exact load, numerical gates and 256/512 updates; 32.000 s /33.172 s | 1.740 GiB | 1024 backward returned; free VRAM 0.161 GiB below 0.5 GiB reserve |
| Complete CE512 v2 | Eight checks, four updates, bitwise resume and unchanged base; 32.297 s /33.360 s | 1.7486 GiB | Single unpadded CE through 512; sustained E1 remains unmeasured |

The full attempt failed its first left-padding likelihood comparison against
maximum error 0.03125 and mean error 0.01. Its original receipt did not store the
error magnitude. Failure-only telemetry now preserves errors and uses null for
nonfinite entries; the instrumented retry stopped before loading. Source inspection
found aligned positions and plausible masks. BF16 shape-dependent rounding is a
hypothesis, not an established cause. No threshold was relaxed.

Two non-author assistant reviews checked native source/config/journal/cost and
scope consistency; this is not human approval. Native verified loads total four and disposable optimizer calls total six.
Completed-step input positions total 2560; actual training forwards used 3584
positions across the two latest trials. No generation, teacher calls or capability
measurements occurred. The historical 71 stdlib controls and nine actual tiny CPU
loader controls passed; earlier
12 tiny numerical controls remain a separate run. One historical-helper test
initially failed after adding a new dependency; the corrected fixture verifies
actual old dependencies and rejects inventing a dependency in an old commit.
The failed log/receipt remains preserved. CPU controls do not establish GPU fit.

## App-independent execution and detailed sequence

The corrected standalone controller completed its admission window after
601.063 seconds. It recorded 256 observations, with 0.581-0.955 GiB available,
and never invoked a native supervisor. Its process is terminal and the raw wait
receipts are preserved. A later OS reading was 1.397 GiB with no Python workers.
Headroom subsequently rose to 5.221 GiB; the controller measured 5.118/5.121 GiB
and admitted the streamed unpadded trial. No user application was terminated. A regular PowerShell launcher can run
outside the desktop app. It waits at most ten minutes for two observations at
least 3.0625 GiB, including supervisor allowance, then invokes one contained
30-minute scoped qualification trial. The latest 512 successor admission window
was deliberately capped at 60 seconds and returned after 62.552 seconds, with
27 observations of 1.441-1.853 GiB. No native supervisor was invoked; the controller
is verified terminal and its wait receipts are preserved. A later actual headroom
change to 4.303 GiB enabled two admission readings of 4.275/4.278 GiB and the
complete CE512 trial. That controller is terminal after 37.248 seconds. It aborts on source/config changes,
resource failure, timeout or numerical failure and records durable results.
Desktop working-set observations near 2 GiB do not guarantee that closing it
reclaims exactly that amount. Actual headroom is checked before model work.

The earlier backward_resume resource stage retains the exact loader, base and zero-adapter
comparisons, positional EOS/PAD label mechanics, 256/512/1024 gradients, five
disposable optimizer calls, checkpoint save/reload and bitwise within-stage resume.
It omits unresolved padded-forward/causality and sampler checks and explicitly
records the changed sampler RNG trajectory. Its distinct resource-only PASS
cannot satisfy full E0 or authorize E1. See LAPTOP_RUNBOOK and the launcher files.

After resource and numerical qualification, bind measured laptop successors for
native collection, queue and E1. Prioritize one resident GPU worker, pretokenized
streamed microbatches, GPU adapters/optimizer state, adapter-only checkpoints and
sequential evaluation panels. Avoid retaining a second CPU model or concurrent
CPU-heavy data preparation. Register activation checkpointing or shorter contexts
only if measured VRAM requires them. Existing E1 plans remain historical until
rebound to qualified current source/runtime.

## Scope matching the actual SFT inputs

AMENDMENT_012 registers UNPADDED_SINGLE_SEQUENCE_CE_V1. Prepared E1 data require
microbatch one, zero padding, all-one masks and completion-only labels. The
new scope validates those boundaries, omits a redundant validated all-one mask
for primary forwards and rejects cache, custom positions and packing metadata.
Semantic data eligibility still requires actual row/token/label binding; IDs alone
cannot prove a sequence is eligible.

The scoped worker adds mandatory native 80-token unpadded causality (bitwise
77-token prefix agreement after perturbing the last three tokens) and an actual
explicit-all-one-mask comparison at the unchanged 0.03125/0.01 likelihood bounds.
It retains EOS labels, causal CE, finite confined gradients, all three context
profiles, five disposable calls, resume and unchanged-base checks. Padding,
batched/cached inference, sampled-policy likelihood and evaluation capture remain
unsupported or unqualified. The known full E0 padding failure remains preserved.

The actual trial passed bitwise 77-token causality and all-one-mask likelihood
with maximum and mean error zero. Finite confined gradients, causal CE and updates
completed at 256/512. The 512 single-step profile had a 3.089 GiB CUDA allocation
peak and 0.203-second duration; this is a mechanics fixture, not sustained training
throughput. At 1024, backward and gradient checks returned, then the VRAM guard
stopped before the third optimizer call. Peak allocated/reserved VRAM were
6.254/6.639 GiB; minimum sampled host headroom was 3.032 GiB. All 387 accounted
operations returned; no checkpoint save/reload or final frozen-base check occurred.

AMENDMENT_013 registers UNPADDED_SINGLE_SEQUENCE_CE512_V1 with contexts 256/512,
four disposable updates and bitwise 512-token resume. Its hard input ceiling is
512 tokens. Prepared TRAIN sequences are at most 398 tokens, so the complete
data dose and targets fit without truncation. Model math, eval mode, optimizer,
RNG preservation and numerical/resource floors remain unchanged. The narrower
fixture changes the declared workload; resume is checked within its own stage,
without asserting cross-stage RNG equivalence. Activation checkpointing
remains a separately registered future option for longer contexts. Long-context
evaluation and preservation still require independent native qualification.

The scoped consistency checker requires explicit selection of the new 512 stage,
with distinct config SHA and PASS status. It binds 393 calls, 26608 positions,
1792 training positions, 75 resource samples and 865 journal records; timing
measurements must cover the recorded intent/return intervals. It verifies native
checks, current/committed source, runtime, inventories, containment, opaque resume
bytes and exact source assets. It never imports the model framework or grants E1.
This is local byte identity and consistency, not authenticated execution proof.

All 107 current stdlib controls passed in 6.107 seconds (process 6.538 seconds).
Separate non-author assistant reviews found no blocking native/checker source
issue; this is not human or native approval. Old/failed/resource-only statuses
and synthetic values cannot qualify a bundle. Windows PowerShell AuditOnly also
passed. The default launcher is now unpadded_ce512. The subsequent complete native CE512 result and its 37 byte/source bindings passed
the consistency checker in 2.207 seconds. Independent actual metadata/source
review found no blocking contradiction. All 393 calls returned: 71 forwards,
four backwards/updates and one save/reload; 26608 positions, 1792 training positions,
75 resource samples and 865 journal records. The eight native checks all passed.

Observed process-tree RSS was 1.748646 GiB, only 1.386719 MiB below the 1.75 GiB
cap. Minimum sampled host headroom and GPU free memory were 2.498425/3.326172 GiB.
Peak allocated/reserved CUDA memory were 3.098094/3.474609 GiB. Timers agree with
the ordered journal. This short fixture does not measure gradient accumulation,
data-reader overhead, sustained training, response capture or preservation.
No controller or model worker is running. E1 remains unauthorized.

## Actual TRAIN compatibility with the low-RAM interface

The existing numeric binder only syntax-checked source-row hashes. A prospectively
frozen 28-file contract now joins every actual TRAIN source body to its token/label
array and the CE512 consumer interface. Pure deterministic helpers are byte-pinned
before import, rebuild semantic parents/TRAIN allocations and independently verify
prompts and targets. Language joins complete curated conversations, original-body
lineage and measured token profiles; all 81 prior assistant turns remain masked.
One conversation can contain multiple turns without being sequence packing.

The actual committed producer `e717d87a59263e84db7b555dd902f48a7b928eac` passed all
5120 rows, 680532 input and 63546 supervised positions, maximum length 398. No row
or target was truncated or rewritten. The unchanged historical data SHA is
`51b87fe4fd70d2fb23d16f270d1023716172dbbe1c88efd79dff8c3548623d6d`;
source/array join SHA is
`555776a347ea7a82e7a111b90ab041f5947bb0e54bd32c8b306e9064fbd5fc68`.
The production audit took 3.329 seconds (process 3.801 seconds), with a 39.004 MiB
reader working-set peak. Eighteen new stdlib controls passed in 0.270 seconds;
the earlier 107-control run remains separate. No model/tokenizer/weight access,
new data generation, GPU work or closed-bank access occurred in this audit.

Independent review found and fixed a production provenance gap: receipt creation
now snapshots committed producer/manifest bytes upfront and rechecks them, HEAD
and the budget before publication. Dirty, untracked and changed producers are
rejected. Resource checks are cooperative with OS high-water memory observations;
there is no hard CPU containment. Language original TRAIN origin and prior quality
selection rely on frozen earlier provenance; this is not new human grading,
retokenization or complete pretraining exclusion. The data pass leaves every E1
and native launch grant false. Actual capture/decoding, preservation, language
calibration, long-context runtime and current E1 execution remain prerequisites.

## Prepared science and milestone audit

Preparation remains available: 4096 acquisition TRAIN parents, 768 deterministic
and 256 curated language replay rows; 5120 total sequences with 680532 nonpadding
and 63546 supervised tokens. The historical seed-101 plan has 256 steps, each
16 acquisition plus 4 replay, 13 warmup steps, final LR zero and final-only
selection. Later seeds 211/307 require its registered DEV screen.

The DEV prompt/scorer binding contains 832 rows over 544 parent groups and 890874
prompt tokens, with evaluator references separate. Language has 64 DEV/16 CAL
candidate worlds and unassigned review packets; grading remains unqualified.
Long-context DEV has 32 worlds and 256 related rows at exact 2024/4072 prompt
lengths; native 2048/4096 preservation remains unqualified. Parent grouping,
source-quality exclusions, original failures and scoped fingerprint checks are
retained in their existing receipts and ledgers. CHECK/FINAL remain closed.

| Milestone item | Current evidence | State | Remaining work |
|---|---|---|---|
| Current state | Complete CE512 mechanics and actual TRAIN joins | Partial progress | Preservation/capture/current E1 |
| Exact model | Titan devfreeze; 596049920 parameters | Bound | Same checkpoint retained |
| Git/source | Complete CE512 producer 15901919 | Committed | Delivery commit records final docs |
| Verified | Eight CE512 checks; 5120 source/array/consumer joins | Scoped | Full E0 and E1 not passed |
| Unmeasured | Sustained E1 fit, preservation and capability gain | Open | Current capture and E1 successors |
| Completed experiment | Complete native CE512 mechanics | Recorded | Prior failures and partial work retained |
| Exact result | Latest RAM peak 1.7486 GiB; minimum GPU free 3.326 GiB | Measured | Sustained worker overhead |
| Acquisition effect | Disposable mechanics only | Unmeasured | Qualified E1 |
| Retention effect | No scored native preservation | Unmeasured | Capture and grading |
| Compute/memory | Six workers 103.328 s; supervisors 108.891 s | Measured wall | No sustained throughput inference |
| Failure modes | Padding, 1024 VRAM reserve and RAM admission | Recorded | No tolerance or guard auto-relaxation |
| Interpretation | 7.4 GiB was inherited reserve, not model demand | Source audit | Complete training envelope still needs proof |
| Ruled out | Admission failures alone prove Titan needs 7.4 GiB free | Invalid inference | No efficacy conclusion |
| Next experiment | Qualified capture and graded preservation | Open | Current native plans and E1 fit |
| Heavy compute justified | Bounded qualification | Conditional | E1 waits for scientific gates |

The original capability/retention research objective remains incomplete. Current
best remains unchanged Titan with null acquisition and retention effects. No
preparation or partial resource pass promotes a recipe. Historical blocked audits
and 7.4/6 GiB configs remain evidence of earlier conditions; they are superseded
operationally by the user's authorized laptop route, not rewritten into successes.
