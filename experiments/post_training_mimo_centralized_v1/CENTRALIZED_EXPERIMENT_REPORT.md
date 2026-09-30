# Centralized Rineya Titan 0.6B: laptop execution revision

30 September 2026, America/Chicago. **The inherited 7.4 GiB admission guard was
not measured Titan RAM demand. Exact GPU loading and frozen inference have now
run successfully on this laptop, with observed process-tree RAM peaks below
2 GiB. The complete 512-token CE mechanics trial now passed all eight checks,
four disposable updates, bitwise save/resume and unchanged-base verification.
Its observed process-tree RAM peak was 1.7486 GiB. The earlier 1024 VRAM failure
remains preserved. The observed RSS margin is only 1.39 MiB; sustained E1 fit,
acquisition and retention remain unmeasured. The first capture pilot then stopped on a raw-model likelihood disagreement
with comfortable memory headroom. A separate diagnostic reproduced the failure
and localized the difference upstream of the output head; FP32 head projection
alone still fails. A resumed causal attempt then stopped after imports before any model call.
A separately registered post-import admission wait is ready but has not run
natively. Best remains unchanged Titan.**

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
| Short greedy capture | First fixture generated three tokens; likelihood failed; 25.437 s /26.640 s | 1.405 GiB internal sample | Resource checks passed; no DEV collection |
| Held-hidden diagnostic | Original failure reproduced; eight forwards and sixteen projections; 20.344 s /21.266 s | 1.409 GiB external peak | Difference localized upstream of head; no qualification |
| Causal D2 preload | Runtime imported; dynamic RAM admission failed; 24.422 s /25.360 s | 0.672 GiB external peak | No model constructor/load/forward; causality remains unmeasured |

The full attempt failed its first left-padding likelihood comparison against
maximum error 0.03125 and mean error 0.01. Its original receipt did not store the
error magnitude. Failure-only telemetry now preserves errors and uses null for
nonfinite entries; the instrumented retry stopped before loading. Source inspection
found aligned positions and plausible masks. BF16 shape-dependent rounding is a
hypothesis, not an established cause. No threshold was relaxed.

Two non-author assistant reviews checked native source/config/journal/cost and
scope consistency; this is not human approval. Native verified loads total six and disposable optimizer calls total six.
Completed-step input positions total 2560; actual training forwards used 3584
positions across the two latest trials. The capture pilot emitted three mechanical tokens; no DEV generation, teacher
calls or capability measurements occurred. The historical 71 stdlib controls and nine actual tiny CPU
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

## Frozen-source greedy capture and likelihood failure

AMENDMENT_014 prospectively registered a separate bare-model greedy qualifier,
with no adapters, updates or DEV bank access. It reuses the exact GPU BF16 model
and 2 MiB loader, finite raw FP32 argmax with lowest-ID ties, no cache/padding or
inherited sampling defaults, and fixed 128/384-token mechanical prompts capped at
128 new tokens each. Terminal EOS IDs remain in generated IDs; visible text removes
only one terminal EOS ID before decoding, with special filtering and cleanup off.
Greedy behavior log-probability zero is separate from raw model-NLL diagnostics.
The new provisional resource profile permits 2 GiB worker RSS and retains a 1.25
GiB reserve, GPU4/0.5 GiB and disk4 GiB floors, 600 seconds and Windows containment.
Earlier 1.75 GiB profiles remain unchanged; the allowance is not model RAM demand.

All 44 new stdlib controls passed in 1.130 seconds (process1.381 seconds), and
independent native/checker source reviews found no remaining blocker. The first
contained pilot from `3ed5777167817edd7032f8ed3125afd2e6daa3ad` loaded exactly and
returned all 318 selected calls: 310 copies, six forwards, 1286 input positions.
The first fixture generated IDs16,22,151645 and visible text17, then natural EOS.
No answer filtering was used. Online raw-model log-probabilities were
-0.5026993752,-0.4105024338,-0.0005460678; full131-token teacher forcing gave
-0.4612669647,-0.3731182814,-0.0006275591. Maximum0.0414324105 and mean0.0262993514
exceeded the unchanged0.03125/0.01 bounds. The worker stopped; fixture384 was never
attempted and final base/source checks were not reached. Capture remains unqualified.

Worker/supervisor times were25.437/26.640seconds. Internal sampled RSS peaked at
1.404602GiB, host available RAM stayed at least2.554611GiB, GPU free at least
5.400391GiB and CUDA allocation peak1.400038GiB. This failure is numerical rather
than an admission or memory failure. The100ms supervisor RSS sample can miss the
instantaneous internal peak. Generation's measured instrumented wall was0.250s
for3tokens; it includes durable logging and guards and is not sustained throughput.

Raw failure/result/cost/journal copies are byte-exact. The early failure receipt
contains stale initial zero-call accounting; the final result, cost ledger and
journal correctly record318 and are authoritative. Both records are preserved.
The pure capture checker rejects the failed status before reading model assets.
No source-correct membership, DEV response or E1 grant follows from the pilot.

AMENDMENT_015's fixed-trace diagnostic completed from producer
`0b48e3cd322399827b30766caf122d829b93269b`: 320 returned logical calls, eight
forwards, 1036 positions and sixteen separately journaled direct projections;
zero generation, updates or teacher calls. The 14 new stdlib controls passed in
0.213 seconds (process 0.431 seconds). A non-author actual source/receipt/journal
review reconciled all 718 records and exact endpoint logit hashes. This is local
consistency review, not authenticated execution proof or a reviewer model-payload
reread.

The original likelihood arrays and errors reproduced exactly. Original endpoints
matched the held BF16 projections exactly. Holding hidden values fixed while
changing output-head rows from one to four gave zero BF16 logit or log-probability
difference. Prefix/full hidden states differed by maximum 1.25 and mean 0.055434;
either BF16 head layout carried the original likelihood gap. A GPU-only FP32 head
still gave maximum 0.039300263 and mean 0.013127337 in the one-row comparison,
exceeding both frozen limits. Thus FP32 head projection alone is insufficient.
The experiment localizes this trace's difference upstream of the head, without
identifying a layer, kernel, causality error or rounding mechanism.

Worker/supervisor times were 20.344/21.266 seconds. External process-tree RAM
peaked at 1.409 GiB, minimum sampled available host RAM was 2.249592 GiB and CUDA
allocation peak was 1.737812 GiB. Final base and source hashes remained unchanged.
Every raw output was copied byte-exactly. The diagnostic grants no capture,
source enrollment, language grading or E1 execution. A distinct upstream diagnostic
or exact-prefix scoring scope must be registered before the next experiment;
agreement in a successor cannot relabel the failed bulk gate.

The detailed operational plan V3 keeps GPU state, streamed inputs and sequential
evaluation. The measured 1.75 GiB short CE cap leaves only 1.387 MiB margin;
sustained training requires a separate reader/accumulation/RSS measurement. A
proposed 2 GiB training cap remains unqualified. The historical seed schedule,
all TRAIN targets and numerical limits remain preserved.


AMENDMENT_016 now freezes four same-length 131-token future-suffix comparisons,
with actual mask/rotary/final-hidden witnesses and zero generation or updates.
Root completed the producer after a builder model-capacity error. Fourteen stdlib
controls passed in 0.014 seconds (process 0.209 seconds); independent review
resolved a missing exposed FP16 split-K flag. Existing reduction flags are now
recorded without mutation. Final producer is
`39add8a9d9ea7df69ac838860c3dcab15639e43d`.

The next native diagnostic did not start. Its initial RAM pair was 2.833652 and
3.498203 GiB, failing the required two observations at least 3.3125 GiB. A bounded
30.897-second continuation within the same dispatch recorded twelve observations
from 3.228485 down to 0.221680 GiB. No native supervisor was invoked; all model
counts are zero and causality remains unmeasured for these new comparisons.
Both admission receipts are byte-exact and the controller is terminal. A real
available-memory change is needed before one admitted run; the code/control
pass is not a native causal or capture qualification.

The next short DEV input scope is frozen at512rows/448parent groups, with full
prompt hashes and128new-token caps: at most65536generated IDs and17433472processed
prefix positions. Production time/runtime budgets remain unmeasured and no
collection is authorized. Language requires independently qualified reference
labels, grader/provider/runtime/prompt/decoding identities, blinding and complete
cost budgets. Current packets and managed-agent calls are unassigned diagnostics;
no paid or teacher grading is authorized. A user preference question is pending
while deterministic qualification continues. Long-context capture remains separate.

## Recovered training components and actual scheduled reader

The last automatic coding turn separately failed with the service error
"Selected model is at capacity." The app marked the goal blocked; the user's
status message resumed work in the chat and a later goal continuation is active.
This interruption is distinct from the laptop RAM and numerical capture gates.
No training process was running or lost. Saved code was recovered and pushed.

AMENDMENT_017 adds a bounded random-access TRAIN reader, a separate E1 call journal
and deterministic accumulation/checkpoint components. The old E0 journal's five
optimizer-call limit is preserved; the E1 component permits256, with explicit
failed/pending/returned work and no reset on restore. The engine divides each
sequence's mean completion loss by20, resets/clips/updates once per group and
still advances optimizer state on the final LR0 update. Synthetic controls cover
unequal target counts and restored parameters/moments/RNG at a saved boundary.
The later internal backend source below implements operations; owned worker
connection,physical accumulation/serialization/cross-process resume and launch
validators remain unqualified.

All53 combined stdlib controls passed in6.525 seconds (process7.120 seconds).
The initial reader run had21 pass/one failed drift-check expectation: the selected
line hash caught a same-size edit, while Windows creation time plus restored
mtime missed it at the earlier metadata check. Added NTFS ChangeTime and strict
typed metadata checks. The observed failure is preserved; its original streams
and initial source hash are explicitly unavailable. Source reviews found no
remaining blocker within component scope; these are not native approvals.

From producer `c2fd2619412d9ac6f138ab970aa75795a0cd99a3`, Python3.11.15 passed all
5120 actual scheduled TRAIN lookups, preserving4096 acquisition,768 deterministic
and256 language replay rows,680532 input/63546 supervised positions,maximum398
and81 masked prior assistant replies. Order was checked against separately parsed
byte-pinned PLAN rows; selected-line and finalwholefile hashes and source/HEAD
checks passed. Trace SHA is
`8483ddbd19f9cfec8c573cc45d2663b2221d0c2e162cef13a90269e2e0fc8269`.
The reader took141.094 seconds, with50.797 MiB OS process high-water and5.392 MiB
retained metadata,zero retained token populations. Enforcement is cooperative
128 MiB/600 seconds with32-row high-water sampling,not hard containment. All
model/update/generation/teacher counts are zero; no efficacy or E1 grant follows.

## Causal admission failure and post-import wait successor

On goal resumption, fresh headroom enabled an admitted D2 window at3.797119 and
3.801788 GiB. Producer `dd48234e0f1000ab227d4a459476826fbc3785a8` imported the
pinned runtime, then found1.986759 GiB available versus dynamic2.581776 GiB
required. Worker RSS was0.668224 GiB; all constructor/load/copy/forward counters
remained zero. Worker/supervisor times were24.422/25.360 seconds. External RSS
peak was721870848 bytes. No model or causal result exists; every raw failure and
admission receipt is byte-exact. A new ledger prose typo24.437 seconds is explicitly
corrected; the raw result, CSV and indexed receipt remain authoritative24.422.

AMENDMENT_018 preserves the full2 GiB RSS/1.25 GiB reserve/GPU4-0.5/disk4/600-second
bounds and registers up to60 seconds of resource-only waiting after imports,
polling every2 seconds for two consecutive current growth-budget observations.
No model call may precede admission and strictfresh preload checks. Runtime floors,
source/HEAD/byte checks and external100 ms monitoring/Windows containment remain.
A timeout grants no admission and triggers no model retry. Original four131-token
forwards,524 positions,316 calls and all numerical fixtures remain unchanged.

Independent review reproduced a deadline gap after the initial26-control pass:
a slow second journal could admit after60 seconds using stale time. Resampling
after journal/measurement and at return fixes it; both negative cases are now
covered. Final28 controls passed in0.015 seconds (process0.220 seconds), with final
source review. Registered wait source was committed at `fadef896` (abbreviated).
Its first preworker observation window ended after30.198 seconds with12 samples
2.061134-2.100109 GiB below3.3125. No native supervisor started; the post-import
wait is source-ready and not yet physically exercised. A real headroom change is
still required. No scientific gate or recipe is promoted.

## Internal native backend source and publication

AMENDMENT_020 adds real injected-Torch method implementations and a versioned
engine publication path. Source is committed at
`c4b67c726544b727940e26eff186e8fdaeb905d9`; no framework/model/GPU run occurred.
Thirty-five combined CPU controls passed in5.629 seconds (process5.918 seconds):
21 fake-backend dispatch controls and14 synthetic V2 engine controls. They verify
Python orchestration and rejection behavior,not real tensor math or native fit.

The backend checks exact frozen-base/adapter inventories and source hashes,keeps
eager/eval/cache-off CE512 math,uses the qualified completion-loss helper and
AdamW defaults,and scales sequence mean loss1/20 once. The engine supplies exact
planned components. Adapterparameters/optimizer/Python+TorchCPU+CUDA0 RNG are
serialized through bounded owned32 MiB artifacts; returned references avoid five
resident tensor snapshots. Restore verifies and deserializes the SAME bounded
BytesIO with weights_only/mapCPU,requiring its own durably published manifest.
Current support is a controlled same-process saved boundary; cross-process
reconstruction and rollback after later failed work remain unsupported.

Publication uses typed V2 cursor and POST-save cost metadata before exposing a
checkpoint. Primitive attempt/returned/failed/pending witnesses distinguish each
actual Torch call from its outer backend-method span,retaining a returned update
if a later finite/resource/journal check fails. The constructor factory attaches
primitive costs to the ORIGINAL exception if construction does not return. The
future owned worker must preserve both ledgers on failures. Root source review
fixed unbounded reads,component guessing,envelope version and failure visibility
before registration; earlier fixture/discovery errors remain separate history.

An owned GPU qualifier,physical accumulation/checkpoint/RNG/bitwise continuation,
frozen-base integrity and sustained resource measurement remain required. Current
host readings were1.32-1.38 GiB with no Python workers,so no model retry was made.
Every scientific/native launch flag remains false; the full objective is open.

## Bounded backend qualification: registered, physically unproven

AMENDMENT_021 registers one disposable two-update native backend test. It uses
the first 40 actual TRAIN rows from the unchanged seed-101 schedule as mechanics
fixtures, without reading DEV, CHECK or FINAL. An independent actual-data audit
verified every selected source/input/label binding: 5,248 input positions and
463 supervised positions, with 15 distinct completion lengths. It took 2.032
seconds and peaked at 53,714,944 bytes (51.23 MiB) of process RAM. It imported no
framework and performed no model call or optimizer update. An initial audit code
assumption about a stored label digest failed; the corrected audit independently
hashes the actual label array. Both observations are retained.

The worker reuses the qualified streamed CUDA loader, native backend, TRAIN
reader, post-import wait and Windows watchdog. It checks zero-adapter parity,
two 20-sequence accumulations, independent gradients, adapter-only publication,
restoration against an earlier independent snapshot and copied-gradient Adam
continuation. The frozen base stays on GPU. Cross-process recovery, sustained
throughput and capability gains remain unproven.

Including reference and parity passes, the prospective maximum is 62 model
forwards, 8,326 input positions and 736 supervised forward positions. Two native
updates and one reference update are counted separately. Restoration contributes
one actual reset in addition to the two training resets. CPU source controls
cannot substitute for these physical measurements. Review corrected the copied
global no-grad descriptor and requires RNG shapes to be frozen before model work
and live gradients to remain unchanged during reference passes.

The 17 worker CPU controls and independent source review passed; source was
committed and pushed at `5cb79fe7`. Its admission window then timed out after
15 observations at 2.208-2.245 GiB available RAM versus the unchanged 3.3125 GiB
budget. No worker, framework import or model call occurred. Raw records and byte
copies are retained; the terminal zero-model controller was archived for a later
resource-changed admission. Keep the 2 GiB worker cap and 1.25 GiB system reserve.
Physical qualification, capture, calibrated language grading, long-context
baseline and sustained E1 remain open. Best stays unchanged Titan; acquisition
and retention gain remain unmeasured.

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
| Current state | CE512/data passed; capture failure localized | Partial progress | Distinct numerical scope and qualified capture |
| Exact model | Titan devfreeze; 596049920 parameters | Bound | Same checkpoint retained |
| Git/source | Qualifier 5cb79fe7; earlier native sources retained | Committed | Delivery receipt binds final report |
| Verified | CE512 mechanics; full TRAIN reader; 17 worker controls | Scoped | Full E0 and E1 not passed |
| Unmeasured | Sustained E1 fit, preservation and capability gain | Open | Current capture and E1 successors |
| Completed experiment | Capture failure, geometry diagnostic, D2 admission stop | Recorded | Every failed/partial attempt retained |
| Exact result | CE512 RAM 1.7486 GiB; diagnostic 1.409 GiB | Measured | Sustained worker overhead |
| Acquisition effect | Disposable mechanics only | Unmeasured | Qualified E1 |
| Retention effect | No scored native preservation | Unmeasured | Capture and grading |
| Compute/memory | Nine laptop workers 173.531 s; supervisors 182.157 s | Measured wall | No sustained throughput inference |
| Failure modes | Padding, 1024 VRAM, capture NLL, stale failure counter | Recorded | Correct final accounting and diagnosis |
| Interpretation | Old reserve overstated this route; capture gap is upstream | Measured scope | Sustained fit and numerical mechanism still open |
| Ruled out | Admission failures alone prove Titan needs 7.4 GiB free | Invalid inference | No efficacy conclusion |
| Next experiment | Registered 40-row/two-update backend qualification | Physically unproven; RAM gate | Capture and preservation prerequisites remain open |
| Heavy compute justified | Bounded qualification | Conditional | E1 waits for scientific gates |
