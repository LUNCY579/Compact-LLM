# Cross-scale composable skills: source, data, and resource audit

Audit date: 2026-09-20. Study parent: `07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd`.
This audit read existing worktrees, source files, safetensors JSON headers, and evidence
records. It ran no pretrained model, tokenizer, optimizer, or GPU work. Existing branches,
model files, pinned environments, and evidence were not modified.

The proposed student is the existing Supa 1.7 source. Its exact count is **596,049,920**,
not 596,000,000. The user explicitly accepted the existing model after this rounding
difference was raised. The 4B model is a donor only. An unmerged adapter has additional
parameters; a final dense merge preserves the student's original count. The distinction
must remain visible in future parameter-budget receipts.

## Verified binding and architecture

`experiments/cross_scale_composable_001/evidence/BINDING_ASSETS.json` contains paths,
sizes, and freshly computed SHA-256 values for 52 files. All were present. Its SHA-256 is
`b46a4b1e2c23628911b0be9805de6b6cc21f3fb0aba01883df1fb4c466b56faf`.
These are hashes of raw local bytes, not hashes copied from an old result. Safetensors
counts below were independently summed from headers without loading tensor payloads.
The evidence repository HEAD observed during this audit was
`7f188d5c992f9a615839f40d0ee605354df5bcd8`; external assets remain individually bound
because a repository commit alone does not bind ignored model binaries.

| Property | Proposed deliverable base | Donor base |
|---|---:|---:|
| Identity | Supa 1.7.0-devfreeze, ungated dense source | Qwen/Qwen3-4B |
| Official donor revision | not applicable | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Unique serialized parameters | 596,049,920 | 4,022,468,096 |
| Serialized tensors / dtype | 310 / BF16 | 398 / BF16 |
| Blocks | 28 | 36 |
| Residual width | 1,024 | 2,560 |
| FFN width | 3,072 | 9,728 |
| Q heads / KV heads | 16 / 8 | 32 / 8 |
| Explicit head dimension | 128 | 128 |
| Q projection output width | 2,048 | 4,096 |
| KV projection output width | 1,024 | 1,024 |
| Vocabulary / tied embedding | 151,936 / yes | 151,936 / yes |
| Maximum configured positions | 40,960 | 40,960 |
| RoPE theta | 1,000,000 | 1,000,000 |

Student assets are at
`C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/Supa`.
Donor assets are at
`C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`.
The inspected donor files are regular files, not cache symlinks. The source files in the
new Git worktree contain configuration/metadata; the complete binaries are external.

| Asset | Rehashed SHA-256 |
|---|---|
| Student model | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` |
| Student config | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| Donor config | `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a` |
| Donor shard index | `6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2` |
| Donor shard 1 | `328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223` |
| Donor shard 2 | `6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5` |
| Donor shard 3 | `e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1` |
| Shared tokenizer.json | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Shared tokenizer_config.json | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| Shared vocab.json | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| Shared merges.txt | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |

Both models' four tokenizer files are byte-identical. This supports an aligned physical
token grid; it does not establish residual or computational equivalence. Configured
position length is not a qualified execution budget. Attention width must not be inferred
from residual width. The frozen map should use post-block residuals and zero-based layer
indices, with nearest normalized-depth correspondence and an explicit tie rule. Three
initial inspection points are student `[0,13,27]` to donor `[0,17,35]`. Mapping based on
activation similarity must be a separately registered arm, never selected by skill score.

The cached Qwen3-0.6B config has the same dimensions and exact parameter count as Supa;
it does not solve a literal 596,000,000 limit. No additional smaller source is needed under
the user's accepted existing-model interpretation. No source was downloaded.

## Existing skilled donor: available, with a precision qualification gap

An actual adapter exists at
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/qwenft001-20260918/run/adapter`.

| Artifact | Rehashed SHA-256 |
|---|---|
| adapter_model.safetensors | `e7ab0919192dd10845c5f0769ce942028737750367c736e8bff4664fa020164a` |
| adapter_config.json | `1d0ce7c548142d13c66636ec403dc56efa71ce999f0fb70e27319387d8813295` |

The adapter header has 504 FP32 tensors and **33,030,144** parameters. Its metadata and
training receipt identify rank 16, alpha 32, all 36 layers, and q/k/v/o/gate/up/down
projections; the training receipt reports 384 rows, 28 optimizer groups, and the same
RRP-015 control rows used in the small-model program. Its base path names the exact
donor revision above. This establishes an available trained artifact and source linkage,
not a newly qualified behavioral effect in the current study.

**It was trained with an NF4 double-quantized base and BF16 compute.** The residual
`h_donor_skill - h_donor_base` must compare matched precision, quantization configuration,
runtime, token IDs, masks, position IDs, and hook locations. An NF4 skilled activation
minus a BF16 base activation confounds skill with quantization and must be rejected.
Applying the adapter to dense BF16 is mechanically possible but constitutes a separately
qualified donor realization. The old `TRAIN.json` and adapter configuration do not alone
bind every package version, quantizer realization, or activation-capture implementation.
Those bindings remain pending before native execution. A pre-existing adapter is not
authorization to execute it.

This donor learned a mixed transition dataset, so it can support the first **aggregate
transition-skill** pilot. It is not two independently acquired donor skills. Stage E must
use separately trained and independently qualified donor A/B checkpoints, or restrict
its claim to small-model composition and mark the missing cross-scale pair explicitly.

## Open skill data, calibration, and instrument boundaries

The preferred source is the existing open Forge RL v2 release at
`C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/rineya_reasoning_forge_rl_v2`.
Its data card declares CC0-1.0 intent and original code MIT. Shared grammar/parser and
authorship limit independence; no independent benchmark claim follows.

The raw transition source is `process/train_transitions.jsonl.gz`, SHA-256
`386bc03c7a9a883f60986cfe59ea0cac2cce34ea7dbaec71595e018eb1e0ce43`.
This audit reread all **36,752** rows and replayed every label with the source-bound
`forge_rl.oracle.replay_transition`, with bytecode writes disabled. All labels agreed.
Only `valid=True` rows are generation targets; negatives intentionally contain wrong states.

| Independently defined task | Positive rows / parents | Proposed role |
|---|---:|---|
| ledger_replay | 8,000 / 2,000 | prospective skill A: signed account-state updates |
| register_machine | 8,000 / 2,000 | prospective skill B: modular register/branch updates |
| prerequisite_plan | 2,376 / 1,155 | included in the existing aggregate donor pilot |

A/B are distinct rule systems and family-specific datasets, not independent authors.
Future A/B selection must exclude all calibration and evaluation parents, freeze the
exact row sets and optimizer order before results, and train each skill independently
from the same base. Their final train manifests and separate donor adapters are not yet
available. One mixed adapter cannot be relabeled as two independently learned deltas.

The existing aggregate donor training source is
`rineya-dataset-retention-records/rrp015-20260917/control-prepared/microfit-train.jsonl`,
SHA-256 `0881c027d166f7cdd0286518aff20e5966f92120f4f3877cbca6855fbee2d9e1`.
Its 384 parents are excluded from calibration, in addition to the two existing 384-parent
acquisition instruments. This audit verified that these three parent sets are disjoint.

The active calibration manifest is
`experiments/cross_scale_composable_001/evidence/CALIBRATION_BINDING.v2.json`, with
`role=open_calibration`. It supersedes the preserved v1 file only to add the explicit role;
the selected rows and their digest are unchanged. Active file SHA-256:
`1c84e250d5e40bb8112ac8192dc0a162b653ad90b5b67049c0bafc011ca3cb24`.
The canonical row-list digest is
`a1119a2d12f2d65d63144102cb38bf2834cf664e32406c949a0fa6ed81ec8d96`.

Selection uses salt `cross-scale-composable-001/calibration-v1`, one positive step per
parent, and a deterministic hash order over all eligible parents. There were 4,003
eligible parents after 1,152 exclusions. The first 32 yield 24 map-fit and eight map-check
parents: 12 ledger, 13 register, seven prerequisite. The receipt contains metadata IDs
and prompt hashes, not copied prompts or targets. System rules came from open donor
training rows; evaluation files supplied parent IDs only. No sealed examples were used.
`build_calibration.py` verifies input hashes before and after selection, replays every
source label, reconstructs exact prompt hashes, and refuses an existing output path.
The v2 creation run reproduced all v1 rows and digests with zero native calls.

`prompt_sha256` hashes canonical system/user message JSON before native chat rendering;
it is explicitly **not** a token-ID hash. Native rendering, attention masks, alignment,
and the fixed **256-token maximum per prompt** remain pending qualification. Overflow
must stop the session, with no truncation or replacement. Native token eligibility cannot
be inferred from historical average row lengths.

The `rineya_long_context_cc0_v1_main` candidate is excluded: inherited
`LCR_CC0_AUDIT.md` explicitly records `DO_NOT_USE`, native-window mismatch, and an
abstention-label shortcut. Other LCR variants and the binding/new-signal intervention
are separate lanes and are not silently substituted into this pilot.

## Reusable probes and comparability

| Source machinery | Reusable behavior | Required boundary |
|---|---|---|
| `experiments/reasoning_posttraining_v1/process_arm.py` | source-label replay, positive selection, parent exclusions, rendering | its existing training grant excludes larger teachers; do not borrow its authorization |
| `experiments/reasoning_posttraining_v1/select_views.py` | SELECT-VIEWS-001 metadata/floors | selection/dev only, never training or confirmation |
| `experiments/rab001_exploration/process_probe.py` | same-token transition scoring, strict/canonical separation | bind actual `views_sha256`, implementation and tested weights |
| `experiments/rab001_exploration/retention_probe.py` | 150-row panel and retained source-correct subset | preserve exact source readout, baseline and receipt gate |
| `experiments/rab001_exploration/long_context_probe.py` | registered generation contract, no truncation, separate abstention | spent report-only endpoint; never optimize on it |
| `experiments/model_quality_gpu_v1/native.py` | local-only source verification and runtime checks | its pins/FP32 student loader are not a donor qualification |
| `experiments/architecture_benchmark_001/gates.py` | implementation digest and evidence gates | file hashes in this audit are source pins, not a new valid probe receipt |

SELECT-VIEWS-001 source digest is
`7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3`.
The older report-only transition views digest is
`45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e`.
Equal row counts are not bank equality. Before inspecting future behavioral endpoints,
verify views, dataset, source commit, implementation, runtime/tokenizer, base, delta,
training-parent/order and probe `weight_sha256` identities. If any disagree, return
`BLOCKED_ON_COMPARABILITY` before interpreting the score. Report acquisition as
`(net, nd, exact p)`; cross-training comparisons retain the `|net| >= 20` program floor.
Deterministic transformations of one checkpoint retain their appropriate paired analysis.

## Resource estimates and future native work

All estimates below use GiB = 2^30 bytes and are analytic planning values, not measured
fit or permission. Let `P_S=596049920`, `P_L=4022468096`, and `b` be bytes per value.
Weights cost `b*P`; KV cache costs `2*layers*batch*tokens*KV_heads*head_dim*b`.
Activation captures cost `rows*tokens*captured_layers*hidden_width*b`; all-token capture
should stream per prompt and release each model before loading the next.

| Stage | Explicit lower bound / persisted data | Planning implication |
|---|---|---|
| A/B audit and fixtures | no pretrained allocation; current input hashing uses 1 MiB chunks | CPU only; no native grant consumed |
| C student base capture | FP32 weights 2.22046 GiB; BF16 weights 1.11023 GiB | retain registered precision; plan roughly 4–6 GiB working memory plus process margin, measure before permission |
| C donor dense base | BF16 weights 7.49243 GiB; FP32 weights 14.98486 GiB | full BF16 already exceeds historical free GPU; full FP32 is unsuitable on the recorded 15.22 GiB host before overhead |
| C donor skilled capture | same base plus FP32 adapter 0.12305 GiB | NF4 base and adapter-off/on capture is a possible matched pair after fresh quantizer/runtime qualification; never change precision just to defeat a guard |
| C three sequential capture streams | 32*256*3*(1024+2560+2560)*4 = 0.5625 GiB stored FP32 at maximum prompt length | per-model arrays need not coexist in RAM; last-token-only would use 2.25 MiB but gives only 24 fit observations and a severe rank ceiling |
| C/D offline transport fit | FP64 donor Gram 50 MiB; one dense map 20 MiB; donor design at maximum 8,192 positions is 160 MiB | solve one layer at a time; account for factorization workspace; no claim that this equals peak RSS |
| Historical tail20 rank16 adapter reference (not the CSC pilot recipe) | 7,208,960 trainable parameters; FP32 parameters+gradients+two Adam moments 0.10742 GiB; with frozen FP32 student, floor 2.32788 GiB | activations, logits, loading copies, capture targets and allocator overhead remain extra; a 4–6 GiB working-memory estimate needs native qualification and retained guard margin |
| D full-model FP32 Adam alternative | 16*P_S = 8.88184 GiB before activations | not the preferred pilot under host pressure |
| E independent donor skill training if required | rank16 donor adapter state 16*33,030,144 = 0.49219 GiB plus frozen base | NF4 payload alone is not a fit estimate: embeddings, scales, gradients/workspaces and imports add memory; fresh donor training authority required |
| E student composition/probing | one student plus at most two small adapters or streamed dense merges | evaluate arms sequentially; preserve originals; persistent KV at 4096 is 0.875 GiB for student FP32 and 0.5625 GiB for donor BF16 |
| F replication | approximately the per-run stage-D/E peak, multiplied storage and total time | run sequentially; register draws and budget independently |

The tail20 student adapter count uses `r * sum(in_features+out_features)` over seven
projections per block: `20*16*(3072+2048+2048+3072+4096+4096+4096)`.
All-28-layer rank16 would instead add 10,092,544 parameters; final module scope must be
frozen and identical for norm/rank controls. Optimizer estimates assume FP32 states and
exclude any additional master-copy policy. A full logits tensor at sequence length 256
costs another `256*151936*4` bytes (about 148.4 MiB); calling a backbone-only capture path
avoids logits only after that hook path has been qualified.

The successor's actual proposed bounded Stage D recipe is smaller: rank8 q/v adapters
on four final layers, 163,840 trainable scalars and 2.5 MiB for FP32 parameters,
gradients and two Adam moments. PREREGISTRATION.md and NATIVE_NEXT_STEPS.md govern that
proposal; historical counts above are reusable alternatives, not permission to expand it.

The current authorization covers offline preparation. Still-required native work is:

1. New clean-source/runtime/lease/resource qualification and native tokenizer qualification
   for the frozen 32 calibration parents. Any failure stops before pretrained calls.
2. Three sequential capture streams: student base, matched donor base, matched donor
   skilled. At batch one this is **96 prompt-forward evaluations**, plus separately
   registered qualification checks; zero training steps and zero generated tokens.
3. Frozen map fitting on 24 parents and held-out correspondence checks on eight. Good
   prediction is a representation result, not skill-transfer evidence.
4. A separately authorized one-skill student pilot with fixed rows, step counts, seed,
   trainable modules and optimization budget; S0/S1/X0/X1/X2/O0 and R0 where defensible,
   with all final checkpoints source-bound before behavioral probes. Number of student
   training runs and probe legs must be frozen in that stage's execution registration.
5. Independent A/B donor and student skill qualifications, composition, and later
   replication only after their stage gates; no such training or probes happened here.

The smallest useful **next native session** is the frozen 32-parent sequential capture
qualification: it can expose an unusable donor residual or unlearnable representation
map cheaply. It cannot establish acquired skill. The smallest actual transfer experiment
is one matched aggregate-transition S0/X0 training draw with the mandatory derived
norm/projection/random controls and paired acquisition/retention measurements. Reusing the
existing mixed donor adapter avoids a new donor training run for that first question.

## DOSELORA remains separate and terminal

The read-only `highdose-lora001-20260920/preload-attempts/wait-002/PRELOAD_TERMINAL.json`
records `WAIT_EXPIRED_NO_WORKER`, `controller_launched=false`, `worker_launched=false`,
and no dispatch reservation. The historical session deadline is **2026-09-20T09:39:05Z**,
which is expired. The source runbook's old wait-002 command is historical, not current
authority. The user explicitly forbids reusing that envelope or starting another wait
without external resource recovery.

DOSELORA's unchanged launcher gate is 7 GiB available host RAM and its separate child
guard is 6 GiB after imports. Do not weaken guards, switch precision, change pinned venvs,
terminate unrelated applications, or restart polling. A future dose resumption needs a
real external RAM change, inspection of terminal/grant/journal/shared
`rineya-gpu-0.lock`, a fresh bounded registration, and only then fresh wait ID `wait-003`.
The cross-scale study did not read dose behavioral scores or change any dose evidence.
The evidence repository's load-bearing `* -text` attribute was inspected and preserved.

## Claim status

**Mechanically established:** exact source bytes and tensor inventory; available donor
adapter and its training provenance limits; open calibration identities, exclusion
disjointness and oracle agreement; explicit dimension/memory formulas.

**Scientifically untested:** useful cross-scale correspondence, residual transfer,
direction-specific acquisition after norm matching, protected-skill preservation, and
independent skill composition. No scientific improvement claim has been opened.

**Blocked before native work:** fresh native authority and execution envelope; matched
donor precision/runtime/capture qualification; native calibration token eligibility;
resource/lease gates; independently trained A/B donors for the later composability claim.

**Authorized next action:** finish offline mechanics, tests, source-bound registration
and reviewable commit. No pretrained execution follows automatically from readiness.
