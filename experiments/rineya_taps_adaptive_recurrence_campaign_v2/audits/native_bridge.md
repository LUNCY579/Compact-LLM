# Native bridge audit and successor control apparatus

Audit date: 2026-10-01 America/Chicago. Native model-worker time used by this
engineer: **0 seconds**; no tokenizer/model loader or locked entrypoint invoked.
Inspection used read-only metadata, filesystem hashes, source, and an analytic
fake host. Files outside campaign_v2 were not edited. Coordinator owns native
dispatch, runtime qualification, resource admission, and experiment selection.

## Exact historical assets and environment

The controlling source is the inherited ARMR receipt
`experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json`.
Its current SHA256 is
`daa5bc8e947602a4ed3d7fb19076ec6000e27f7afcc2a1426e544491bff95dfe`.
Its historical status is PASS, source commit
`af7c49d432a652c55cbafc5155c378d87497cf24`, integrated immutable base
`03db218ceebc61a2bc04f2482cf7e094045e01c1`. A read-only `git diff --name-only`
against the integrated base found no differences in ARMR src/scripts or the
cross-scale src tree. The v1 preflight must still execute its bytewise checks.

Canonical source found at:
`C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze`.
All ten receipt-listed files were independently hashed and matched:

| File | Bytes | SHA256 |
|---|---:|---|
| model.safetensors | 1192134784 | a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026 |
| config.json | 1588 | 89c4ae7398449895d044750c858614584b07db8d67130074920ef524c6eb5705 |
| generation_config.json | 171 | 0a6ef79013177858dd5820698b12270863e03baf60ec36521e5c6a34da63dbdc |
| modeling_supa_a11.py | 2071 | b77b7d709fac3690d86f5978d46de409fd7ad3548c66a9d400188c225d99fa3d |
| tokenizer_config.json | 9732 | d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 |
| tokenizer.json | 11422654 | aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 |
| vocab.json | 2776833 | ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 |
| merges.txt | 1671853 | 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 |
| CORE.json | 4866 | bf640ab323d29c2fc18c306294c7e6f7c53e19cd67364f5df8aa041992075914 |
| FILES.json | 3594 | 4446aef8d1f19f4c968aaa20636200094da0df7d6e518bca6acc4041896b8b6e |

The original compatible interpreter exists:
`C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`.
Read-only importlib.metadata inspection reports Python3.11.15, Torch2.12.1+cu126,
Transformers5.14.1, psutil7.2.2, safetensors0.8.0, pytest9.1.1,
accelerate1.14.0, numpy2.4.6, tokenizers0.22.2, huggingface_hub1.24.0.
No dependency installation or environment mutation occurred. The exact file
`.../.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py`
matches historical SHA256
`fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab`.

Metadata-ready does not mean native-qualified. The safe exact preflight command
from this worktree is (choose a new, exclusive output directory):

```powershell
& 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' -B -m experiments.rineya_taps_adaptive_recurrence_v1 native-preflight --source 'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze' --out 'C:\Users\clclc\BusinessProject\Chat-Bubble-rineya-taps-adaptive-recurrence-v1\experiments\rineya_taps_adaptive_recurrence_campaign_v2\runs\historical_preflight_NEW'
```

This imports Torch/Qwen and hashes source/assets; it does not load weights.
Exit2 is an expected BLOCKED refusal. Historical CPU admission requires at
least ceil(7.4*1024**3) available bytes; its worker RSS ceiling is6GiB. Preserve
this floor and exact runtime checks. The coordinator's initial host inventory
has only ~3.66GiB available RAM, so the CPU lane is not presently admitted by
that snapshot; only a fresh complete preflight determines current status.

## Operator, readout and tensor identity

Model class Qwen3AbstainForCausalLM, 28 layers numbered0..27, hidden1024,
intermediate3072, 16 attention heads of dimension128, 8 KV heads, all layers
full_attention, vocab151936. Stored dtype is F16; the native loader explicitly
converts to FP32. A header-only safetensors read found310 stored tensors,
596049920 stored parameters and2384199680 FP32 tensor bytes (~2.22GiB).
Embedding shape is[151936,1024]; no separate stored lm_head tensor, consistent
with tied embeddings. Actual loaded identity remains unverified here.
Historical loaded FP32 state hash is
`b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551`.

Custom head adds5.156249985098839 to UNKNOWN token51378 at every returned
logit position, including each generated token step. This is a readout offset,
not a learned residual gate. Preserve the custom class, actual code, tied head,
and offset. Never substitute stock Qwen3 or another similarly named core.
RoPE is default theta1000000, config max_position_embeddings40960; that config
capacity is not a campaign shape qualification.

Native first block pass is ordinary L12 thenL13. The proposal for each extra
application recomputes L12 thenL13 on the live output, preserving positions,
mask and conditioning. The raw whole-block residual is F(h)-h. One first-pass
unit observation seeds moments; one extra pass uses observation index1.
Budget0 applies no extra layers. Budget1 adds two physical layer applications
per full-model call. No deeper budget, padding, cache or KV reuse is admitted.
The v1 `apply_update` returns the proposal object at unit alpha, avoiding
FP32 subtraction/re-addition cancellation. A3 parameters remain unchanged.

## Gates and unresolved implementation requirements

1. **A5 reset and seed defect corrected in successor apparatus.** In v1 each
   model entry recreates the scheduler, so the same per-example seeded first
   random draw repeats across all generated prefixes. Its runner sets Torch's
   seed but leaves config.seed17 for nominal seed29, duplicating A5. The
   successor `control.py` keeps the RNG stream across prefix moment resets,
   hashes actual config.seed/arm/sample into independent streams, and restarts
   only on a fresh evaluation. A runner must use `replace(cfg, seed=run_seed)`;
   setting Torch.manual_seed alone is insufficient. This is a control-apparatus
   correction; neither v1 nor successor has efficacy evidence.
2. **Full-gold-continuation NLL stays disabled.** v1 diagnostics makes a fresh
   evaluator globals dictionary overriding canonical_answer to returnNone.
   Scorer and generation bytecode are retained. It prevents sequence-wide
   adaptive reduction from seeing future gold tokens and prevents diagnostic
   alphas entering A5's generation pool. Gold NLL/rank are null, notzero. A
   successor runner must use the generation-only body or reproduce and test this
   guard; causal teacher-forcing would require separate prefix replay.
3. **Limits are checks, not enforcement.** v1 runtime_guard and ARMR generation
   check elapsed time/RSS between forwards. A hung forward cannot be interrupted.
   Loading and final state/source hashing are outside per-case guards. A parent
   supervisor must start one worker, track its exactPID/creation identity and
   descendants, enforce wall/RSS limits, sample resources, and terminate only
   that worker tree. Preserve terminal receipts even after forced termination.
   Never terminate unrelated Python/Unity/browser processes or remove their locks.
4. **4096 is an API refusal cap, not a measured maximum.** V1 checks batch1,
   prefix<=4096, all-one optional attention mask and explicit use_cache=False,
   but does not register admission shape before loading. One eager FP32 attention
   score tensor at4096 has16*4096**2*4=1073741824 bytes; softmax and intermediates
   add memory. Bind actual rendered prompt tokens, max output24 (including
   per-case smaller caps), maximum full prefix and representative shape tests.
   Empty input, dtype/device and explicit position semantics should also be
   enforced in the prospective host contract. CPU/GPU shape success does not
   qualify another device or larger context.
5. **Source/code qualification remains incomplete.** Native preflight verifies
   inherited Python trees, historical receipt/config and receipt-listed assets.
   It rejects unlisted Python below source, compares Transformers and Qwen hash.
   It does not hash allowlisted smoke data there, require CPU/FP32 in the wrapper,
   inspect the actually loaded remote-code module, or qualify determinism on GPU.
   Successor registration must bind exact smoke/DEV/scorer hashes, actual loaded
   custom-code path/hash, all state tensors/device/dtype, zero trainable parameters,
   no optimizer, output cache absence and before/after canonical identity.
6. **Qualification and costs are distinct.** Exact old/new no-extra/unit token
   and first-logit equality on smoke is a mechanical gate, not all-shape identity
   proof or efficacy. V1 then smoke-gates all scale arms against A0 for norms,
   finite logits, cosine>=.9, KL<=.5, zero score/invalid regression. The minimum
   requirements transfer; empirical scaled-arm qualification is still required.
   First-pass core timing coversL13 only, extra timing coversL12-L13; end-to-end
   timing includes overhead. Dense-matmul proxy excludes attention/norms/logging.
7. **Matrix and seeds.** With8 smoke and16 DEV items, two nominal seeds, the v1
   physical layout forecasts392 evaluate_case calls:32 legacy/new qualification,
   64 regular smoke,256 regular DEV,8 A5 smoke and32 A5 DEV. A4 is derived, not
   another physical rollout. A1 duplicates fixed alpha1 mechanics; deterministic
   seeds do not create independent tasks. A prospective smaller deduplicated
   design can preserve comparisons, but must be frozen before outcomes. A
   qualified pilot cost is recommended for a new runtime/device. For the
   explicitly authorized original runner, exact historical native costs may
   support a prospective complete forecast with uncertainty and a hard parent
   deadline. Charge loading, failed work, hashes and retries in either case.

## CPU and GPU successor prospects

The exact environment and source are present; host RAM is the immediate CPU
admission uncertainty. No lowering of the7.4GiB exact CPU floor is legitimate.
The inventory reports RTX4070 Laptop,8188MiB total, ~6324MiB free, driver566.07
(reported CUDA12.7), with active unrelated graphics/Unity work. FP32 stored
weights are~2.22GiB, so a separate short-prefix GPU lane is plausible. This is a
feasibility hypothesis, not a reservation or qualification. Torch CUDA startup,
transient host conversion, available VRAM under WDDM, attention workspace and
state-hash copies must be measured. Existing accelerate is available for an
explicit device_map; a direct GPU load may reduce host FP32 residency but must
be registered and independently reviewed before any loader call. No CPU-only
runner may be called with undocumented GPU fallback. DisableTF32/autocast,
register deterministic/CUBLAS settings, synchronize timed forwards, and retain
same frozen eager FP32 operator/head/cache/generation semantics. Other dtypes,
quantization/cache/deeper recurrence remain outside this successor lane.

## Successor apparatus validation

Added only assigned `control.py`, `tests/test_control.py`, and this audit.
Test command (no native load):

```powershell
& 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider experiments/rineya_taps_adaptive_recurrence_campaign_v2/tests/test_control.py
```

Initial control result:13 passed in3.18s, analyticCPU host only. Tests cover RNG variation
across prefixes, reproducibility, actual seed/arm/sample binding, unchanged
moment reset/identity, permutation and per-example stream isolation, rejected
delta without draws, ordinary warmup without draws, exact v1 A3/unit/fixed
numerics, budget0 policy identity, identical position/mask objects, inherited
cache/overlap refusal, exception restoration and deterministic context re-entry.
Independent reviewer must inspect resulting files; these tests do not license
native dispatch by themselves. The next executable engineer step is to receive
review dispositions and a coordinator-frozen successor host/run contract.

## Bounded tokenizer/data pre-admission packet

`binding.py` is a no-model command for the coordinator to execute after review:

```powershell
& 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' -B -m experiments.rineya_taps_adaptive_recurrence_campaign_v2.binding --source 'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze' --out 'C:\Users\clclc\BusinessProject\Chat-Bubble-rineya-taps-adaptive-recurrence-v1\experiments\rineya_taps_adaptive_recurrence_campaign_v2\runs\binding_NEW'
```

It verifies exact canonical path, all ten asset hashes, integrated-base Python
source/receipt/config bytes and runtime/Qwen identity before loading only the
tokenizer (local_files_only=True, trust_remote_code=False). All HF cache paths
are run-local with offline mode set in that process. It reads only exact
MERGEDEV-fast and SMOKE files, requiring historical hashes. SMOKE controlling
dataset hash is404c866f64f36422e5e9e5d52170a358d6c2dd28d276492a38f348ce18f2050f;
DEV isb75df9368e27cfc7de58e55370090e55bd5bcb4d1bd86aeb079b486555925da0.
The operator/scorer sources are unchanged; canonical model weights are never
loaded. Source tree checks are inherited in meaning, expressed read-only here.

The packet stores raw prompts, rendered chat templates, native tokenIDs and
per-item shape caps. It reports scenario_id as provisional clustering with
semantic/generator independence unverified, target-string mentions as surface
flags, choice option positions, exact/near duplicate checks, prompt-copy and
fixed answer-blind constant floors. These checks do not certify semantic
reasoning, answerability, a solver, or human review. Solvers and human review
remain explicitly absent. No source bank is renamed to independent validation.
No target-bearing field becomes a live controller input.

The independently calculated historical caps areSMOKE192 andDEV214 total
generated-token caps per pass. Thus13*192+18*214=6348 model forwards is the
complete392-evaluation worst-case cap before earlyEOS. Maximum processed
positions and actual prompt/full-prefix maxima require tokenizer execution and
are computed by the packet. Native maximum-shape mechanics remainNOT_RUN;
wall-time forecast remainsnull in this metadata packet; the coordinator may bind
an estimate from historical exact-lane costs or a qualified successor pilot.
The packet has a terminal
receipt even on refusal and leaves loaded model/custom-code identitynull.

After adding two metadata regression tests, the same test command reports15
passed in8.24s. These additional tests use temporary allowlist fixtures and a
fake tokenizer: exact render/tokenizer argument semantics, maximum prefix and
position-sum formulas, duplicate flags, unresolved exposure/review labels and
changed-data refusal before tokenization. This remains apparatus evidence only.

The coordinator executed `runs/binding-001` and the engineer independently read
its raw JSON. Terminal status isTOKENIZER_METADATA_BOUND_NATIVE_NOT_QUALIFIED;
source_unchanged=true, native_model_loads=0. Binding SHA256 is
82590a8f76eca27e7b15694b724123477e21ae0cc38ed2b33b56641ab9e70f26 and prompt
audit SHA256 isf4b5b21638bc2d8a8da4fac707072208d3a8a1aa2f1918c7574a326cc996bf7d.
Actual tokenizer class isQwen2Tokenizer, len151669, versus model vocabulary
151936 (distinct quantities). Actual scorer SHA256 is
5a22e1c1ff7b0a56623931c0782f81dd2b3a9df83ccfe77ce3724d37cf320086.
Max input2237 and max full-prefix2248 occur in the DEV long_context_retrieval
and lost_in_the_middle cases, each cap12. Full matrix cap is6348 calls and
1248966 processed prefix positions. One2248 FP32 eager attention-score tensor
is308.4MiB; this calculation excludes softmax/intermediates and is not measured
peak memory. Max-shape mechanics remainunrun.

There are13 provisional DEV scenario clusters for16 items: one shared cluster
has three paraphrases and another has two long-source tasks. SMOKE has8
scenario clusters. Five crossSMOKE/DEV near duplicates are flagged and no
normalized exact duplicates. Prompt-copy wins3/16DEV (general_reasoning,
multi_hop_binding, unanswerable_abstention) and2/8SMOKE (multi_hop_binding,
unanswerable_abstention). Best tested blind constant solves1/16DEV and1/8SMOKE.
These are the original bank/scorer floors and require contextual interpretation;
none demonstrates semantic reasoning or solver competence. The later source
edit only clarifies metadata forecast wording (historical-cost estimate or
qualified pilot); binding-001's source digest and outputs remain historical.
