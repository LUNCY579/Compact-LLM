# Independent raw GPU qualification audit

Scope: read-only, standard-library audit of native-gpu-reference-001. This auditor
started no model worker, imported no model/Torch code, changed no source/assets,
and read no protected/TEST material. Only this report is mutable.

**Finding: the recorded GPU reference-operator qualification passed; DEV did not
start because the complete remaining-cost forecast refused.** The run's final
worker status is FAIL/exit2 and its parent status is WORKER_NONZERO. Neither is a
completed scientific comparison, an adaptive benefit, or a production claim.

## Binding and identity

The copied admission SHA256 is
5f9f2d2921ea9cb49d6a9f5fa0748964c3ac628c9b90268bc3cd267471524238.
Both copied review and registration bind this exact hash. Review status is
PASS_GPU_FP32_DISPATCH; its approval applies to the recorded source hashes and
built-in qualification/cost gates, not model efficacy.

Recorded inherited HEAD is7ece5cead285ba65c7d73ab57c0327fae2dc40bb; campaign code
was uncommitted and bound by per-file digests. Source_manifest binds the same
recorded dispatch code. Its dictionary does not duplicate five admission entries
(data JSONLs, AUTHORITY.txt and two binding metadata files); those entries are
instead bound explicitly by the copied admission/binding/registration.

The audited GPU native source digest is
4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f.
The admitted GPU parent digest is
227888015222897c126fa28699a00b3f083d94a64d1f50937b8cc964dc8a5a6c.

At audit time all current admitted file bytes still match, except
GPU_HOST_REGISTRATION.md, which has already received the prospective resource
allocation revision. That new document requires a new admission/review binding;
the old qualification remains attributed to its original copied admission.

All ten canonical model/tokenizer/config/custom-code asset hashes were freshly
recomputed read-only and match the admission, including model.safetensors
a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026.
The loaded campaign-local custom-code copy independently hashes to
b77b7d709fac3690d86f5978d46de409fd7ad3548c66a9d400188c225d99fa3d
and equals the canonical source file.

The worker records one native load, custom Qwen3AbstainForCausalLM, zero trainable
parameters, loaded FP32 state
b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551.
Qualification before/after state hashes are equal. This digest covers state_dict
parameters and persistent buffers; it is not a separate hash of nonpersistent
buffer contents. Exact state equality is a native receipt, not re-executed by this
no-model auditor.

## What the source-executed qualification checked

The source checks all parameters are frozen FP32 on cuda:0 and all buffers are on
that device, with floating buffers FP32. It requires the actual loaded custom
class/code, abstain token51378 and bias5.156249985098839. Registered runtime is
Python3.11.15, Torch2.12.1+cu126/CUDA12.6, Transformers5.14.1, eager attention,
TF32 disabled and CUBLAS_WORKSPACE_CONFIG=:4096:8. This is a separately qualified
GPU successor, not the historical CPU execution reproduced.

Raw outcomes independently reproduce24 pairwise token/response/EOS identities:
eight cases each for no-extra legacy/new, unit legacy/new and original/reference
A3 versus successor A3. The source compares actual first-token logits using
torch.equal at each pair and aborts on mismatch. Full logit arrays or byte hashes
were not journaled, so this auditor cannot independently recompute that numerical
array equality from saved bytes; it is supported by the executed source/qualification
receipt. The same limitation applies to maximum-shape logit equality.

The longest bound input has2237 tokens. The source extends its actual input IDs by
eleven punctuation tokens to2248, without reading a gold continuation or scorer
target. A retrieval value may legitimately appear in the original evidence; the
non-gold extension claim concerns appended tokens, not absence of answer strings
from evidence.

At2248, source-executed checks compare intact/noextra, legacy/new unit and original/new
A3 logits. An intentional exception occurs in the wrapper sink during attempt7;
after cleanup the layer methods equal their saved originals and intact logits
are reproduced. The source also rejects residual recurrence owner tags, gradients
and any state/assets/custom-code mutation immediately before writing qualification.

Two raw cache probes reject use_cache=True and a cache object even with
use_cache=False. Both record wrapper_forward_count0 and model_compute_started=False.
They are API attempts, not completed model forwards.

Runtime hooks enforce L0–L2 exactly once per completed full-model call; native L12
positions equal CUDA arange(0,prefix_length); the full-prefix input mask is all ones.
Replay uses the identical position IDs, position-embedding and causal-mask objects.
Causal-mask device and dimensions are checked. Full triangular-mask values were
not separately compared, and the receipt explicitly retains that limitation.
Qualified prefix generation remains cache-free, batch1, unpadded and <=2248.

All A0/A1/five fixed-grid/A3 scaled-arm smoke gates and maximum-shape attempts passed.
A5 is **PENDING_DEV_POOL**: the generation-derived DEV marginal does not exist, so
its required fresh frozen-pool smoke and maximum-shape qualification remain unrun.

## Raw work and cost recomputation

The raw journals contain112 physical outcomes, all SMOKE, and no DEV rows:

-48 identity generations:8 cases ×6 legacy/new/reference paths.
-64 arm smoke generations:8 cases ×8 arms.
-713 completed generation model calls; their summed prefix positions are27,669.
-15 completed maximum-shape calls,33,720 prefix positions.
-One intentionally failed maximum-shape attempt at2248,0 completed-model calls;
  partial computation remains UNKNOWN and its observed time is0.3121182 seconds.
-Two cache-rejection API attempts,0 completed model calls, no model computation.

Thus the run made731 API attempts,728 completed model forwards and one partially
executed intentional failure. Outcome model-call counters sum to713, matching the
generation journal. Output-token counts also sum to713 and include emitted EOS.
All gold-answer/NLL fields are null. The synchronized completed-forward time sums
to58.2457571 seconds; this excludes loading, source/state hashing, other diagnostics,
journaling, the partial attempt and parent/bootstrap overhead.

Parent model-worker charge is145.343 seconds, including the entire failed run;
worker receipt wall time is141.1023058 seconds. Parent supervisor wall time is
146.968 seconds. There is no hidden zero-cost qualification or failed attempt.

Resource snapshots/peaks recompute as follows:

| Measurement | Bytes |
|---|---:|
| Parent process-tree RSS peak,124 samples |2,266,378,240|
| Largest recorded worker RSS snapshot |1,639,735,296|
| CUDA allocator peak allocated |3,252,241,408|
| CUDA allocator peak reserved |3,573,547,008|

The allocated peak is below the registered5GiB ceiling. Parent RSS and worker
snapshots have different scopes. Snapshots are not guaranteed transient host peaks;
global GPU inventory is not per-worker allocated memory. Instrumented smoke timing
does not establish deployment speed or a CPU/GPU matched-quality speedup.

## Forecast independently recomputed

SMOKE completion caps sum to192 and DEV caps sum to214. The prospective complete
matrix therefore has:

    Physical case calls =15*8 +18*16 =408
    Generation-call cap =15*192 +18*214 =6732
    Generation-prefix-position cap =1,267,542
    Maximum-shape API attempts =17
    Maximum-shape attempted prefix positions =17*2248 =38,216
    Cache-rejection API attempts =2
    Overall model API attempt cap =6732+17+2 =6751
    Retrospective constant-oracle rows =32, no extra physical calls

Of the17 maximum-shape attempts,16 were attempted before DEV; the seventeenth is
the future frozen-pool A5 maximum check. For the point of refusal, remaining work
is18*214 DEV generation calls,192 A5 smoke calls and one A5 maximum call =4045.

The maximum observed synchronized completed-forward duration is0.876529 seconds,
independently matching the saved forecast. Therefore:

    4045 *0.876529 *1.5 =5318.3397075062385 seconds forecast
    141.1016373 elapsed +5318.3397075 forecast +1800 reserve
        =7259.44134480624 seconds required
    Required minus7200 admitted deadline =59.44134480624 seconds

The refusal is correct under the registered reserve rule. No DEV outcomes were
discarded: none was measured. The forecast is a conservative estimate from the
observed largest diagnostic forward, not an assured runtime bound or speed claim.

## Smoke mechanism observations only

A0, A1, each of the five fixed alphas and A3 each score6/8 with8/8 EOS and zero
safety stops in their smoke rows. These are qualification outcomes; they must not
select or promote a controller or establish H1/H2/H4 efficacy.

A3's52 extra-update smoke-prefix alphas range1.1416130066–1.1661778688,
mean1.1547713394; none is upper-saturated at1.2. This rules out exact fixed1.2
collapse on this inspected smoke trajectory only. It neither establishes useful
adaptive headroom nor proves that the feature predicts task improvement. There is
no DEV-selected fixed comparator, frozen A5 pool, complete control comparison,
oracle headroom, retention/abstention analysis or useful latency tradeoff yet.

## Prospective10800-second allocation

Completed native charges across the campaign are18.734 CPU-refusal seconds plus
145.343 GPU seconds =164.077. Remaining initial allowance is21,435.923 seconds.

One10800-second dispatch ceiling plus1800 finalization reserve fits, leaving
8835.923 seconds after reserving that ceiling and reserve. Two full10800-second
attempts plus the reserve do not fit the21600-second total. The single authorized
infrastructure retry must remain conditional on actual settled costs and a complete
matrix forecast fitting the remaining allowance/reserve. A late failure could make
that full retry inadmissible.

A larger job-time allocation with the same scientific matrix is a prospective
resource revision; it needs new admission/source/review bindings and the same fresh
mechanical/smoke/cost gates. It must preserve this failed run and qualification,
keep A5 qualification pending until its actual pool freeze, and add no native depth,
dtype, data exposure or controller revision merely to fit the budget.

## Artifact hashes and audit limits

| Run-relative file | SHA256 |
|---|---|
| admission.json |5f9f2d2921ea9cb49d6a9f5fa0748964c3ac628c9b90268bc3cd267471524238|
| review.json |10dc0d0842163ec9eb4eb825de2fef5a2b7cf29ecef586526fca89d6837d6a8d|
| source_manifest.json |add0bf496b25fe4ea760d6ad310815f31852bcd50180294f8699153ad052f700|
| terminal_receipt.json |aadb5170c82dd4069f78e4b33403ea7b034bf8916f3145b161bf0dfe39c06792|
| result/registration.json |71d6ee323da356f25a45aadb17a5d1d1c0f2e9068624b6f41026af78a0533e0e|
| result/qualification.json |d697df5823cf4a6cf45f825082415450efd5ad58d4813c79a0eead8691bf065b|
| result/matrix_forecast.json |d35e518ad1a30badc7f05e471f6d9ab0cf935a19432ac91ef9baf5b39d4aeb57|
| result/receipt.json |9b855ab74a73b026ee8742860f8715df3bc180f1b443653142547e7d97efdb70|
| result/forwards.jsonl |f2862c4d3a699bfab4f0bed02211eea7acc4da5ac1c3d7be572f77bd51420f57|
| result/outcomes.jsonl |81dff6d3ccf4ba4a12b34efbe5029cfc1e310ce7a20befc9ff3f6af1c9394eb9|
| result/trajectories.jsonl |e1686e1d60baa5dfb5fe298ea9b104a72a4f964a36d3041c748f8ce4e6fae8b4|
| resources.jsonl |10f08de94175bd681c357c9761a17da02dc0f1d9ce35c1402715ea1e77a4f9fd|

This report independently recomputes raw journals, hashes and forecasts. It does
not issue a code-author or dispatch-review PASS, rerun the model, verify unlogged
logit bytes, certify every nonpersistent buffer's final contents, or establish
scientific efficacy. The next gate remains a newly bound/reviewed resource revision
and complete qualified native DEV matrix within the cumulative contract.

