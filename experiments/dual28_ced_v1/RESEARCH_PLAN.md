# Research plan and evidence boundaries

## 1. Question and relation to existing work

The user's first revision was to establish architectural viability at the original
source dimensions before combining it with width or FFN compression. The second
revision, implemented here, gives each architectural role a complete pretrained
28-layer stack. Both copies start from exactly the same pinned checkpoint.
Combining different family members or tokenizers is a different experiment.

The parent FWT-001 experiment adds reads from the midpoint of one intact 28-layer
backbone. It remains the smaller interface experiment. D28-001 adds a complete
encoder alongside a complete decoder and exposes a separate serial pure endpoint.
Neither experiment supersedes the other's receipts or authorizes the other's runs.
The original plan remains at `../full_width_transition_v1/RESEARCH_PLAN.md`.
Historical RAB-001 split-conversion/recovery observations remain exploratory and
are not rewritten as evidence for this new model or treated as impossibility laws.

The hypothesis is that preserving complete donor depth and original dimensions
may make a useful encoder/decoder specialization easier to learn. Identical copies
do not initially provide independent knowledge. Greater capacity and compute are
real confounders, so a gain over a frozen 28-layer model alone would not establish
an architectural advantage.

## 2. Exact warm start, rather than a shape-only claim

H56/P56 copy every source encoder block to E0..E27 and separately to D0..D27.
Their original causal attention and SwiGLU FFNs remain intact. They share one tied
vocabulary embedding/output tensor, not their block tensors. The decoder starts
from embeddings; the encoder affects it only through the new branches. Those
branches have donor-initialized Q and normalization and zero output matrices.
A zero path preserves a function but says nothing about the quality of the encoder
copy. P1 therefore checks *all* donor values/storage identities and temporarily
activates the outputs at the fixed scale 0.001 before testing the pathway.

Memory normalization copies source final_norm. Memory K/V and K normalization
copy layer 0. Encoder final residual states, before that bank normalization, are
the memory input. Read j copies the input/Q norms and Q projection of donor layer j.
There is no second zero scalar gate. Initially only the read output matrices can
receive gradients; subsequent updates must reach every other interface tensor.
The source block parameters are independently cloned, not aliases into the source.

C56 is a separate hard conversion: encoder layers 0..27 are retained; decoder read
blocks 0..27 take Q/O/FFN/norms from donor layers 0..27. The decoder starts from the
encoder residual stream. Decoder-private K/V parameters are absent. Width, head
dimensions and FFN sizes are unchanged, but the computation is not the source
computation. No zero-training behavior or recoverability guarantee is made.

C28 is constructed afresh in this package using encoder donors 0..13 and decoder
donors 14..27, the same source-final-norm memory convention, and memory K/V donor
14. It is not a reuse or relabeling of an old C28 receipt. The C28/C56 contrast
changes role depth and donor assignment together; it is not an isolated scalar
"depth effect" with every other factor held fixed.

D56 retains the original 28-layer path followed by another 28 donor-initialized
ordinary blocks. Added attention outputs and FFN down projections are zero, so
these extra residual blocks initially implement identity. Naive serial stacking
is not used as the ordinary baseline. Useful extra depth still requires training.

## 3. A hybrid-to-pure bridge remains an independent research problem

Removing H56 self-attention is not sufficient to obtain C56. The models differ in
where reads occur, which read projections are used, and whether decoder input is
an embedding or a final encoder state. A future migration must separately review
and register attention replacement, residual-input transition, donor mapping,
training schedules, retention stops and the exact endpoint. There is no silent
interpolation switch, pruning command or pure-CED export in this version.

An improved hybrid would establish that an additional complete causal encoder can
supply useful memory under the tested recipe. It would not establish that private
decoder attention can be removed. Conversely, a damaged hard C56 initialization
would not prove that an adequately trained full-width architecture is unworkable.
A short smoke or tiny-data null cannot settle recoverability.

## 4. What the present executable stages settle

P1 is mechanics on the exact ungated Supa source and pinned CPU FP32/eager runtime.
The imported source loader is hash-pinned, local-only and unchanged. It verifies
all source asset hashes, the 310-tensor BF16 source inventory, parameter count and
mapping before conversion to the declared FP32 reference. Native parity precedes
new-arm construction. The run is bound to implementation, dependencies, fixtures,
source assets, complete runtime fingerprint and exact clean tracked checkout head.
There is no GPU, BF16 execution, quantized source or replacement checkpoint fallback.

P1 checks source-value inheritance, independent storage, exact budgets, bitwise H/P
and D56 identity, full/token/chunk decoding, active future-token independence,
repeated positions, near-limit RoPE positions, padding and packed-segment isolation.
H/P inactive identity is not allowed to hide broken active code. C56 also must
actually respond to a memory-value intervention. Serialization is create-only and
H/P/C reload outputs must match bitwise. Checkpoints and failed attempts remain
in their new attempt directory; the source is never overwritten.

P2 is explicitly **25 steps for H and 25 for P**, not 25 examples of validation or
registered recovery. Both use the same deterministic row order and optimizer
settings. Full-paragraph next-token CE supplies a learning signal; matching a
teacher clone with pure KL alone would initially be stationary in exact arithmetic.
Teacher KL here constrains behavior on the engineering rows, not general knowledge
retention or correctness. Eight original artificial paragraphs are engineering
fixtures, not an adequate recovery corpus or scientific benchmark.

P2 must see finite optimization, nonzero gradients in every added tensor, awakened
outputs, unchanged frozen encoder/decoder tensors, matched trainable budgets and
post-update causal/cache mechanics. A loss decrease is not required for this tiny
mixed-row engineering exercise and would not be a capability result. No smoke
checkpoint is released or promoted to a recovered model. A PASS opens no campaign.

## 5. Requirements before an efficacy campaign

A later, separately reviewed registration must supply family-disjoint training,
development and sealed manifests, source provenance, contamination checks and
human-reviewed scoring of raw outputs. Define meaningful closed-context reasoning
or retrieval endpoints alongside retained abilities demonstrated by B28. Record
ordinary continued-training B28, not just frozen B28. Specify replay and teacher
policy; narrow-task KL is not broad retention insurance.

Set minimum worthwhile effects, per-family regression margins, checkpoint selection
and stopping rules before outcomes. Perform a cluster-aware power calculation on
independent task families. Repeated paraphrases and training seeds are not additional
independent test questions. Use at least three genuine training seeds; account for
failed and interrupted attempts and equal tuning opportunity across arms.

H56 vs P56 is the first matched-interface capacity contrast. Include an appropriately
trained ordinary continuation reference. C56 vs D56 tests different organization at
similar depth, not exact parameters/FLOPs. C28 vs C56 tests the full-donor-role proposal
with different depth and donor assignment. Report total/trainable parameters, token
exposure, optimization policy, wall time and peak memory rather than forcing a match
by changing FFN widths. An unequal optimization budget must not masquerade as a fair
architecture comparison. The current code contains no such campaign runner.

After training, disabled-memory and appropriate memory interventions must test actual
usefulness, alongside untouched retention evaluation. Nonzero gradients, nonzero
outputs, fluency, CE or teacher agreement alone do not demonstrate intelligence.
A useful hybrid that fails at the pure endpoint remains a hybrid result.

## 6. Resources and causal semantics

The exact analytic unique-parameter totals are in README and `inspect`. H/P retain
56 private layer caches plus one shared bank; C56 retains 28 encoder caches plus one
bank. Thus C56 has 29 banks versus B28's 28, not a cache saving over B28. It has fewer
banks than D56's 56 under the same dense dimensions. All counts exclude temporary
attention tensors, gradients, optimizer state, allocator overhead and model weights.

The current full-width reference is unoptimized dense attention. Tail-only mode in
DualStack saves only final-logit work, not encoder or decoder block work. Cache
validation also intentionally favors correctness over speed. No latency or GPU-fit
claim follows. The encoder processes each generated token as it joins history;
freezing its parameters does not make it a one-time prompt-only computation.

During training, position t predicts token t+1. Encoder and decoder reads are bounded
by both physical order and allowed position/segment masks, so later encoder states
cannot leak a target into an earlier query. Packed segments need unique IDs per
independent document. Cache/context limits are enforced; eviction is unimplemented.

Qualification keeps three mechanical checkpoints on disk and reloads one model at
a time. Its pinned CPU lane is not an 8-GB GPU qualification. A future efficient
training runtime requires measured resource preflight and fresh mechanics before
introducing quantization, adapters or a changed precision/backend. Compression and
cross-family donors remain separate, deferred interventions.
