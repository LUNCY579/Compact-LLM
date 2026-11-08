# Amendment 014: qualify frozen greedy capture before collecting DEV

30 September 2026. Complete CE512 mechanics and actual TRAIN source joins have
passed. That evidence generated no model tokens and grants no response capture,
language grading, long-context inference or E1 execution. The original padding
and 1024-token VRAM failures remain unchanged evidence.

Prospectively register CAPTURE_GREEDY_SHORT_QUAL_V1.json, a non-DEV frozen-source
qualifier on the same exact Titan BF16 base. There are no adapters, optimizer,
training updates, benchmark payloads or teacher calls. Preserve the direct CUDA
constructor, fixed 2 MiB streamed checkpoint loader, all 310 source/GPU value
hashes, tied embeddings, 596049920 unique parameters, tokenizer/template/asset
identities and pinned native packages. Keep eager attention, deterministic
settings, disabled TF32 and torch.no_grad(). Pin reused and new producer sources
plus the installed Qwen3 and masking source files before use.

The intended capture interface is one contiguous unpadded prompt, batch one,
validated all-one mask then omission, no cache/packing/custom positions, and at
most 512 total tokens. Compute raw FP32 last-position logits with logits_to_keep=1;
choose their lowest-ID argmax without sampling, inherited generation defaults,
penalties, suppression, processors or beam search. The constructor/runtime seed
is 20260928. Greedy choice probability is one, so its behavior log-probability is
zero; raw model-softmax selected-token log-probability is a separate diagnostic.

Qualify two fixed mechanical prompts at lengths 128 and 384 with a 128-token cap
each, giving total envelopes 256/512. Also run finite last-position vocabulary
probes at full input lengths 256/512. These probes do not claim all-position
vocabulary finiteness. Check each observed generated sequence against independent
teacher forcing: full prompt plus generated IDs, retain the last N+1 logits and
select their first N positions. Compare selected raw model log-probabilities at
the unchanged maximum 0.03125 and mean 0.01 tolerances. Preserve every discrepancy
and nonfinite value as failed evidence; never compare these values with greedy
behavior log-probability or silently change numerical thresholds.

Retain every emitted token, including terminal EOS151645 or EOS151643. Stop at
the first newly emitted EOS or the cap; EOS at the final allowed position takes
precedence over length termination. Preserve complete decoded output with special
token filtering and whitespace cleanup disabled. Scoring text removes only one
terminal EOS token by ID before decoding, preserving all other tokens, fences,
reasoning text and whitespace. Require at least one natural native EOS termination
across the fixed fixtures; report the actual EOS IDs and cap branches observed.
Both EOS IDs and cap/extraction branches need CPU controls but synthetic coverage
is never labeled native observation. Never force EOS, filter for answer correctness,
retry a fixture or change it after an outcome.

Use the distinct provisional LAPTOP_CAPTURE_RESIDENT_V1 resource profile: process
tree RSS ceiling 2 GiB, 1.25 GiB system reserve, dynamic admission
1.25 + max(0, 2-currentRSS) GiB, GPU admission/runtime floors 4/0.5 GiB, disk floor
4 GiB and a 600-second worker ceiling. Before-worker budgeting is 3.25 GiB plus
64 MiB supervisor allowance. The extra allowance budgets rendering, records and
driver overhead after CE512's observed 1.39 MiB margin; it is not measured fit or
model RAM demand. Keep all earlier 1.75 GiB and 7.4/6 GiB profiles unchanged.
Retain Windows Job containment, external 100 ms RAM polling, one GPU worker,
write-once outcomes and no automatic guard relaxation or retry.

Maximum accounting is 256 generated tokens, 260 forwards, 83328 forward positions
and 572 logical calls including the 310 tensor-copy spans. These are ceilings.
Derive actual counts from both observed rollout lengths and their ordered prefix
hashes; natural EOS must not be replaced with full-cap counts. Preserve attempted,
returned, failed and pending calls and incomplete fixtures faithfully. A success
requires source/config/model/runtime/cost/journal/containment consistency and
unchanged base/source assets. An exception never fabricates an unattempted row.

After this bounded qualifier passes, freeze a measured short DEV collection plan
for 256 acquisition and 256 deterministic-retention rows, 448 independent parent
groups. Each row has one greedy attempt and at most 128 new tokens, including EOS;
the longest prompt is 381, hence at most 509 total tokens. The worst-case ceilings
are 65536 new tokens and 17433472 full-prefix input positions. Do not infer wall
time or sustained throughput from CE512, which performed no decoding. Keep routing
metadata and evaluator references outside model inputs and the GPU worker.

Language remains behind its independently qualified rubric/labels/grader and
blinding/cost registration. Current unassigned packets or internal assistant calls
cannot supply those grants. Long-context 2024/4072-prompt capture with a 24-token
cap requires separate 2048/4096 inference qualification. Deterministic measurements
may precede these prerequisites but cannot qualify broad preservation or E1.
The original one-Titan capability/retention objective, CHECK/FINAL closure and
unchanged best recipe remain intact.
