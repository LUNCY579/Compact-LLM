# Amendment 022: exact-prefix teacher-forcing capture qualification

Register a distinct, bare-Titan scoring/capture mechanics scope after the fixed-
shape causal diagnostic passed. The original full-sequence bulk likelihood check
failed at its fixed maximum/mean limits 0.03125/0.01. Those limits, source, config
and negative records remain unchanged. This amendment neither relabels that
failure nor claims to identify its numerical cause or prove shape invariance.

Use the exact frozen BF16 Titan, eager/eval/no-grad execution, no adapter/cache,
the qualified direct-CUDA 2 MiB streaming loader and pinned tokenizer/template.
Keep the original non-evaluation mechanical prompts at 128 and 384 tokens, the
128-token generation cap and total envelope at most 512. Compare tokenizer-built
fixture IDs with prospectively frozen IDs/hashes before any model construction.
The 256/512 finite probes remain. Answer correctness is not a mechanics gate.

Online generation uses the raw full-vocabulary FP32 argmax of the BF16 last-logit
row, lowest token ID on ties, with no processors or sampling. Persist each emitted
token, exact preceding-prefix hash, raw BF16 logits hash/shape/dtype, raw argmax
witness and selected model logprob durably. Greedy behavior logprob is separately
0 and must not be compared as a model probability. Stop at the first newly emitted
EOS or exact cap; retain terminal EOS in generated IDs. EOS inside the prompt is
not a generated termination event. Preserve raw and visible output, removing only
one terminal EOS ID from visible decoding, without cleanup or filtering.

Independently teacher-force each recorded target using prompt IDs plus only the
previous generated IDs: target t is generated[t], input is prompt+generated[:t].
Use a fresh forward, cache off and logits_to_keep=1. Never reuse online logits or
model logprobs, and never decode/reencode to rebuild inputs. Require matching
prefix IDs/hashes, shape [1,1,151936], BF16 dtype, all-vocabulary finite values and
bitwise raw-logit hashes. Compare selected raw-model logprob maximum AND mean
absolute error at prospective 1e-4. Freeze these requirements before physical
results; do not change a tolerance after failure. Include the terminal EOS target.

For observed generation lengths N1,N2 in 1..128, completed forwards are
2+2*(N1+N2), logical calls are 312+forwards, and processed input positions are
768+2*(128*N1+N1*(N1-1)/2+384*N2+N2*(N2-1)/2). Maximum: 514 forwards,
826 logical calls, 164352 input positions and 256 generated tokens. Count every
physical attempt/return before later validation or journal failure. Partial online
and replay costs are independent; the completed formula never replaces partial
accounting. Retain failed/pending calls, partial tokens, witnesses and errors.
There are zero optimizer updates, training positions and teacher/grader calls.

Keep worker RSS <=2 GiB, system reserve >=1.25 GiB, pre-worker available RAM
>=3.3125 GiB, GPU admission free >=4 GiB and runtime free >=0.5 GiB, disk free
>=4 GiB and total worker time <=600 seconds. Use the existing external 100 ms
Windows Job watchdog. Require two current full-budget pre-worker observations,
fresh checks before launch, and the registered 60-second resource-only post-import
wait with two consecutive dynamic-budget observations and a fresh pre-load check.
No model retry follows timeout. Do not terminate other applications or GPU jobs.

Bind clean committed source/config/runtime/asset bytes and the original failed
trace plus completed causal diagnostic. Recheck frozen base hashes, assets and
source after success. Require natural model EOS in at least one fixture; CPU
controls cover both EOS IDs, EOS-at-cap and cap-only paths without forcing native
EOS. A distinct scoped PASS would establish this exact-prefix greedy capture and
scoring mechanics only. It would not establish the old bulk method, stochastic RL,
adapter capture, long context, calibration, DEV enrollment or production budgets.
DEV, CHECK, FINAL, language grading and E1 remain closed until separately qualified
and registered. The full one-Titan acquisition/retention objective stays intact.
