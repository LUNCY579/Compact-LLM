# Exact-prefix deterministic reader qualification v1

This separately registered endpoint is `CRS_EXACT_PREFIX_GREEDY_SERVING_576_V1`.
It uses one frozen Titan BF16 checkpoint, stock eager Qwen3, batch one, unpadded
full prefixes, no cache/adapters/gradients, up to512 complete input tokens and64
emissions. Input length includes instructions/source/state/procedure/schema and
chat-template overhead. Unsupported inputs fail; nothing is silently truncated.
The original bulk capture remains FAILED at unchanged maximum0.03125/mean0.01
limits. New serving qualification cannot qualify bulk likelihood, training or RL.

The complete scientific design review is
`audits/READER_QUALIFICATION_DESIGN_REVIEW_1.md`. This implementation is source-ready
continuation, not a native receipt. Host RAM0.715GiB and unregistered Photos/Unity
GPU processes block admission. No model/framework is initialized by module import,
and no model run, tokenizer load, gold generation or resource exception is made.

Frozen upper schedule: direct/no-op wrapper parity at128,512,576 (6forwards,
2432positions); historical natural owned/stock/replay (9,1161);512-prompt natural
owned/stock/replay (≤192,104352);64 frozen mechanical prefixes512–575 plus a
nonemitting576 probe (65,35360); historical131 and long≤576 bulk diagnostics
(2,707); maximum576 baseline and three suffix controls (4,2304). Total≤278forwards,
146316positions and590logical construction/load/310copy/forward calls. Maximum
owned-plus-stock natural emissions134; replay and mechanical inputs are separate
exposure, never generated model tokens. Actual counts depend on natural EOS.

Natural serving stops on its first newly emitted EOS151645/151643, including EOS
at the output cap. EOS already in a prompt does not stop a new response. Raw
rendering retains special tokens and whitespace; visible text removes only one
terminal EOS by token ID. Historical output is frozen to16,22,151645 and drift
fails its scope. Long natural cap-stop may remain unobserved. The explicit64-token
continuation is labeled forced mechanical shape/resource evidence; it cannot
stand in for observed unconstrained64-token generation. Final576 forward emits
no65th token.

The owned loop and independent pinned stock `generate` run sequentially. Effective
logits processors must be empty; stopping criteria are exactly length and EOS.
Raw stock FP32 copies must preserve BF16 logits; processed scores are unused.
Raw lowest-ID argmax/token/stop/rendering, no-op forward logits/normalized hidden,
full-prefix replay and unchanged probability tolerances are checked. Passive
actual causal-mask/position/hidden witnesses bind every forward. Maximum576 suffix
controls require protected logits and normalized hidden raw-byte identity.

Tokenizer/template identity and rendered UTF-8/token equivalence are native-worker
prerequisites, deferred until admission; exact tokenizer sidecars/templates and
installed generation source files are pinned. Mechanical prompt streams are
prospectively fixed qualification inputs, not benchmark cases or gold retrieval.

`reader_execution.py` adapts the exact reviewed supervisor control flow into this
namespace, using source-pinned native fencing, Windows Job/query-handle transfer,
whole-tree RAM/GPU/time/process monitoring, durable idempotency and failure costs.
It never mutates or invokes the old D2 launch contract. Keep600seconds,2GiBprocess
RSS,1.25GiBreserve,3.3125GiBpre-worker, GPU4/0.5GiB,disk4GiB,100mswatchdog and
oneworker. Qualification/failures count within7200seconds/epoch and14400/rolling24h.
A timeout/unknown outcome requires reconciliation, not automatic duplication.

Only a complete source/model/tokenizer/runtime-bound worker result, durable
journal, fully successful contained supervisor and independent exact review can
publish `PASS_EXACT_PREFIX_SERVING_576_SCOPED`. A caller PASS dictionary or prior
D2 receipt cannot open serving. `ReaderSession.serve` requires that immutable
qualification bundle. CLI `--serve` stays closed pending separately registered
public request/DEV collection. Existing central CHECK/FINAL and all training
remain closed. The16-case DEV extraction diagnostic requires its own frozen
output-token fit and registration; complete event JSON may exceed64 tokens.

Executable continuation after fresh source review, clean commit and host change:

`python -B -m experiments.rineya_centralized_reasoning_system_v1.reader --preflight --contract-sha256 SHA --review REVIEW`

`python -B -m experiments.rineya_centralized_reasoning_system_v1.reader --qualify --contract-sha256 SHA --review REVIEW --output CRS/results/NEW_RUN --job-key FROZEN_KEY --epoch EPOCH`

Review must identify `APPROVED_FOR_READER_QUALIFICATION` and the exact binding.
These commands are gated; source readiness and CPU tests establish no native
serving, reasoning gain, calibrated confidence, procedure discovery or learning.
