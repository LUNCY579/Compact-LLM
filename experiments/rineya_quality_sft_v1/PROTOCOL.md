# RQSFT-001 prospective quality successor

User-approved direction: balanced quality, existing 0.6B first, verified data and
SFT first, preserve key skills. This is a new experiment, not an amendment to MQ,
DSD or RRP. The user requested implementation and conditional native comparison;
actual data admission/qualification remain scientific prerequisites.

Exact source is ungated Supa1.7,596,049,920 parameters, weight SHA256
962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b.
Frozen source versus whole-sequence and final-assistant-response+EOS SFT. Train
exact blocks8..27 (314,618,880 parameters/220 tensors), freeze everything else.
FP32 eager execution with forced math SDPA, deterministic, no TF32; LR1e-5;
WSD warmup20/stable60/decay20 to.1peak;
AdamW(.9,.95),epsilon1e-8,decay0,clip1, CPU FP32 moments, microbatch1. Reuse audited
numerical primitives, never another experiment's permit. Source/optimizer/mask
changes need fresh binding and qualification.

TRAIN200: grounded80, solving60, preservation_dialogue40, output_contract20.
DEV512: acquisition128, harder_transfer128, grounding_contracts128,
preservation_dialogue128. Dataset owner builds/admit exports; this consumer
generates no real bank. Rights/gold/pair/family/scenario/lexicon/shortcut reviews
must be actual, independently attributable and byte-bound. Developer fixtures and
EXCLUDED ancestry can never be upgraded to TRAIN. No protected-bank reads.

Messages include complete histories and a final assistant target. Prior assistant
turns remain in input. Response labels cover only final answer+EOS, never prior
answers/header/thinking stub/trailing newline; whole labels cover all next-token
positions. Explicit verified empty targets supervise EOS only; missing capture
is not a valid empty answer. Causal shift occurs exactly once. Native rendered
prefix/full template token IDs must agree with the native chat-template API.

Consume ALL200 TRAIN rows once, no prefix subset or outcome selection. Parent
groups (including answerability pairs) remain atomic during seeded17 ordering and
update grouping. Each training full sequence<=512, group processed tokens<=2048,
total<=25,000, updates<=32. Oversized/underfilled quotas or incomplete parents
refuse; no truncation, padding for budget, silent dropped cases or refill.
DEV complete prompt+64 generation allowance<=2048; target+EOS<=64. Final
checkpoints only. Raw generations, EOS/cap behavior and per-case failures persist.

Five dimensions: solving, harder transfer, grounding/answerability, contracts,
communication. Literal UNKNOWN and semantic abstention are separate. Natural
abstention and communication require blinded, independent, prediction-hash-bound
adjudication; missing/uncertain results block advancement, not become false/zero.
Error categories require evidence, never infer missing knowledge/retrieval/execution
mechanisms from an unexplained wrong string. Historical reuse requires exact
input IDs/prefix/EOS/source parameter/runtime/generation identity and separate
attribution. Reuse raw text/tokens only, never old judgments; score again with
current independent gold and frozen evaluator. Invalidated procedural prompts
and history-stripped rows are excluded.

Advancement (DEV only): acquisition macro gain>=5pp; harder transfer gain>0;
zero new critical grounding failures (correct-to-wrong damage or a reviewed safe
refusal becoming an unsupported assertion on a critical unanswerable case);
source-success retention>=.98 with denominator
nonzero; macro preservation change>=-2pp; blinded communication delta>=0; complete
coverage/finite updates/termination/cost accounting. Report damage and repairs
separately. A smaller update alone is not retention success; no pooled seed/row
p-values or automatic release. Missing ratings or insufficient preservation
evidence yields INCOMPLETE/INCONCLUSIVE. Follow-up choices match the approved plan.

Native qualification: exact source/tokenizer/runtime/code/plan/owner reviews,
representative and maximum512 training backward,2048 evaluation forward/cache
parity, whole/response loss-gradient checks, frozen params/buffers, actual math
SDPA, clean-boundary checkpoint restore without updates. Qualification creates
no optimizer updates. Full official-state recovery must pass before relying on
resume. All stages independently bound; native fixtures cannot authorize them.

Resource minima preserved from audited MQ: training before6GiB host/6GiB GPU;
qualification before4.6GiB host/3GiB GPU; post-load1.5GiB host/.5GiB GPU.
Check complete checkpoint/disk requirements. Foreground owned worker/watchdog,
one local model job at a time, no foreign process eviction/stale-lock removal.
No automated retries. Preserve partial cases, all completed checkpoints, source
rehashes, worker failures/timeouts and supervision receipts. Cached generation is
measured, not assumed equal to other loaders. New byte/recipe changes invalidate
qualification. No model promotion or serving change in this implementation.
