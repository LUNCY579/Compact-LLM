# Learning tokenization scope review 1

Reviewer `/root/runtime_audit`, 2026-10-02. **Eligible independent CPU prerequisite
in principle; actual token measurements remain closed until a separately frozen,
reviewed registration/source and reliable resource observation exist.** This is a
read-only scope assessment. No tokenizer/backend/framework was imported or called,
no model/weights/host database/probe/holdout/training operation occurred, and no
frozen artifact or quarantined helper was edited. Only this short report is added.

The owner authorizes local CPU tests and read-only relevant artifacts in the
retained pasted request. This work would measure accepted fixed TRAIN32 under
learning-data 837bd/contract 37f3; it neither implements an extra pipeline repair
round nor retries a failed worker. Dataset source/science acceptance and the CPU
scope's fresh non-builder review remain prerequisites.

## Smallest meaningful scope

Freeze one ordinary-adaptation user prompt made solely from model_input(case)
(source/query/rules) plus one fixed structured-answer instruction. Freeze ordered
TRAIN targets from training_targets: canonical status/value JSON, with no explanation,
procedure graph, gold/IDs/family/split/oracle/feedback in user content. Cover all 32
registered parents, draw 101/order 211; unsupported cases stay in the denominator.
Do not create another bank. The existing reader_codec is a DEV-only per-line compact
extraction codec; its90-span prompt/output schema is a different task and should
not be repurposed for this ordinary answer-supervision prerequisite.

Load the exact local tokenizers.JSON once through the lightweight tokenizers CPU
backend. Render only the pinned single-user and user+assistant branches with
Jinja2 sandbox/StrictUndefined and compare byte-exactly to a fixed manual subset:
no tools/system/multi-turn/assistant reasoning or arbitrary template selection.
Use enable_thinking=False. Both the generation prefix and final assistant branch
contain the fixed empty <think> block. Freeze content serialization before measuring;
reject source/target drift, duplicate fields, unsupported roles, embedded control
sentinels or any unregistered template branch. Instructions in source remain data.

## Boundary, EOS and mask controls

Encode the prompt-only generation prefix and the complete joined transcript with
add_special_tokens=False, no padding/truncation or implicit BOS/EOS. Require the
prompt-only ID sequence to equal the corresponding prefix of the joined sequence;
a BPE boundary merge fails the case rather than assigning masks by text/character
length or by separately encoded target length. Compare Jinja/manual render and
encode paths, decoded special-token-preserving representations, known special IDs,
Unicode/normalization and hostile control-sentinel fixtures. The backend declares
NFC normalization and ByteLevel postprocessing/decoding: character, UTF8-byte and
token offsets are not interchangeable. Accepted TRAIN32 is controlled ASCII;
normalization behavior on other input is a control, not a general roundtrip claim.

Explicit training EOS is one 151645 (<|im_end|>). 151643 is pad/endoftext and a reader
stop alternative, not a replacement training EOS. 151644 is im_start. The template's
final newline after im_end must be recorded in the full-render hash, then excluded
by an explicit predeclared training-sequence transformation after verifying it is
exactly that terminal suffix. Preserve both render/training-sequence bytes/hashes;
never silently strip or truncate content. The empty thinking/header region belongs
to the prompt and receives no target loss.

For joined training IDs with prompt lengthP, labels at positionsj<P are ignore
(-100); labelsj>=P own only canonical response tokens and the one terminal EOS.
The causal prediction ownership is shifted: logit positionj-1 predicts targetj.
Record unshifted labels, shifted ownership/mask, one-hot attention, selected EOS
position and exact counts; no prompt/header/post-EOS newline/padding loss. Controls
must detect off-by-one boundary/EOS masking, prompt leakage, extra EOS, target
alteration, nonfinite/malformed metadata and unsupported sizes. No tensor or loss
is computed in this CPU stage.

## Available actual dependencies and pins

Read-only installed metadata/source files confirm CPython 3.11.15, tokenizers 0.22.2,
Jinja2 3.1.6, MarkupSafe 3.0.3 and psutil 7.2.2. No install/download is necessary.
The checked tokenizers import wrapper uses the native CPU backend; this does not
attest every compiled/transitive dependency or guarantee unseen side effects.

| Selected dependency or asset | SHA256 |
| --- | --- |
| Base CPython executable | 28d2965d061656abbe3a46100bd3342d1152185289445d489782a21a8cfe9b57 |
| tokenizers/tokenizers.pyd (7395328 bytes) | 7acb83f5b89136597e0d14b788d82917bf2870df94575bf77e12731c4e49c4df |
| tokenizers/__init__.py | 510d5e23458612433da9f1fe430b5009fb88ba90508ee4340d2397b32c319080 |
| jinja2/environment.py | f6786b3fb0a1f8d6c65f4d30bf2af8cb2fae84d1ead8e09ceb48201ab9e2fdf9 |
| jinja2/sandbox.py | 330d9a8ad958d88f256bb15885c5f6606f41b5461c2e70f41a1ddddbd7035ab6 |
| MarkupSafe speedups.pyd | 755b26c10a4ece4acfb14a30201b5a5ff97b930784a0800b1bbc9bd93407a9fd |
| psutil/_psutil_windows.pyd | 0035450801bd7d938e9e146c5ec28e619cb5a5f4a18cdc53ac7e9734c7f94f78 |
| Titan tokenizer.json (11422654 bytes) | aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 |
| tokenizer_config.json | d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 |
| generation_config.json | 2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 |
| Chat template UTF8 (4168 bytes) | a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8 |

The three sidecar and template hashes were freshly verified at the existing
read-only Titan external directory. No model binary was opened. Read JSON as UTF8
bytes explicitly: an initial static inspection using Windows default cp1252 failed
on tokenizer.JSON; UTF8 JSON-byte inspection succeeded without loading it. Backend
padding/truncation areNull. Metadata model_max_length 131072 is not qualification
or permission to expand the512+64/max576 study envelope.

## Concrete ceiling and review before measurements

Recommend one disposable CPU subprocess, one loaded tokenizer, all 32 cases, at
most 160 registered encode calls plus 24 bounded syntactic controls, 60 seconds wall,
512 MiB observed process/tree RSS,1.25 GiB host reserve (prelaunch 1.75 GiB available),
32 MiB maximum tokenizer asset,8192-byte prompt text admission and 8 MiB total receipt.
Set TOKENIZERS_PARALLELISM=false/RAYON_NUM_THREADS=1 and offline flags. Use a pinned
ordinary subprocess/timeout with actual PID/time/RSS/cost observation and no child,
model/framework/API side-effect claim; reject unexpected descendants or unavailable
resource observation. Prefer the already pinned direct base CPython executable with
explicit pinned venv site-packages to avoid an unobserved launcher child. No new
containment framework or copying/changing quarantined native/probe helpers belongs
in this prerequisite. Source/fixture work may proceed if observation is unavailable;
actual token measurement remains closed.

Require prompt 1..512, target including EOS 1..64 and joined training length<=576,
without truncation/resampling. Thus 32-case processed-position accounting is at
most 18432 and supervised target positions at most 2048, below the preliminary
microstudy 32768/8192 ceilings. Count full teacher-forward length separately from
the served prefix needed to predict the last EOS; retain every failure and actual
CPU time/RSS. These are predeclared ceilings, not measured fit or peak claims.

Before creating an actual token artifact, independently review a new source/contract
pinning the accepted bank, prompt instruction/serialization, tokenizer/template and
CPU dependency files, exact Jinja subset/manual parity, prefix/EOS/shifted-mask
controls, all 32 denominator, hard limits, immutable receipts and reliable timeout/
RSS instrumentation. That review must be separate from this assessment and from
the builder. No actual tokenizer call was made here. Successful measurements would
establish reproducible CPU token IDs/fit/boundaries/target ownership for those
exact bytes only. They would not establish native serving parity, trainable loss/
gradient correctness, max-shape training fit, bulk equivalence, resume/export,
learning/retention or model eligibility. Existing frozen learning-data flags and
pipeline quarantines remain unchanged; a new dependent receipt records the result.
