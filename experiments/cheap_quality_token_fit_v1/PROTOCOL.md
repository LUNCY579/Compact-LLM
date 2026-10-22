# Frozen tokenizer suitability study

Population: inherited bridge allocator, fixed seed 20260929, exactly 128 TRAIN
parents and zero TRAIN_SELECTION, DEV, CHECK, FINAL quotas. Select canonical
parents before rendering or solving, by the original role/structure policy.
Verify all selected parents are TRAIN before any rendering. Preserve both public
render variants 0 and 1 unchanged and their identical canonical JSON answer.
Use existing independent verifier and positive/mutated-answer controls; record
selection counts and exact parent/generated-row hashes. No response is generated
by a model, and no sealed parent is rendered, solved or verified.

Bind exact bridge_data.py and direct serialization.py bytes. Load these files as
isolated modules after checking their standard-library import declarations; do
not import either experiment package. The fixed system message is in config.json.
Prompt messages are exactly system,user; mode direct; enable_thinking false;
thinking stub remains in the generation prompt and is never supervised.

Tokenizer assets are exactly tokenizer.json, tokenizer_config.json (including
native chat template and special-token definitions), vocab.json and merges.txt
from the declared local Titan directory. Verify their byte hashes and the chat
template UTF-8 hash. Copy only these four files into a new output asset directory
and load the tokenizer there, offline, without remote code. Thus AutoTokenizer
cannot open a model config or weights from the published core. Pin Transformers
4.57.1/tokenizers0.22.2, EOS token <|im_end|> and ID151645. Record executable,
Python, dependency versions, disabled backends and exact asset identities before
token measurement. No Torch/TF/Flax import, model instantiation or weight access.

Independent checks for each rendered row: template-rendered generation tokens
equal explicit encoding; full native assistant transcript is prefix + JSON target
+ exactly one response EOS + newline; full tokens preserve the exact generation
prefix; target token segment equals independently encoded JSON; one response EOS;
post-EOS tokens decode exactly newline; labels supervise only JSON and response
EOS, excluding system,user,thinking stub and trailing newline. Compare IDs,
prefix, boundaries and labels with the pinned inherited serializer for every
fitting profile. Store full IDs/masks as measured mechanics, not native-model
qualification. Unqualified serialization invalidates the whole study.

Three profiles: 512/1024/2048 processed tokens, each with 128 completion reserve.
Processed includes the native trailing newline; supervised includes target+EOS;
final content excludes EOS. Reject if processed exceeds profile, prompt+reserve
exceeds profile, or supervised completion exceeds 128. Never truncate, silently
skip, shorten the public task or change target. Preserve every rejection and its
reason. Run all128*2*3=768 decisions and account for each family/profile/variant.

Report retained/rejected counts, processed/prompt/supervised/final distributions,
reserve headroom, supervised fraction and token totals per pass. Report paired
variant1-minus-variant0 token differences on exactly the same parents/targets.
Rank efficient fitting rows within each family by fewer processed tokens per
supervised token, with deterministic ties. These lists guide later selection;
they are not a training policy or evidence of better learning. Preserve small
families rather than claiming higher ratio alone means better intelligence.
No inference on model quality, training convergence, native admission, hardware
requirements for training, or changes to existing token budgets/gates.

Fail closed on hash/runtime/identity/target/parity mismatch; create-only outputs.
Outer supervisor supplies measured hardware conditions and the 512MiB/900s/50MiB
limits. The script additionally checks local elapsed time and output bytes.
