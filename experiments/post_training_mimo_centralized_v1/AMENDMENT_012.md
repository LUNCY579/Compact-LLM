# Amendment 012: qualify the actual unpadded single-sequence CE runtime

30 September 2026, America/Chicago. The headless admission controller exhausted
its ten-minute window and did not invoke a native supervisor. Preserve that actual
wait and the missing native evidence. The user's authorized laptop plan remains
the operational route; do not relaunch below its registered resource envelope.

Source review identified a specific scope mismatch. E1_PLAN_V1 fixes microbatch
one and accumulation twenty; e1_data requires zero padding and all-one attention
masks, and DEV input exports also use unpadded prompts. Native padded-forward
equivalence is broader than those prepared completion-CE inputs. The original
E0 padding failure remains a failure; it is not converted into a full pass.

Prospectively register UNPADDED_SINGLE_SEQUENCE_CE_V1 with qualification stage
unpadded_ce under LAPTOP_STREAMED_V1. Primary forwards use one contiguous sequence
and omit the redundant all-one attention mask. Validate a supplied mask before
omission: reject any non-one entry, mismatched shape, batching above one, packed
sequence metadata, cache/past state and custom position arguments. Do not infer
that an omitted mask makes a padded or packed row eligible.

Retain exact model/tokenizer/template/value binding, constructor and adapter
initialization order, 32 base and zero-adapter comparisons, positional EOS/PAD
label mechanics, independent causal-shift completion CE, finite confined gradients,
all three 256/512/1024 profiles, five disposable optimizer calls, unchanged base
and bitwise within-stage save/resume. Native unpadded causality is mandatory and
separate: use the original 80-token fixture, perturb its final three tokens and
require bitwise identity of earlier logits. Also compare a fixed explicit all-one
mask with the omitted-mask reference at the same shape, retaining the original
maximum 0.03125 and mean 0.01 likelihood tolerances and finite-value requirements.
The diagnostic actually passes its all-one mask after the same input validation;
it is not a comparison of two normalized omitted-mask calls.

Generation budget is zero; sampled online/offline behavior likelihood is not
qualified by this CE-only stage. Its omitted sampler draws and resulting RNG
trajectory are disclosed as in AMENDMENT_010. Native padding, packing, batched or
cached inference, probability-based RL and generation capture remain unsupported
or unqualified. The eventual evaluator needs a separately frozen one-prompt
decoding/capture contract; the current structural importer is insufficient.

Retain the fixed-buffer loader, 1.25 GiB system reserve, 1.75 GiB worker RSS cap,
dynamic admission, existing GPU/disk floors, 1800-second/five-call bounds and
external containment. This change qualifies a concrete supported operation rather
than changing model scale, data dose, optimizer, numerical thresholds or the
capability/retention research objective.

Success must use PASS_E0_UNPADDED_CE_SCOPED, distinct from a full E0 pass and all
resource-only passes. e1_authorized remains false. Before E1, bind a reviewed
scope-specific evidence validator and current native plan, qualify actual TRAIN
inputs and unchanged-source capture, and satisfy preservation, language grading
and long-context gates. The historical E1 checker remains fail-closed for this new
config. No scope registration, CPU test or future partial pass opens CHECK/FINAL,
promotes a recipe or claims intelligence gain.
