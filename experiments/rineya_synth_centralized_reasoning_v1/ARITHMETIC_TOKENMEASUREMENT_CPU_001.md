# Arithmetic tokenizer fit measured and projection corrected

The actual tokenizer-only run measured all512 arithmetic rows with the pinned
local Titan tokenizer and nonthinking chat template. No overflows occurred:
maximum225 prompt tokens,9 completion tokens including EOS and234 total tokens,
within480/32/512. All prompt/target/EOS/newline joint boundaries and decode
roundtrips were checked. Policy capture IDs remain prompt-only; gold arrays,
labels and masks are separate exposed CPU oracle diagnostics.

[Original run](evidence/arithmetic_tokenmeasurement_attempt001_20261009/ROOT_PRESERVATION.json)
preserves four output files and exact reviewed source/current inventory. Current
1,392-file/104,065,928-byte identities matched before/after. The original
cooperative ResourceWatch measured28.895209s and161,841,152-byte peak working
set after output writes, within600s/512MiB. These are CPU mechanics and process
measurements, not hard containment or native model-resource qualification.
Seven inert wrapper controls and independent actual byte/row/mask audits pass.

A late review found that the original projected metadata retained the full-result
SHA after separating arrays. That original standalone seal is unsupported and
remains preserved. The [additive corrected projection](evidence/arithmetic_token_projection_reviewed_cpu_20261009/ROOT_PRESERVATION.json)
has its own seal, explicit original full-result digest and byte-bound policy/
oracle references. Pure JSON conversion verified both seals/full rejoin and
original file preservation; no tokenizer repeat occurred. Three authored/Root/
independent controls cover stale references, malformed metadata and bounded
reads through the helper's `read_join` entry. The actual additive converter used
`read_bytes` before validation; those bounded-read controls do not qualify the
complete converter. Its valid corrected output and original files remain
preserved. Future converter execution needs bounded, prospectively selected
reads. Future runner source is retained separately and still requires a fresh
current byte selection before execution.

No weights/model config, framework/model/GPU/optimizer, native scoring or M0
enrollment was performed. Token fit is now measured; scorer/model qualification,
available-lineage disjointness, source-correct membership and retention precision
remain pending. Qualified dialogue and separate long-context authority remain
open. The M0/M1/M2 transfer result is still unanswered.
