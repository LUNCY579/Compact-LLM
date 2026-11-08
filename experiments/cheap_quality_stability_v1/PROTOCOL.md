# Prospectively frozen analysis protocol

The input set is exactly monitor_probe.jsonl, the machine reference-only label
audit, the existing typed scoring implementation/contract, and four completed
historical B28last20 lr=1e-5, 5,200-token, WSD 0.2/0.2, beta2=0.95 draws at seeds
17, 23, 29, 41. Their byte hashes and expected source asset identities are in
config.json. No CHECK, FINAL, sealed split, new reference, checkpoint binary,
or train corpus is opened. Train overlap uses only recorded consumed row IDs.

Before analysis: create a new run directory; bind exact input bytes, this script,
protocol, config, tests and README; record actual Python runtime and cached source,
data, scorer and harness identities in freeze.json. Verify all pins; then validate
exactly 150 unique monitor/audit/output IDs, full monitor row/messages/prompt hashes,
original references, family/class identities, cached monitor/audit hashes, expected
arm/recipe/completion/device-integrity fields, consistent initial raw outputs,
reference sizes, consumed-ID count/digest, initial/final stored totals and stored
permissive per-item correctness maps. Any mismatch invalidates the whole run.
The cached source/harness hashes are recorded historical claims; this study does
not reverify binary weights or historical native execution.

Rescore raw text plus EOS termination using the pinned retention.py, loaded
directly without importing the architecture package. Its only imports must be
the Python standard library math, re and fractions. Policy differences change
only the existing named bare-number convention. Exact typed rational/clock
comparison consumes the entire answer. Correctness requires EOS. Labels/aliases
are not fitted to candidate outputs. Preserve parsing reasons and raw answers.
Machine target review is still pending human review; the bounded parser excludes
some semantically correct free-form answers. Four historical entity-label issues
use the existing frozen audit overlay and are identified in the report.

For every item and policy: record source/final success, gain and loss per draw;
counts over four draws; stable gain = source incorrect and gained in all four;
recurrent loss = source correct and lost in at least two draws; universal loss =
lost in all four. Loss rate is conditional on the fixed source-correct set, with
denominator shown. Separately report persistent source failures, mixed outcomes,
termination changes, typed-value mismatch versus parser rejection, and policy
disagreements. Compare gain/loss sets for every draw pair using intersection,
union, Jaccard (null for two empty sets) and item IDs. Break down by reply class,
typed kind and audited arithmetic template; group_id is retained per item.
Compute recorded training-ID intersections and Jaccard without interpreting zero
train overlap as independence of repeated evaluation outcomes.

Report all draws and both policies. No best-seed selection, pooled p-values,
confidence intervals assuming 600 independent rows, causal schedule attribution,
or new model intelligence claim. The 150 prompts were historically exposed and
reused across draws, and several share arithmetic templates. Results prioritize
preservation targets and prospective controls for later resource-admitted native
experiments. A stable gain identifies historical repeatability on this panel;
it does not establish generalization. This study adds no parameters or model
size. Compactness and intelligence gains require subsequent native evidence.

One CPU process; no subprocess creation or GPU operations; standard library only.
Prospective limits: process tree RSS 512 MiB, elapsed 900 seconds, outputs 50 MiB.
Close hardware observation is supplied by the outer supervisor and must accompany
the result. The script performs a local elapsed/output check and preserves failure
receipts. Inputs limited to 4 MiB each to prevent unexpected JSON memory use.
