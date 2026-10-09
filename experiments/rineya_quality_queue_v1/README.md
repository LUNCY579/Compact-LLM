# Rineya retention-first quality queue

This is an offline evidence index and proposal review tool. It never loads models,
tokenizers or weights, discovers task banks, dispatches owner workers, trains,
evaluates native models, promotes candidates or changes serving policy.

It preserves the five programme priorities: CE8 qualification, one-donor pruning
versus acquisition-matched scalar attenuation, matched MLP at the actual live
bias3.75, reviewed data/ordinary-SFT parent, then the RSD three-arm step2 screen.
Existing lane owners and their registrations remain authoritative. RSD, DSD and
RDD are distinct experiments. SRCL and twelve-draw pruning expansion are deferred.

## Build and inspect

From repository root, use an existing interpreter; no dependencies are needed:

```powershell
$py = "C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe"
& $py -B -S -m experiments.rineya_quality_queue_v1 build --output "C:/Users/clclc/BusinessProject/rineya-quality-queue-NEW/packet"
& $py -B -S -m experiments.rineya_quality_queue_v1 verify --packet "C:/Users/clclc/BusinessProject/rineya-quality-queue-NEW/packet"
```

Build is create-only and rejects output inside any tracked checkout or `.codex`.
Inputs are explicitly listed source/protocol/receipt/proposal metadata, not weight
payloads or task banks. Existing source worktrees under `.codex` may be inspected
read-only; every new output belongs under BusinessProject outside tracked source.
Run files and logs remain immutable. Choose a new packet after source drift.

`QUEUE.json` fixes priority, owner, blockers and a bounded next proposal for each
lane. Five `*.quality-card.json` files carry exact file hashes, source/runtime and
candidate status, historical measurements with denominators, exposure, registered
rules, and current preservation gates. `REVIEW.txt` summarizes the queue.
`evidence/` contains byte-identical listed receipts/protocols; `specs/` freezes the
card definitions and `software/` preserves the queue implementation. `MANIFEST.json`
binds all packet files. Models, tokenizers and independently supplied raw-bank or
response files are refused; source receipts may retain their historical diagnostics.

Verification checks both immutable capsules and current source bytes by default.
It replays numeric assertions and asset metadata against capsules, compares cards
with their frozen specifications, and checks every path and nested permission flag.
`--capsules-only` can audit old packet integrity, while explicitly making no claim
that the current source still matches. Hashes prove byte equality, not the truth
of source assertions or that a model executed.

## Evidence and advancement contract

Fixture/preparation evidence cannot support measured model-quality numbers.
Absent measurements require null numerator/denominator and `NOT_MEASURED`;
unknown is not zero. JSON assertions resolve an exact pointer; text assertions
retain the exact supporting excerpt and denominator. Historical outcomes stay
historical and never fill the new candidate's gates.
Prose carrying only evidence IDs is an indexed interpretation, rather than a
mechanically checked statement or semantic certification.

Every required gate and every extra registered preservation gate is mandatory.
Acquisition, retention, reliable answering, termination, response length, exact
native qualification and independent replication remain separate. Missing or
failed evidence yields `HOLD`; gains in one axis cannot compensate for another.
Even all-PASS fixture gate metadata yields only eligibility for a separate review,
with native/release/promotion permissions false. The built programme snapshot
keeps every current proposed candidate on HOLD.

Source-specific retention, original source-correct cohorts, independent replay,
family/parent split closure, reference review, complete target+EOS, no silent
truncation, independent training draws, stopped/incomplete attempts and uncertain
spending retain their original contracts. Do not relabel a spent bank as fresh,
an unfinished E1/SYNTH run as an SFT parent, or proposed resource ceilings as
measured prices. Sealed/CHECK/FINAL access and serving changes stay separately gated.

## Tests

```powershell
& $py -B -m unittest discover -s experiments/rineya_quality_queue_v1/tests -v
```

Tests use synthetic metadata/opaque bytes and temporary directories under
BusinessProject. They verify contracts and source binding only; no Rineya
capability, memory fit, throughput or native qualification follows from them.

## Lossless vector archive prototype

`vector_archive.pack_rows` accepts explicit opaque little-endian F16 row bytes,
unique logical slots, domains and independently supplied source hashes. It
stores the actual two-byte UNKNOWN value with each row, canonicalizes only that
slot to zero, and shares blobs only when the remaining bytes are exactly equal.
The optional `zlib` codec is lossless; no floating-point conversion occurs.
`restore_rows` verifies the separately trusted manifest root and every blob and
row hash before returning the original bytes, order and provenance.

The helper is in memory only, with a default 64 MiB logical-byte allowance and
at most 4,096 rows per archive. Callers retain the trusted root, source bindings,
file headers and cumulative storage/I/O accounting outside the helper. It has
no writer, model loader or worker integration. Compression can expand data;
small-sample savings do not prove that the complete 34-profile MLP calibration
fits its registered storage cap. Every diverged continuation and full vector
must remain reconstructable, with independent restoration checks before any
reviewed downstream adoption.

`f16_inspection.inspect_finite_argmax_row` separately checks every supplied F16
value for finiteness and computes the lowest token ID at the numerical maximum.
It compares integer bit patterns without converting or rounding floats; signed
zeros tie. The optional expected ID must agree. This checks explicit row bytes
only: native origin, fixed case/arm/bias/prefix, runtime and full capture custody
still require independent binding. It performs no file or model I/O. Nonfinite
rows fail inspection but remain eligible for unchanged lossless archival as
rejected evidence; no bytes are silently removed or repaired.

## Unselected fixed-3.75 analysis prototypes

`primary375_analysis.validate_primary375_registration` checks bounded declarations
for the fixed parent/unchanged-MLP comparison, 96 CAL/192 CHECK, 1024/64 decoding,
explicit margin/confidence/claim scope, scorer domain and review commitments.
It validates their shape only. It does not verify the truth of a review, select
actual method values, qualify confidence inference or grant bank/native access.
`primary375_paired_tally` requires every enrolled CHECK pair and the exact
48 answerable/16 unanswerable cases per family. It reports supplied correctness
and refusal axes separately, including repairs, losses, family net changes and
the exact inherited proxy. A positive aggregate never overrides a failed
observed preservation gate. Confidence, shortcut floors, semantic adjudication,
retention, costs and actual source/data/native qualification stay external.

`cluster_statistics.stratified_cluster_bootstrap_interval` is an exploratory
metadata prototype. It resamples whole declared parent components within fixed
families, keeps case-weighted family ratios and combines an equal-family mean.
Inputs must supply complete, unique typed joins; declared IDs do not establish
semantic ancestry or independence. Seed and draw count are explicit, and the
local RNG leaves caller inputs and the global RNG unchanged. Percentile ranks
use the exact supplied binary-float confidence ratio; they differ from treating
a printed decimal as an exact rational. Resampled rates use floating-point
arithmetic, and Monte Carlo percentiles have no exact coverage guarantee.
Zero observed differences can yield a degenerate interval without excluding
future harms. Neither prototype is selected as an experimental acceptance rule;
all fixtures are synthetic metadata and provide no Rineya capability evidence.

## Unselected atomic-R coverage diagnostic

`r_lookup_diagnostic.analyze_atomic_r_fixture` accepts only explicitly supplied
synthetic atomic-R metadata. It validates the existing finite public grammar,
then compares separate request-traversal and synchronous graph-closure routes.
Repeated requested aliases are valid. Distinct aliases can share an endpoint.
A reachable cycle or dangling alias requires UNKNOWN; unrequested bad components
are distractors. Any unresolved request prevents a partial endpoint set from
becoming an answer. Malformed input, operational failure and route disagreement
raise; they do not become semantic UNKNOWN.

The diagnostic labels these coverage dimensions and returns one sorted fixture
reference, without scoring responses, rendering prompts, generating banks,
loading tokenizers/models or selecting a treatment. Its partial logical counts
are not inclusive resource prices. Tests use hand-checked edge cases,
invariances, maximum-domain cases and 300 bounded two-alias graphs/request
sequences. Such agreement is software evidence within a shared grammar, not
epistemic independence, actual-bank coverage or Rineya quality evidence. The
canonical-only CE8 study, stopped warm path, thresholds and enrolled data stay
unchanged. Actual public-prompt/reference, ancestry, native token/mask/EOS/fit
and retention evidence remain separately required.

## Unselected source-preservation diagnostic

`source_preservation.source_preservation_tally` accepts an explicit enrollment,
paired judgement metadata and a separately declared required-cell list. It groups
results by retention data source, family, context bin and answerability. Model
baseline identity, data-source ancestry and family labels are separate bindings;
the helper does not authenticate any of them or grade response text.

Strict and semantic source-correct membership is frozen in the enrollment.
Known source-success denominators survive missing or failed candidate pairs;
unknown membership stays unknown and cannot be filled from later outcomes.
Complete paired baseline flags must match known frozen membership. Required
cells are not inferred from observed responses, and empty cells or source-success
populations remain unmeasured. Partial evidence stays caller-owned rather than
being erased or converted into a judgement.

The tally exposes kept successes, losses and repairs separately, alongside
justified/false abstention, unsupported answers, natural termination, truncation
and lengths by correctness. A relative retention win or pooled acquisition gain
can coexist with source losses. The diagnostic always leaves quality advancement
on HOLD; complete supplied pairs with no observed damage establish neither a
confidence guarantee nor general preservation. No lane's thresholds, scorers or
permits are transferred, and no real analysis method is selected.

The 16,384-case object-shape limit accommodates a proposed PRSC raw retention
panel but establishes no physical fit, resource price or native allowance.
Token counts must already follow a prospectively defined EOS convention. Tests
use synthetic judgement metadata and remain excluded from scientific datasets.
They exercise hidden source/context losses, absolute source damage despite a
relative win, incomplete accounting, baseline drift and invariant ordering.

## Standing research consent

The programme records direct human standing consent in each card as preparation
evidence. It removes repeated consent questions within the recommended bounded
research scope. Existing owners still bind each concrete phase to reviewed inputs,
source, runtime, budgets and genuine prerequisite evidence. This offline index
issues no execution grant; its closed permission fields describe the packet.
Scientific admission, resource fit and every preservation gate remain independent
requirements. Deployment and serving changes are outside this standing scope.
