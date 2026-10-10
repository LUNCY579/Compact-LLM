# Open historical quality pattern review

This is an offline reconstruction of already open historical DEV records and
training metadata. No model, tokenizer, optimizer, protected bank or confirmation
content was accessed. The machine report is `OPEN_MQ_RAB_PATTERN_AUDIT_001.json`;
its file SHA256 is `23335dc922c6be3d6991856e6be259a3ee328886fb8adba39db5e434bca603c7`.

## Most defensible positive contrast

At matched input dose, response-only supervision outscored whole-sequence
supervision in each of three MQ order repetitions. These reuse selection seed 17,
the same 238 training rows and the same 64 DEV rows, of which 51 were scorable.
Each completed arm consumed 24,910 input tokens in 13 token-normalized groups.
Response-only supervised 1,944 targets; whole-sequence supervised 24,672.
Both used 1e-5 peak LR, WSD 20/60/20, AdamW betas .9/.95, clip 1, decay 0 and the last 20
blocks (314,618,880 trainable parameters). Within each pair the entire LR trace,
input occurrence order, source and plan hashes agree.

| Order | Response correct /51 | Whole correct /51 | Response better / whole better | Source-correct retained / lost, both arms |
|---|---:|---:|---:|---:|
|17|13|7|7 /1|3 /0|
|23|13|7|7 /1|3 /0|
|29|13|8|6 /1|3 /0|

The net response advantages are 6/51, 6/51 and 5/51 (11.76, 11.76 and 9.80 percentage
points). This supports a prospective objective comparison. It does not prove
fresh-family transfer or broad preservation: only three source-correct DEV rows
were available, and zero DEV rows were marked unanswerable. The same
`rineya_v3_sft__estimation_sanity__reas_estimation_sanity_s04__01456` was correct
under whole-sequence and incorrect under response-only in every order. No claim
of universal row-wise improvement is supported.

The original corrected runs lack launch/supervisor receipts, and every archived
checkpoint is absent. The first-pair41-scorable-row results are retained
separately rather than pooled with the corrected 51-row instrument. Failed
`cell-s17-whole` RAM admission remains a failed attempt. Three order repetitions
are not three independent training-data draws or new test samples.

## Repeated acquisition with forgetting

RAB last20 independent training draws repeatedly improved aggregate arithmetic
accuracy while losing source-correct answers. The same 150-row exposed panel had
47 source-correct rows. Whole-sequence CE was used despite the misleading
historical `response_only` artifact tag.

| Training draw | Rows / updates | Targets / input tokens | Cosine correct / source losses | WSD correct / source losses | WSD paired better / worse |
|---|---:|---:|---:|---:|---:|
|17|41|5,233 /5,274|67 /7|74 /6|10 /3|
|23|40|5,284 /5,324|58 /8|62 /10|10 /6|
|29|39|5,259 /5,298|60 /8|73 /5|19 /6|
|41|39|5,260 /5,299|62 /8|68 /8|9 /3|

Within each draw, source, data and occurrence hashes match between recipes. WSD
changes the schedule and warmup fraction together with beta2 (.95 versus.999);
the +7, +4, +13, +6 direction cannot establish a schedule-only effect. Repeated
evaluation rows do not become independent because training draws differ.
The WSD arms lost these same source-correct IDs in all four draws:

- `rineya_v3_sft__time_duration__reas_time_duration_s04__01408`
- `rineya_v3_sft__time_duration__reas_time_duration_s07__02345`
- `rineya_v3_sft__time_duration__reas_time_duration_s08__02681`

Thus aggregate gains repeatedly fail a zero-new-loss rule. The table retains
all failed/partial RAB records separately; an unfinished run is not a successful
reduced-cohort result. The machine report reconstructs all 32 original RAB
artifact baseline/final boolean vectors, not just the eight selected endpoints.

## Secondary MLP evidence and exact-source limit

The already open CAPABILITY1 count summary at bias 0 records answerable 285→339/480
(75 gains, 21 losses; 264/285 retained) and unanswerable 120→231/240 (111 gains,
zero losses). It is one 450-update/3,600-presentation training run, seed 20260903,
on the narrow same-generator BSEL2 bank with one shared 156,225-parameter MLP.
CAPABILITY1's unseen phrasing is not a new task family.

Existing TRANSFER3 summaries give answerable 54 gains/43 losses at bias 0, versus
99 gains/6 losses at canonical bias 5.156249985098839. They do not measure actual
serving bias 3.75. This is evidence that readout policy changes the apparent
gain-and-damage balance. The report labels MLP material as summary evidence;
the snapshot omitted raw outputs, and this audit does not pretend to reconstruct
them. CALIB2's already-correct-target restriction still admitted task gains;
CALIB3's commitment-only objective improved the abstention count while failing
retention; CALIB4's one-parameter neutral-bias result was harmful at canonical.

MQ and ordinary RAB bind ungated Supa1.7 weight
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
The MLP evidence binds a different gated FP16 Supa-A1.1 weight
`a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`.
SYNTH Titan's source weight begins`537201676814`; none of these historical gains
establishes its present capability or native qualification.

The RAB historical audit hashes LF-normalized Git text, whereas the current
preserved files use CRLF. The report retains both hash domains explicitly; the
normalization matches the historical hashes. No model bytes or checkpoint
references were followed to resolve this text distinction.

## Independent parser inputs

`mq_records` and `rab_records` contain exact record IDs, source/data/recipe/dose
metadata, original path/member hash references and baseline/final boolean maps.
Recompute retained as both true, gained as false→true and lost as true→false;
repairs never cancel losses. The contrast arrays name the exact paired record
IDs. Group MQ by selection seed and order seed; group RAB by training draw.
Do not pool initial and corrected MQ scorer denominators, treat incomplete
records as completed, or transfer historical results to a different source.
