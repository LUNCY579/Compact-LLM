# Common CE8 registration proposal

**UNAPPROVED.** The JSON beside this document proposes concrete common settings
for either operator choice. Every qualification, training and evaluation grant
is false. The original warm study remains `STOP_NO_SIGNAL`; neither arm is
selected or launched by this proposal.

## Fixed treatment and comparison

Each selected arm starts independently from deployed warm anchor
`ff516ff86259c101a8cf55b4bf548e267f74719dc450ad34538f4383ca55d84b`.
Canonical-only selects one arm; the paired choice selects canonical and fixed
permutation arms. Both use the same 3,000 R TRAIN rows, 375 CE8 boundaries,
154 named tensors in blocks 14–27, 220,233,216 trainable parameters and fresh
AdamW. Static recipe names come from weight-header metadata and reproduce the
prior verified inventory name hash.

The proposed native seed is **20261001**. Teacher permutation seed **1301** is
separate. Proposed checkpoint boundaries are **0, 1, 75, 150, 225, 300, 375**.
One documented technical resume is proposed, with at most 450 charged physical
boundary attempts and 74 rolled-back boundaries; logical committed exposure
remains exactly 375. No failed DEV gate may trigger retraining or a recipe retry.

## Exact evaluation populations

| Bank | Fixed rows | Use |
| --- | ---: | --- |
| DEV core R | 250 | H1: strict pass@1 versus warm anchor; net gain at least 13 and the adjusted exact one-sided McNemar threshold. |
| DEV core S | 250 | Exposed source-skill diagnostics; proposed loss margin 10%. |
| DEV core P | 250 | Exposed source-skill diagnostics; new proposed loss margin 10%. |
| guard_dev | 1,200 | All eight verifiable legacy families, 150 rows each; exposed diagnostics, proposed loss margin 5%. |

The JSON records every sorted row ID and parent ID, their digests, and actual
manifest/canonical-file hashes. R/S digests reproduce the prior DEV scope; P is
bound independently from the same core manifest. No held-out model output was
read to construct these populations.

H1 familywise alpha is .05 over all selected arms, using Bonferroni:
canonical-only uses .05; both paired arms use .025 each. Minimum discordance
is 5 at .05 and 6 at .025; zero or insufficient discordance is
**UNCERTIFIABLE**. Net gain must still be at least 13/250. No arm is selected
after outputs. The teacher-order arm-to-arm contrast remains descriptive
single-seed feasibility unless separately registered.

Source-correct denominators require future baseline outcomes. They are fixed
as **all** correct IDs from each baseline's first complete bank read, with a
second independent process requiring zero token-sequence churn. Baselines are
the warm anchor and original Supa. Keeping the Supa-correct population prevents
warm-start losses from disappearing when enrollment changes.

Losses and repairs remain separate. One-sided 95% Clopper–Pearson upper bounds
are compared with the named diagnostic margins. Zero-loss denominators below
29 at 10%, or below 59 at 5%, are **UNCERTIFIABLE**, including empty populations.
These exposed banks cannot support a confirmatory no-forgetting claim.
Each legacy family's denominator, losses and repairs must also be reported;
the pooled diagnostic margin cannot certify an individual family.
Fixed-bank losses are exact counts. The binomial confidence interpretation of
Clopper–Pearson requires conditional independent Bernoulli losses with a common
probability, or another valid binomial justification; exchangeability alone is
insufficient. Repeated parents require parent/world clusters and a separately
registered clustered analysis. R/S/P share worlds across banks and cannot be
pooled as 750 independent worlds.

There is **no natural-language preservation bank** in this scope. Language
preservation is `NOT_MEASURED`, requiring another explicit bank decision.
TEST, CHALLENGE, guard_test and guard_challenge model outputs stay sealed.

## Qualification and budgets

Eight stages are proposed, each with one attempt and a finite cap: ingestion,
runtime/anchor, longest masks, likelihood/gradient parity, Adam/clipping,
fresh-process checkpoint resume, delta roundtrip, and resource/IO measurement.
The proposed physical job deadline is **19,200 seconds per arm**. Charged wall
reservations total **57,600 seconds**: 19,200 for nested stage-operation
reservations, 19,200 for prepaid cleanup spanning the whole model lifetime,
and 19,200 for the native journal's separate `tick_wall` charge.
This does not extend the physical deadline. Exact stage deadlines, reservation
caps, operation, read/write, hash and metadata bounds are in the JSON.
Outer controller summaries reference native charges instead of charging them
again; their own bootstrap/verification work fits the stage-operation envelope.
An additional independent outer clock would require another approved allowance.
Checkpoint reservations include the backend wrapper and inherited storage.
The actual child control plan requires **eight restore attempts**: three
corrupt-latest rejections, three verified good-ancestor recoveries, one fresh
resumed-child prefix restore, and the final parent endpoint restore. Every
failed attempt retains its full charge. Four saves are scheduled (parent U2,
uninterrupted U2, prefix U1, resumed U2), with two bounded optional C0 allowances.
The base save primitive validates complete history and permits a first U1/U2
save; restore requires an actually retained checkpoint.

With six save allowances, eight restore attempts and three isolated corruption
copies, the analytical combined reservation is **738.43 GiB read / 79.41 GiB
write**. Proposed checkpoint stage ceilings are **1,024 GiB read / 128 GiB
write**. Twelve complete live-state fingerprints receive a separate finite
96 GiB hash allowance within the stage's 4 TiB hash ceiling. These cumulative
reservations are separate from active disk capacity and are not measured costs.
Child bootstrap, source/handle, RPC, lease, reload and publication costs still
need complete separately approved profiles. The earlier metadata validation
receipt remains evidence for its prior proposal revision.

The actual tokenizer-only receipt identifies the maxima **P = 1,259**,
**C = 29**, and **S = 1,287** independently. For each teacher arm, one row
attains max P, 89 rows attain max C, and one row attains max S; max P and max S
share a row. The JSON binds all actual IDs and prepared-row hashes to both
V2 ingestion receipts. They also record exactly **20,311,962 prepared-hash input
bytes per arm**. V1 receipts remain preserved. `maxP + maxC = 1,288` is not a
measured joint row.
The eight-row gradient reference begins with one representative of each
maximum, deduplicates, then fills from the witness union in exposure order.
Masks and likelihood parity cover every witness. This metadata establishes
token lengths and identities; it does not qualify native model execution.

Likelihood parity is proposed at 1e-4 nats on every selected token, including
EOS. Gradient parity uses L2 difference ≤ absolute tolerance + relative
tolerance × reference L2, with both proposed tolerances 1e-6; the coordinate
error bound is separate. A zero reference does not disable the L2 criterion.
Native checkpoints must reproduce actual parameters, Adam, module modes and
Python/NumPy/CPU/CUDA RNG bytes across separate OS processes. Delta roundtrip
reports the deployed endpoint and its measured FP32 rounding residual rather
than assuming equality with the trained master.

Proposed hard resource gates are launch free RAM 5.5 GiB, ongoing free RAM
1 GiB, worker peak RSS ≤ 4.5 GiB, allocator fraction .78, peak reserved memory
≤ .95 of that cap, GPU free memory ≥ .5 GiB, and launch free disk ≥ 32 GiB.
The inherited disk floor is 14 GiB, with a proposed 16 GiB active reservation.
Qualification must establish actual fit. Failure stops; gates are not lowered.

**These are budget proposals, not measured prices.** Per-operation profiles
and prospective 12-hour-per-arm production ceilings do not qualify production.
The actual native worker/verifier, source freeze, capability issuer, artifact
readback, ledger replay and verification accounting must be implemented and
approved. The JSON defines the exact native ledger artifact interface and four
owned construction targets; those artifacts are **unapproved and not created**,
with no invented raw hashes. Cleanup must be reserved before model allocation
for the whole model lifetime. Production retains its proposed 12-hour physical
deadline and needs measured full-unit projections within the hard ceilings and
its own frozen scope. Qualification weights are discarded.
Production's proposed charged-wall ceiling is 129,600 seconds: operation
reservations, lifetime cleanup and `tick_wall` each receive 43,200 seconds.

Native ledger `bindings.metadata_roots` must equal the one exact admitted
output directory. This permits checkpoint and receipt metadata inside that
owned output root while retaining path containment checks. The CPU fixture
rule remains limited to its journal directory.

The backend proposal includes the exact pinned asset receipt's UTF8 text,
preserving its raw CRLF bytes, plus asset/tokenizer/warm-file byte counts.
These support pure analytical lower bounds for known loader/hash work before
asset reads. They do not supply measured operation prices.

No R-only result establishes the broader acquisition/composition/retention goal.
