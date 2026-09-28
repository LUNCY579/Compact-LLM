# Fresh long-context preservation sentinel for exact Titan

**Status: PROPOSED_UNQUALIFIED.** Source-only design, 2026-09-28, audited execution
HEAD `43d2ed3c438a64fdd6a4088db0a151f7e604d0ee`. No model, tokenizer runtime,
dataset generator or protected bank was opened by this design. No run is
authorized by it. The companion proposal is
`configs/LONG_CONTEXT_SENTINEL_PROPOSAL.json`; root must reconcile and freeze it
with the pending retention amendment before implementation or native execution.

## Evidence and scope

The exact source is Titan Experimental 1.0.0-devfreeze,
`rineya.core.titan-experimental.v1.0.0-devfreeze.28L.supa17-rrp015-control-seed17`,
596,049,920 parameters, weight SHA256
`537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
The native tokenizer/template binding is in `receipts/PRIMARY_STUDENT_RECEIPT.json`.

VERIFIED SOURCE: `configs/E0_TITAN_BF16.json` and `scripts/qualify_native.py`
register only 256/512/1024-token mechanics. The latter explicitly leaves context
beyond 1024 unqualified. Its mechanical prefix filler is not a reading task.
Even a successful E0 would not establish long-context capability or retention.
The current stored E0 v2 result is BLOCKED_E0 before loading any weights.

RECORDED FROM PRIOR WORK: the published Titan `CORE.json`, lines 99-117, records
`max_context=4096`, `max_new_tokens=24`, no overflow skips, and 11 source-correct
rows lost despite a net gain of one on the historical long-context instrument.
Its README identifies the FP32 DenseReference evaluation contract. These are
historical results, not present BF16 runtime qualification. A 4096-token ceiling
does not show that every historical prompt contained 4096 tokens; exact occupied
lengths were not inspected. The demo README records 853-1183 words, which cannot
be converted into native token counts by assumption. The checkpoint's declared
40960 maximum is likewise not a measured operating length.

The historical demo is SPENT and excluded from TRAIN, SELECTION and CONFIRMATION.
Historical LCR training/DEV material belongs to the Supa/Titan ancestry audit and
cannot be re-split into a fresh preservation sentinel. Only metadata and source
documentation were used here, not the demo rows, historical predictions or
protected banks. Fresh worlds must also respect all exclusions listed in
`manifests/RETENTION_REQUIREMENTS.json`.

## Separate forward-only resource qualification

Propose two genuinely larger total sequence envelopes: **2048 and 4096 tokens**.
Each includes a 24-token generation reserve, so scored prompts occupy exactly
2024 or 4072 tokens after the complete native chat template. Both lengths must
pass an independently registered forward-only extension before sentinel scoring.
These are proposed lengths, not historical occupied-length or native-fit claims.

Keep the exact E0 model, stock Qwen3 implementation, BF16 frozen base, FP32
rank-8/alpha-16 q/v adapter representation, eager attention, no KV cache,
microbatch one, TF32 off, deterministic settings, tokenizer, template and EOS
contract. The extension performs **zero optimizer calls**. Verify frozen tensor
identity before/after and exact zero-adapter output parity. Training beyond 1024
is outside this proposal. A switch to SDPA, cache, quantization, checkpointing or
CPU execution needs its own prospective runtime qualification.

Use four non-evaluation mechanical fixtures per length: two at the prompt cap
and two at the full sequence envelope. Run each unchanged and with a zero adapter
(16 forwards total). For one prompt-cap fixture per length, perform greedy
full-prefix generation up to 24 tokens and one teacher-forced likelihood check
on those generated tokens. Reuse the frozen E0 maximum/mean likelihood tolerances
0.03125/0.01 and exact zero-adapter tolerance; do not relax them after failure.
Require finite logits and unchanged causal/EOS semantics. Keep only the required
last-token or supervised suffix logits, not all-vocabulary prompt caches.

This is at most 66 model forwards, 48 generated tokens and 201,960 processed
input tokens, including repeated full prefixes; the input count is a conservative
bound using full-envelope teacher-forced passes. Record actual tokens, all failed
attempts, load time, peak allocated/reserved/free VRAM, host available/RSS and
throughput separately for each length. Qualification wall ceiling: 1800 seconds.
No claim of affordable scoring follows until actual throughput is available.

Keep all admission/worker gates unchanged: host available >=7.4 GiB at admission
after native imports, host available >=6.0 GiB while running, GPU free >=4.0 GiB
at admission and >=0.5 GiB while running, disk free >=4.0 GiB, process-tree RSS
<=4.0 GiB, and one contained GPU worker. A cheap pre-import failure prevents launch;
passing that check does not bypass the post-import check. The current reported
7.18 GiB pre-import availability does not meet admission and grants no launch.
Use the contained Windows supervisor on this host. A failed or interrupted
profile leaves its length unqualified; do not retry automatically or drop a
length after inspecting capability outcomes. A separately frozen 2048-only
amendment could support that restricted envelope, never 4096 preservation.

## Minimum operational panel

The smallest recommended *operational sentinel*, not a powered noninferiority
study, is 32 fresh DEV parents and 64 separately reserved CHECK parents. Cross
two deterministic reasoning families (three-fact keyed joins; latest authorized
record selection) with four evidence placements. DEV has four distinct worlds
per family/placement cell; CHECK reserves eight. All facts controlling a target
must be derivable by an independent symbolic solver.

Each parent has two length renderings and a full 2-by-2 answerability/distractor
factorial: answerable/neutral, answerable/adversarial distractors,
missing-required-evidence/neutral, and missing-required-evidence/adversarial
distractors. This gives eight related rows per parent. The factorial is necessary
to distinguish answer retrieval, distractor interference, and unsupported
answering; the eight rows never count as eight independent tasks.

| Bank | Parents | Rows | Per-model new-token ceiling | Per-model input-token ceiling |
| --- | ---: | ---: | ---: | ---: |
| DEV | 32 | 256 | 6144 | 18,797,568 |
| CHECK, reservation only | 64 | 512 | 12,288 | 37,595,136 |
| Total per unchanged source or candidate | 96 | 768 | 18,432 | 56,392,704 |

Input ceilings count every full-prefix no-cache generation forward at the maximum
24-token completion. They are not training tokens: for each row with prompt P,
the cap is `24*P + 0+...+23`. At P=2024 and 4072 this is 48,852 and 98,004 input
tokens respectively. New-token ceilings include EOS. Actual early stops reduce
cost; failed/partial calls and any retries still consume the same bounded budget.
There are no teacher calls, reference-model calls or training updates.

The four-domain retention proposal currently has DEV 256 parents/320 rows and
CHECK 512 parents/640 rows. This additional sentinel makes prospective combined
totals **DEV 288 parents/576 rows** and **CHECK 576 parents/1152 rows**. It does not
replace short retrieval, language or answerability coverage. FINAL remains closed
and has no count assigned by this design.

## Freshness, pairing and controlled positions

Allocate canonical semantic parent/family keys and role commitments before text
rendering, length fitting, solver calls or model scoring. CHECK receives only
reserved counts/opaque role commitments until its gate; do not render or inspect
its examples now. A seed or entity-name change alone does not establish a new
parent. Pair all answerability, distractor, length and wording derivatives under
the originating semantic world. Keep acquisition, replay, DEV, CHECK and FINAL
worlds disjoint. Audit permitted historical ID/fingerprint inventories without
opening protected contents; unavailable family-level exclusions remain a blocker.

Use three necessary, nonredundant support facts for each answer. For early,
middle and late parents, all required support spans lie inside 10-20%, 45-55%, or
80-90% of the **context-body token interval**, respectively. Distributed parents
place one required fact in each interval. Record native-token span boundaries
and both body-relative and whole-prompt offsets; verify every support span and
its intended dependency. Percentages apply after final template/tokenization,
not character or word positions. The query/instructions remain identical across
paired variants, and the semantic graph, target and evidence placement stay fixed
when length increases. Additional context consists only of predeclared irrelevant
facts that the solver proves cannot change answerability or the answer.

Adversarial rows replace neutral filler with plausible decoys: a different key,
an obsolete timestamp, or an unauthorized record, balanced across parents. Decoys
must neither supply the missing fact nor introduce an alternative valid answer.
Unanswerable rows remove exactly one required fact, counterbalanced across the
three dependencies, and replace it with irrelevant text occupying the same
controlled region. Other support facts and the question remain unchanged. Verify
that alternate paths or query wording cannot leak the answer. Preserve the full
2-by-2 pairing, so distractors also occur on unanswerable rows.

Fit exact 2024/4072-token prompts using deterministic irrelevant prose, then
recheck all span and solver contracts. Never truncate, insert masked PAD/EOS as
fake context, selectively drop hard rows, or move evidence to fit a budget.
If the frozen builder cannot render a parent under these constraints, the bank
fails construction before model scoring; fix/version the generator and rebuild
the entire affected role under an explicit amendment. Freeze generator, verifier,
template, bank and prompt-token hashes before native enrollment. Original authored
text still requires recorded provenance; no external text or teacher generation
is required.

## Frozen source enrollment and evaluation

Use the exact unchanged Titan source first under the qualified runtime, same
greedy decoding (`do_sample=false`, no cache), 24-token cap, EOS IDs 151645/151643
and PAD=151645. Targets are short canonical symbols or literal `UNKNOWN`, with
native target length <=16 tokens; long reasoning is not solicited. Preserve each
raw output and token sequence. Freeze the primary strict single-answer parser
(whitespace trim, otherwise case-sensitive exact answer; no extra prose) before
seeing outputs. A separately frozen content diagnostic may accept only an entire
canonical symbol after a fixed optional `Answer:` prefix; substring matches and
post-hoc tolerant rescoring cannot decide the primary endpoint. Report formatting,
false abstention and unsupported answers separately.

Run every frozen DEV row on the source before any candidate DEV sentinel result,
then write an immutable enrollment receipt binding source/tokenizer/runtime/
scorer/bank hashes and per-row correctness. Do not use historical success labels,
replace source-failed parents, or select only easy native successes. Keep every
row, including source errors; enrollment marks a subset, not a filtered bank.
For CHECK, keep the bank closed until frozen candidate checkpoints and the seed
decision exist; then score the unchanged source first, freeze membership, and
only then score candidates without further selection. Retain failures and costs.

For each length, family, placement, answerability and distractor stratum report
baseline/candidate accuracy, source-correct losses, retained source successes,
source-incorrect repairs, formatting/EOS rates and all relevant denominators.
Count source-correct parents as well as rows. A zero source-correct denominator
is undefined; sparse support is inconclusive, never preservation. Report strict
all-eight-rows parent success as a secondary diagnostic. Bootstrap entire parents
within family/placement strata, keeping both lengths, all variants and all seeds
together (10,000 draws, seed 20260927); never bootstrap rows or seeds independently.
Show raw discordant parents rather than interpreting a broad interval or a
nonsignificant result as equivalence. Repairs cannot cancel source-correct losses.

## Cost ceiling and E1 decision implications

Propose sentinel scoring only at the frozen **final E1 checkpoint**, including
seed101 DEV before authorizing seeds211/307, rather than adding long inference to
every intermediate checkpoint. This cadence must be explicitly accepted in the
retention amendment; intermediate short-panel DEV remains as registered. Each
source/candidate DEV call has a 7200-second wall ceiling; each CHECK call has a
14,400-second ceiling. These are fail-closed planning caps, not measured duration
predictions. Stage completed banks across separate contained runs if necessary,
using immutable row receipts and verified checkpoint identity; never silently
increase the total bank budget or count incomplete scoring as a pass.

Unchanged source plus three candidate final endpoints cost at most 73,728 new
tokens, 225,570,816 processed generation-input tokens, and 86,400 seconds of
scoring wall time across all roles, plus the separate 1800-second runtime profile
and 48 profile generation tokens. Each candidate's DEV+CHECK ceiling is
**18,432 new tokens and 21,600 seconds**; all three together
are 55,296 new tokens and 64,800 seconds. Source enrollment accounts for the
remaining 18,432 new tokens and 21,600 seconds. Charge every candidate's sentinel
cost against its existing E1 262,144-generation-token and 24-hour per-seed
ceilings, leaving room for the acquisition and other preservation work. Root must
freeze a combined cost ledger before launch; these additional caps do not expand
the registered E1 total. Failed attempts count and no paid compute is proposed.

Both resource qualification and current-source baseline coverage are prerequisites
for treating this sentinel as represented in E1. If the source cannot solve a
stratum, that stratum cannot certify preservation; record its absolute behavior
and leave preservation unqualified. Freeze exact decision endpoints and margins
before source scoring. The existing exploratory 2pp aggregate/5pp family screens
are not a noninferiority certificate, especially at 32/64 parents; this proposal
does not convert them into one or allow long-context loss to be hidden in other
domains. Any native source-correct loss must be disclosed and diagnosed.

An E1 result can at most support preservation on this narrow, synthetic,
2048/4096-token controlled reading sentinel, with its uncertainty and source
coverage stated. It cannot establish natural-document competence, the configured
40960-token context, language quality, or full broad retention. The independent
language sentinel and every other retention requirement remain required. If
either long-context length, language coverage, provenance, baseline enrollment or
resource gate remains missing, broad E1 preservation/promotion remains blocked;
do not relabel <=1024 retrieval or a reasoning-only replay bank as full retention.
