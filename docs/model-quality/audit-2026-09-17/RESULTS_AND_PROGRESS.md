# Fine-tuned candidates and current progress

Updated after pilot completion, 2026-09-17 UTC. Historical scores below are
preserved observations, not fresh independent confirmation. All planned RRP-015
training and evaluation runs have now finished.

## D1: six seeds exist

The user's quoted D1 table is correct for seed 17. Six selection draws have now been
measured under that historical recipe. They vary selected training examples, not
merely minibatch order.

| Candidate | Transition skill /384 | Arithmetic /150 | Source answers retained /47 | Long context /256 | Paired long-context p | Abstention /64 |
|---|---:|---:|---:|---:|---:|---:|
| Supa source | 14 | 47 | 47 | 161 | â€” | 13 |
| D1 seed 17 | 239 | 77 | 40 | 160 | 1.0000 | 8 |
| D1 seed 23 | 217 | 71 | 38 | 159 | .8450 | 8 |
| D1 seed 29 | 236 | 67 | 36 | 148 | .0351 | 5 |
| D1 seed 41 | 227 | 72 | 39 | 147 | .0288 | 5 |
| D1 seed 47 | 225 | 64 | 37 | 157 | .5034 | 12 |
| D1 seed 53 | 228 | 67 | 36 | 154 | .1435 | 12 |

All six clear all three historical acquisition floors. Four of six pass the
historical gate (arithmetic>=62 and retained>=37); seeds 29/53 fail retention.
The gate pass-rate exact 95% interval is .223â€“.957: six seeds do not establish a
reliable success rate. Seeds29/41 have detected long-context losses. The p-values
above are the historical unadjusted two-sided paired McNemar results, recomputed
from the 256 raw paired outcomes for this report.

**D1 seed 17 loses 11 source-correct long-context rows and gains 10 others.** Its net
change isâˆ’1 and p=1, but nonsignificance is not equivalence or proof of no forgetting.
It also loses 7/47 source arithmetic successes and 5/64 abstentions. The one-step
transition improvement is substantial; broad superiority is not demonstrated.

Sources: [six-seed report](../../../PROCESS_DOSE_RELIABILITY.md) and its linked raw
artifacts; [first replication](../../../D1_REPLICATION.md).

## Later asymmetric-dose candidates and M4

The asymmetric arm uses 128 ledger parents but 64 parents for each other family,
instead of D1's 128 in every family. It has three measured seeds:

| Candidate | Transition skill /384 | Arithmetic /150 | Retained /47 | Long context /256 | Abstention /64 | Historical gate |
|---|---:|---:|---:|---:|---:|---|
| Asymmetric seed 17 | 213 | 72 | 43 | 157 | 12 | PASS |
| Asymmetric seed 23 | 208 | 77 | 42 | 158 | 12 | PASS |
| Asymmetric seed 29 | 204 | 61 | 38 | 156 | 14 | FAIL: arithmetic |
| M4, single seed | 267 | 77 | 39 | 137 | 13* | PASS |

The asymmetric arm offers a different tradeoff: less acquisition than D1, with
more retained arithmetic answers and abstentions in all three matched seeds.
Every long-context total remains below Supa 161. None of those three differences
was detected by its paired test; preservation/equivalence is still unproved.

M4 loses 33 source-correct long-context rows and gains 9: netâˆ’24,
paired p=.000272. **Its 13 reported abstentions include four `UNKNOWN.` answers;
only 9 are the exact literal `UNKNOWN`.** The four affected rows are RLC-019-U,
RLC-034-U, RLC-035-U and RLC-043-U. The old statement that it fully restored the
literal sentinel is therefore too strong. The Demo's normalized score is not
being changed retroactively.

The new dataset audit also found 12 short arithmetic-template contacts in M4's
consumed rehearsal rows. That qualifies interpreting all its arithmetic gain as
off-task transfer; the size of any score effect is unknown. These are reasons to
preserve M4 as research evidence, not treat it as an overall winning core.

Sources: [asymmetric-dose report](../../../PROCESS_ASYMMETRIC_DOSE.md),
[M4 comparison](../../../REHEARSAL_COMPOSITION.md), and
[consumed-data audit](DATASET_AUDIT.md).

## What has been preserved

**Yes, existing candidates are worth retaining as separately identified research
models.** All three surviving D1 exports (seeds 17/23/29) were freshly hashed against
their materialization receipts and marked for preservation. No weight copies or
published-core changes were made. Keep the failing seed too, so the archive does
not silently become a best-seed selection.

The retained seed 17 export is a later reconstruction: its recorded full final-state
hash exactly matches the original D1 run. Old optimizer checkpoints were reclaimed
before this task; these three complete BF16 model exports survive. They can be used
for research inference, but do not provide the removed optimizer state.

The inspected M4 and asymmetric-dose run directories have no surviving weight
files. Their data, plans and results remain, so rebuilding would be a separate
recorded reproduction, not recovery of an existing binary. No claim is made about
uninspected external backups.

[PRESERVED_CANDIDATES.json](PRESERVED_CANDIDATES.json) records exact paths, hashes,
sidecars, preservation markers and limitations. Markers are policy records, not
filesystem access controls or an independent backup.

## Current trained candidates and comparison plan

RRP-015 has already trained two independently initialized Supa 1.7 candidates:

| Candidate | Training data | Processed tokens | Updates | Training status |
|---|---|---:|---:|---|
| Clean control C | 384 transitions, seed 17 | 54,172 | 28 | COMPLETE; native qualification PASS |
| Clean rehearsal R | Same 384 transitions + 648 filtered rehearsal examples | 108,389 | 55 | COMPLETE; native qualification PASS |

Both full resumable checkpoint sets are retained. This differs from the historical
cleanup policy. Both exclude 768 scored parents (the historical probe and the new
selection bank); rehearsal additionally excludes short arithmetic text contacts.

The matched acquisition comparison on **SELECT-VIEWS-001** is complete:

| Model | Correct /384 | Family shortcut floors cleared |
|---|---:|---:|
| Supa 1.7 | 14 | 0/3 |
| Opallo 1.0 | 8 | 0/3 |
| Clean control C | 234 | 3/3 |
| Clean rehearsal R | 225 | 3/3 |

The remaining battery is complete. Arithmetic uses the inherited permissive
policy; source figures are the same frozen baselines rechecked by the scorers.

| Candidate | Arithmetic /150 | Retained source answers /47 | Long context /256 | Exact `UNKNOWN` /64 | Gate v2 |
|---|---:|---:|---:|---:|---|
| Supa source | 47 | 47 | 161 | 13 | Source comparator |
| Clean control C | 69 | 39 | 162 | 10 | PASS |
| Clean rehearsal R | 63 | 35 | 134 | 8 | FAIL: retention |

Under strict arithmetic scoring the corresponding totals are 41, 62 and 50, so
the control's arithmetic gain is not merely the permissive formatting policy.
The legacy per-class monitor still reports FAIL for both trained models. Gate v2
is a different, weaker conjunction: arithmetic >=62, source successes retained
>=37, and every acquisition family clearing its floor. Report both verdicts.

Control long context is net +1: 11 source successes lost, 12 gained, paired p=1.
Rehearsal is net -27: 34 lost, 7 gained, paired p=.0000253. Both are report-only
diagnostics; no semantic abstention review or independent confirmation is claimed.
The tested rehearsal intervention doubled training exposure and did not establish
a general benefit from replay. This does not imply all replay methods are harmful.

Both training runs passed actual-source native qualification and completed all
updates. All eight planned evaluation outputs are complete and supervised. Across
13 total attempts (including the three retained failures), observed supervisor
wall time is 3,653.936 seconds, within the amended 8,400-second model-work ceiling.
No experiment model worker is left running. Full training states and all raw
predictions are retained. The successful control also has a preserved HF export
at `rineya-dataset-retention-records/rrp015-20260917/control-preserved-hf`, whose
weight hash exactly matches the export used for evaluation.

The complete hash/source/qualification audit is [RRP015_SUMMARY.json](RRP015_SUMMARY.json).
The control is the candidate to replicate under the permitted experimental gate;
it is not a validated general replacement for Supa or Opallo.

**The historical D1/M4 bank and SELECT-VIEWS-001 are different 384-row instruments.**
Their shared denominatorsâ€”and Supa's coincidentally identical 14 totalâ€”do not make
their scores interchangeable. Historical candidates may have trained on parents
inside the new selection bank. Do not rank D1 against C/R by directly subtracting
those two banks' headline numbers.

Three failed evaluation attempts are retained: missing monitoring dependency,
600-second baseline timeout, and an owned stale-lock preflight refusal. The
published-baseline limits were prospectively changed to 900 seconds under the same
overall compute ceiling; lock recovery is now tested. No scoring threshold, model,
training recipe or memory guard changed to accommodate an outcome.

After this comparison, the next useful work is a usable long-context rehearsal
bank and fresh evaluation instruments, then additional independent training draws
of recipes selected on permitted development evidence. The current spent
long-context suite remains report-only. [LCR plan](LCR_NEXT_EXPERIMENT.md) and
[preference audit](PREFERENCE_AUDIT.md) explain the concrete data gaps and which
new banks can be used safely after further qualification.
