# Registered CPU-budget replication audit

Status: PASS for execution and fidelity. Existing router-scaffold development gate: True.

Question: does the identical A0/A1/A3 matrix complete and preserve quality when
the generation deadline alone increases from 90 to 900 seconds per case?
Preregistration was committed before execution. Setup: same 16 MERGEDEV-fast
cases, eight smoke cases per arm, selected layers, CPU FP32/eager, four threads,
seed 17, greedy token caps, no cache, scoring and existing gates. The source
manifest identifies the exact execution commit, runtime and checkpoint hashes.

| Arm | Completed | Correct among completed | EOS | Token cap | Max case seconds |
|---|---:|---:|---:|---:|---:|
| A0_dev | 16/16 | 12 | 16 | 0 | 195.9 |
| A1_dev | 16/16 | 12 | 16 | 0 | 196.4 |
| A3_dev | 16/16 | 13 | 16 | 0 | 224.3 |

- A1_dev vs A0: 0 paired wins, 0 paired losses; preservation=True; router scaffold gate=False.
- A3_dev vs A0: 1 paired wins, 0 paired losses; preservation=True; router scaffold gate=True.

Fidelity checks: 19/19 pass. The prior run had
66 non-censored case-runs and six timeout case-runs across smoke/development.
The identity CSV compares complete token IDs for the former and old prefixes
for the latter, plus first-token logit metrics for all 72. Configuration,
data, source, runtime, selected arms and checkpoint/state invariants are checked
independently; a mismatching audit cannot license escalation.

Canonical restoration and frozen-parameter checks are in decision.json. Hidden,
NLL, prefill/decode, FLOP and RSS evidence is retained in the source run and
dynamics_efficiency.csv. Deadlines remain cooperative between Torch forwards;
the reporting latency also includes optional gold NLL. Token-cap completions
retain the original scoring policy and are identified separately from EOS.

Interpretation: this resolves the registered time-budget question only if
coverage and fidelity pass. A passing existing development gate licenses a
future scaffold experiment, not a routing or intelligence claim. All 16 cases
are development data previously inspected in the original run. There is no
sealed, training, compute-matched efficacy or long-context reasoning result.
The prior failed/inconclusive evidence remains unchanged.

Scope complete at this matrix and audit: no automatic router implementation,
PEFT, additional layer selection, or sealed evaluation was performed.
