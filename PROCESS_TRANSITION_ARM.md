# Teaching one state transition moved every family, and the arithmetic panel with it

**RRP-007-PROCESS-A1, seed 1, single-seed dose probe. Not a claim.**

Supervised fine-tuning on 1,536 one-step state transitions — 512 per family, positives
only, from parents **disjoint from every scored view** — moved the held-out instrument from
below its shortcut floor to well above it on all three families, and moved the arithmetic
panel it was never trained on from 47/150 to 71/150.

## 1. The instrument it was built for

384 frozen views, 128 per family, one per parent. Floors published before any model ran.

| family | Supa (frozen) | **RRP-007 seed 1** | binding floor | clears at | 95% CI |
|---|---:|---:|---:|---:|---|
| `ledger_replay` | 14/128 = 0.109 | **122/128 = 0.953** | 0.2422 | k ≥ 40 | [0.901, 0.983] |
| `prerequisite_plan` | 0/128 = 0.000 | **92/128 = 0.719** | 0.0703 | k ≥ 15 | [0.633, 0.795] |
| `register_machine` | 0/128 = 0.000 | **75/128 = 0.586** | 0.0547 | k ≥ 12 | [0.496, 0.672] |
| aggregate | 14/384 = 0.037 | **289/384 = 0.753** | 0.1224 | — | |

**All three clear, every lower bound far above its floor.** Strict-exact is 288 against
canonical 289, so the output contract is essentially perfect — one presentation-only row,
against ten for the frozen source.

### It is not the shortcut

The obvious failure mode is a model that learned to echo its input. It did not:

- predictions equal to `before`: **39/384**, of which **38 are the correct no-ops**. It
  emits the copy exactly when the copy is right, which means it learned *when* an event is a
  no-op — part of the rule, not an evasion of it.
- correct on rows where the state actually **changes**: **251/346**.
- posted `transfer` rows, which move value between two accounts simultaneously and produced
  the dominant two-values-wrong bucket at baseline: **3/53 → 47/53**.
- `register_machine` requires arithmetic modulo 97 and the frozen core never reached the
  contract at all (0/128 correct key sets). Remaining errors are arithmetic slips with the
  correct shape and correct untouched registers.

## 2. The unexpected result: it transferred to the arithmetic panel

The 150-row arithmetic panel is word problems from v3. Nothing in this arm trained on it.

| | source | arm | net | lost | class detail |
|---|---:|---:|---:|---:|---|
| **permissive** (registered) | 47 | **71** | **+24** | 10 | pct 21→30, ratio 16→18, time 10→23 |
| **strict** | 41 | **53** | **+12** | 15 | pct 21→20, ratio 10→10, time 10→23 |

**The robust part is `time_duration`: 10 → 23 under *both* policies.** That is a real
arithmetic gain, not a formatting one.

`percentage_reasoning` gains only under the permissive policy (21→30) and slightly regresses
under strict (21→20, 7 lost), so a meaningful part of the headline +24 is **output
discipline**, not new arithmetic. `ratio_proportion` is flat under both. Reporting +24
without this split would overstate the result.

## 3. Retention: passes the binding gate, exactly, and fails the stricter one

Registered adjudication **D3** binds `DATASET_PROGRAMME_PREREG.md:140-141`.

| gate | value | verdict |
|---|---|---|
| arithmetic panel ≥ 62 | **71** | **PASS** |
| source references retained ≥ 37 of 47 | **37** (lost 10) | **PASS, exactly at the threshold** |
| `protocol.json` `G1_RETENTION`, zero correct→incorrect per class | lost 1 / 4 / 5 | **FAIL** |

Both cross-checks agree: `arm_total − outside_source = 71 − 34 = 37`, and
`source − lost = 47 − 10 = 37`.

Three things must be said plainly rather than buried:

1. **Retention passes on the boundary.** One more lost answer and the registered gate fails.
   This is not a comfortable margin and the next seed may not reproduce it.
2. **The stricter registered rule fails.** `G1_RETENTION` allows zero correct→incorrect per
   class and this arm lost 1, 4 and 5. D3 chose the softer gate *before* any number existed,
   and Opallo's own shipped recipe also fails the strict rule — but an arm that clears the
   binding gate has not cleared that one.
3. **Under the strict bare policy retention is worse**: 26 of 41 retained, 15 lost. The D3
   threshold is written against the permissive/47 reference, which is how this programme's
   ledger reports Opallo's 41/47, so permissive is the matched comparison — but the strict
   column is the less flattering one and it is reported here for that reason.

Termination is 50/50 in every class under both policies. Nothing degenerated.

## 4. What this does and does not license

**Licensed.** On held-out worlds of the same three families, this 596M model went from
unable to apply one state-update rule to applying it correctly most of the time, and the
capability transferred to at least one untrained arithmetic class. The programme's
"reasoning cannot be taught here" reading was established against final-answer SFT on
long-prompt instruments; it does not survive contact with a short, atomic, verifiable target.

**Not licensed.** This is a **single seed** and the prereg forbids single-seed claims. It is
in-distribution generalisation — new worlds and new values, the same three families and the
same rule text — and says nothing about novel task types. It does not establish multi-step
search or planning, which the puzzle probe measured at literal zero. `prerequisite_plan` and
`register_machine` remain at 0.72 and 0.59, not solved. And no release or quality claim
attaches: the run's own receipt records
`status: MICROFIT_EXECUTED_NOT_QUALITY_CONFIRMATION` and
`quality_improvement_established: false`.

## 5. Training record

110 update groups, 216,896 processed / 21,100 supervised tokens, one exposure, blocks 8–27,
fp32, CpuAdamW, peak LR 1e-5, WSD 20/60/20, 2048 window — the qualified RRP-001 recipe
unchanged. The arm changed the data and nothing else.

Mean objective CE fell 0.4244 (first ten updates) → 0.1100 (last ten). The engine's own
probes, which score **training** rows and are a health check rather than a result, went
0/24 → 17/24 → 18/24.

Checkpoint `checkpoint-00110`: 220 tensors, all sha256-verified against the sealed manifest,
197 changed by training, 23 identical, 90 source tensors untouched (embeddings, blocks 0–7,
final norm). Frozen source digest unchanged across the run.

## Reproduce

```bash
PYG=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
$PYG experiments/rab001_exploration/materialize_checkpoint.py --source Supa \
     --checkpoint <journal>/runs/train-001/microfit/checkpoint-00110 \
     --out <dir> --research-format
$PYG experiments/rab001_exploration/process_probe.py --model Supa --label rrp007_a1 \
     --checkpoint <journal>/runs/train-001/microfit/checkpoint-00110
$PYG experiments/rab001_exploration/retention_probe.py rrp007_a1 <dir> cuda
```

Views sha256 `45674251e38e…`, checked on every scoring run. Floors and both pre-run
amendments in `forge-v2-prereg-001/PROCESS_PROBE_FLOORS.md`; the four registered decisions in
`ADJUDICATION_001.md`. The retention control reproduced the frozen source reference exactly
(permissive 47, strict 41) before the arm was scored; without that the comparison would mean
nothing. The key-set decomposition and the no-op analysis in §1 were **not** pre-registered
and are reported as mechanism.
