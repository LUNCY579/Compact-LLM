# DMRF-001 — the rank-16 result is magnitude, not direction selection

**One arm, no training run, one seed. Registered in `DMRF001_PREREG.md` and committed
(`c46b6ba`) before the arm was built. All eight registered predictions held. Long context and
abstention are report-only on a SPENT suite. Nothing selected, nothing promoted.**

Arm `502260c9b28994a4335e9a9ff9f8c954da42032a2818eee3cec16430a9055c36`, built twice and
byte-identical both times. Matched conditions verified on every artifact: implementation digest
`a329eb7bb0…`, transformers 4.51.3, `.venv-rab001-cu126`, `max_context` 4096, greedy, TF32 off,
cuda, **0 rows skipped for overflow**. The retention probe bound the arm's own weight hash and
the source hash in its receipt.

---

## The control, and why it was the missing piece

`P16` does two things at once: it keeps 16 singular directions per matrix **and** it shrinks the
update norm to 0.812 of the full delta. RANK-001 measured the *combination* and read the result
as the skill being carried by 16 directions. N16 separates them.

For each 2D trained matrix, `s_t = ‖P16(Delta_t)‖_F / ‖Delta_t‖_F`, applied to the **full-rank**
delta. Measured `s_t` range **0.712–0.960**; realised global norm fraction
**0.812249925436939**, matching RANK-001's r=16 arm to all 15 digits. So N16 has r=16's
per-tensor norm profile and **every** singular direction.

---

## The frontier at matched norm

| arm | norm ratio | rank | acq /384 | floors | arith perm/strict | retained /47 | long ctx | paired p vs Supa | `UNKNOWN` | gate v2 |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|:--:|
| frozen Supa | 0 | — | 14 | 0/3 | 47 / 41 | 47 | 161 | — | 13 | fail |
| α = 0.75 *(INT-001)* | 0.750 | full | 234 | 3/3 | 68 / 61 | 39 | 163 | 0.824 | 10 | PASS |
| **N16** *(this screen)* | **0.812** | **full** | **236** | **3/3** | **69 / 62** | **39** | **162** | — | **10** | **PASS** |
| r = 16 *(RANK-001)* | 0.812 | 16 | 235 | 3/3 | 68 / 60 | 39 | 159 | 0.845 | 10 | PASS |
| k = 0.25 *(the promoted core)* | 0.836 | full | 232 | 3/3 | 72 / 63 | 41 | 164 | 0.629 | 12 | PASS |
| full delta | 1.000 | full | 235 | 3/3 | 71 / 64 | 40 | 157 | 0.572 | 9 | PASS |

Per-family canonical rates for N16: `ledger_replay` 0.4297, `prerequisite_plan` 0.7422,
`register_machine` 0.6719, against floors 0.2344 / 0.0781 / 0.0703.

---

## Finding 1 — at identical norm, truncation and full rank are indistinguishable

The comparison the screen exists to make, against REP-001's single-draw spreads:

| endpoint | r = 16 | N16 | difference | draw spread | verdict |
|---|---:|---:|---:|---:|---|
| acquisition /384 | 235 | **236** | +1 | 7 | **within noise** |
| arithmetic permissive | 68 | **69** | +1 | — | within noise |
| arithmetic strict | 60 | **62** | +2 | 13 | **within noise** |
| references retained /47 | 39 | **39** | 0 | — | identical |
| long context /256 | 159 | **162** | +3 | 13 | **within noise** |
| literal `UNKNOWN` /64 | 10 | **10** | 0 | — | identical |
| gate v2 | PASS | **PASS** | — | — | identical |

**A full-rank update scaled to the same norm profile performs identically to the rank-16
truncation — marginally better on four of six endpoints, identical on two, worse on none.**
Every difference is inside the spread the same recipe shows across independent training draws.

So discarding 1008 of 1024 directions per matrix costs nothing *because it was never the
directions doing the work*. The 235 is explained by the update being 0.812 of full magnitude.

**What this changes.** RANK-001's Finding 1 is correct as a **storage** result and unsupported
as a **mechanism** result:

- **Survives, and it is the valuable part.** This delta *stores* at rank 16 — 7.26M parameters,
  27.7 MB, 43× smaller than the 314.62M-parameter dense delta — with no behavioural loss on any
  instrument here. Independently reproduced: I recompute 7.21M in the 2D factors plus 0.0461M
  1D kept in full = **7.26M**, matching exactly.
- **Does not survive.** "A skill is a rank-16 object" in the sense of the capability *living in*
  16 directions. A full-rank update of the same magnitude is behaviourally the same model.

---

## Finding 2 — this arm's own premise came out negative too

The prereg registered a falsification condition for N16 itself: if N16 landed within 1 row of
α = 0.75 on every endpoint, then the per-tensor norm *profile* carries nothing beyond the global
norm and the uniform control was already sufficient.

| endpoint | α = 0.75 (uniform, norm 0.750) | N16 (per-tensor, norm 0.812) | difference |
|---|---:|---:|---:|
| acquisition | 234 | 236 | +2 |
| arithmetic permissive | 68 | 69 | +1 |
| references retained | 39 | 39 | 0 |
| long context | 163 | 162 | −1 |
| literal `UNKNOWN` | 10 | 10 | 0 |

**That condition is met.** A heterogeneous per-tensor norm match (`s_t` spanning 0.712–0.960)
behaves the same as a single uniform scalar at a similar global norm. So the answer to "does
the per-tensor profile matter?" is no, and INT-001's α line was already an adequate magnitude
control — nobody had joined it to the rank result.

Reporting this because it is a negative result about the arm I built, not only about someone
else's.

---

## Finding 3 — nothing on this line repairs long context, and the ordering is noise

N16's 162 sits 1 row above frozen Supa's 161. k = 0.25's 164 sits 3 above. r = 16's 159 sits 2
below. The whole observed range across every arm of three screens — **157 to 164** — is 7 rows
against a **13-row** single-draw spread.

**The honest statement is that no arm on this delta is distinguishable from the frozen source on
long context, and none is distinguishable from another.** The apparent ordering
k=0.25 > α=0.75 > N16 > r=16 > full is not a ranking; it is five draws from one distribution.
Prediction 6 held: N16 does not exceed Supa by more than the noise band.

Abstention regresses on every arm, 13 → 9–12, with no exception. N16 is 10.

---

## Predictions versus outcome

| # | prediction | outcome |
|---|---|---|
| 1 | N16 acquisition in 232–237 — the prediction the arm exists to test | **held** at 236 |
| 2 | N16 acquisition ≥ 220 (the weak form; failure would vindicate the mechanism reading) | **held** at 236 |
| 3 | N16 clears all three family floors | **held**: 3/3 |
| 4 | N16 retained ∈ [39, 43] of 47 | **held** at 39 |
| 5 | N16 long context ∈ [157, 165] and ≥ 159 | **held** at 162 |
| 6 | no arm repairs long context beyond the noise band | **held**: 162 vs 161 |
| 7 | gate v2 PASS | **held**: 69 ≥ 62, 39 ≥ 37, 3/3 |
| 8 | the legacy per-class retention monitor FAILS | **held**: FAIL, both policies, all three classes |

**8 of 8.** The screen made a specific prediction about a control nobody had built and the
control landed where the audit's bracketing argument said it would.

---

## Damage, stated as paired counts

N16 against the frozen source on the gate's own instrument, permissive policy:

| | count |
|---|---:|
| source answers retained | **39** of 47 |
| source answers **lost** | **8** |
| correct outside the source set | 30 |
| arm total | 69 of 150 |

**The strict preservation criterion FAILS: 8 of the source's 47 correct answers were lost.**
Per class, lost = 1 `percentage_reasoning`, 3 `ratio_proportion`, 4 `time_duration`. Strict
policy: 8 lost of 41, arm total 62.

`references_retained` 39 clears gate v2's ≥ 37 while the legacy per-class monitor fails all
three classes under both policies — the same disagreement every arm in this programme produces.
Both verdicts are reported. No threshold was changed.

---

## Cost

| item | value |
|---|---|
| training runs | **0** |
| optimizer steps | **0** |
| model calls to construct | **0** |
| build wall time | ~3.5 min, twice (byte-identical) |
| build peak host memory | 2.38 GiB FP32 state dict; refused below 4.0 GiB free |
| battery wall time | process 310 s, retention 55 s, long context 163 s = **528 s** |
| **deployable artifact** | **2,384,234,992 bytes** — a dense FP32 model, the same size as every other arm |
| stored delta if factored | 7.26M params / 27.7 MB — **not a servable artifact on this path** |
| inference overhead, dense vs factored | **UNAVAILABLE** — never measured, no factored serving path exists |

The two artifact figures are both reported because the report under audit quoted only the
second. No factored serving path exists or is qualified, so a 27.7 MB skill is a storage
representation that must be expanded to a ~2.38 GB dense model to run.

Net gain against r=16 is +1 acquisition row, inside the noise band, so **no cost-per-gain ratio
is reported** — `analysis/paired.py::cost_per_net_correct` refuses to compute one when the gain
is not established, and the fixture tests assert that refusal.

---

## Reproducibility, and a defect inherited from the shared builder

The arm was built twice, three minutes apart, and produced
`502260c9b28994a4335e9a9ff9f8c954da42032a2818eee3cec16430a9055c36` both times.

But RANK-001 Erratum 2 measured that `torch.linalg.svd` on this build is deterministic at a
**fixed** thread count and not **across** thread counts. **Amendment, recorded post-hoc:** the
measured arm was built at this host's torch default of **16 threads**, and its
`MATERIALIZATION.json` — written before that erratum's diagnosis reached me — does not record
the count. `build_n16.py` now pins `torch.set_num_threads` and records it; the default is 16
rather than 1 so that the shipped default still reproduces the measured hash. The original
receipt is left unedited and this paragraph is the amendment.

One datum that narrows the shared defect: **this builder calls only `svdvals`, never the
singular vectors.** Singular values are unique; singular subspaces are not. That this arm
rebuilds byte-identical while RANK-001's r=4 reconstruction did not points at the vector
computation rather than the values.

---

## What this licenses, and what it does not

**Licenses.** One sentence, and it is a correction rather than a discovery: the rank-16
compression of this delta is a **storage** result, not evidence that the capability occupies a
16-dimensional subspace. Compression by 43× at no behavioural cost stands and is worth having.
The mechanism claim should be withdrawn.

**Does not license.** Anything about *two* skills composing — that is untested and needs a
second delta. Any confirmation: long context and abstention are report-only on a SPENT suite
and nothing on this host can confirm them. Any statement about deployment size or latency
beyond the dense artifact measured above. Any claim that preservation was demonstrated — 8 of
47 source answers were lost and the strict criterion fails.

One draw, one update direction, one base model.
`quality_improvement_established` stays false. `release_permitted` stays false. Nothing
promoted.
