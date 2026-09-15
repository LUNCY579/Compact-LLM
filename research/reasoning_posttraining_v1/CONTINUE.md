# Continue RRP-001 / RRP-002

## Mandatory scope

Read [GOAL.md](GOAL.md), [STATUS.json](STATUS.json), [RESULTS.md](RESULTS.md),
[CLAIM_LEDGER.json](CLAIM_LEDGER.json) (28 entries), [RRP002_INSTRUMENT_PREREG.md](RRP002_INSTRUMENT_PREREG.md),
[BOTTLENECK_CANDIDATES.md](BOTTLENECK_CANDIDATES.md), [SOURCE_AUDIT.md](SOURCE_AUDIT.md) and the
package [AGENTS.md](../../experiments/reasoning_posttraining_v1/AGENTS.md).

**The goal is active and not close to complete.** A qualified lane, a met microfit criterion, an
audited verifier and a validated DEV bank are prerequisites, not the deliverable. **No reasoning,
preservation or generalization claim has been measured, and none may be reported.** The operator
has attested the data and authorized GPU fine-tuning, so the ladder is now open — but it still
needs the instrument gaps closed and its margins predeclared first.

Worktree: `C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1`.
Branch: `research/rineya-reasoning-posttraining-v1`. Published to `origin` with operator
authorization. Never force-push; never rewrite history.

```powershell
git status --short; git rev-parse HEAD
$env:PYTHONPATH = 'C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1'
$py = 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe'
& $py -m pytest experiments/reasoning_posttraining_v1/tests -q   # 118 passed, 0 skipped
```

## RRP-001: done, and exactly what it proves

Three supervised launches on real Supa 1.7 weights, all settled, receipts committed under
`evidence/inspection-20260915/rrp001-runs/` with all 346 sealed budget events. Checkpoint tensors
(~11.3 GiB) live only under `rineya-reasoning-posttraining-records/rrp001-journal/`.

| run | outcome |
| --- | --- |
| `qualify-001` | PASS. 596,049,920 unique / 314,618,880 trainable in 220 tensors, recounted natively. Source state byte-identical before and after. 0 optimizer steps. 32/32 targets reproduced. 2048-token shape qualified. |
| `train-001` | 105/105 updates. Fixed 32-world TRAIN probe 8/32 → 29/32 → **31/32**; response CE 1.496 → 0.0855 → 0.0122. Criterion was ≥31/32: **met**. |
| `resume-001` | PASS, 15/15 checks. 103 updates from `checkpoint-00002` in a fresh process reproduced the candidate's final parameters **bit-identically** and every prediction. |

The probe rows are the rows it trained on, so 31/32 is fitting, not generalization. It establishes
that the optimizer, masking, EOS handling, generation, scorer, checkpoint/resume and budget
accounting all work on real weights. **`checkpoint-00105` is not a quality candidate — do not
export, merge, quantize or register it.**

Final RRP-001 spend: 1,067 of 2,400 GPU-seconds, 333,987 of 400,000 processed training tokens,
130 of 160 evaluation calls, 11.34 of 16 GiB disk. **Its single adaptive round is spent.**

## RRP-002: instrument built, statistics layer outstanding

Designed and pre-registered in `RRP002_INSTRUMENT_PREREG.md`, with the impossibilities recorded
*before* any result depends on them. Built so far, all CPU-only:

- **Verifier adversarially audited and it holds.** 280 v2 worlds × 22 attack classes = 4,390
  attacks, 0 violations. Nothing that must never earn credit did: 6 malformed classes, 5
  reward-manipulation classes (including forged `reward`/`correct` keys and prompt-injection keys),
  false UNKNOWN and false IMPOSSIBLE both directions, off-by-one, nonoptimal, infeasible,
  truncated, duplicated, invalid-token. **20 of 82 reversed plans WERE accepted**, confirming
  equally-optimal plans are not penalized — so **never score v2 by text-matching `reference_text`**;
  call `rewards.score_response`. Module `verifier_audit.py`, tested against five deliberately
  broken stub verifiers so a pass means something.
- **`RRP2-DEV-V1` built and proven**: 384 Forge v1 validation worlds (48 × 8 microfit families),
  strata ANSWER 328 / UNKNOWN 56, zero overlap with the 32 microfit worlds, 384 distinct
  `case_sha256`. **Positive control 384/384** — the independent exact solver reproduces every gold.
  File `evidence/inspection-20260915/RRP2_DEV_V1_BANK.jsonl`, sha256 `26206606a2b9be1a…`.
- **Shortcut floor measured: 0.1458 strict accuracy (56/384)**, reached by three trivial
  strategies. Report it beside any model number.
- **A 4096 EVALUATION window is qualified** (CL-027): cached/uncached agreement 5.29e-05, peak
  VRAM 5.38 GiB, source bytes unchanged, 0 backward passes. **Inference only** — training at
  4096 needs its own qualification.
- **The first held-out measurement exists, replicated across both windows** (CL-024, CL-028):

| | 2048 | 4096 |
| --- | --- | --- |
| evaluated / refused overlength | 240 / 144 | 333 / 51 |
| model strict accuracy | 0.2625 (63/240) | 0.2553 (85/333) |
| shortcut floor, same subset | 0.1417 | 0.1411 |
| positive control, same subset | 1.0000 | 1.0000 (333/333) |
| ANSWER stratum | 47/206 | 62/286 |
| UNKNOWN stratum | 16/34 | 23/47 |
| false abstention on answerable | 27.7% | 28.3% |
| unsupported answer on unanswerable | 52.9% | 51.1% |
| sibling collision rate | 0.2147 | 0.2056 |

  The source clears the floor by ~11 points, so it is doing something real. But
  `constant_unknown` **beats it outright on the UNKNOWN stratum** (47/47 vs 23/47), and both
  abstention error directions are large. `dialogue_memory` is **0/43**. Every pattern
  replicated across a 39% larger sample, so the 2048 length-bias caveat is now bounded rather
  than hypothetical.

## Findings that change what you should do next

1. **`test_length` is unreachable on this host** (CL-016). v1 11k–36k tokens, v2 27k–61k; zero rows
   fit any window to 16,384. With `eager` attention, one fp32 layer's scores cost ~1.07 GiB at
   4,096, ~4.3 GiB at 8,192, ~17 GiB at 16,384 against ~7.2 GiB free VRAM. **4,096 is the realistic
   next qualified window; 8,192 marginal; 16K out.** Host limit, not a model finding.
2. **No independent confirmation bank is possible here** (CL-017). Nothing is label-sealed,
   authorship is shared, a new seed changes instances only. Best honest label:
   `FRESH_INSTANCE_WITHIN_GRAMMAR_SAME_AUTHOR_NOT_INDEPENDENT`. Do not let "fresh" imply independence.
3. **The v1 question text leaks its own sentinel** (CL-022). Every question ends "…or UNKNOWN.", so
   copying the last capitalized token scores 56/56 on the whole UNKNOWN stratum. That is why the
   floor is 0.1458 and why abstention gains must be compared against `copy_question_sentinel`.
4. **v1's non-UNKNOWN strata are confounded with family** (CL-019): IMPOSSIBLE only in
   `temporal_order`, UNREACHABLE only in `weighted_routes`, INCONSISTENT only in `truth_tellers`.
   `BOTH` is an ANSWER, never abstention. Do the stratified abstention analysis on **v2**, which has
   a structured status spanning families, with v1's UNKNOWN as the cross-grammar replication.
5. **B-1 was reinterpreted and is still unresolved** (CL-015). The one unfitted microfit world's
   `211` is not a recency shortcut — the untrained source said `UNKNOWN`, and `211` is the gold of a
   structurally near-identical sibling world. Cross-world memorization interference is the live
   hypothesis, n=1, default expectation non-replication. Comparison baselines are now measured:
   answer-agnostic predictors collide with a *specific* sibling gold at 0.0000–0.0550.

## Ordered next work

CPU-only, no budget needed:

1. **`RRP2-DEV-V2`** on v2 `validation` (1,600 worlds; ANSWER 1,218 / IMPOSSIBLE 244 / UNKNOWN 138),
   scored with the audited `rewards.score_response`. Note only 532 of 1,600 fit 2,048 — either
   qualify 4,096 first or label the bank explicitly short-context and biased.
2. ~~**`RRP2-PRESERVE`**~~ — **done**: `RRP2-PRES-DEV`, 768 rows, see below. Its build forced
   six corrections to its own pre-registration (PRES-C1..C6, amendment A1).
3. **The statistics layer — now the single blocking prerequisite.** MQ-001 has none: `classify`
   grades only `exact` and `number`, and `analyze.summarize` skips unscoreable rows. World-paired
   comparison with cluster dependence and an uncertainty method must be built. Two hard
   constraints it must honour, both measured rather than assumed:
   - **Cluster on `cluster_id`, which every `RRP2-PRES-DEV` row already carries** (CL-038).
     Sub-bank (b) has **16 effective units for 192 rows**; treating them as independent would
     make an interval there roughly three times too narrow.
   - **Use the 40 complete minimal pairs as the paired unit for abstention calibration**
     (CL-039), not as 80 independent rows.
   The inherited `max_dev_rows = 64` cap does **not** bind this successor (CL-020).
4. **Predeclare** primary contrasts, family weights, cluster units, practical margins and the
   allowed number of model-selection reads — before any arm runs.

Needs a new registered experiment and envelope:

5. ~~Measure the source on `RRP2-DEV-V1`~~ — **done**, twice, see the table above.
6. ~~Register the 4,096-window qualification~~ — **done for inference** (CL-027). Training at
   4096 still needs its own qualification, with a backward pass and optimizer state.
7. **Measure a candidate's sibling-collision rate against the source's 0.2056** — that is the
   fluency-matched null for B-1 (CL-025). Until then B-1 is untested, not supported.
8. Only then the A/B/C ladder, reported review-pending unless the packet is signed.

Budget lanes now registered in `budget.REGISTERED`: `RRP-001-MICROFIT` (spent),
`RRP-002-BASELINE` (launch/retry counters exhausted) and `RRP-002-BASELINE-A2`
(amendment A2, launches 4→8 and retries 2→6 — raised because **two launches died on my own
implementation defects**, not scientific need; see `BUDGET_AMENDMENT_A2.json`). A2 has 7
launches, 1,263 evaluation calls and 3,069 GPU-seconds left, with training, model selection,
RL and cloud still pinned at zero.

## Hard constraints that bit during this work

- **Binding equality forbids repo changes mid-experiment** (CL-013). `binding` contains the RRP
  package code hash and HEAD, and `validate_checkpoint` enforces equality, while `checkout()`
  requires an empty `git status --porcelain --untracked-files=all`. You cannot edit or commit **any**
  repo file between a candidate attempt and its resume. Note `budget.py` changed after RRP-001
  finished, so RRP-001's checkpoints will no longer validate against a current binding — expected,
  and harmless because RRP-001 is complete.
- **Data approval is an operator ATTESTATION, not a per-world review** (CL-029,
  `OPERATOR_DATA_APPROVAL.json`). The operator holds the authority and approved in session, so the
  admissibility blocker is lifted. But no reviewer worked through the worlds: every "Human
  decision" in `REVIEW_PACKET.md` still reads PENDING. **Critically, the 85 packet ids and the 384
  `RRP2-DEV-V1` ids are DISJOINT** — the packet covers all 32 microfit TRAIN worlds and none of
  the DEV bank. So the bank every held-out number comes from is
  `MACHINE_VERIFIED_NOT_HUMAN_REVIEWED`: byte-verified release, `validate_row` on all 384 rows,
  384/384 positive control, zero prior exposure. Do not describe it as human-reviewed.
- RL stays shut. The v2 verifier now passes adversarial audit, which is a prerequisite, but the
  readiness gate and on-policy pilot are not attempted and `on_policy_rl_rollout_tokens` is 0.
- **`instrument.py` must stay torch-free at module level** — the foreground controller imports
  it, and importing torch there costs the child ~0.6 GiB of the headroom its own pre-load guard
  measures. Every torch-dependent name must be imported lazily inside its function; forgetting
  `parameter_digest` cost a supervised launch. A subprocess test now asserts this.
- **`native_runtime.evaluate` is `@torch.no_grad()`-decorated and must stay so.** Without it a
  caller builds an autograd graph over 28 layers of eager attention and exhausts VRAM on the
  first long row. That cost another launch.
- Instrument defects to fix in the successor: CL-014 (`final_evaluation_sha256` hashes wall-clock
  timing and can never match across runs), CL-008 (`qualify` charges 2 calls for ~16 forwards),
  CL-009 (30 evaluation calls of headroom left — a retried resume would be refused), CL-011
  (checkpoints only at update 2 and the last, so a late abort forfeits ~102 updates), CL-018
  (regeneration must diff against the shipped case hashes; 7 of 480 sampled new-seed cases collided).
- Promote `evidence/inspection-20260915/rrp001-runs/verify_resume.py` into the package.

Preserve negative results. Never extend a test until significance appears, never pool seed
measurements into artificial sample size, and keep `CLAIM_LEDGER.json` growing rather than
resolving contradictions silently — five of its entries are corrections to this agent's own
earlier claims.

## RRP2-PRES-DEV: the preservation / abstention bank (built 2026-09-15)

The third lane, and the one that decides whether a reasoning gain was paid for. Ordinary useful
short answers plus appropriate abstention, from two **non-Forge** ordinary-language corpora, so
it is a genuinely different composition domain from both Forge grammars.

| | rows | effective clusters | contract |
| --- | --- | --- | --- |
| (a) `residual_v23` direct_reasoning dev | 384 | 384 | prompt-contained reasoning |
| (b) `residual_v23` procedural_rag dev | 192 | **16** | retrieval-grounded procedures |
| (c) v3 slice `answer_or_abstain` | 192 | 189 | ordinary language, 40 minimal pairs |
| **total** | **768** | **589** | 384 answerable / 384 unanswerable |

- **Positive control 768/768.** Every gold is its own answer under the declared contract.
- **Leakage: zero on all four axes** against `sealed.jsonl` (110),
  `sealed_roleplay_v2_DO_NOT_TRAIN.jsonl` (400), `dev.jsonl` (510), `dev_roleplay_v2.jsonl` (400)
  and the 32 selected TRAIN worlds. All 768 rows pose distinct questions verbatim.
- **Shortcut floor 0.4375** (336/768), by `constant_abstain`. Strict and tolerant agree.
- File `evidence/inspection-20260915/RRP2_PRES_DEV_BANK.jsonl`, receipt
  `RRP2_PRES_DEV_BUILD.json`.

**Three things about this bank that are easy to get wrong.**

1. **There are TWO abstention sentences, not one** (CL-037). `procedural_rag` conflict rows
   require *"The available sources conflict, so I can't give one certain answer."*; everything
   else requires *"I don't know based on the available information."* And `direct_reasoning`
   requires the **missing-information** sentence for 405/405 of its *conflicting_evidence* rows
   too — so in (a) the two abstention causes are indistinguishable from the response, and the
   missing-vs-conflict comparison is a **contract** distinction only in (b). `wrong_abstention_kind`
   is its own scored channel; abstaining for the wrong reason is neither correct nor a plain miss.
   This is also why the floor is 0.4375 rather than 0.5: `constant_abstain` cannot reach the 48
   conflict rows.
2. **Row counts are not denominators** (CL-038). 768 rows are 589 digit-blinded clusters, and
   sub-bank (b)'s 192 rows are **16** — one question template per procedure family, numbers
   varying. Internal redundancy is measured and published rather than gated, because deleting 198
   legitimate rows would only flatter a statistic.
3. **40 genuine decisive-evidence pairs exist** (CL-039), which the pre-registration said were
   impossible here. Each pair differs by the presence of the decisive fact alone, so it tests
   abstention *calibration*: `evidence_sensitive` / `abstains_regardless` / `answers_regardless`
   / `inverted`. The bank's best analysis unit was nearly lost to a confident sentence in my own
   design document.

**DEV only, and machine-enforced.** `residual_v23` dev was read by the completed v2.3 run's dose
calibration and DEV/OOD selection; every admitted v3 row is `split: train` with
`human_review_status: pending`. `quarantine.refuse(..., use="DEV")` passes;
`use="CONFIRMATION"` does not. Label: `MACHINE_VERIFIED_NOT_HUMAN_REVIEWED`.

## RRP2-QUARANTINE: what is barred, and from what

`QUARANTINE.json`, 18 entries, create-only, checked by code rather than read by a human at
decision time — the precedent being `model_soup.py`, which already refuses to run while
`splits_onpanel/CONTAMINATED.md` exists.

Entries are **scoped**, not boolean (CL-034). Each carries `bars` ⊆ (`ANY_READ`, `TRAIN`,
`SELECTION`, `DEV`, `CONFIRMATION`), because `residual_v23` dev must never be trained on yet is
the legitimate source of the preservation bank, while both `fresh_sealed` splits must not be
opened at all. One flag would either leak a seal or refuse a bank we are entitled to build.

Every row count the inventory asserted was re-measured and **matched exactly** — forge test_iid
200, test_transfer 200, `splits_onpanel` 845/120, residual dev/ood_dev 4860/4860/1080/1080, the
four consumed sealed TESTs 738/792/1320/4050, Long_Context_Demo 256, ILL test 1024. Zero
mismatches. One entry, `mq-data-001`, is registered `present: false`: the inventory cites a
`SUPERSEDED.txt` that exists in neither checkout, so it carries no digest.

**RRP-002 declines a sealed split it could have used** (CL-035). This bounds a claim in the
pre-registration, which says flatly that "nothing is label-sealed". True of the Forge releases;
**not** true of the v2.3 corpus, which ships 3,960 unconsumed `fresh_sealed` rows under a
one-shot policy — the strongest held-out surface on this machine. It belongs to the v2.3
programme's pre-registration, so spending it here would consume there what it bought here. Both
files are `ANY_READ`: digested, never parsed. The honest sentence is *"no independent
confirmation bank is available **to RRP-002**"*, not *"nothing is sealed"*.
