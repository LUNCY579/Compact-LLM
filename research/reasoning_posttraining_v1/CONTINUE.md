# Continue RRP-001 / RRP-002

## Mandatory scope

Read [GOAL.md](GOAL.md), [STATUS.json](STATUS.json), [RESULTS.md](RESULTS.md),
[CLAIM_LEDGER.json](CLAIM_LEDGER.json) (23 entries), [RRP002_INSTRUMENT_PREREG.md](RRP002_INSTRUMENT_PREREG.md),
[BOTTLENECK_CANDIDATES.md](BOTTLENECK_CANDIDATES.md), [SOURCE_AUDIT.md](SOURCE_AUDIT.md) and the
package [AGENTS.md](../../experiments/reasoning_posttraining_v1/AGENTS.md).

**The goal is active and not close to complete.** A qualified lane, a met microfit criterion, an
audited verifier and a validated DEV bank are prerequisites, not the deliverable. **No reasoning,
preservation or generalization claim has been measured, and none may be reported.**

Worktree: `C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1`.
Branch: `research/rineya-reasoning-posttraining-v1`. **Local only — never pushed.** Publishing is
outward-facing and was not authorized; ask before pushing.

```powershell
git status --short; git rev-parse HEAD
$env:PYTHONPATH = 'C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1'
$py = 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe'
& $py -m pytest experiments/reasoning_posttraining_v1/tests -q   # 114 passed, 0 skipped
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

## RRP-002: instrument partly built

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

## Five findings that change what you should do next

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
2. **`RRP2-PRESERVE`** from `residual_capability_v23` (direct_reasoning dev 4,860 with a verbatim
   UNKNOWN contract; procedural_rag dev 1,080 three-way answer/abstain/conflict) plus the v3
   14,000-row preservation slice's 125-answer/125-abstain stratum. DEV only — those dev/ood_dev
   splits were already consumed for dose calibration. **`splits_onpanel` is quarantined; keep it out.**
3. **The statistics layer.** MQ-001 has none: `classify` grades only `exact` and `number`, and
   `analyze.summarize` skips unscoreable rows. World-paired comparison with family-cluster
   dependence and an uncertainty method must be built. Note the inherited `max_dev_rows = 64` cap
   does **not** bind this successor (CL-020).
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
- Human data review is **NOT PERFORMED**. `REVIEW_PACKET.md` covers only the 32 microfit worlds and
  is hand-authored with every decision PENDING; the machine packet
  (`prepared-004/REVIEW_PACKET.jsonl`, 85 worlds = 32 v1 + 53 v2) covers EXACT_PRIMITIVE 32,
  ANSWER 32, IMPOSSIBLE 15, UNKNOWN 6 and no PROSE. **This agent will not self-certify it.**
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
