# TRISKILL-GEO-001 PARENTQ: parent qualification before the GEO screen

**Status.** DRAFT, written 2026-10-02. Not frozen, not independently reviewed. **No native execution
is authorized.** No model has been loaded or run by this lane.

**Authorization.** The operator instructed, in session: "Start the draft of the parent-check
pre-screen". That covers design, implementation and CPU verification. Running it natively needs:
1. an independent review;
2. a freeze (`build_parentq.py freeze`);
3. an explicit operator grant, passed to `parentq_launch_wmi.py --launch`, which the supervisor records
   verbatim.

**Branch.** `research/triskill-geo001-parentq`, created from `research/triskill-geo001-sectionh-fixes` @
`9b44e1ed`. Directory: `experiments/triskill_geo001_parentq_v1/`.

**GEO is not modified.** Every file pinned by GEO's `FREEZE_SECTIONH.json` (sha256 `f18d4867…b687`) is
byte-identical on this branch. This lane executes GEO's code only from those pinned bytes. It does not
amend GEO's registration, rows, gate or verdict.

## 1. Why this exists

The frozen GEO screen tests its parents inside its single model load. Its parent gate is
`benchmark.parent_gate` on 4 T rows and 4 H rows. A parent-gate stop therefore **spends the one-shot
screen and says nothing about the geometric hypothesis** (AMENDMENT_SECTIONH section 9). The review
estimated the probability of reaching construction at about 0.08–0.38; that is a reviewer's model, not
a measurement.

**The H side is the larger unknown.** These facts were verified first-hand on 2026-10-02 from δ_H's
training rows (`compose001-20260918/H/H-prepared/microfit-train.jsonl`, sha256 `cb3df333…`, 648 rows):
- **The screen's H contract appeared only on answer rows.** The HSKILL answerable contract ("… If one
  answer is not supported, reply exactly: "I don't know based on the available information."") is on 10
  training rows. **All 10 have answer gold** ("Alex.", "11.", "160 miles.").
- **The exact abstention reply came only under two other prompts.** All 56 exact "I don't know based on
  the available information." replies sit under:
  - prompt A: "Solve the problem silently and give only the concise final answer. Use only the supplied
    facts; do not show chain-of-thought or invent missing information." (28 rows, all abstentions);
  - prompt B: "Use the stated facts to reach the result. Return only the compact conclusion, not the
    reasoning steps." (28 rows, all abstentions).
- **So the contract cue and the abstention cue point in opposite directions.** H learned to answer under
  the screen's contract, and to abstain under prompts A and B. Its abstention under the screen's contract
  is unmeasured (AMENDMENT section 5 already discloses this).

## 2. Questions

**Q1 (decision).** What is the probability that the frozen GEO parent gate passes? This is computed as
the posterior predictive probability over screens built by the same constructors with the screen's
composition:
- T: one row of each subtype (posted transfer, posted scale, register add, register conditional);
- H: answerable, answerable, distractor, cue.

**Q2 (diagnostic).** On missing-value rows, does H abstain more than BASE under the screen's contract? If
not, does it under its own abstention training prompts? And is that specific to H, or does BASE abstain
too?

**No outcome is evidence for or against the geometric hypothesis, and none is a capability claim.**

## 3. What runs, and what never runs

- **Arms.** BASE (Supa 1.7), T (Titan-Sparse) and H (H-alone = FP32(Supa + TT − Sparse)). These are GEO's
  own arm constructions, via GEO's `load_arm_weights`.
- **Hook.** GEO's block-14 hook stays registered at IDENTITY (change = 0), exactly as in the screen's
  parent units. The worker refuses to generate if the change is ever nonzero.
- **Never run:** TH, ROTATION, REVERSE, RANDOM, PROMPT, and any J or K row.
- **Never read:** any protected bank. C_FRESH, T_BANK_V5, H_FRESH and HSKILL_CONFIRM_V2 stay unopened, and
  TRISKILL Stage B stays sealed. The only banks read are exposed ones: T_BANK_V3X/V4 and δ_T's training
  rows (for the collision scan), δ_H's training rows (prompt text only), and HSKILL_VIEWS_V2 with
  CONFIRM-002's stored predictions on it (for the canary).

## 4. Rows (`parentq_bank.json`, built by `parentq_rows.build_bank`)

Every row is made by GEO's frozen constructors (`_transition_h`, `_answerability_h`). The registered
seed is 20261002, with row order shuffled by seed 20261003.

**T: 24 rows, 6 per subtype.**
- Ledger balances are −30..40 (the screen's ledger rows span −27..37). Transfer amounts are 1..12 (screen:
  8) and scale factors 2..4 (screen: 3). Status is always posted.
- Register values are 0..96, with operand −8..11 (never 0) and no modulo-97 wrap.
- Every row passes GEO's `t_support_violations`.
- Collisions: 0 against the 1,408 exposed ledger/register payloads, checked with GEO's own `t_collisions`.

**H_ON: 32 rows** (12 answerable, 10 distractor, 10 cue).
- The screen's template and the screen's system prompt (`ANSWERABILITY_SYSTEM_H`, verbatim).
- 32 fresh worlds (tree names). None occurs as a word in any GEO row or in δ_H's training prompts.

**H_OFF: the same 32 user texts**, with the system prompt replaced by H's training prompt A or B
(alternating within each kind: 6/6, 5/5, 5/5). Gold answers are unchanged.

**Membership checks (CPU validation receipt).**
- All four GEO signatures (row id, world, state, rule) are disjoint from every GEO screen, development,
  legacy, format and conjunction row.
- Row ids, worlds and (state, event) pairs are unique within the bank.
- One ledger event recurs on two different states, which is allowed.
- Longest prompt: 144 tokens (≤ 288). Longest expected reply: 29 tokens (< 32).

## 5. Arm-identity canary (`parentq_canary.json`)

**Purpose.** To catch a wrong arm before its data are used: wrong weights, a wrong H accumulation, or a
wrong template.

**Construction.**
- Source: the exposed HSKILL_VIEWS_V2 bank and CONFIRM-002's stored greedy FP32 predictions on it
  (`STAGE_A_H_VIEWS_BASE_SPARSE.json`, `STAGE_A_H_VIEWS_NEW.json`, committed in `1d54b93f`).
- Eligible rows are those where the stored Supa, Titan-Sparse and H25 texts are **pairwise different**
  (56 of 224). On those rows a wrong arm cannot match the right source.
- Each eligible row's stored replies are ≤ 31 tokens.
- Selected: 4 answerable and 4 boundary rows, in sha256(id) order.
- CPU validation shows the GEO renderer equals CONFIRM-002's `apply_chat_template(…, enable_thinking=False)`
  on all 8 rows.

**Rule.** Each arm runs its 8 canary rows before any of its qualification rows. It must reproduce its
own source's stripped text on at least 7 of 8, or the run stops: `STOP_ARM_CANARY:<arm>`.

**Why one flip is tolerated.** Two harness differences are known:
- CONFIRM-002 used HF `generate`, unpadded and with default attention. This lane decodes left-padded to
  288 slots with eager attention.
- H25 = FP32(Supa + 0.25·P16(δ_H)), while this lane's H = FP32(Supa + TT − Sparse). They are equal only up
  to FP32 rounding of TT.

CONFIRM-002 itself reproduced an independent FP32 implementation on 224/224 rows, so flips should be rare.
A wrong arm or template fails on most rows.

## 6. The decision rule (`parentq_decide.decide`; first match wins)

1. **Padding.** GEO's padding-parity procedure on GEO's two development mechanics prompts. An explicit
   failure gives `STOP_NATIVE_MECHANICS_PADDING`; not reached gives `STOPPED_BEFORE:padding`.
2. **Canary,** per arm in schedule order (BASE, H, T). Incomplete gives `STOPPED_BEFORE:canary:<arm>`;
   failed gives `STOP_ARM_CANARY:<arm>`.
3. **Coverage.** A missing (BASE, T) unit on the T rows, or a missing (BASE, H) unit on H_ON, gives
   `INCOMPLETE_COVERAGE`.
4. **Q1, from the posterior predictive probability P:**
   - P ≥ 0.70: `PARENTS_QUALIFIED`;
   - 0.40 ≤ P < 0.70: `PARENTS_MARGINAL`;
   - P < 0.40: `PARENTS_NOT_QUALIFIED:<T|H>`, naming the parent with the lower component probability.

**How P is computed.**
- Each subtype's paired cell probabilities (BASE credited, parent credited) get an independent Jeffreys
  Dirichlet(½,½,½,½) posterior from the qualification rows.
- P = P_T × P_H, where each factor is the exact posterior predictive that the frozen gate passes on the
  screen's 4-row composition. The two answerable H rows share their subtype's posterior.
- The gate's truth tables come from calling GEO's frozen `parent_gate` on GEO's frozen screen rows with
  synthetic credited values. This rule therefore cannot drift from the screen's own gate. A CPU test
  also checks the tables against 4,000 directly simulated screens scored by `parent_gate`.
- Also reported:
  - the plug-in probability;
  - the 10/50/90% posterior quantiles of P(π) (20,000 draws, seed 20261004);
  - bank-level exact McNemar tests: T vs BASE on T rows, and H vs BASE on H_ON.

**Q2 label** (descriptive). It is reported whenever BASE and H passed their canaries and covered both H
blocks. All tests are on missing-value rows, with exact two-sided McNemar, p ≤ 0.05 and gains > losses:
- `H_ABSTAINS_UNDER_CONTRACT`: H credited abstentions exceed BASE's under the screen contract;
- else `NATIVE_PROMPT_ELICITS_H_ABSTENTION`: H under A/B exceeds H under the contract, and H exceeds BASE
  under A/B;
- else `NATIVE_PROMPT_ELICITS_ABSTENTION_NOT_SPECIFIC_TO_H`: H under A/B exceeds H under the contract, but
  BASE also abstains under A/B;
- else `H_ABSTENTION_NOT_SHOWN`.

Over-abstention on answerable rows is reported for each arm and contract. So is the counterfactual H gate
probability under A/B. With 20 missing rows, a difference smaller than 6 one-directional discordant rows
cannot reach p ≤ 0.05.

**Credited scoring is GEO's, unchanged:** `benchmark.score_response`, with the published boundary rule
plus GEO's registered extensions and integer veto. A lenient answer rule for A/B rows is reported but is
never used in a decision: a non-abstaining first line whose only integer is the gold.

**What the rule says on constructed data** (from the CPU tests; illustrations, not predictions):

| scenario | P | status |
|---|---:|---|
| T right on every row, BASE on none; H answers and abstains correctly, BASE never abstains | ≥ 0.70 | QUALIFIED |
| T right on 2/3, H abstains on about 2/3 of missing rows | 0.668 | MARGINAL |
| T right on 1/2, H abstains on 1/2 | 0.366 | NOT_QUALIFIED:H |
| H never abstains | < 0.40 | NOT_QUALIFIED:H |
| H always abstains, BASE answers correctly | < 0.40 | NOT_QUALIFIED:H |
| T no better than BASE | < 0.40 | NOT_QUALIFIED:T |

The screen's 4-row gate is intrinsically noisy, so QUALIFIED needs strong parents on both sides.

## 7. Compute budget and retry rule

| quantity | allocated / cap |
|---|---:|
| model loads per attempt | 1 |
| generations | 200 (canary 24, T block 48, H_ON 64, H_OFF 64) |
| forwards (upper bound, every generation at 32 slots) | 6,416 |
| token positions incl. padding (upper bound) | 64,964 |
| worker wall | 540 s |
| supervisor hard wall | 600 s |
| loaded attempts, ever | 3 |

**Throughput.** CONFIRM-002-era FP32 decoding on this RTX 4070 Laptop GPU measured about 0.05 s per decode
step. That puts generation at 160–320 s, depending on reply lengths, plus two weight loads. This is an
estimate from records, not a measurement of this lane.

**Order.** One model load (BASE), then H and then T in-memory weight loads. Within each arm the canary
runs first, then its blocks. Weight loads therefore go BASE → H → T: if the wall stops the run, T, whose
subtype rates have historical priors, is what is lost.

**Decoding.** Each generation records its tokens and margins to the attempt's progress file as it
completes. Decoding is GEO's padded greedy decode, stopped at the first EOS. A CPU test shows its tokens
equal GEO's fixed-slot decode for every EOS position.

**Output policy.**
- An attempt that never loads the model writes `blocked_result_parentq.json` in its attempt directory.
- A loaded attempt with a terminal verdict (`PARENTS_*`, `STOP_ARM_CANARY:*`, padding failure) writes the
  create-only `native_parentq.json`. The supervisor then refuses all further attempts.
- A loaded attempt that stops short (resources, wall, supervisor loss) writes
  `incomplete_result_parentq.json` in its attempt directory. A later attempt, with its own grant, may
  rerun from scratch.
- Greedy FP32 decoding is deterministic, so a rerun cannot select a different outcome. The overlap
  between attempts is compared in the report.

**Supervision.** Supervisor and launcher are GEO's reviewed ones with this lane's names. `derive_from_geo.py`
lists every replacement, and a CPU test requires byte equality. They give:
- lease label `triskill-geo001-parentq`;
- never reclaiming a foreign lock;
- ShowWindow=0 WMI launch;
- killing only our own worker tree;
- releasing our lease only once the worker is dead.

## 8. What each outcome means for GEO

These are operator procedure; this lane changes nothing in GEO.

- **`PARENTS_QUALIFIED`**: the operator may grant the frozen GEO screen unchanged.
- **`PARENTS_MARGINAL`**: the operator's call, with P and its band.
- **`PARENTS_NOT_QUALIFIED:T`**: do not spend the screen. The T subtypes need a new GEO amendment.
- **`PARENTS_NOT_QUALIFIED:H`**: do not spend the screen. Then:
  - with `NATIVE_PROMPT_ELICITS_H_ABSTENTION`: a GEO amendment could move the H rows (and the chains'
    abstention contract) to H's training prompts, re-registered and re-frozen before any screen arm;
  - with `…NOT_SPECIFIC_TO_H` or `H_ABSTENTION_NOT_SHOWN`: H's skill does not transport to this template.
    The third-skill question then needs a different H instrument, or the δ_C question (TRISKILL-002,
    unowned) instead.
- **Constraints on any such redesign.** It is parent-side only and J-blind (no J output exists anywhere),
  and it is registered before any screen arm. The PARENTQ rows become exposed development data and are
  never reused as screen rows.

## 9. Limits stated before any data

- **Exchangeability.** P treats the screen's 8 parent rows as exchangeable with rows from the same
  constructors. The screen rows were hand-chosen, without model outputs. The T subtypes were chosen from
  historical TT rates (disclosed in AMENDMENT section 5).
- **Judgment calls.** The prior (Jeffreys) and thresholds (0.70 / 0.40) are judgment calls, registered
  here before any data. Six rows per T subtype give wide intervals; the band is reported for that reason.
- **Q2 is descriptive.** Prompts A and B are pooled. Its labels cannot license any claim about H's skill
  beyond these templates.
- **The canary tolerates one mismatch in 8 per arm.** It checks arm identity, not quality.
- **No outcome is confirmation.** This lane does not test the GEO hypothesis.

## 10. Files

| file | role |
|---|---|
| `parentq_rows.py` | bank generator (frozen GEO constructors) |
| `parentq_decide.py` | decision rule (self-contained; loaded from pinned bytes) |
| `parentq_worker.py` | native worker (frozen GEO internals via GEO's binding) |
| `parentq_supervise.py`, `parentq_launch_wmi.py` | derived from GEO's (`derive_from_geo.py`) |
| `geo_frozen.py` | CPU-side loader of GEO modules from their pinned bytes |
| `build_parentq.py` | membership, CPU validation, freeze |
| `parentq_bank.json`, `parentq_canary.json` | frozen membership |
| `test_parentq_*.py` | CPU tests, including end-to-end runs of the real worker on a tiny random Qwen3 |

**Launch sequence, after review, freeze and grant:**

    python parentq_launch_wmi.py --freeze-sha256 <FREEZE_PARENTQ.json sha256> --grant "<operator grant text>" --launch
