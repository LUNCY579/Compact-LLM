# TRISKILL-GEO-001 — Section H amendment (frozen before any native forward)

**Status.** Written 2026-10-01 and revised 2026-10-02 after an independent review of its first
implementation (section 10). No model has ever been loaded or run by this pilot or its successors.

This amendment applies the "minimum changes before any native forward" (section H) of an
independent read-only review of the pilot at `09a0514e`. It builds on the Codex successor
`codex/triskill-geometric-gapfix-20261001` @ `332b7f9b`.
- Branch: `research/triskill-geo001-sectionh-fixes`.
- Research ancestor: `codex/triskill-stage-a` @ `01f47694`.

**Authorization.** The operator instructed, in session, "apply the section H fixes to GEO". That
covers implementation, CPU verification and review. **It does not authorize native execution.**
`supervise.py` refuses to start without an explicit operator grant, which it records verbatim. The
worker refuses to run unless the supervisor's start receipt carries that grant and its lease token.

**Preserved.** These files are unchanged and remain interpretable under their own implementations:
`PREREG.md`, `REPORT.md`, `FREEZE.json`, `FOLLOWUP_PREREG.md`, `REPORT_GAPFIX.md`,
`FREEZE_GAPFIX.json`, `geometry.npz/json`, `screen.json`, `development.json` and every receipt.
This document governs the amended implementation and **supersedes their decision rules** (section 6).

**Unchanged construction choices.**
- Layer 14 `self_attn.o_proj`; rank 4 per parent; η = 0.1 (Cayley coefficient η/2 = 0.05).
- T = Titan-Sparse − Supa(BF16→FP32). H = TT − Titan-Sparse. H-alone = FP32(Supa + TT − Sparse).
- The nine arms: BASE, T, H, TH, IDENTITY, ROTATION, REVERSE (= Rᵀ, from the gapfix successor),
  RANDOM and PROMPT. The PROMPT prefix text is also unchanged.
- No third-skill (J) labels, gradients, J-trained delta or J outcomes are used to construct or select
  anything. No model has ever generated on any J row.

## 1. Run gates on measured resources, not process attribution (item 1)

The original preflight blocked on any `nvidia-smi --query-compute-apps` PID other than its own, and on
any "[Insufficient Permissions]" entry. Under WDDM this desktop always lists `dwm.exe` and
`explorer.exe`, so the pilot could never run.

`worker.preflight` now blocks on:
- host RAM free < 5 GiB, disk < 6 GiB, measured GPU free < 6 GiB, or a failed GPU query;
- any lease that is not ours: the lease must name the worker's supervisor, found among its ancestors
  because the venv `python.exe` is a launcher, plus the worker's token and label;
- a foreign research/model process (python/ollama/llama/vmmem) with ≥ 512 MiB RSS, or one whose memory
  is not visible.

The compute-apps list is recorded as information only. Other safeguards:
- CUDA is capped with `set_per_process_memory_fraction(0.78)`.
- Active reserves are GPU 0.5 GiB and host 1.5 GiB, checked every forward and every tensor.
- The H arm now accumulates in place, at 20 B per element. For the tied embedding that is a 2.9 GiB
  working set, so the 5 GiB preload gate covers it with the active reserve.

## 2. In-lane launch (item 2)

The dependency on another lane's containment verifier is removed. That lane (MiMo exact-prefix /
Windows containment) is not modified.

**`launch_wmi.py`**
- Starts the supervisor through WMI `Win32_Process.Create`, outside the session's process tree.
- Passes **ShowWindow=0**, so there is no console to receive a Ctrl+C.
- After creating the process it waits up to 30 s for the supervisor's start receipt or failure
  receipt, and records which it saw.
- Defaults to a dry run.

**`supervise.py`**
- Checks the freeze digest and the bytes it will execute.
- Loads the programme's `gpu_lease.py` only from its pinned bytes.
- **Refuses if any lock exists.** It never calls a reclaim path on another lane's lock.
- Refuses if a foreign GPU python is present, or if the screen output already exists.
- Writes the worker launcher pid into the lease, retrying on Windows sharing violations.
- Gives the worker an explicit environment.
- Enforces a **600 s hard wall** (the worker's own wall is **540 s**), and kills only its own worker
  tree on overrun.
- Then kills the real CUDA worker pid that the worker recorded, if it is still alive.
- Releases its own lease, token-checked and retried, **only once both are dead**. Otherwise it keeps
  the lease and says so.
- Every refusal or crash leaves a `SUPERVISOR_FAILURE_*.json`.

**Worker side.** At every forward the worker checks that the lease-holding supervisor is still alive,
and stops with `STOP_SUPERVISOR_LOST` if not. So an orphaned worker ends itself.

**Attempts and the retry rule.**
- Each launch writes into its own `native_attempts/<utc>-<pid>/`: start receipt, worker pid, log,
  per-generation progress JSONL, and receipt.
- The create-only screen output `native_screen_sectionh.json` is written **only once the model has
  loaded**.
- An attempt that stops earlier records `blocked_result_sectionh.json` in its attempt directory and
  leaves the screen unspent.
- Once any attempt has loaded the model, the screen is spent and the supervisor refuses further
  attempts.

**Launch sequence** (the grant is required):

    python launch_wmi.py --freeze-sha256 <FREEZE_SECTIONH.json sha256> --grant "<operator grant text>" --launch

## 3. Exact geometry (item 3)

`left_subspace` is now an exact float64 SVD. The original randomized finder missed the exact top-4
subspaces by a 4th principal angle of 22.9° (T) and 22.6° (H), because the spectral gaps are
σ₄/σ₅ = 1.109 and 1.098.
- Its operator is 39% (relative Frobenius) from the exact one.
- Five other seeds of the same finder give operators 40–55% from the frozen one.

**RANDOM frame.**
- It is a sign-corrected Haar QR, seed 20261003.
- It spans exactly the legacy frame's subspace; three of its eight column signs flip.
- It is still one draw. Its rank, spectrum and compute match; its realized displacement does not
  (section 9).

`geometry_sectionh.json` records:
- the singular spectra and gaps;
- a rank-1..8 sensitivity table (descriptive; rank 4 is not re-selected);
- the Haar null for ‖K‖₂ (15.6% of random pairs reach the observed 0.1169);
- the legacy operator's distance from the exact one (0.393).

## 4. Making a null informative (item 4)

**Hook liveness** runs before any TH-hosted unit, on the two DEVELOPMENT mechanics prompts.
- It takes each arm's (basis, change) **from the same construction the screen uses**. The analytic side
  uses only the registered arrays.
- IDENTITY must displace exactly 0.
- For ROTATION, REVERSE and RANDOM, the displaced vectors must match h·B(R−I)ᵀBᵀ element-wise
  (relative error ≤ 1e-3), in float64, and their energy must match and be positive.
- An energy-only check cannot see direction errors. R vs Rᵀ, or a column-reversed union basis, displace
  identical energy; CPU tests show both are caught by the vector check.
- ROTATION must change the logits.
- A hook that never fires fails the check.

**Teacher-forced sensitivity readout**, with no extra generations. TH's greedy tokens are re-scored under
TH, ROTATION, REVERSE and RANDOM. Each row records:
- argmax agreement;
- max |Δlogit|;
- mean and max KL(TH‖arm);
- TH's minimum top-2 margin;
- the target-margin change.

**Every generation** records per-step top-1 margins and its scored token ids, as soon as it completes.

**`UNDETECTABLE_ON_J`** is registered as distinct from rejection: ROTATION's text equals TH's on every
dependent row.

## 5. Rebuilt rows and contract (items 5 and 7)

There are 18 screen rows (`screen_sectionh.json`, built by `benchmark.build_screen()`) and 8
development rows.

**T: 4 rows (2 ledger, 2 register)**
- Each row sits inside T's trained and measured support: operand ranges, every event key including
  `tick`, and **no modulo-97 wrap** (T_BANK_V3X has no wrapping add; TT scores 0/3 on wrapping
  conditionals, and every arm 0/5 on V4 wrapping adds).
- No row matches any of the 1,408 ledger/register payloads in T_BANK_V3X, T_BANK_V4 and δ_T's training
  rows, by full payload, operative event, or (before, answer). A known historical payload collides on
  all three keys (canary).
- Subtypes were chosen from exposed historical T_BANK_V3X accuracy. **This is development information,
  declared here.** The goal was a gate that can both pass and fail:

| subtype | TT historical rate |
|---|---:|
| posted positive transfer | 52% |
| posted scale | 64% |
| non-wrapping register add | 95% |
| register conditional | 45% |

  The original negative-transfer subtype scored 4/60.
- **T credited endpoint:** typed JSON equality after stripping whitespace and at most one trailing
  period. This is *more lenient* than the programme's exact-string T rule: it accepts reordered keys and
  internal spaces. Strict is reported separately.

**H: 4 rows**
- 2 answerable; 1 near-miss distractor (only the crate's code is given); 1 with the explicit cue H was
  trained on ("The seal's code was not recorded.").
- The "Complete notes" opener, a completeness counter-cue, is removed.
- The system prompt is the HSKILL answerable contract, verbatim (a CPU test checks it against
  HSKILL_VIEWS_V2): "…If one answer is not supported, reply exactly: "I don't know based on the
  available information."".
- **Disclosure:** in H's measurement banks this prompt only ever accompanied answerable rows. H's
  abstention under it is unmeasured, and so are its answers on rows like these.
- The original literal-`UNKNOWN` contract is retired. It was BASE's own trained format (2,567/33,761 Supa
  SFT replies), and H's native abstention scored zero under it.
- Literal UNKNOWN is still *credited* as an abstention, on every arm alike. That raises BASE's H ceiling,
  which is why the parent gate has a separate ceiling stop.

**Credited abstention**
- The published `boundary_rule.abstains` patterns, copied verbatim (sha and pattern equality are checked
  by a CPU test), applied to the first line.
- Registered extensions: "insufficient", "cannot be determined / cannot determine", "undetermined /
  indeterminate", "unknown", "values differ", "multiple values", and typographic apostrophes normalized.
- On abstention-gold rows, **a first line that contains an integer is never credited**: "No, the value
  is 5" and "5?" are answers.
- Answers are credited by canonical exact match (trailing .!? stripped), with no abstention.
- Strict = the exact reply.
- This first-line rule strips leading whitespace, unlike the published `first_line`.

**Dependent chains J: 8 rows**
- Rules are affine maps a·x + b, or T's *conditional* update applied to the chain register:
  x + v if x > t, otherwise x − v (mod 97).
- **4 answerable:**
  - a visible 0·x reset;
  - a disguised 291·x reset;
  - two conflicting records that a conditional rule maps to the same value (collisions).
- **4 abstain:**
  - decoys 195 ≡ 1 and 96 ≡ −1;
  - two conditionals whose candidates do not collide.
- Starts are balanced: 4 missing, 4 conflicting. Conditionals appear on both sides (2/2), so neither
  "has a conditional" nor "has a zero multiplier" decides answerability. Resolving the conflict rows
  requires tracking both candidate states through T-style conditional transitions.
- No gold is copyable: none equals the marker 23 or any rule offset.

**Known-state controls K: 2 rows**, one of them through a conditional.

**Shortcut floors** (credited, J /8), recorded by `screen_floors`:

| policy | J /8 |
|---|---:|
| always abstain | 4 |
| visible-zero / literal 0-or-97 / magnitude ≥97 reset scans, each with correct later arithmetic | 5 |
| affine-reset oracle (recognizes every multiple of 97, ignores collisions) | 6 |
| abstain on missing, solve conflicts | 6 |
| first record only | 4 |
| copy last reset offset | 4 |
| copy marker | 0 |
| best constant integer | 1 |
| oracle | 8 |

**Decoding**
- Text ends at either EOS id, 151645 or 151643, both from `generation_config.json` and checked at load.
- Slots: 288 input (the longest prompt, with the PROMPT prefix, is 281 tokens); 32 output (the longest
  acceptable single-line reply, spaced ledger JSON, is 27); 319 for the readout.
- A pretty-printed multi-line ledger JSON (34 tokens) would be truncated.

**Parent gate (credited)**
- T: at least 2/4 credited and more than BASE.
- H: more credited H rows than BASE, and at least 1 credited abstention on a missing-value row.
- When BASE already scores every row of a group, the stop is `STOP_SCREEN_PARENT_CEILING:<parent>`,
  not "no parent signal".
- Parent damage is reported, not gated.

## 6. The single decision rule (item 6)

This rule supersedes:
- PREREG lines 106–114;
- BENCHMARK_MATRIX lines 121–131;
- REPORT lines 110–111;
- FOLLOWUP_PREREG's continuation rule.

It is coded as `benchmark.decide_screen` and applied by the worker. A mechanics value that was never
measured is **never** a failure: it gives `STOPPED_BEFORE:<stage>`, with the stopping cause in
`stopped_by`. In execution order, first match wins:

1. Padding: an explicit failure gives `STOP_NATIVE_MECHANICS_PADDING`.
2. The parent gate gives `STOP_SCREEN_NO_PARENT_SIGNAL:T|H` or `STOP_SCREEN_PARENT_CEILING:T|H`.
3. Liveness: an explicit failure gives `STOP_MECHANICS_HOOK_LIVENESS`.
4. `STOP_IDENTITY_CANARY`: IDENTITY text differs from TH on any row both have, even when later arms
   never ran.
5. `INCOMPLETE_COVERAGE`: an arm is missing a row it must cover. Every arm covers all 18 rows except
   PROMPT, which covers the 10 chain rows.
6. `UNDETECTABLE_ON_J`: ROTATION text equals TH on every dependent row (**not a rejection**). Movement
   elsewhere is reported.
7. `SCREEN_SIGNAL_EXPLORATORY` if all of S1–S6 hold, otherwise `NO_SIGNAL`.

A **signal gain** is a J row ROTATION is credited on and TH is not, excluding two kinds of row:
- **format repairs:** TH was semantically right;
- **parent repairs:** T-alone or H-alone was already credited there (interference repair, as
  BENCHMARK_MATRIX defined it).

The conditions:
- **S1:** ROTATION's credited J score is above every comparator (BASE, T, H, TH, IDENTITY, RANDOM,
  PROMPT, REVERSE). Beating REVERSE makes the signal sign-specific.
- **S2:** at least 1 signal gain, and more signal gains than credited J losses vs TH.
- **S3:** no credited loss vs TH on any answerable J row.
- **S4:** no credited loss vs TH on any T, H or K row.
- **S5:** at least 1 signal gain on an answerable J row, i.e. a computed value. Abstention or wording
  changes alone cannot pass.
- **S6:** ROTATION's J score is above the always-abstain floor (4/8).

**Disclosed changes to superseded clauses.** FOLLOWUP_PREREG required "both strict and semantic gain".
That is replaced by the credited endpoint, with format-only and parent repairs excluded from the gains.
Strict is still reported. The exact McNemar p (ROTATION vs TH on J) and the floors are reported with
every verdict. **No outcome is confirmation.**

**NO_SIGNAL is not evidence of absence.** For the first implementation's S1–S4 rule, the review's
model gave power of 0.2–0.6 for a moderate effect, and at most about 0.9 for a perfect one, because S4
is a one-loss veto. Under small null perturbations it gave a false-signal rate of up to about 10%.
These are model-based reviewer estimates, not recomputed for S1–S6. The added S5–S6 can only lower
both rates.

## 7. Compute budget (frozen)

| quantity | limit | allocated |
|---|---:|---:|
| model loads | 1 | 1 |
| generations | 154 | 154 |
| forwards | 5,100 | 5,024 |
| token positions incl. padding | 76,000 | 75,562 |
| worker wall (s) | 540 | — |
| supervisor hard wall (s) | 600 | — |

The 5,024 forwards break down as:
- padding parity: 16;
- liveness: 8;
- generation: 154 × 32;
- readout: 72.

The review estimated 245–345 s of native wall time from recorded throughput (a reviewer estimate,
not measured). The historical 48-token scores are not comparable.

## 8. Benchmark matrix status (erratum)

**Relabelled "diagnostic rows exist, unqualified"** (not "Available"): instruction following, math,
answerability, state tracking, unfamiliar rules, contradiction detection and skill composition.

**Still unmeasured:** roleplaying, language, long context, error correction and broad evidence
integration.

Per-family retention needs at least 35 source-correct rows with zero damage before a Wilson 95% damage
bound can fall below 10%. No family claim follows from this screen.

## 9. Limits stated before any data

- **The prior for a geometry-only lever is low.**
  - Orthogonality was not the lever, and a norm-matched random update was inert.
  - ‖R−I‖₂ = 0.0117, and the T–H overlap sits at the Haar null.
- **A parent-gate stop is a likely outcome.**
  - For the first implementation's rows, the review estimated about 0.08–0.38 probability of reaching
    construction: the T gate about 0.53–0.86, the H gate about 0.15–0.44. These are model-based reviewer
    estimates.
  - The non-wrapping T3 and the cued H row should raise them. They were not recomputed, and H's
    behaviour under this contract is unmeasured.
  - A parent-gate stop says nothing about the hypothesis.
- **Exclusion against protected banks rests on construction, not on opening them.**
  - That construction is: new surface forms, the legacy pool cue-word scan, and T-row operand collision
    checks against exposed banks.
  - The T rows now sit *inside* the support T_BANK_V5 was sampled from, so the earlier out-of-support
    argument no longer applies. An exact collision with its sealed rows remains astronomically unlikely
    but is unverified.
  - C_FRESH, T_BANK_V5, H_FRESH and HSKILL_CONFIRM_V2 remain unopened, and TRISKILL Stage B stays sealed.
- **Controls C2–C4 are unchanged.**
  - RANDOM is one frame, and its realized displacement is not matched. REVERSE is exactly
    displacement-matched to ROTATION.
  - There is no wrong-partner control.
  - PROMPT is TH plus a hint.
  - IDENTITY is a determinism canary.
- **If the supervisor process itself dies,** the worker stops itself (`STOP_SUPERVISOR_LOST`). The lock
  stays held with `worker_pids` set until the start receipt's token is used for a manual own-lease
  release.
- **Scale.** One draw each of the parents, the random frame and the screen rows. A screen signal would
  justify a separately registered, powered study, never a capability claim.

## 10. Review of the first implementation (2026-10-02) and dispositions

An independent five-lens review (native path, verdict, supervisor safety, registration integrity,
scope) plus a completeness critic examined `5a82e8d7`. The lead re-verified the findings first-hand
before fixing them. Fixed here:
- an abstention-only or hedged-wording SCREEN_SIGNAL (S5, S6, integer rule);
- format-only and parent-repair gains counted as signal;
- T3 needing a mod-97 wrap;
- surface-cue floors reaching 7/8 (collision rows; floors now max 6/8);
- not-reached stages reported as mechanics failures;
- the identity canary hidden by the coverage rung;
- UNDETECTABLE defined over all rows;
- an H-arm RAM reserve above the preload gate;
- lost in-flight generations, and no per-unit checkpoint;
- a zero-forward block spending the screen;
- liveness bypassing the screen's arm construction;
- several surviving mutants (now pinned by tests);
- an unpinned CPU validation receipt (now a freeze build gate);
- the WMI launch without ShowWindow=0;
- reclaiming another lane's stale lock;
- single-shot lock writes;
- unrecorded supervisor refusals;
- an unsupervised worker after supervisor death;
- tracking only the launcher pid;
- several wording errors.

Recorded as limitations (section 9): power, the parent-gate stop probability, and controls C2–C4.
