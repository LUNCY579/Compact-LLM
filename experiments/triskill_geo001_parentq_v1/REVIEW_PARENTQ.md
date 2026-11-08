# PARENTQ independent review: consolidated findings (draft `3e07fd87`)

**Review structure.** The review ran from 2026-10-02 to 2026-10-05 and had three stages:
1. **Five lenses.** Native worker path, decision rule and statistics, supervisor and launch safety, data and
   registration integrity, and premise and scope.
2. **Adversarial verification.** Every finding then went to verifiers prompted to refute it: one verifier for
   MINOR and NOTE findings, two (reproduction and impact) for MAJOR and BLOCKER findings.
3. **A completeness critic.**

**How it was carried out.** Session limits interrupted the run three times; it was resumed from cache each
time. All work was read-only and CPU-only, with no GPU and no lock touched. The lead re-checked the key
mechanical claims first-hand (marked **lead**).

**Tally.**
- **47 findings verified:** 45 CONFIRMED, 1 PLAUSIBLE (CUDA deterministic mode: cannot be shown without a
  GPU), and 1 REFUTED (canary false-fail as a MINOR; its duplicate L1#7 survives as a NOTE).
- **Also included:** 9 findings from the critic, and 6 first-pass supervisor findings that the resumed
  passes did not re-list.

**Verdict.**
- **No defect in the native path.** The real `parentq_worker.py run` bootstrap was executed on CPU through
  GEO's full binding and freeze validation, up to preflight (1.8 s).
- **Design and supervision need revision before any freeze.**

## A. Design: decisions for the operator

1. **The decision quantity is miscalibrated, and the gate it predicts is noisy.** (L5#1, L2#4, L5#7:
   CONFIRMED; lead: perfect data gives P ≤ 0.967 × 0.904 = 0.875.)
   - **The prior deflates P.** The 4-cell Dirichlet(½) prior gives each arm a uniform marginal.
   - **MARGINAL is the modal verdict for good parents.** At a true pass probability of 0.82, the rule returns
     QUALIFIED about 30% of the time and MARGINAL about 70%.
   - **The gate passes parents that have no skill.** GEO's 4-row gate passes a null H 12–35% of the time.
2. **The exact quantity can be measured directly.** (L5#5: both verifiers say the fix is sound; L2#9.)
   - **What would be measured:** GEO's parent gate is a deterministic function of BASE, T and H greedy
     outputs on GEO's own 8 parent rows. That costs 16–24 generations and exposes no J, K, TH or ROTATION
     output.
   - **What the verifiers checked:** they reproduced the gate bit-for-bit on CPU across processes, decoders,
     arm orders and interleavings.
   - **The tension it creates** (critic C#4): PREREG section 8 makes measured rows "exposed". Adopting this
     needs an explicit ruling that rows measured early stay valid screen rows (deterministic outputs, no row
     selection, no row swap).
3. **Section 8 leaves outcome-driven redesign open.** (L5#2, L5#3, L5#4, critic C#2: CONFIRMED.)
   - **The A/B branch** can fire when H abstains on everything. It also proposes changing the chains'
     contract, which also carries J's semantics.
   - **A BASE ceiling** reads as "H does not transport".
   - **The T branch** lets gate subtypes be re-selected from measured results.
   - **No route** exists for a canary failure, a padding failure, or exhausted attempts.
   - **J's dependence on T is unstated.** 2 J rows (sorrel, tansy) and K yarrow need T's conditional, and
     this is never linked to T's conditional cell.
4. **The T bank does not represent the screen rows it predicts.** (L2#3, L4-data#1: CONFIRMED. Lead: all 6
   conditional rows have a positive value, while the screen's conditional is −6 on the true branch.)
   - **History shows the classes differ.** Measured rates are 80% vs 48–58% for the transfer sign classes, and
     67% vs 47–53% for the conditional classes.
   - **One bank row copies a screen row's arithmetic.** T14 computes 19 × 3 = 57, which is screen T2's
     arithmetic (lead; L4#5; C#4).
5. **The Q2 labels have holes.** (L2#1, L2#2, L2#8, critic C#7: CONFIRMED.)
   - **They ignore answerable rows**, so blanket abstention earns NATIVE_PROMPT_ELICITS_H_ABSTENTION.
   - **NOT_SPECIFIC can come from a null**, and there is no BASE-ceiling label.
   - **Natural abstentions are not credited** ("not recorded", "not provided").
   - **There is no lenient or semantic answer count on H_ON**, where answer-format import from the delta
     could fail H.

## B. Mechanical fixes (no operator decision needed)

**Supervision and ownership**
- **The attempt cap leaks.** (L1#1, L3#1, L4#2.) A loaded attempt killed at the hard wall, or crashing
  natively, is never counted. Fix: a create-only LOAD_BEGUN marker before `from_pretrained`; count every
  directory that has one.
- **Foreign lease schemes are invisible.** (L3#2; lead verified the files.) `<git-common>/rineya-central-gpu0.lock`
  (MiMo, EFE, RVS) and `ChatBubble-agent-prompts-20261002/hardware-worker.lock` (an OS exclusive handle;
  HARDWARE_POLICY) are never checked. The CRS SQLite lease isn't either. Fix:
  - refuse while either lock is present or held;
  - dual-lease the central lock (the EFE convention: create with `x`, write a JSON token, remove only our
    token);
  - re-check at every unit.
- **A reclaim race.** (First-pass L3#3; lead read `gpu_lease.acquire`.) A foreign lock created between
  `exists()` and `open('x')` falls into `_try_reclaim`. That deletes a dead-pid lock whose record has no
  `created` field, which is what TLF writes. Fix: set `_try_reclaim = lambda: False` on the pinned module.
- **Kill and cleanup paths:**
  - PID-identity check (create time) before `taskkill`;
  - a bounded wait after the kill;
  - kill our own child if the supervisor fails after `popen`;
  - wrap `taskkill` timeouts so the receipt is always written;
  - parse arguments inside `try`;
  - microsecond launch-receipt stamps;
  - re-check lease ownership during the run (L3 #3–#6, first-pass #4–#8 and #12).
- **Grant replay.** (Critic C#3.) The previous START receipt stores the grant verbatim. Fix: the grant must
  name the attempt number and freeze sha, and a string already used is refused.

**Worker**
- **Blocked records.** Pre-`try` refusals write no `blocked_result` (L1#2, L3 first-pass #9). Wrap the whole
  pre-load section.
- **The RAM gate is too low.** (L1#3.) The 5 GiB host-RAM gate is below what the H load needs once the model
  is resident (2.90 GiB working set + 1.5 GiB reserve + CUDA/model host overhead). Fix: a measured
  lane-specific preload gate of about 6.5 GiB, so a doomed attempt is BLOCKED rather than spent.
- **Progress events** get timestamps (L1#6).
- **`bind_geo`** must hash and parse the same bytes (L1#8).
- **Canary:**
  - record first-divergence margins on any miss (L1#7);
  - reconsider the 7/8 threshold, since no wrong stored source reaches 3/8 (C#6);
  - register a route for every outcome.
- **Attempt overlap:** register the comparison between attempts and a NONDETERMINISM_OBSERVED status (C#8).

**Validation and registration**
- **Gate counts that are now only recorded:** 10 contract rows, 56 = A + B exact abstentions, and 56 eligible
  canary rows (L4#3, L4#4).
- **Pin both T-bank hashes,** and require the gpu_lease pin to equal GEO's (L4#6, L4#12).
- **Add an operative-number collision key** (touched inputs plus answer) against GEO's screen and development
  rows and the exposed banks (L4#5, C#5).
- **The worker must require `receipt['passed']`** (L4-data first-pass #2).
- **Freeze builder:** run the CPU tests, or move test-only facts into `validate` (L4#10).
- **Untested rungs:** add tests for all of them. Fix the test that never checks the H-cell invariance. `decide`
  must never raise inside `finally` (L2#10).
- **PREREG wording:**
  - 58 rows are pairwise different and 56 remain after the length filter;
  - δ_H's gold replies are read, not only its prompts;
  - correct the illustrative table, and test every value in it;
  - name the EOS positions the decode test actually checks;
  - 224/224 holds for BASE and SPARSE only;
  - measured throughput is 0.039–0.043 s per step;
  - restore GEO's qualifier on the 0.08–0.38 estimate (C#9);
  - add GEO's T_BANK_V5 caveat (C#5);
  - note that δ_T trained ledger scale factors only {±1, ±2} (L4#6);
  - remove the literal `protected_banks_opened` (L4-data first-pass #2c).

## C. Out of lane: GEO itself (frozen; needs a GEO amendment and a re-freeze)

- **The one-shot rule can be broken.** (Critic C#1: VERIFIED.) GEO's supervisor refuses only if
  `native_screen_sectionh.json` exists, and the worker writes that file only in `finally`. A loaded GEO attempt
  killed at the hard wall leaves the screen "unspent", with its J texts already in the progress file. A second
  loaded attempt could then be chosen after seeing J outputs.
- **GEO's supervisor shares B's gaps:** foreign lease schemes, the reclaim race, PID identity, and the cleanup
  paths.
- **Two of GEO's chain worlds** ("quartz", "willow") occur as framing words in δ_H's A/B training prompts
  (L4-data first-pass #8).
- **Order of work:** a GEO fix changes FREEZE_SECTIONH.json, which every PARENTQ pin binds. Any GEO supervisor
  amendment must therefore come first, with PARENTQ re-pinned afterwards.

## D. Disclosures

- **A reviewer's search touched the protected directory.** One recursive grep run by the first-pass data
  reviewer traversed `experiments/titan_third_skill_v1/`. It printed two non-bank file names
  (PHASE0_SCORES.json, STORAGE_DELETE_20260926.json) and no bank content. Later searches excluded the
  directory, and HSKILL_CONFIRM_V2 was never opened.
- **The branch was pushed without this lane's involvement.** This branch (with `3e07fd87`, and `9b44e1ed`
  beneath it) is on `origin` (github.com/LUNCY579/ChatBubble). It went up in a bulk push of dozens of local
  branches at 2026-10-02 23:40:17 −0500, followed by `codex/worktree-save-20261002/*` branches at 23:50. This
  lane never ran `git push`.
- **The TT source core is untracked in the primary checkout** (it is backed up in the private HF vault, per
  the critic). Back it up before any grant.
