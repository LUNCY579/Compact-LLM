# COMP-RS-001 Amendment 1 — operator scope decisions and pre-registration repairs

**Status: amendment to the working protocol (`COMP_RS_001_PROTOCOL_DRAFT.md`). Not a training grant and not
a frozen statistical registration.** No model was loaded, trained or scored for this amendment. No TEST,
CHALLENGE, guard_test or guard_challenge file was opened. Every number below comes from specs, code or committed
records. Base: `codex/comp-rs-native-q1@8f6afcf5`. Where this amendment and the draft disagree, this amendment
governs.

**Revision 2.** Revision 1 was reviewed adversarially before being shown to anyone. That review found:
- H2 was degenerate: gains are zero by construction on the solo-correct denominator.
- Every margin was chosen after an exchangeable DEV preview.
- MT's matched coefficient was chosen on DEV.
- F_draw was ~0 on RS.
- The plateau rule depended on churn rather than accuracy.
- Tier-2 rules could be set after Tier-1 TEST outcomes.
- Endpoints lacked the `tag=='core'` filter.
- The shipped scorer refuses CALIBRATION.

Each finding was re-checked by hand before it was acted on (§14). This revision replaces revision 1 entirely.

## 0. Operator decisions (2026-09-26)

Verbatim: *"One objective first, trained P donor, include multi-task, calibrate on CALIBRATION."*

| id | decision | consequence |
|---|---|---|
| D1 | One objective first | Every trained arm uses objective **O1**, and GRPO-vs-MaxRL becomes **COMP-RS-001b** (§12). **O1 = MaxRL** (confirmed by the operator, 2026-09-26). |
| D2 | Trained P donor | A precedence donor, trained exactly like R and S, is the size-matched trained non-partner for the composition nulls (§5). |
| D3 | Include multi-task | An interleaved R∪S arm **MT** is a primary H4 comparator (§6). |
| D4 | Calibrate on CALIBRATION | Coefficients are chosen only on CALIBRATION. DEV never chooses a coefficient, checkpoint or recipe (§1). |

**Signed off by the operator on 2026-09-26, verbatim: "Add S′, keep the defaults, commit it."**
- Replicate **S′** is added (+1 training unit), so H4 on RS accuracy and atomic S can be confirmatory (§3).
- O1 = MaxRL, m_R = m_S = 10% and m_L = 5% are confirmed. The margins may still change before S1, never after.

## 1. Bank roles, scope and read discipline

**Endpoint scope.** Every endpoint uses rows with `tag == 'core'` of the named signature. Boundary
(answerability-twin) rows are never read.
- DEV has 50 boundary S and 50 boundary RS rows, on 25 parents with two rows each.
- CALIBRATION has 68 boundary S rows.

Under the core filter, DEV/TEST/CALIBRATION carry one row per parent per signature, which is what makes the
row-level exact tests of §9 valid.

**Scope receipts.** Before any donor read, freeze scope receipts for CALIBRATION and TEST in the form of
`evidence/Q1_DEV_SCOPE.json` (per-signature `ids_sha256` and `parents_sha256`, core filter). Every scorer asserts
the receipt digest.

**What counts as a read.** A **read** is generating or scoring any model output on a row.
- Computations on specs are not reads: floors, shortcut rates, content keys, triple structure.
- A teacher-forced forward pass that is neither decoded nor scored (§5 engagement) is not a read.

| bank (core only) | contents (verified) | permitted | forbidden |
|---|---|---|---|
| TRAIN `atomic_R/S/P` | 3,000 rows each (= the manifests) | donor and MT training | evaluation |
| CALIBRATION | R, S, P: 150 rows each, **same 150 parents**, 50 per level; no RS | source competence (sampled groups, R, S and P); coefficient reads for R, S, MT (§4); churn floor (§8) | — |
| DEV | 250 parents; R, S, P, RS: 250 rows each | acquisition gates (§3 H1) for every trained arm, as trained; one end-to-end dry run per tier (§9), exposed and read by no rule; §5 engagement pass (not a read); §10 diagnostics after all TEST reads | choosing any coefficient, checkpoint or recipe; retrain-and-retry after a failed gate |
| guard_dev | 1,200 rows on 1,200 parents | B enrollment; legacy losses on a hash-ordered prefix of ≤ 300 B-correct rows at each dose point (§4) | confirmatory retention |
| TEST | 250 parents per signature; guard_test 1,600 | one read per arm, after that arm is frozen and the whole analysis plan is frozen (§9) | pooling with DEV |

**Why coefficient choice left DEV [V].** In 62/250 DEV worlds the atomic-S sibling has the same (k, target) as the
RS row. In 33/250 the S sibling's first oracle answer passes the RS row. Choosing α on DEV atomic S would read DEV
RS.

**No retry on DEV.** A failed DEV gate removes that arm and is never retried against DEV.

## 2. The arm panel

**Storage.** Donors are stored only as FP32 deltas over the trainable inventory:
δ_X = θ_X(FP32 master) − upcast(θ0).

**Arm construction.** Arms are built by streaming:
- each δ tensor is memory-mapped and added in place into the GPU-resident FP32 upcast(θ0);
- the order is fixed (R, S, P), with α applied in FP32, and the result is never cast to BF16;
- an arm's identity is the sha256 of its per-tensor data digests in sorted key order;
- a rebuild must reproduce the digest, or STOP.

**Tier 1 (from the R, S and P donors; no extra training)**

| arm | weights | role |
|---|---|---|
| B | θ0 | source |
| R, S | θ0 + α_R δ_R ; θ0 + α_S δ_S | solos at the composed coefficients |
| RS | θ0 + α_R δ_R + α_S δ_S | composition under test; the sum is never rescaled |
| R_a100, S_a100 | θ0 + δ_R ; θ0 + δ_S | trained strength; enrollment sources for H4; "full donor effect" wording |
| RS_a100 | θ0 + δ_R + δ_S | untuned composition at trained strength; the primary H4 comparator |
| R_N, S_N | θ0 + c_X δ_X, with ‖c_X δ_X‖ = ‖α_R δ_R + α_S δ_S‖ | magnitude twins: legacy rows, plus S_N in H3c's secondary. Never on atomic rows of their own donor's skill. Built only if c_X ≤ 1; never extrapolated |
| P_R | θ0 + p_R δ_P, with ‖p_R δ_P‖ = ‖α_R δ_R‖ | size-matched stand-in for R |
| SP | θ0 + α_S δ_S + p_R δ_P | S with a trained non-partner of R's size |
| RP | θ0 + α_R δ_R + p_S δ_P, with ‖p_S δ_P‖ = ‖α_S δ_S‖ | R with a trained non-partner of S's size |

**Tier 2 (launched on DEV gates alone; §12)**

| arm | construction | role |
|---|---|---|
| MT | θ0 trained on stratified `atomic_R ∪ atomic_S`, 2 units (§6) | joint-training comparator, scored at α = 1 |
| R→S, S→R | second skill trained from θ0 + δ_first, 1 unit each | sequential comparators, as trained |
| R′ | replicate of R (registered different seed), 1 unit | floor for atomic-R endpoints |
| S′ | replicate of S, 1 unit (signed off 2026-09-26) | with R′, builds **RS_a100′ = θ0 + δ_R′ + δ_S′** (no training), the replicate of the H4 comparator |
| MT_cal | MT at its CALIBRATION coefficient (§4) | descriptive "deployed vs deployed" comparison against RS |

Norms are measured on these FP32 arms, per tensor and in total. A training **unit** is one donor's registered
number of prompts × G rollouts and optimizer updates. Generated tokens are a recorded ceiling, not the matching
currency.

## 3. Hypotheses (registered forms)

**Confirmatory metric and testing.**
- Strict verifier outcomes are the only confirmatory metric.
- Every McNemar limb is an **exact, one-sided test in the registered direction**. The shipped
  `behaviour.exact_mcnemar` is two-sided and descriptive.
- A McNemar limb with discordance below nd_min is **UNCERTIFIABLE, never null**. nd_min is computed in code as the
  smallest nd with 0.5^nd ≤ its level (nd_min = 5 at 0.05; 8 at 0.05/12).

**H1 — acquisition gate** (DEV core atomic rows, arm as trained):
- **Criterion.** θ0 + δ_X against B, paired: an exact one-sided McNemar p ≤ 0.05 **and** net ≥ Δ_acq rows.
- **Scope.** X ∈ {R, S, P}. MT needs both R and S. R→S needs S, S→R needs R. R′ and S′ need their own skill.
- **The S shortcut.** Arms carrying S must also beat the "first k indices" shortcut, paired against the
  shortcut's own responses (it passes 36/250 DEV atomic S rows) [V].
- **When P fails.** A failed P relabels SP/RP as "trained non-skill partner"; it does not stop the study.

**H2 — atomic preservation** (TEST; an intersection-union claim "RS preserves both", each limb at one-sided 0.05):

*Statistic.*
- D_X is the TEST core atomic-X rows that the X solo answers correctly, enrolled from X's TEST read before RS is
  scored.
- L_X = #(RS = 0 on D_X).
- **H2-X passes iff the one-sided 95% Clopper–Pearson upper bound on L_X/|D_X| ≤ m_X.**

*Margins and attainability.*
- **m_R = m_S = 10%**, fixed now. The operator may change these before S1, never after.
- H2-X is UNCERTIFIABLE iff the bound at L_X = 0 exceeds m_X, i.e. 1 − 0.05^(1/|D_X|) > m_X (|D_X| < 29).
  This depends on the denominator only.
- Gains, #(X=0, RS=1) on X-wrong rows, are reported and never netted. The nd_min rule does not apply (one-sample
  limb).

*H2-X-specific (secondary).* On D_X, run an exact McNemar of RS against the size-matched non-partner arm (SP for
X = S, RP for X = R), subject to the §5 labels. Magnitude twins are never used on atomic rows of their own donor's
skill.

**H3 — joint skill** (TEST core RS rows; fixed sequence at one-sided 0.05; each limb is tested only if the previous
one passed):

*Background facts [V]:*
- Atomic R's verifier demands the full resolved set, and an R-contract answer passes 0/250 DEV RS rows. So "RS
  beats both singles" is essentially RS vs S.
- The lookup-blind floor is 0.344 on DEV RS (0.444 at k=2, 0.293 at k=3).

| step | limb | statistic | passes iff |
|---|---|---|---|
| 1 | **H3a** accuracy | exact McNemar, RS vs S | RS > S |
| 2 | **H3b** R use | e_arm,i = 1[response is an int list ⊆ row i's resolved set] − π_blind(response); UNKNOWN, unparseable and non-list responses get e = 0. Statistic Σ_i (e_RS,i − e_S,i), paired sign-flip test (null: per-row exchangeability of RS and S), Monte Carlo with 10^6 flips, registered seed, +1 correction | Σ > 0 at p ≤ 0.05 |
| 3 | **H3d** | exact McNemar, RS vs R | RS > R |
| 4 | **H3c** excess repairs | rep_RS = [B=R=S=0, RS=1] vs rep_SP = [B=P_R=S=0, SP=1], exact McNemar | RS > SP (subject to §5 labels) |

- **H3 is supported iff H3a, H3b and H3d all pass.**
- **H3c** is interpreted only if reached, and its secondary contrast (RS vs S_N) is descriptive.

**π_blind** is exact for any lookup-blind response. Every value class is a triple, exactly two members are
resolved, and the excluded member is uniform from the model's point of view.
- **Definition.** π_blind(L) = Π over value triples g of h(|L∩g|), with h(0) = 1, h(1) = 2/3, h(2) = 1/3, h(3) = 0, counting distinct indices. π_blind(L) = 0 if L holds an index outside every triple; such a list is also never ⊆ the resolved set, so e = 0.
- **Why it holds [V].** `generator.py` builds the groups from a random permutation and requests `group[:2]`.
  `canonical()` sorts keys and `requested` is shuffled, so no ordering cue reveals the excluded member.
- **Descriptive companion.** A design-randomization test of RS alone (re-draw the excluded member per triple,
  recompute inside-ness for the fixed responses) is reported descriptively.
- **Check at freeze.** The 2-of-3 structure is re-verified on TEST specs.

**H3b null cell [V].** Run on DEV specs, with 400 re-draws of the excluded member per triple and two pairs of
deterministic lookup-blind policies (lex-first vs lex-last; lex-first vs a fixed random choice):
- **Centring.** Mean e for a blind policy is −0.0015 (sd across re-draws 0.030).
- **False-rejection rate** at nominal 0.05: **0.062** and **0.050** (Monte Carlo SE ≈ 0.011).
- **Power.** Power was ~1 when RS used the lookup on ≥ 10% of rows. This is optimistic, because real arms also
  differ on rows for reasons unrelated to the lookup.

**Rule.** At arm freeze, re-run the null cell on TEST specs with ≥ 2,000 re-draws. If the one-sided 95% upper bound
of the null rejection rate exceeds 0.075, H3b's reference distribution switches to the design-randomization
distribution of Σ(e_RS − e_S): re-draw the excluded member per triple, with the responses held fixed.

**H4 — composition against training alternatives** (TEST, Tier 2; Holm across the registered family):

*Family.* The comparators are MT, R→S and S→R. Each is compared against **RS_a100**, all at trained strength and
all untuned.
- **Endpoints:**
  - RS accuracy (core RS rows);
  - atomic-R losses (on R_a100-correct rows);
  - atomic-S losses (on S_a100-correct rows);
  - legacy losses (on B-correct guard_test rows).
- The family has m = 12 limbs.
- Each limb is an exact one-sided McNemar on one fixed row set, direction registered as "RS_a100 better".

*Cross-draw floor.* A limb is claimed only if it passes Holm **and** |net| ≥ F_E. F_E is the minimum detectable
|net| of an exact one-sided McNemar at 0.05, given the discordance between RS_a100 and RS_a100′ on the same rows.
- **No replicate, no claim.** If R′ or S′ fails its DEV gate, RS_a100′ cannot be built. The floor is then undefined on RS accuracy, atomic S and legacy, and those limbs are **descriptive**.
- **Atomic R fallback.** Atomic R then takes F from R_a100 vs R′_a100 (if R′ passed).

*Descriptive only.* Acquisition levels are always reported beside H4, because the comparators may acquire
differently. Also descriptive: RS vs MT_cal, order asymmetry, and checkpoint trajectories. The draft's "or a
detectably different gain/loss structure" limb is deleted.

**Legacy retention** (every arm that reads guard_test):
- **Statistic.** L/|C| over B-correct guard_test rows, one-sided 95% CP upper bound against **m_L = 5%** (fixed now;
  operator may change before S1). |C| is B's TEST enrollment.
- **Per family.** Descriptive: the pooled rate is a B-competence-weighted mean and cannot certify one family.

*Magnitude states.* An RS legacy loss is read as interference, or as its absence, only if:
- at least one twin is buildable, and the RS loss exceeds, or does not exceed, **every** buildable twin; and
- ‖RS‖ lies inside the norm range of a **pooled** R∪S guard_dev dose fit (retained vs ln‖·‖), whose prediction
  at ‖RS‖ is reported beside it.

Otherwise the loss is labelled **MAGNITUDE_UNCONTROLLED** and supports no interference claim either way.

## 4. Coefficient choice on CALIBRATION (D4)

This is an **operating-point choice, not an equivalence claim**. H2/H3 compare arms at the same coefficients, so
the choice needs to be fixed and pre-registered; it does not need to be inferential.

1. **Grid** for R, S and MT: α ∈ {0.125, 0.25, 0.375, 0.50, 0.75, 1.00}. The recorded SFT plateau for δ_H is
   ~0.25–0.675 (`MAGNITUDE_IS_THE_MASTER_VARIABLE_20260920.md:45-64`). 0.125 is added because RL deltas may
   saturate lower, not because a measured plateau reached it.
2. **Reads at each point:**
   - greedy strict pass@1 on the donor's own 150 core atomic rows;
   - legacy losses on a hash-ordered prefix of ≤ 300 B-correct guard_dev rows (frozen at S0 enrollment);
   - ‖αδ‖ per tensor and in total.
3. **α\*** is the **smallest** α with the maximal acquisition. **On-plateau** means acq(α\*) − acq(α) ≤ **T = 9
   rows (6% of 150)**, on point estimates. Every pair's (c, b, nd) is published.
4. **Select the median on-plateau α** (the lower median if the count is even). Not the smallest: that is the
   knee edge, and the programme's composed-at-0.25 arm failed its share-of-solo-effect limb (bound 0.718 vs 0.75).
   Not α\*: that is the winner's curse.
5. **MT** follows the same steps with a joint criterion. α is on MT's plateau iff it is within T of MT's own
   maximum on **both** atomic R and atomic S. If no grid point qualifies, MT_cal is NOT_AVAILABLE.
6. **Frozen now.** The grid, T and the rule are fixed here and cannot change after any CALIBRATION donor read. The
   draft's Stage 2 rule is superseded.

## 5. The size-matched trained non-partner P (D2)

**Training.** P is trained from θ0 on the 3,000 TRAIN `atomic_P` rows with O1 and an identical recipe, inventory,
unit, seed policy and ceilings. S0 measures P's CALIBRATION mixed-group fraction alongside R's and S's. This is
recorded, not a stop rule, so D2 stands.

**Why P and not U.**
- **P's skill is inert on RS.** RS specs carry no `precedence`, and the verifier checks order only when it is
  present.
- **U would interfere.** It rewrites values (every TRAIN atomic-U row uses add/copy/swap with cutoff 0; the
  contract also permits set). That would interfere with S's target sums.

**What P does not match [V: `contracts.render`].** P is **not** a matched null for R's habit. The two differ in
prompt cues:

| prompt | "Return ALL eligible items" | "All item indices are initial candidates" | R's REFERENCE BINDING rule |
|---|---|---|---|
| atomic R | yes | no | yes |
| atomic P | yes | yes | no |
| atomic S | no | yes | no |
| RS | no | no | yes |

On RS prompts δ_R is on-cue and δ_P is off-cue. SP and RP are therefore **size-matched trained non-partner
compositions**.

**Labels, applied mechanically:**
- **PLACEBO_UNDERSIZED(shortfall %).** If p_R or p_S would exceed 1, it is set to 1 and this label is applied. An
  undersized placebo can support only verdicts that survive the shortfall: H3c FAIL, or H2-specific "no excess
  cost". It can never support H3c PASS or "excess cost".
- **PLACEBO_UNDERENGAGED.** At S2, compute the mean KL between B's and each arm's next-token distributions at the
  first answer position of each DEV RS prompt: one teacher-forced forward pass, no generation, no scoring. If
  KL(P_R) < 0.5 × KL(R), H3c can be read only as FAIL.

## 6. The multi-task comparator MT (D3)

- **Training.**
  - Budget: 2 units.
  - Every optimizer step holds exactly half atomic_R and half atomic_S prompts, each in registered hash order within
    its stratum. MT's R and S rollout counts each therefore equal one donor unit exactly.
  - O1, with the same inventory and group size.
  - No RS, bridge or boundary rows (RS signatures are withheld from training by the dataset rule).
- **Primary use:** MT at α = 1 against RS_a100. Both are 2 units of training and both are untuned.
- **MT_cal** (§4.5) against RS is descriptive.
- **No DEV selection.** There is no DEV read in any MT choice.
- **Gate failure.** If MT fails its gate, H4-MT is reported GATE_FAILED.

## 7. Sequential comparators

- **Start and training.** R→S starts from θ0 + δ_R and trains S for 1 unit on `atomic_S`. S→R mirrors it.
- **KL reference.** If O1 has an anchor-KL term, its reference is each run's own start state, with an identical
  coefficient in every run.
- **Retention.**
  - First-skill retention is scored on R_a100's (S_a100's) TEST-correct core atomic rows.
  - It is compared against RS_a100 on the same rows (H4).
- **Checkpoint records.** At fixed update counts, record ‖θ_t − θ0‖, ‖θ_t − θ_first‖ and the retained coefficient
  ⟨θ_t − θ0, δ_first⟩/‖δ_first‖². These are weight quantities only, with no decoding and no selection, so
  checkpoints need not be retained.
- **Order asymmetry is descriptive.** R→S′ is not registered, so there is no floor for it.

## 8. Evaluation protocol

**0. Scoped scorer (before S0).** The shipped `behaviour` path refuses role `calibration` (`corpus.py:26`,
`behaviour.py:21`) and rejects partial response sets (`behaviour.py:40`). Before any S0 read, commit a hash-bound
scoped scorer in `maxrl_comp002_pretraining_v1` (not in the source-locked `maxrl_composable_skills_v2`). It must:
- admit roles {calibration, dev, guard_dev, test, guard_test};
- accept a declared row subset whose ID digest is bound in the receipt;
- score with the unchanged `corpus.reward`;
- reproduce the full-bank `behaviour.enroll` scores on the declared subset (fixture).

**1. Decoding.** For every scored read:
- greedy (`do_sample=False`), `max_new_tokens=128`, `enable_thinking=False`;
- EOS/PAD IDs from the tokenizer receipt;
- batch size 1.

The Q2 sampler (T = 1.0) is for competence and training only.

**2. Precision.** Every scored read uses the FP32 streamed arms of §2, B included:
- FP32 compute;
- `torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`;
- `torch.use_deterministic_algorithms(True)`;
- a fixed thread count.

The `decoding_sha256` preimage includes the dtype, both TF32 flags, the determinism flag, the thread count, GPU
name, driver version and the torch/transformers/tokenizers versions. The draft's Q1 receipts certify BF16, so the
loader identity and the EOS/logprob/cache parity checks are re-run at FP32 before S0 enrollment.

**3. Churn floor = 0 or STOP.** Decode B twice, in separate processes, on guard_dev and on CALIBRATION core atomic
R/S/P. Any differing row is STOP_NONDETERMINISTIC until repaired, because churn enters every discordance count
directly. Each composite arm's rebuild digest check (§2) covers weight determinism.

**4. Scorer repairs**, committed with fixtures before S0:

*(a) Legacy content path.* Through `maxrl_verifiable_data_v1.verifier.parse_answer` (same `{"answer": …}`
envelope). The current `content_scoring.py` raises a host error on legacy specs.

*(b) Contract self-contradiction, resolved for its rejection clause.* An answer object whose non-`answer` fields
contain, at any depth, an int list or `"UNKNOWN"` fails content as `multiple_candidates_in_object`. Other extra
fields remain allowed. Fixtures:
- `{"answer":[1],"alt":[2]}` fails;
- `{"answer":[1],"reason":"ok"}` passes.

*(c) Strict-failure subtyper* (diagnostic). Each strict failure outside the five normalisations gets exactly one
type, with one positive-control fixture per type:
- LENGTH_CAP, EMPTY, STRING_ENCODED_ANSWER, NON_INT_ELEMENTS, ALIAS_STRINGS, KEY_VARIANT;
- PROSE_WRAPPED, MULTI_OBJECT, OTHER.

*(d) RS response classifier.* Exhaustive, with this precedence:
1. UNPARSEABLE
2. UNKNOWN_ANSWER
3. NOT_INT_LIST
4. RS_PASS
5. FULL_RESOLVED_SET (equals the resolved set; R's contract leaking)
6. RESOLVED_WRONG (⊆ resolved, not passing)
7. R_BLIND_S_VALID (not ⊆ resolved; length k; sums to target over all indices)
8. OTHER_WRONG

**5. Scorer canary.** Runs through the single scoring entrypoint before S0 and in the same pass as **every**
scored batch:
- oracle references score n/n strict;
- each reference in each **applicable** wrapper scores strict 0 and content n/n. The wrappers are code fence,
  trailing period, bare list and non-candidate extra field. Bare UNKNOWN applies only to UNKNOWN references, and
  core rows have none.
- within-signature deranged references reproduce a precomputed strict count;
- constructed lookup-blind certificates classify as R_BLIND_S_VALID;
- legacy references and wrappers are included for all 8 families.

The TEST entrypoint's code hash must equal the one used at S1/S2.

## 9. Statistics, dry runs and TEST timing

**Exact methods.** Clopper–Pearson and exact conditional McNemar throughout.
- **No percentile bootstrap decides a verdict.** At n = 250 its nominal 95% interval covers 0.39/0.63/0.86/0.91/0.95
  at true counts 0.5/1/2/4/8, and returns [0,0] 61% of the time at 0.5 [V].
- **§10 diagnostics** may use a bootstrap, and are descriptive.

**No free parameter remains.** Every margin (m_R, m_S, m_L), floor rule, attainability rule, label rule and
arm-construction rule is a closed formula fixed in this amendment. Any operator change to m_* must happen before
S1.

**Dry runs.**
- **What they are:** the DEV dry run of each tier checks the pipeline end to end and prices reads.
- **Visibility:** they are fully visible and exposed.
- **Effect:** **no rule reads them**. Attainability is computed on TEST from denominators and nd alone, by the
  closed rules of §3.

**TEST timing.**
- **Tier-2 launch.** Tier 2 is launched on DEV gates alone, whatever the Tier-1 TEST read shows. Its analysis plan
  is this amendment.
- **S′ decision.** Recorded in §0 (2026-09-26), before any TEST read.
- **Result.** The Tier-2 TEST read is therefore non-adaptive, even though it happens after the Tier-1 read.

## 10. Mechanism diagnostics (descriptive; DEV rows; after all TEST reads)

**Readout f.** Gold-token readouts are undefined here, because the verifier accepts any permutation (R) and any
valid k-subset in any order (S, RS).
- **Canonical path.** Fix the canonical path π(x), the oracle's first reference, and teacher-force it. Score the
  envelope prefix separately.
- **Per position.** At each list position t, f_t = log-odds of the probability mass on tokens that keep π_<t
  completable to **some** accepted answer. Completability comes from the verifier's own `concrete()`/`feasible()`,
  never from the ≤ 3 recovered references.

**Interaction.** I = f_RS − f_R − f_S + f_B, computed separately on atomic R, atomic S and RS prompts.
- **Normalisation.** κ_R uses (f_R − f_B) on R prompts, and κ_S uses (f_S − f_B) on S prompts. Each ratio is
  reported only if its denominator's interval excludes 0.
- **RS prompts.** Raw I only.

**Scale rule.** A SUB- or SUPER-ADDITIVE label needs same-sign resolved intervals on two scales: f unclipped, and
the completable-mass log-probability. Otherwise the label is SCALE_DEPENDENT. FA-001's registered label changed
with the clip, and its sign flipped at clip 5.

**Controls.**
- **Bind rates.** Report per-arm bind rates for any clip.
- **Saturation null.** I_RR = f(α_R + b) − f(α_R) − f(b) + f_B along R's own ray, with b = ‖α_S δ_S‖/‖δ_R‖, only
  where α_R + b ≤ 1. I_SS likewise.
- **Positive control.** I_self = f(2α_R) − 2f(α_R) + f_B on R prompts must be resolved nonzero. If it is not, any
  I ≈ 0 is INSTRUMENT_INSENSITIVE.

**Activation cross term.** Stream C_{l,m} = Σ_t (δR h)·(δS h) with hooks. Σ is never materialised, and C_RR, C_SS
and ρ = C_RS/√(C_RR·C_SS) are computed alongside.
- **Inputs.** h is θ0's own **uncentered** input to module m:
  - q/k/v: 1024;
  - o_proj: 2048;
  - gate/up: 1024;
  - down_proj: 3072.
- **Scope.** Inventory modules only.

**Geometry.** As in the draft.

**Nothing in §10 decides success.**

## 11. Data gates (spec-only)

| gate | value [V] | rule |
|---|---|---|
| RS lookup-blind floor (DEV) | mean 0.344 (k=2 0.444; k=3 0.293); the lexicographically first blind subset passes 83/250 | recomputed on TEST specs at freeze |
| Triple structure | every value class a triple; `requested` = `group[:2]`; no ordering cue | re-verified on TEST specs at freeze |
| DEV S→RS leak | same (k, target) 62/250; S's first oracle answer passes RS 33/250 | the reason for D4 |
| S shortcut, "first k indices" (`quality_diagnostics.json` definition) | TRAIN 424/3000, CALIBRATION 19/150, DEV 36/250 | H1 comparator for S |
| TRAIN→held-out content overlap | key = (sorted(initial), k, target); pool = the 3,000 `manifests/atomic_S.json` ids; DEV atomic S 1/250, RS 5/250 | frozen now. Controls: a permuted TRAIN spec must be recovered; target+1 must not. At TEST, endpoints are reported overall and with exposed rows removed; a claim stands only if both agree in sign and side of margin |

## 12. Stages, stop rules and resources

| stage | work | stop rule |
|---|---|---|
| S0 | §8.0 scoped scorer, then the canary; native qualification **at FP32**; on-policy throughput pilot (25 hash-selected `atomic_R` TRAIN groups, weights discarded; records per-group wall, peak RSS, peak CUDA memory, and eval-decode wall per completion); freeze runtime and `decoding_sha256`; CALIBRATION competence for R, S, P; guard_dev enrollment (= churn run 1); churn run 2 | STOP_HOST_INFEASIBLE if the pilot exceeds the host floor or the 0.78 VRAM cap (the training host floor is registered from the pilot, not inherited from the loader's 5.5 GiB). STOP_NO_SIGNAL if R's or S's mixed-group fraction < m_min. STOP_NONDETERMINISTIC if churn > 0 |
| S1 | Train the skill with the lower mixed-group fraction first (tie: R), then gate it (H1). Then the other. | Either failing stops the study: no P, no Tier 2 |
| S1b | Train P, then gate it | Failure only relabels SP/RP |
| S2 | CALIBRATION α reads (§4); guard_dev dose losses and pooled fit; build Tier 1 (digests); §5 engagement pass | — |
| S3 | Tier-1 DEV dry run; canary; **Tier-1 TEST read** | — |
| S4 | MT, R→S, S→R, R′, S′; their DEV gates; MT's CALIBRATION reads; Tier-2 dry run; canary; **Tier-2 TEST read** | A failed gate removes that arm from TEST |
| S5 | §10 diagnostics | — |
| S6 | F-WANDA on the damaging donor, a separate registration. It must add a norm-matched scalar control (it matches keep count, not norm) and a 1-D tensor policy. | — |
| 001b | GRPO-vs-MaxRL. Sizing from DEV and CALIBRATION only. Any contrast with an already-read MaxRL arm is registered before COMP-RS-001's Tier-1 TEST read, or runs on a bank COMP-RS-001 has not read. | — |

**Training units:**
- **H1–H3:** R, S, P = 3.
- **Tier 2:** MT 2 + R→S 1 + S→R 1 + R′ 1 + S′ 1.
- **Total:** **9**.

**Read scopes (TEST, core rows).** Each arm decodes only what its limbs use:

| arm | atomic R | atomic S | RS | guard_test B-correct |
|---|---|---|---|---|
| B | — | — | ✓ | full guard_test (enrollment) |
| R | ✓ | — | ✓ | — |
| S | — | ✓ | ✓ | — |
| RS | ✓ | ✓ | ✓ | ✓ |
| RS_a100, MT, R→S, S→R | ✓ | ✓ | ✓ | ✓ |
| R_a100, R′_a100 | ✓ | — | — | — |
| S_a100 | — | ✓ | — | — |
| MT_cal | ✓ | ✓ | ✓ | — |
| RS_a100′ | ✓ | ✓ | ✓ | ✓ |
| R_N | — | — | — | ✓ |
| S_N | — | — | ✓ | ✓ |
| P_R | — | — | ✓ | — |
| SP | — | ✓ | ✓ | — |
| RP | ✓ | — | — | — |

The measured S0 wall per completion, times these counts, sets each stage's read-wall ceiling. A stage that would
exceed it does not launch.

**Disk.**
- **Per unit, the ledger reserves:** one FP32 delta (4·|inventory| B) and two rolling checkpoints (12·|inventory| B
  each: parameters and both Adam moments). The checkpoints are deleted once the delta's data digest verifies.
- **Launch gate.** Free disk is re-read before every launch and every checkpoint write. It must be ≥ the ledger
  requirement plus the inherited 14 GiB floor.
- **Current state.** C: had 13 GiB free on 2026-09-26, and the whole-study peak is published at S0.

**Execution contract:**
- **Launch.** WMI `Win32_Process.Create` (detached); the GPU lease is held for the whole run.
- **Checkpoints.** Rolling and verified by the safetensors **data-section** digest. Each holds FP32 parameters, both
  Adam moments, all RNG states and the rollout cursor.
- **Resume and completion.** A resume-determinism acceptance test applies. "Complete" is a count, never a log line.
- **Clean checkout.** Receipts and interim outputs go outside the bound checkout. `git status --porcelain
  --untracked-files=all` must be empty before every launch and resume, with no commits between an attempt and its
  resume.
- **Read runs.** Rows are appended one at a time with a per-row hash under a fixed arm digest and `decoding_sha256`,
  and an interrupted read resumes at the first missing row.
  - A resumed TEST read is the same read iff the digests are unchanged and the interruption is documented (nvlddmkm
    14/153 events by XML SystemTime, or an operator restart record).
  - Otherwise that arm's TEST read is INCOMPLETE and is never re-read.

**Recipe.** Any anchor-KL term and the ratio clip are fixed and identical for every trained arm.

## 13. Still open before a training grant

- **Gates and budgets:**
  - Δ_acq and m_min, frozen before the full CALIBRATION competence run;
  - G;
  - per-unit ceilings (after the S0 pilot) and seeds.
- **Recipe:** KL and clip; inventory and precision.
- **Margins:** confirmed at the defaults; optional operator changes to m_R, m_S, m_L before S1.
- **From the draft's open list:** membership; independent evaluation; historical-exclusion limits.

## 14. Provenance and review record

**Verified by me on 2026-09-26 ([V]).** Checks were on committed code or by spec-only recomputation over the local
Rineya-Dataset `data/` directory (manifest `86ac8f0d…`). Only TRAIN, DEV, CALIBRATION and guard_dev metadata were
opened.

*Bank facts:*
- CALIBRATION holds 1,250 rows on 250 parents; its core R, S, P are 150 rows each on the same parents, 50 per level.
- DEV S and RS carry 50 boundary rows each (25 two-row parents), hence the core filter.

*RS bank facts:*
- The RS floor, first-lex count, S→RS leak, shortcut counts and overlap counts are as in §11.
- An R-contract answer passes 0/250 DEV RS rows.

*Code facts:*
- `generator.py` L13-21 (triples and equal values) and L40 (`group[:2]`);
- `contracts.py` L33-34 (`canonical`, `sort_keys=True`) and L108-126 (`render` cue rules);
- `verifier.py` L49-53;
- `corpus.py:26` and `behaviour.py:21,40` (no calibration role; full-bank rule);
- `content_scoring.py` (composition-verifier import; `extra_object_fields` path).

*Statistics:* the bootstrap coverage was computed exactly by binomial enumeration.

*Precedents:*
- CONFIRM-002 C3 FAIL (`composition_confirmation_v2/RESULTS.md:44`);
- FA-001 Erratum 1 (`functional_additivity_v1/RESULTS.md:107-118`);
- TH16s attenuation (`delta_multi_reasoning_v1/analysis/COMPOSITION_PAIRED_20260920.md:79-81`);
- δ_H plateau as an open DEV result (`MAGNITUDE_IS_THE_MASTER_VARIABLE_20260920.md:45-64`).

**Taken from the review without independent re-derivation.**
- **Operational facts from memory notes:** the 0.78 VRAM cap, driver-reset events and WMI detachment.
- **The inherited 14 GiB disk floor.**
- **Cost figures.** The decode-cost estimates behind the read scopes. The S0 pilot replaces them with measurements.

## 15. Clarifications 1–9 (2026-09-27, pre-data)

Registered before any model output exists. No arm has been trained, built or scored. Each clarification settles
something the text above left open, or states it more narrowly; none changes a margin, an endpoint or a
comparison. They came from an adversarial review of the pre-S0 implementation, and each was checked against the
code before being adopted.

- **C1 — UNCERTIFIABLE is for non-significant limbs only.** A limb that is significant under its registered
  procedure is never UNCERTIFIABLE. Significant with net ≥ floor is PASS. Significant with net < floor is FAIL
  (below floor). Not significant with nd < nd_min at the procedure's smallest level is UNCERTIFIABLE. Anything
  else is FAIL. This matters for H4, where Holm's later steps can reject a limb whose nd is below 8.
- **C2 — The H4 family size is fixed at m = 12.** A gate-failed comparator's limbs stay in the family as
  GATE_FAILED and are never rejected, so m never shrinks. Shrinking it would move the other limbs' Holm
  thresholds.
- **C3 — F_E when the replicates barely disagree.** If the replicate pair's discordance on E is too small for any
  net to be significant, F_E = 0: draw noise on E is negligible, and Holm significance alone decides.
- **C4 — "Int list" means a non-empty list whose elements all have type int** (bools excluded), everywhere. An
  empty answer list has e = 0 either way.
- **C5 — H3b's candidate is the content-contract candidate.** This is the same parse that `classify_rs` uses, so a
  wrapper cannot hide lookup use. A multiple-candidate object, UNKNOWN, unparseable output or a non-int-list gives
  e = 0.
- **C6 — Rows normalised as `multiple_candidates_in_object` are named by that category and get no subtype.**
- **C7 — The null cell's registered counts are ≥ 2,000 re-draws and 10^6 flips per sign-flip test.** Only a
  compliant run may emit the §3 rule outcome. The freeze run uses H3b's registered flip seed.
  - **A 400-re-draw run cannot certify the rule.** Revision 1's own 0.062 at 400 re-draws has a CP upper bound of
    0.086, which is above the 0.075 switch bound.
  - **DEV at the registered counts** (seed 20260926): 0.0435 and 0.0465 for the two pairs, with upper bounds 0.052
    and 0.055. The outcome is the sign-flip reference.
- **C8 — Scope receipts are mandatory when scoring CALIBRATION, DEV composition rows and TEST.**
  - **Each receipt** binds the bank sha, and every signature of a multi-signature scope.
  - **The DEV receipt** is extended to include P, which the H1 P gate needs.
  - **Guard banks** are bound by their manifest-checked bank sha until guard scope receipts exist.
  - **Boundary rows are refused** by the scorer unless the call is flagged diagnostic. The flag is recorded, and
    no endpoint may use a diagnostic call.
- **C9 — "No ordering cue" rests on code facts, not a spec-level test.** The facts are `canonical()` sorting keys
  and `requested` being shuffled; `contracts.py` is pinned in the scorer's code bundle, and `generator.py` by the Q0 source lock (`source_lock.json`). The
  ordering statistics stay descriptive. On DEV the excluded member's rank within its triple is lowest 630, middle
  702, highest 666 (χ² ≈ 3.9 on 2 df), which is consistent with uniform.

## 16. S0 registration (2026-09-27, before any native output)

Fixed before the first native model load in this study. The operator authorised S0 on 2026-09-27: *"Go ahead
with S0 once the fixes land"*, with the standing instruction to keep host RAM low and use VRAM freely. Nothing
here authorises S1.

**Runtime.**
- **Interpreter.** `C:\Users\clclc\BusinessProject\.venv-mq001` (Python 3.11.15, torch 2.10.0+cu126, transformers
  4.57.1, tokenizers 0.22.2, safetensors 0.7.0), asserted at start.
- **Determinism settings, before CUDA initialises:**
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - `torch.use_deterministic_algorithms(True)`;
  - both TF32 flags off;
  - `torch.set_num_threads(4)`;
  - `torch.cuda.set_per_process_memory_fraction(0.78, 0)`.
- **GPU lease.** Every GPU stage holds the shared lease for its whole run: `gpu_lease.py` from
  `rineya-dataset-retention-records/hplat-rehearsal-20260921`, sha `049742734eea…`, lock `rineya-gpu-0.lock`.
- **Host-RAM guard.** Host RAM is guarded with the programme's `hostbudget.py`, sha `9f9dcf35…`, used unmodified:
  `require` at launch, `guard` before every unit.

**Loader (low host RAM).**
- **Build on the GPU.** The model is instantiated from `config.json` directly on the GPU in FP32.
- **Stream the weights.** Each of the 310 stored BF16 tensors is streamed from `model.safetensors` and upcast into
  place. lm_head is tied to the input embedding.
- **Identity.** Every loaded FP32 tensor cast back to BF16 must equal its stored tensor bitwise, with no missing,
  extra or re-initialised tensors.
- **No CPU materialisation.** The model is never built on the CPU.

**Q1-FP32 probe (stage S0-Q1).** Launch floor: `hostbudget.require(2.5)`, then `guard(1.0)`.

*Prompts.* Eight CALIBRATION core prompts: the R and S prompts of the first four parents, in the order recorded in
`evidence/Q2_SOURCE_SMOKE_SELECTION.json`.

*Checks, with tolerances frozen now (FP32, TF32 off):*
- **T1 identity.** As above, plus the tied-head alias and the `parameter_inventory` record.
- **T2 determinism.** Two forwards produce bitwise-equal logits.
- **T3 generation vs teacher forcing.** Greedy generation with raw scores. The selected-token log-probability at
  every generated position, EOS included, must match a teacher-forced full-sequence forward within **1e-4 nats**,
  with identical argmax.
- **T4 cache vs full prefix.** Incremental KV-cache logprobs must equal the full-prefix logprobs within **1e-4**.
- **T5 padded vs unpadded.** A left-padded batch of the eight prompts plus completions must match the individual
  forwards within **1e-4** on real positions, and `mechanics.completion_labels` must mask padding and include EOS.
- **T6 backward only.** Trainable scope is candidate inventory A (§16 pilot). The loss is the mean selected
  completion logprob on two prompts. Every trainable gradient must be finite, every frozen gradient None, and the
  full state fingerprint bitwise identical before and after. No optimizer is constructed.
- **T7 resources.** Record peak CUDA allocated and reserved, peak worker RSS, and GPU shared-usage memory once at
  the end.

**The later S0 stages** use a host floor equal to Q1's measured peak worker RSS plus 1.0 GiB.

**On-policy throughput pilot (S0-P).**
- **Prompts and sampling.** 25 `atomic_R` TRAIN groups, selected by sorting
  `SHA256("COMP-RS-001-S0-PILOT|" + id)`. G = 8 sampled completions per prompt, with the Q2 sampler (T = 1.0,
  top-p 1.0, top-k 0, repetition penalty 1.0, `max_new_tokens=128`, thinking disabled).
- **Batching.** The 8 samples share one prompt and so are an unpadded batch.
- **Update.** One update per group, with MaxRL advantages from the inherited estimator (zero for uniform groups).
  The loss is −(1/G) Σ_i A_i · Σ_t log π(y_it). AdamW runs on the GPU with lr 1e-6 and weight decay 0. Gradient
  checkpointing is on, and logits are computed only at completion positions.
- **Weights are discarded.** This stage measures cost and memory; it fixes no training recipe.
- **Inventory rule, frozen now:**
  - Candidate A is blocks 8–27 (314,618,880 parameters), with FP32 parameters, gradients and Adam moments all on
    the GPU (5.74 GiB of state against the 6.24 GiB cap).
  - **If the A pilot completes with no OOM and `max_memory_reserved ≤ 0.95 × cap`, inventory A is the training
    inventory.**
  - Otherwise the pilot is re-run with candidate B, blocks 14–27 (220,233,216 parameters), which is adopted if it
    fits.
  - Otherwise STOP_HOST_INFEASIBLE.
- **Host check.** STOP_HOST_INFEASIBLE also applies if the pilot's peak worker RSS + 1.0 GiB exceeds the free RAM
  measured at its launch.
- **Other outputs.** The pilot records eval-decode wall time per greedy completion, which prices the reads, and the
  pilot groups' mixed/all-fail/all-success counts (descriptive).

**Runtime freeze (S0-F).** After the pilot, write `S0_RUNTIME_FREEZE.json` containing:
- the `decoding_sha256` preimage (§8.2);
- the inventory decision;
- the host floor;
- the code bundle of the scoring entrypoint.

Every later stage asserts it.

**Source competence (S0-C).**
- **Prompts and sampling.** All 150 CALIBRATION core rows each of R, S and P, G = 8, with the pilot's sampler and
  batching.
- **Seed.** Per prompt: `int(SHA256("COMP-RS-001-S0-COMP|" + id)[:16], 16) mod 2^63`.
- **What is kept.** Every attempt is kept, with no refill.
- **Reported:**
  - pass@1 (the mean over the 8 samples);
  - pass@8;
  - all-fail, mixed and all-success group counts;
  - UNKNOWN, format and cutoff diagnostics;
  - token costs.
- **m_min = 0.10.** STOP_NO_SIGNAL if R's or S's mixed-group fraction is below 0.10 (fewer than 15 of 150). P's
  fraction is recorded, not gated (D2).
- **Δ_acq = 13 rows** (≈ 5 points of the 250 DEV rows), for the §3 H1 gate. Both values are fixed here, before the
  competence run.

**Enrollment and churn (S0-E).**
- **Run 1.** Greedy, batch size 1, in one process: all 1,200 guard_dev rows, then the 450 CALIBRATION core R, S and
  P rows. guard_dev is enrolled through the unchanged `behaviour.enroll`, a full bank. The CALIBRATION rows go
  through the scoped scorer with `score_with_canary`.
- **Run 2.** The same rows in a separate process.
- **Churn.** Churn is the number of rows whose generated token IDs differ between runs. **It must be 0**, otherwise
  STOP_NONDETERMINISTIC.
- **Reads are resumable.** A read appends each row with its hash and resumes at the first missing row (§12).

**Records.** Native outputs are written to `C:\Users\clclc\BusinessProject\comp-rs-001-records\s0\`, outside the
bound checkout. All code is committed and `git status` is clean before every launch. Receipts are copied into
`evidence/` and committed after each stage.

## 17. Clarifications 10–12 (2026-09-28, pre-data)

Registered before the first native model load. They came from building the S0 harness and checking its memory
arithmetic against the Supa 1.7 shapes. None changes an estimand, an endpoint or a threshold.

**C10 — Rollouts are sampled one at a time, and gradients are accumulated one sequence at a time.**

*Why.* The FP32 KV cache is 224 KiB per token per sequence (28 layers × K/V × 8 heads × 128 dims × 4 bytes).
Sampling all G = 8 in one batch while the optimizer state is resident needs about 2.3 GiB at 1,300 tokens and
3.2 GiB at 1,900. That breaks the 6.24 GiB cap for both inventories purely through execution batching, so the
§16 inventory rule would have measured the batching rather than the inventory.

*What changes, in the pilot (and, unless re-registered, in training):*
- **Sampling.** The G completions of a group are drawn as G sequential single-sample calls with the registered
  sampler. The seed for sample k is `int(SHA256("COMP-RS-001-S0-PILOT-SAMPLE|" + id + "|" + k)[:16], 16) mod 2^63`.
- **Update.** The loss −(1/G) Σ_i A_i Σ_t log π(y_it) is computed one unpadded sequence per micro-batch, with
  gradients accumulated over the G sequences before one optimizer step. The gradient is unchanged up to
  floating-point summation order.

*Competence (S0-C)* has no optimizer state resident, so it keeps one unpadded batch of 8 per prompt. At the
1,763-token CALIBRATION maximum that is about 5.2 GiB. Samples are i.i.d. under either scheme; only the RNG stream
differs, and it is recorded.

*The registered ladder still runs.* The arithmetic predicts that inventory A cannot meet the 0.95 × cap rule even
at one sample at a time: 5.74 GiB of state plus 0.26–0.40 GiB of KV cache, against 5.93 GiB. Inventory B is
predicted to fit (about 5.1 GiB). **The A → B ladder is run as registered anyway:** a prediction from shapes is not
a measurement, and A fails fast at the cap.

**C11 — One response rule in every stage: the strict body.** The response string passed to `corpus.reward` is the
completion minus its terminal EOS (removed by position), decoded with `skip_special_tokens=False`. An ID the
tokenizer cannot render becomes `<|unrenderable:ID|>`. The pilot's earlier lenient rule (strip every special token)
would have turned a special token inside the body into a success, which QUALIFICATION.md forbids. The lenient score
is recorded as a diagnostic only.

**C12 — A pilot with no mixed group defers the inventory decision to S0-C.** If all 25 pilot groups are uniform,
the pilot ends UNMEASURED_NO_UPDATE and no inventory is adopted. S0-C then runs with the runtime frozen from S0-Q1
(the inventory field reads PENDING).
- **If S0-C stops on STOP_NO_SIGNAL,** the study stops there, and the inventory question is moot.
- **If S0-C passes,** the pilot is re-run on the next 25 hash-ordered `atomic_R` TRAIN groups before any S1 run.

**The runtime freeze (S0-F)** is written by `s0_freeze.py` from the passed S0-Q1 receipt and the adopted pilot
receipt (plus the A receipt when B is adopted). The decoding preimage is the one S0-Q1 made on cuda:0, and the pilot
must have run under the same `decoding_sha256`.

**C13 — S0-C samples one at a time as well (2026-09-28, after attempt 1 failed on memory).**

*What happened.* S0-C attempt 1 used the registered one-batch-of-8 sampling. It journaled 2 groups and then died
with CUDA OOM on the third prompt: 5.62 GiB allocated against the 6.24 GiB cap. C10's 5.2 GiB estimate counted the
weights and the KV cache but not the attention intermediates of an 8-row prefill on long prompts.

*The change.* S0-C now draws a prompt's 8 samples as 8 sequential single-sample calls. Sample k is seeded with
`int(SHA256("COMP-RS-001-S0-COMP|" + id + "|" + k)[:16], 16) mod 2^63`. The samples remain i.i.d. draws from the
unchanged registered sampler; only the RNG stream differs. The estimand (mixed-group fraction) and every rule are
unchanged.

*Why this is admissible after attempt 1.* The failure is infrastructural, a memory limit, not an outcome. The 2
journaled groups were both R and both 0/8. That result is kept and reported, and it cannot have steered the change:
any sampling scheme would have produced the same verdict on them.

*How the rerun happens.* The journal binds the code bundle, so a resume under changed code is refused, as §12
requires. Attempt 1's journal and attempts log are therefore preserved under an `attempt1_OOM` name, and all 450
groups are re-sampled under one scheme in a fresh journal. The receipt reports attempt 1 beside the result.

## 18. Amendment 2 — the warm-started anchor θ0′ (2026-09-28, registered before any warm-start output)

**Why.** S0 ended STOP_NO_SIGNAL: `evidence/S0_COMPETENCE.json`, commit `a8070991`. Supa 1.7 passes 0/3,600 strict
CALIBRATION samples on R, S and P.
- **Format.** It writes the `{"answer": …}` envelope in 6/3,600 samples.
- **Content.** Even with bare lists accepted, it solves R 2/1,200, S 56/1,200 and P 0/1,200.

Strict-reward RL from Supa therefore has zero gradient. **The operator chose, verbatim: "Warm-start anchor"**,
meaning SFT Supa on the dataset's own 600-row WARMSTART bank and re-run S0-C.

**The anchor.** θ0′ = Supa 1.7 + δ_warm. From here on, θ0′ replaces θ0 everywhere in this amendment: donors, arms,
twins, placebo, MT, sequential arms, B and every enrollment. The warm-start is a declared change to the experiment
(DATA_CARD: *"Using bridge/boundary training changes the experiment and must be declared"*). Composition is measured
relative to θ0′.

**Warm-start recipe (frozen now):**

*Data.*
- **Rows.** All 600 rows of `data/sft/warmstart.messages.jsonl`: atomic R, U, S and P, 150 parents. Each row has a
  system message, a user message and an assistant message holding the oracle target, e.g. `{"answer":[0,1,3,4,6,7]}`.
- **Validation.** The ids come from `warmstart.lineage.jsonl` and must equal `manifests/warmstart_core.json`. Each
  target's sha must equal its lineage `target_sha256`. The manifest sha must be `86ac8f0d…`.
- **Rendering.** The same chat template as S0 (`enable_thinking=False`, the pre-closed think block). The sequence is
  prompt tokens, then target tokens, then EOS 151645.
- **Loss.** Mean token cross-entropy over target + EOS tokens only. Prompt tokens are masked, using the one-shift
  rule of `mechanics.completion_labels`.

*Optimisation.*
- **Order.** One epoch, rows sorted by `SHA256("COMP-RS-001-WARMSTART|" + id)`.
- **Batches.** Micro-batch of one unpadded row; gradients accumulated over 8 rows, which gives 75 optimizer steps.
- **Optimizer.** AdamW with lr 1e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0, no warmup and no schedule. The
  gradient is clipped to a global norm of 1.0.
- **Numerics.** FP32 with TF32 off, deterministic algorithms, fixed threads, and the §16 memory cap and lease.

*Trainable inventory: blocks 14–27* (inventory B, 220,233,216 parameters).
- **Why B.** By arithmetic, inventory A cannot meet the 0.95 × cap rule (§17 C10), and B fits with margin.
- **First measurement.** This run is the first measured update memory; its `max_memory_reserved` is recorded.
- **Donor inventory.** The donor inventory is still decided by the §16 pilot ladder, re-run on θ0′.

*Output.*
- **Storage.** δ_warm is written create-only as an FP32 delta over inventory B (`arm_build.write_delta`, data-section
  digest), outside the checkout.
- **Identity.** θ0′'s identity is the §2 arm digest of Supa + δ_warm.
- **Rebuild.** Every later stage loads Supa through the S0 FP32 loader and adds δ_warm in FP32. It asserts both the
  delta digest and the arm digest, which are pinned in a runtime freeze for θ0′.

**What runs next, in order:**
1. **The warm-start run** (one GPU run).
2. **A descriptive envelope check.** Greedy decode of the 8 S0-Q1 prompts on θ0′. It is recorded, not gated.
3. **The runtime freeze for θ0′.** The S0-Q1 decoding preimage and host floor are carried over (the numeric path is
   unchanged), and the anchor identity is added.
4. **S0-C on θ0′,** with the unchanged m_min = 0.10 rule on R and S.
5. **If S0-C passes:** the §16 pilot ladder on θ0′, then S0-E (enrollment and churn) on θ0′.

**Stop rules and reporting.**
- **If S0-C on θ0′ is STOP_NO_SIGNAL, the study stops.** No second warm-start dose or recipe without a new operator
  decision. Whether the failure is all-fail (too weak) or all-success (too strong) is reported.
- **The warm-start's own cost** is reported descriptively, not gated: one extra guard_dev read of Supa, compared
  against θ0′'s S0-E enrollment. The legacy retention endpoints of §3 stay relative to θ0′.
