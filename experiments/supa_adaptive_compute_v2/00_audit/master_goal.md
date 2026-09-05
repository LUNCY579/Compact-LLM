
# MASTER GOAL — Two-Track Supa Adaptive-Compute Research Programme

You are acting as a Principal LLM Architecture Research Engineer responsible for the next rigorous phase of the Rineya Family / Supa-A1.1 research programme.

Your task is to build and execute a **shared but scientifically separated experimental framework** for two related research questions:

# Track A — Native ARMR Generalization

\[
\boxed{
\text{Does bounded native middle-layer recurrence within Supa-A1.1
produce a real and generalizable capability gain?}
}
\]

# Track B — Cross-Scale Donor Transformation Transfer

\[
\boxed{
\text{Can a larger Qwen3-family layer transformation be faithfully transferred
into Supa-A1.1 and produce substantial, statistically defensible gains?}
}
\]

These two tracks MAY and SHOULD share infrastructure.

They MUST NOT share:

- experimental decisions made after observing results;
- calibration/evaluation leakage;
- run evidence;
- candidate-selection logic;
- preregistration;
- success criteria;
- mutable output directories.

The governing principle is:

\[
\boxed{
\textbf{shared engineering, independent science}
}
\]

Do not collapse these into one giant experiment.

Do not add routing, MLA, MHA, MTP, PEFT recovery, or additional architecture mechanisms merely because they are part of the long-term roadmap.

First answer the two questions above cleanly.

---

# 1. CURRENT EVIDENCE THAT MUST BE PRESERVED

## 1.1 Native ARMR evidence

The completed CPU-budget replication established:

- A0 intact Supa: 12/16
- A1 L12 repeated: 12/16
- A3 L12-L13 block repeated: 13/16
- A3 versus A0: 1 paired win, 0 paired losses
- all 16 prompts completed after the timeout budget was corrected prospectively
- the corrected counterfactual-transitivity case reproduced
- no preservation/validity regression was observed on this tiny development set
- the evaluation is too small and previously inspected to establish general capability

This is a **hypothesis-generating signal**, not a capability result.

Do not overwrite or reinterpret historical evidence.

## 1.2 Cross-scale Qwen3-4B evidence

The completed cross-scale experiment inserted one frozen Qwen3-4B L16 after ordinary Supa L12 with fitted affine residual bridges and gate 0.1.

Main results:

- aligned graft: 12/16
- intact Supa: 12/16
- 0 paired wins
- 0 paired losses
- 16/16 greedy outputs identical
- mean first-token cosine ~0.9973
- mean KL ~0.005
- approximately +16% observed wall time
- approximately +24% dominant linear-matmul estimate
- ~100.9M frozen donor parameters
- ~5.25M bridge coefficients
- random bridge moved reference-answer NLL more favorably than the aligned bridge
- absolute hidden-state mapping was much stronger than transformation-delta mapping
- held-out donor transformation cosine was only ~0.443
- ordinary text-token alignment error was much worse than misleading global metrics suggested

This establishes:

\[
\boxed{\text{mechanical viability}}
\]

but NOT:

\[
\boxed{\text{useful cross-scale transfer}}
\]

The immediate cross-scale bottleneck is therefore **transformation fidelity**, not merely evaluation sample size.

---

# 2. FIRST TASK — AUDIT EXISTING WORK

Before implementing anything new, inspect the repository comprehensively.

Find and review:

- native ARMR implementation;
- reapplication-screen code;
- A0/A1/A3 callers;
- CPU-budget replication code;
- large-model graft code;
- bridge fitting;
- activation extraction;
- Qwen3-4B loading/extraction;
- existing cross-scale evidence;
- hidden-state diagnostics;
- hash/integrity utilities;
- evaluation datasets;
- statistical utilities;
- scoring;
- timeout handling;
- run manifests;
- evidence/audit tooling;
- sealed-data protections;
- any other recurrence/cross-scale branches that contain reusable components.

For every relevant component classify it as:

- REUSE AS-IS
- REUSE WITH REPAIR
- SUPERSEDED
- HISTORICAL EVIDENCE ONLY
- SCIENTIFICALLY DIFFERENT
- UNSAFE / INCORRECT
- MISSING

Create:

`00_audit/existing_work_inventory.md`

Include exact:

- paths;
- branches;
- commit SHAs;
- scientific purpose;
- implementation quality;
- what should be reused;
- what must remain untouched.

Do not duplicate good code.

Do not merge historical evidence folders.

---

# 3. RECOMMENDED CODEBASE ORGANIZATION

Prefer one shared experiment root with two independent pipelines.

Example:

```text
experiments/
  supa_adaptive_compute_v2/

    README.md
    AGENTS.md
    ROADMAP.md
    STATUS.md

    shared/
      model_loading.py
      supa_hooks.py
      hidden_capture.py
      generation.py
      evaluation.py
      scoring.py
      statistics.py
      manifests.py
      hashing.py
      integrity.py
      guards.py
      timing.py
      datasets.py
      reporting.py

    native_armr/
      README.md
      protocol/
      configs/
      src/
      scripts/
      tests/
      evidence/

    cross_scale/
      README.md
      protocol/
      configs/
      src/
        bridge_fit.py
        delta_alignment.py
        donor_layers.py
        graft.py
        depth_pair_screen.py
      scripts/
      tests/
      evidence/

    data/
      manifests/
      generated/
      frozen_splits/
It is acceptable to adapt the current repository structure instead.
Do not restructure merely for aesthetics if existing auditability would be damaged.

4. SHARED INFRASTRUCTURE POLICY
The following SHOULD be shared:
Supa model loading;
tokenizer loading;
deterministic generation;
hidden-state capture;
checkpoint hashing;
source hashing;
dataset manifests;
validity checks;
Unicode/repetition/EOS diagnostics;
paired evaluation infrastructure;
logit metrics;
NLL metrics;
latency measurement;
memory measurement;
statistical testing;
reporting;
sealed-data guards.
The following MUST remain track-specific:
candidate operators;
calibration sets;
bridge training data;
hyperparameter selection;
primary metrics;
GO/NO-GO criteria;
run manifests;
evidence;
preregistrations.
Never allow Track B bridge fitting to consume Track A efficacy evaluation answers.

5. BUILD A REAL EVALUATION BANK
The existing 16-item development set is inadequate for estimating small effects.
Construct a new, fresh, stratified, unsealed evaluation bank.
Target initial size:
N≈1,000\boxed{N \approx 1,000}
A practical target is 1,024 examples.
The exact count may change slightly to keep family strata balanced.
The bank must contain multiple independent scenario families.
Suggested categories:
relational/transitive reasoning
counterfactual reasoning
compositional reasoning
multi-hop reasoning
mathematical/arithmetic reasoning
symbolic/logic reasoning
factual/ordinary QA preservation
instruction following
evidence sufficiency / abstention
retrieval
distractor/interference
long-context multi-hop, if runtime allows
Do NOT generate 1,000 lexical variants of the already-successful counterfactual example.
Prioritize diversity of reasoning structures.

6. DATA CONSTRUCTION RULES
Each family should contain multiple scenario templates.
Paraphrases of one latent scenario must remain within the same split.
Where possible include paired variants:
original;
paraphrase;
counterfactual;
irrelevant-context;
insufficient-evidence;
conflict;
reordered evidence.
This enables more causal analyses than aggregate accuracy.
Create immutable IDs.
Example:
{
  "example_id": "counterfactual_0042_variant_c",
  "scenario_id": "counterfactual_0042",
  "family": "counterfactual",
  "split": "confirmatory_dev",
  ...
}
Freeze the bank before model evaluation.
Store a hash of the full manifest.

7. IMPORTANT DATA SEPARATION
Use at least three data roles:
Calibration
Used only for bridge fitting or engineering calibration.
Architecture-development
Small set for rejecting broken candidate mechanisms.
Confirmatory development
Large fresh ~1,000-example bank used to evaluate frozen candidates.
Do not tune on the confirmatory bank and then call the same results confirmation.
If substantial design changes are made after reading the confirmatory results, version the next experiment and construct/hold back an additional confirmation subset.
Sealed evaluation remains closed.

8. TRACK A — NATIVE ARMR
Research Question
QA:Does native bounded recurrent depth within Supa produce a real, generalizable advantage?\boxed{ Q_A: \text{Does native bounded recurrent depth within Supa produce a real, generalizable advantage?} }
9. TRACK A CANDIDATES
At minimum evaluate:
A0
Intact Supa.
A1
L12 with one additional application.
Historical control.
A3
L12-L13 block with one additional block application.
Current strongest hypothesis.
A16
A preregistered L16 recurrence arm.
L16 is scientifically important because it showed unusually strong repeatability in the prior reapplication atlas.
If the existing evidence supports another exceptionally interesting native candidate, document the rationale prospectively before running the large bank.
Do not add many layers after seeing partial results.

10. TRACK A COMPUTE-MATCHED CONTROL
Add at least one approximate compute-matched control.
The purpose is to distinguish:
benefit from recurrence\text{benefit from recurrence}
from:
benefit from simply spending more FLOPs.\text{benefit from simply spending more FLOPs}.
Possible control:
h′=h+αM(h)h'=h+\alpha M(h)
where MM is a frozen or independently initialized Supa-width residual transformation with approximately matched dominant matmul cost.
Do not intentionally train the control to lose.
Record:
parameter count;
active parameters;
dominant matmul estimate;
measured latency.
The exact control design must be preregistered.

11. TRACK A PRIMARY ANALYSIS
Because every arm receives the same examples, use paired statistics.
Primary binary comparison:
nwin=#(Ai correct, A0 wrong)n_{\text{win}} = \#(A_i\ correct,\ A0\ wrong) nloss=#(Ai wrong, A0 correct).n_{\text{loss}} = \#(A_i\ wrong,\ A0\ correct).
Report:
raw accuracy;
paired wins/losses;
exact McNemar test;
paired bootstrap confidence interval;
absolute error reduction;
relative error reduction.
Do not rely solely on aggregate accuracy.

12. TRACK A FAMILY ANALYSIS
For every family report:
A0 accuracy;
candidate accuracy;
paired wins;
paired losses;
NLL delta where available;
output identity rate;
confidence interval where meaningful.
A useful outcome may be specialization.
Example:
A3:
  counterfactual: strongly positive
  ordinary QA: neutral
  retrieval: neutral
This is scientifically different from:
A3:
  slightly positive everywhere
The first outcome can later justify routing.

13. TRACK A GENERALIZATION ANALYSIS
Check whether gains survive:
unseen templates;
paraphrases;
changed entity names;
changed relation directions;
counterfactual changes;
longer chains;
distractor insertion;
context reordering.
A recurrence mechanism that only fixes one lexical structure should not be called general capability.

14. TRACK A GO CRITERIA
Predefine a meaningful threshold.
A reasonable screening criterion may be:
at least ~2 absolute accuracy points;
OR
at least ~5% relative error reduction;
AND:
paired confidence interval supportive;
no material preservation regression;
no significant validity degradation;
favorable or explainable quality-per-cost.
Do not mechanically use these numbers if repository standards already define stronger thresholds.
Record the final preregistered values.

15. TRACK A POSSIBLE OUTCOMES
Outcome A
A3 clearly beats A0 and matched compute.
Interpretation:
Native bounded middle recurrence is genuinely useful.
Outcome B
A3 beats A0 but not matched compute.
Interpretation:
Extra computation helps, but recurrence is not specifically supported.
Outcome C
A3 helps one family strongly but is neutral overall.
Interpretation:
Native recurrence is a specialized operator and routing may become scientifically justified.
Outcome D
No meaningful gain.
Interpretation:
Mechanical repeatability does not translate into capability.
Preserve all outcomes.

16. TRACK B — CROSS-SCALE TRANSFORMATION TRANSFER
Research Question
QB:Can a larger Qwen donor’s useful internal transformation be transferred faithfully into Supa and produce statistically meaningful gains?\boxed{ Q_B: \text{Can a larger Qwen donor's useful internal transformation be transferred faithfully into Supa and produce statistically meaningful gains?} }
Do not run the existing G arm across 1,000 cases simply to obtain a more precise zero.
The current mechanism must first solve transformation alignment.

17. CROSS-SCALE CORE DISTINCTION
Absolute-state correspondence is not the final target.
Current experiments indicate:
P↑(hS)≈hBP_\uparrow(h_S)\approx h_B
can be moderately achieved,
while:
LB(P↑hS)−P↑hSL_B(P_\uparrow h_S)-P_\uparrow h_S
does not sufficiently match the donor's native transformation.
Therefore prioritize:
transformation/delta alignment\boxed{\text{transformation/delta alignment}}
rather than raw hidden-state alignment alone.

18. TRANSFORMATION DEFINITION
For native donor state:
hBh_B
define:
ΔB=LjB(hB)−hB.\Delta_B = L^B_j(h_B)-h_B.
For projected Supa state:
z=P↑(hS)z=P_\uparrow(h_S)
define:
ΔP=LjB(z)−z.\Delta_P = L^B_j(z)-z.
Primary bridge goal:
ΔP≈ΔB\boxed{ \Delta_P \approx \Delta_B }
in direction and useful scale.

19. BRIDGE OBJECTIVES TO COMPARE
Prospectively compare at least:
B0 — Historical absolute ridge
Current baseline.
B1 — Token-normalized state alignment
Prevent high-norm control/chat tokens from dominating regression.
B2 — Delta-aware alignment
Optimize transformation correspondence explicitly.
B3 — Token-normalized + delta-aware alignment
Likely strongest candidate.
BS — Shuffled-pair control
Mandatory.
BR — Random bridge control
Mandatory.
Do not remove negative controls because they perform surprisingly well.

20. TOKEN BALANCING
The previous experiment exposed a major issue:
global error metrics were dominated by high-norm tokens while ordinary text tokens were poorly aligned.
Explicitly report metrics separately for:
all tokens;
ordinary text tokens;
special/control tokens;
final prompt token;
optionally semantic/content tokens.
Potential weighting schemes include:
h^t=ht∥ht∥+ϵ\hat h_t= \frac{h_t}{\|h_t\|+\epsilon}
or loss weights approximately inverse to token-state magnitude.
Do not silently choose whichever weighting looks best after efficacy results.
Compare them prospectively.

21. CROSS-SCALE DEPTH-PAIR SCREEN
Do not assume:
Supa L12↔Qwen3-4B L16.Supa\ L12 \leftrightarrow Qwen3\text{-}4B\ L16.
Construct a cheap representation/transform alignment matrix.
Potential Supa insertion points:
i∈{8,12,16,19}i\in\{8,12,16,19\}
Potential donor points approximately spanning the donor middle region.
For each pair:
(i,j)(i,j)
measure held-out:
up-map cosine;
down-map cosine;
ordinary-text normalized error;
donor-delta cosine;
donor-delta relative MSE;
residual magnitude ratio;
numerical stability.
Do NOT run full generation for every pair.
Use this as a mechanism screen.

22. DEPTH-PAIR SELECTION
Candidate selection must occur using only calibration/held-out-alignment data.
Freeze the selected pair(s) before efficacy generation.
Avoid choosing based on the 1,000-example confirmatory efficacy bank.
Document why each selected pair won.

23. CROSS-SCALE ALIGNMENT SUCCESS METRICS
Primary mechanism metrics should include:
cos⁡(ΔP,ΔB)\cos(\Delta_P,\Delta_B)
and normalized transformation error.
The historical value around 0.443 serves as a baseline.
Do not define an arbitrary “success” threshold after seeing results.
Preregister a meaningful required improvement.
Also check whether better alignment actually correlates with:
larger useful NLL shifts;
changed decisions;
better accuracy.
High alignment without useful efficacy remains a negative capability result.

24. PROSPECTIVE GATE-DOSE STUDY
After selecting a new bridge formulation, investigate the donor residual gate prospectively.
Possible grid:
g∈{0.03,0.10,0.20,0.40}g\in \{0.03,0.10,0.20,0.40\}
or another compact preregistered grid.
Measure:
output identity;
first-token cosine;
KL;
hidden RMS;
answer NLL;
correctness;
invalid outputs;
latency.
The objective is to determine whether the cross-scale transformation exhibits:
inert→useful→destructive\text{inert} \rightarrow \text{useful} \rightarrow \text{destructive}
behavior as dose increases.
Do not tune gate on confirmatory data.

25. TRACK B STAGED EVALUATION FUNNEL
Do not immediately spend 1,000 examples per bridge.
B-Stage 1 — Mechanism
No or minimal generation.
Use calibration + scenario-held-out activations.
Questions:
does transformation alignment improve?
which depth pair aligns best?
are ordinary text tokens aligned properly?
B-Stage 2 — Small frozen generation screen
Approximately:
64−12864-128
fresh examples.
Purpose:
reject inert/destructive candidates;
verify behavior changes appropriately;
detect gross regressions.
Not sufficient for efficacy claims.
B-Stage 3 — Confirmatory development
Only strongest frozen candidate(s):
512−1,024512-1,024
fresh stratified examples.
Compare against A0 and required controls.

26. TRACK B CONTROLS
At minimum the confirmatory comparison should include:
X0
Intact Supa.
XN
Best native recurrence candidate from Track A, if frozen by then.
XG
Best aligned 4B donor graft.
XS
Shuffled-pair bridge.
XR
Random bridge.
XM
Approximate parameter/FLOP-matched Supa-width residual transformation.
If feasible:
XD
A distilled or same-scale control in later phases.
The important claim is NOT:
XG>X0XG>X0
alone.
The stronger required claim is:
XG>generic comparable extra transformation\boxed{ XG > \text{generic comparable extra transformation} }
27. CROSS-SCALE PRIMARY ANALYSIS
Use the same paired statistical framework as Track A:
accuracy;
paired wins/losses;
McNemar;
paired bootstrap CI;
family-level effects;
NLL;
output identity;
preservation;
quality-per-FLOP.
Also relate efficacy to bridge metrics.
Ask:
Does higher held-out delta alignment predict larger useful gains?\text{Does higher held-out delta alignment predict larger useful gains?}
That is scientifically important.

28. TRAINING BOUNDARY
Freeze:
Supa backbone;
Qwen3-4B backbone.
Initially train only:
affine/linear bridges;
perhaps explicitly registered bridge normalization;
optional gate parameters only in a later experiment.
No Transformer backbone training.
Verify:
state hashes before/after;
checkpoint hashes;
trainable parameter inventory;
zero gradients in frozen backbones.

29. QWEN3-8B GATE
Do NOT escalate to Qwen3-8B merely because support is available.
8B becomes eligible only if one of the following is established:
Qwen3-4B transfer yields a real positive signal;
bridge fidelity becomes strong but 4B operator capability appears limiting;
a specific scientifically motivated cross-scale question requires 8B.
Otherwise keep 8B unexecuted.

30. SHARED STATISTICS MODULE
Implement reusable paired statistics supporting:
accuracy difference;
paired win/loss table;
exact McNemar;
bootstrap confidence intervals;
family stratification;
NLL differences;
effect sizes;
multiple-comparison annotations.
Do not automatically over-correct every exploratory screen.
Clearly distinguish:
exploratory;
candidate-selection;
confirmatory.

31. SAMPLE-SIZE / POWER PLANNING
Before the large confirmatory run, estimate how many examples are required to detect plausible effect sizes.
Do not pretend exact power is known before paired discordance rates are observed.
Use reasonable planning scenarios such as:
+1 point;
+2 points;
+3 points;
5% relative error reduction.
Document expected sensitivity at:
N=256, 512, 1024, 2048.N=256,\ 512,\ 1024,\ 2048.
This should guide whether ~1,000 is sufficient.

32. DATASET FREEZE
Before any confirmatory evaluation:
generate/assemble examples;
validate them;
freeze manifest;
freeze scorer;
hash inputs;
commit preregistration;
only then execute models.
No editing failed questions after observing model outputs.
Broken examples may be excluded only under preregistered objective validity criteria.

33. SCORING QUALITY
For generated/synthetic logic cases, prefer deterministic programmatic scorers.
For open-ended cases, avoid a single opaque LLM judge if exact/reference scoring is possible.
When LLM judging is required:
freeze judge;
freeze rubric;
blind model identity;
audit a sample manually.
Record score confidence/ambiguity.

34. CPU EXECUTION COST
The current CPU setup can make 1,000 examples × many arms extremely expensive.
Build support for:
resumable immutable runs;
per-arm scheduling;
deterministic shard execution;
multiple independent processes only when safe;
aggregation after completion.
Do not allow two processes to mutate the same evidence directory.
If GPU evaluation becomes available, qualification against CPU/reference behavior must occur before mixing runtimes.

35. PARALLELISM BETWEEN TRACKS
Track A and Track B may be worked on concurrently because their core questions are independent.
Recommended:
Process A
construct/freeze native ARMR large evaluation and execute native arms.
Process B
improve bridge/delta alignment and perform depth-pair screen.
They should NOT compete heavily for RAM/CPU simultaneously on one low-resource machine if this changes latency behavior or causes instability.
Support parallel research workflow, but allow execution scheduling to remain sequential when resources are constrained.

36. EVIDENCE DIRECTORY RULES
Example:
native_armr/evidence/
  00_preregistration/
  01_dataset_validation/
  02_confirmatory_run/
  03_statistics/
  04_independent_audit/

cross_scale/evidence/
  00_preregistration/
  01_alignment_matrix/
  02_bridge_objectives/
  03_gate_dose/
  04_small_generation/
  05_confirmatory_run/
  06_statistics/
  07_independent_audit/
Never write Track B evidence into Track A directories.

37. RUN MANIFEST
Every execution must include:
branch;
commit SHA;
model checkpoint;
model hashes;
donor hashes if applicable;
source hashes;
dataset hash;
config hash;
seed;
threads;
dtype;
device;
cache mode;
start/end;
wall time;
peak memory;
arm;
status;
failure receipt.

38. HARD FAILURE CONDITIONS
Fail closed on:
NaN/Inf;
hidden RMS outside registered guards;
malformed generation explosion;
checkpoint mutation;
unexpected trainable backbone parameter;
unexpected gradient;
incomplete A0 control;
evaluation leakage;
dataset/hash mismatch;
evidence overwrite;
scorer mismatch;
unregistered candidate/gate changes.
Do not hide failed runs.

39. NATIVE VERSUS CROSS-SCALE COMPARISON
Once both Track A and Track B have frozen confirmatory results, compare them explicitly.
Questions:
Capability
Which has larger paired gain?
Specialization
Which task families benefit?
Cost
How much added:
latency;
FLOPs;
parameters;
memory?
Efficiency
Define a simple quality-per-cost frontier.
Do not force a single scalar metric if tradeoffs are important.

40. DECISION AFTER BOTH TRACKS
Possible outcomes:
Case 1 — Native wins, cross-scale fails
Then:
continue native bounded recurrence\boxed{\text{continue native bounded recurrence}}
and deprioritize large-donor grafting.
Case 2 — Cross-scale wins, native fails
Then:
large-model transformation transfer is the stronger mechanism\boxed{\text{large-model transformation transfer is the stronger mechanism}}
and prioritize donor distillation.
Case 3 — Both win on the same examples
Choose the simpler/cheaper operator unless the larger one materially improves capability.
Case 4 — Both win on different families
This is the strongest justification for future routing.
Case 5 — Neither wins
Do NOT proceed to elaborate adaptive routing.
Shift attention to other mechanisms such as attention refinement.

41. ROUTING GATE
Adaptive routing is NOT automatically part of this task.
Routing becomes scientifically justified only if:
at least two operators show positive or complementary effects;
or one operator helps only a clear subset while hurting/adding cost elsewhere.
If a single fixed A3 operator is universally best, use the fixed operator.
Do not add a router just because it is architecturally interesting.

42. MLA / MHA / MTP GATE
Do not combine these mechanisms during the two-track study.
They belong to the next research phase.
They become licensed when we know whether useful extra computation exists and what form it takes.
Potential future question:
When Supa receives extra compute, is it better spent on additional depth or additional contextual rereading?\boxed{ \text{When Supa receives extra compute, is it better spent on additional depth or additional contextual rereading?} }
Only then compare:
native recurrence;
GQA reread;
MHA reread;
MLA reread.
MTP becomes useful later as:
auxiliary training;
uncertainty estimation;
adaptive halting.

43. CROSS-SCALE DISTILLATION GATE
If the 4B donor demonstrably helps, immediately prioritize distilling the donor transformation into a compact Supa-width operator.
Teacher:
ΔT(h)=P↓[L4B(P↑h)−P↑h].\Delta_T(h) = P_\downarrow [ L_{4B}(P_\uparrow h)-P_\uparrow h ].
Student:
Rθ(h)≈ΔT(h).R_\theta(h)\approx\Delta_T(h).
The long-term objective is not necessarily to carry a 100M+ donor layer during deployment.
A much stronger outcome would be:
borrow a large-model transformation during training, then compress it into a small Rineya operator\boxed{ \text{borrow a large-model transformation during training, then compress it into a small Rineya operator} }
Do not implement this until donor efficacy is established.

44. TESTS
Shared tests:
canonical model identity;
deterministic generation;
hash preservation;
scorer determinism;
dataset manifest freeze;
evidence exclusivity;
sealed path rejection;
incomplete baseline rejection;
paired-statistics correctness.
Track A:
layer repetition counts;
block order;
stem untouched;
no fake position advance;
disabled-mode identity;
compute-control execution.
Track B:
bridge dimensions;
donor frozen;
Supa frozen;
correct donor layer;
residual delta formulation;
shuffled pair construction;
random bridge reproducibility;
token-normalized metric correctness;
depth-pair matrix coverage;
gate zero identity;
restore identity.
Do not claim tests for unimplemented future mechanisms.

45. STATUS.md
Maintain a clear programme status table.
Example:
Track
Question
Mechanics
Candidate selection
Large eval
Gate
Native ARMR
Does recurrence generalize?
PASS
frozen
pending/running
...
Cross-scale
Can donor transformation transfer?
PASS mechanically
alignment v2 pending
not licensed
...
Routing
Are multiple operators complementary?
not licensed
—
—
CLOSED
MLA/MHA
Is reread better than depth?
not licensed
—
—
CLOSED
MTP
Can prediction guide compute?
not licensed
—
—
CLOSED

Update this after every meaningful stage.

46. EXPECTED CURRENT-SESSION DELIVERABLES
At minimum, complete:
full repository audit;
unified shared infrastructure plan;
native/cross-scale caller separation;
fresh evaluation-bank design;
data-generation/validation pipeline;
paired statistics module;
Track A preregistration;
Track B transformation-alignment-v2 preregistration;
implementation of Track A large-eval caller;
implementation of Track B alignment/depth-pair caller;
tests;
status documentation.
Execute what is computationally reasonable, but never claim runs not performed.
If long evaluations exceed the available session budget, leave deterministic resumable commands and exact next-run instructions.

47. FINAL RESPONSE FORMAT
Report:
Repository
branch
local HEAD
remote HEAD
clean/dirty state
Existing work audit
What was reused and from where.
Shared architecture
What utilities both tracks use.
Track A
protocol
evaluation-bank size
arms
execution status
preliminary/final results
paired statistics
GO/NO-GO
Track B
bridge objectives
depth pairs
held-out transformation metrics
selected bridge
gate study if licensed
generation status
GO/NO-GO
Statistical power
What effect sizes the current sample can meaningfully detect.
Integrity
hashes
frozen weights
data boundaries
sealed-data status
Evidence paths
Exact directories.
Tests
Exact commands and pass counts.
Limitations
Be explicit.
Exactly one next causal experiment per track
Give:
one next step for Track A;
one next step for Track B.
Do not combine the tracks prematurely.

48. SCIENTIFIC STANDARD
The objective is NOT to make one architecture win.
The objective is to resolve two questions:
A. Can Supa improve itself by selectively reusing its own internal depth?\boxed{ \textbf{A. Can Supa improve itself by selectively reusing its own internal depth?} }
and
B. Can Supa borrow a useful internal transformation from a larger sibling?\boxed{ \textbf{B. Can Supa borrow a useful internal transformation from a larger sibling?} }
These are distinct hypotheses.
A negative answer to either is valuable.
A positive answer must survive:
fresh data;
adequate sample size;
paired controls;
appropriate compute controls;
preservation evaluation;
independent audit.
Only after those questions are answered should the programme decide whether heterogeneous Cross-Mix routing is scientifically warranted.

I would strongly prefer this **single-codebase/two-callers design** over maintaining two entirely independent implementations. It gives you a particularly useful property later: if both experiments produce viable operators, they are already running through the same generation, evaluation, metrics, and integrity interfaces, so a future router can compare or invoke them without a major rewrite.

Conceptually, the API you eventually want is almost this simple:

```python
result = evaluate_operator(
    model=supa,
    operator=native_armr_operator,
    dataset=frozen_eval,
)
versus:
result = evaluate_operator(
    model=supa,
    operator=cross_scale_4b_operator,
    dataset=frozen_eval,
)
with both operators implementing something like:
Refine⁡(h,context)→h′.\operatorname{Refine}(h,\text{context})\rightarrow h'.
That common abstraction becomes extremely valuable later. Native L12–L13 recurrence, a transferred 4B layer, a distilled donor, MLA reread, or MHA reread can all eventually become different implementations of the same optional refinement operator interface.
But for now, the scientific logic stays exactly as you phrased it:
Track A:Can Supa reuse itself and obtain generalizable gains?Track B:Can Supa faithfully borrow a larger model’s transformation and obtain gains?\boxed{ \begin{array}{ll} \textbf{Track A:} & \text{Can Supa reuse itself and obtain generalizable gains?}\\[4pt] \textbf{Track B:} & \text{Can Supa faithfully borrow a larger model's transformation and obtain gains?} \end{array}}
That is a clean enough foundation that whatever result comes next—positive or negative—will meaningfully tell us what kind of Cross-Mix architecture Rineya should pursue.

