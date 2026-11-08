# TRISKILL-GEO-001 prospective pilot

Frozen 2026-10-01 before successor model execution. No training, no delta_C,
no third-task examples used to construct or select the intervention. This is a
single falsifier, not a new full development sweep or original Stage B retry.

## Lineage and source evidence

Verified origin: `https://github.com/LUNCY579/ChatBubble.git`.
Actual research branch: `codex/triskill-stage-a`, commit
`01f47694988a85bdedab7831d1652f0b9bf07e9f`. Child branch:
`codex/triskill-geometric-pilot-20261001`. The original worktree, dirty main
checkout, all unrelated worktrees, checkpoint copies, and concurrent fixes are
preserved. No AGENTS.md exists in this source tree; the source ancestor
`C:/Users/clclc/.codex/AGENTS.md` is empty. Relevant `.agents/skills/cavecrew`
was read for delegation. Other local skills have no research requirements.

Original TRISKILL-001 added a supervised C donor (29 updates; +40 semantic
rows/192 alone) to TT. It stopped `STOP_NO_ELIGIBLE_BETA`. The Stage A T
**net** losses at .25/.5/1 were 16/19/84; **gross source-correct** losses were
25/47/120, with gains 9/28/36. The C-only dose curve was also nonmonotonic
(84/76/73), so collision does not explain its entire dose response. The
norm-matched post-stop random draw was inert. These are historical findings,
not new results from this task.

CONFIRM002 demonstrated narrow parents: T familiar single-event transitions
430/864 vs base5; H25 boundary263/336 vs base185. TT retained T434 and boundary250,
but H's 75% effect-preservation confidence criterion failed (lower bound .718),
and strict answerable correctness fell37→35/109. T is not established general
state reasoning; H is not established structured uncertainty propagation.

ARMPERP tested global flattened Gram–Schmidt on a second **same-T** draw and
parameter addition over220 tensors. Its unresolved boundary control/perp/same
cluster was69/66/68; trueH85 separated. This only rejects that realization as a
boundary generator. This pilot instead intervenes on one live residual using
two different learned parent output subspaces. It repeats neither ARMPERP nor
the old delta_C coefficient sweep. Original C_FRESH/T_BANK_V5/H_FRESH and
HSKILL-CONFIRM-002 remain unopened; CONFIRM002 fresh banks remain spent.

## Construction fixed without a task objective

Use exact official Supa BF16 bytes upcast to FP32, official Titan-Sparse FP32,
and TT FP32, with their weight digests pinned in geometry.py. Recover
`dT=Sparse−Supa` and realized compressed `dH=TT−Sparse`; full H donor is absent.
H-alone is `FP32(Supa + TT − Sparse)`, not a claim to reconstruct the unrounded
full donor. Requalification is necessary.

Freeze layer14 self_attn.o_proj, shape1024×2048. Extract the top **output**
rank4 singular subspace of each delta using one fixed randomized range finder
(8 oversamples, one power step; seeds20261001/2). In their union B (rank≤8),
form orthogonal projectors PT and PH, then

`K=PT PH−PH PT; R=(I−0.05K)^−1(I+0.05K)`.

Apply `h'=h+B(R−I)Bᵀh` once at block14 output on every token. Weights downstream
stay fixed. There is no d×d projector, parameter-space matrix, optimizer or
data-dependent coefficient search. Projector commutators satisfy spectral
norm≤.5; the rotation is at most approximately .05 radians. Commuting subspaces
produce identity and stop with `STOP_NO_GEOMETRIC_INTERACTION`.

This hook is linear. Any nonlinear output interaction arises in subsequent
network computation. A consistent change of coordinates in B leaves the
operator identical; compensating every network coordinate change would leave
behavior identical. This uncompensated surgery can change behavior, but neither
geometry nor K's sign identifies a useful third-skill direction. It may only
recombine a latent ability. Genuine generalization needs held-out task-structure
gains beyond controls. Weight orthogonality proves neither functional
independence nor preservation.

The finite Cayley transform needs no iterative solver. Deep equilibrium models
find a fixed point in a designed, trained recurrent system; that fact supplies
no missing semantic objective or skill generator here. No DEQ solver is added.
Background: [Bai et al.](https://arxiv.org/abs/1909.01377) and
[Turner et al.](https://arxiv.org/abs/2308.10248) motivate distinct equilibrium
and activation-steering mechanisms; neither establishes this construction.

## Controls and tiny stages

Eight arms: BASE, T, H-alone, TH ordinary existing two-skill combination,
TH+IDENTITY, TH+ROTATION, TH+RANDOM, TH+PROMPT. TH is a comparator/host, not the
proposed construction. All arms execute the same compact hook; its change
matrix is zero except ROTATION/RANDOM. RANDOM uses a frozen Haar union basis
and the same R: operator spectrum, norm, rank, and multiplication count match.
Actual activation displacement need not match; log its norm ratio. An inert
random control alone cannot prove semantic specificity.

1. CPU fixtures validate algebra, coordinate invariance, active negative
   canaries, objective-free determinism, dataset oracles/membership and budgets.
   CPU preparation reads three pinned checkpoints but calls no model.
2. Native mechanics checks padded vs unpadded greedy first4 tokens on2 separate
   DEVELOPMENT fixtures (16 forwards). A mismatch stops before screening.
3. SCREEN has8 new rows:2 familiar T contracts,2 answerability diagnostics,
   3 dependent transition-answerability tasks and1 known-state chain control.
   Membership and gold come from
   benchmark.py; they are evaluation only. Before intervention arms, each
   parent must exceed BASE on its2 diagnostic rows and get≥1 strictly right,
   with zero BASE-source-correct damage. Parent rows run first. Otherwise
   `STOP_SCREEN_NO_PARENT_SIGNAL`; small n proves no general competence.
4. A simple independent conjunction is a separate development instrument;
   structurally dependent transfer requires reasoning whether unknown or
   contradictory evidence survives a chain or is erased by reset. It is not
   merely answering two parent questions. Before any gain claim, this conjunction
   control and expanded family retention must run under a new concrete budget.

The tiny screen is a runnable falsifier and can stop this realization. It
cannot confirm an intelligence or third-skill claim. A useful screening signal
requires ROTATION above BASE, either parent, TH, IDENTITY, RANDOM and PROMPT on
primary strict scores of the3 structurally dependent rows
(semantic diagnostics reported separately), at least one net new correct third
row, zero gross source-correct parent damage,
and identity text parity. A gain confined to the known-state chain control
cannot pass. Gains that merely restore a solo parent's ability are interference
repair, not a third capability. Report all paired gains/losses and exact uncertainty;
with n3 dependent rows even perfect discordance cannot meet ordinary p<.05.
Zero result rejects only this fixed layer/rank/angle. No coefficient, layer,
rank, sign or seed adaptation is permitted from SCREEN outcomes.

BENCHMARK_MATRIX.md registers all12 requested families, sources, membership,
grading, n and failure limits. Unavailable panels are unmeasured. Roleplay and
open-ended language need independent calibrated human/rubric grading; no paid
grader calls. Long context requires a separate memory qualification. Broad
retention and confirmation are gates, not inferred from parent scores.

## Prospective hardware and compute ceiling

ASUS TUF A16, Ryzen9 7940HX,31.218GiB usable host RAM,RTX4070 Laptop8GiB.
Preload requires host free≥5GiB, disk free≥6GiB, GPU free≥6GiB, exclusive shared
lease, readable ownership, and no foreign research/model process≥512MiB RSS.
Active reserves: host≥1.5GiB/GPU≥.5GiB (existing MQ reserves). No other job may
be stopped. Every native stage rechecks resources and jobs/lease before load.

One model load in total; in-memory source weights replace one tensor at a time.
Greedy, FP32,eager,TF32off,batch1,thinking disabled. Fixed256 input slots (reject
overlength, no truncation) and24 output slots per screen generation; only text
before firstEOS is scored, remaining EOS slots equalize compute. At most64
screen generations,1600 forwards,20000 scheduled token positions including
padding; full screen+mechanics uses1552 forwards and at most18892 token positions.
No historical48-token comparability claim is made for this changed budget.
Cache/dtype/padding mechanics must qualify first.

Hard wall≤600seconds includes startup, hashing, load and screen. The worker has
cooperative counters but deliberately implements no Windows launcher or
containment repair. Execution requires the other task's independently qualified
live-worker Job/supervisor verifier and pinned receipt. A self-declared receipt
or arbitrary Job membership cannot qualify containment. No owner verifier is
available in this lane; standalone `run` therefore fails closed. The callable
worker body can be integrated by the owning lane after independent qualification.
No native call without both hardware and that gate, and a supervisor-pinned
FREEZE.json digest covering code, geometry and benchmark membership.
Default worker command is preflight only. Current native stages are blocked;
no large suite, long training, fullE1, paid compute, protected edits, checkpoint
overwrites, push/merge/deploy, global setting changes or OS changes are allowed.

Requested task settings remain GPT-6.1 Sol / Ultra / Standard (no fast). This
worker does not inspect or change account/global settings; effective routing
is controlled by the delegating platform and is not independently verified here.
