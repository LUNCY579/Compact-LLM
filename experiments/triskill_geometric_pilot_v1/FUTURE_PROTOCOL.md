# Prospective confirmation proposal; execution not authorized

The eight-row screen is a mechanical/exploratory gate. Three dependent endpoint
wins without losses has exact two-sided McNemar p=0.25. Two examples per parent
cannot establish broad retention. This proposal is not Stage B, a launched
expansion, or permission for additional model calls.

A separately approved confirmation would freeze 512 independent generated
source worlds and all operator, layer, rank, angle, seeds, stop, inference,
membership and scoring choices before any confirmation C outcomes. No C-labelled
training, gradients or parameter selection belongs to the primary construction.
If choices are adapted from development C outcomes, register that adaptation
arm separately and use fresh confirmation worlds. Global training/spent-bank
exclusion receipts must be obtained without reopening protected banks. Until
then membership is local-only and cannot support a protected held-out claim.

The primary endpoint is semantic correctness for transition-conditioned
answering/abstention, tested against BASE, T, H, TH, IDENTITY, RANDOM, PROMPT and
REVERSE. Require a beneficial two-sided exact paired McNemar rejection against
every comparator, family alpha=.05 and per-comparison alpha=.00625. Record all
2x2 tables and multiplicity-adjusted effect intervals; a lower simultaneous
paired-effect bound must exceed zero. Require strict-format retention separately.
The power calculation below addresses the paired rejection requirement only;
the additional confidence-bound and retention requirements may reduce overall
pass power. It does not promise 90% power for that entire compound gate.

| Assumed gain/loss probabilities against each comparator | N=256 power | N=384 power | N=512 power | N=512 all-eight lower bound |
|---|---:|---:|---:|---:|
| .15/.05 (net .10) | .789764 | .954181 | .992281 | .938251 |
| .10/.05 (net .05) | .203309 | .376323 | .540708 | 0 |
| .12/.04 (net .08) | .651657 | .882577 | .967119 | .736950 |

`future_power.py` sums `D ~ Bin(N, p_gain+p_loss)` and
`X | D ~ Bin(D, p_gain/(p_gain+p_loss))` over the exact beneficial rejection
region, using floating-point binomial summation. The all-eight bound is
`max(0, 1 - sum(1-power_i))`; no independence among comparator tests is assumed.
The rates are prospective assumptions, not pilot outcome estimates. Thus 512
supports a >=90% lower bound for simultaneous beneficial paired rejections only
under the .15/.05 assumption for every comparator. It is poorly powered for
smaller gains. No automatic increase or sequential coefficient/sample search is
permitted after seeing outcomes.

Worlds must balance answerable and unanswerable cases and include reset erasure,
surviving missing/conflicting states, known-value dependence, irrelevant missing
fields and evidence precedence. Fully known arithmetic and independent parent
conjunctions are reported separately. Meaning-preserving format variants are
repeated observations inside one source world; either preregister a single
primary variant or aggregate within-world semantic correctness. They cannot
increase N. Strict and semantic endpoints, source-correct damage, answerable
false abstention, unsupported answers and interference require separate reports.

Retention must be assessed per family rather than averaged. Proposed limits
are a multiplicity-adjusted upper noninferiority confidence bound of <=5
percentage points strict loss and an upper confidence bound of <=10%
source-correct damage, with familywise confidence at least 95% and the full
multiplicity and sampling plans frozen before execution. At least
48 rows per available retention panel is a planning floor, not adequate-power
proof for every limit. Before launch, calculate the actual noninferiority sample
size for each planned panel and budget all controls. A two-example parent gate
provides no such retention guarantee. An unavailable panel stays unmeasured.

All twelve families and their initial provenance, scorers, sample sizes and
pass/failure rules remain in [BENCHMARK_MATRIX.md](BENCHMARK_MATRIX.md), amended by
[FOLLOWUP_PREREG.md](FOLLOWUP_PREREG.md). Roleplaying/language need two blinded,
independent calibrated graders and written anchors before claims; no paid grader
calls. Long context needs separately authorized memory qualification at declared
lengths before accuracy testing. The current format fixtures are DEVELOPMENT
only, with native robustness unmeasured. No large suite is launched here.

Before any future native proposal can run, its owner must supply a qualified
containment/source handoff, a full compute/memory budget and fresh exclusive GPU
ownership. An initial screen retains the existing <=600-second ceiling; a
512-world confirmation will need a separately reviewed concrete budget and
explicit authorization. No model calls, Stage B consumption or training are
performed by the power calculator.
