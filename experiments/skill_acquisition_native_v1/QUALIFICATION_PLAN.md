# Native qualification contract (planning only)

Status: **UNAPPROVED; no native qualification runner exists yet.** This document
specifies the remaining implementation and evidence requirements. It authorizes
no optimizer update and supplies no measured prices.

Production admission currently requires a complete
`SKILL-ACQUISITION-NATIVE-QUALIFICATION-1` receipt from a separate qualification
run. A boolean `passed` or a CPU fixture is insufficient.

## Prerequisites

1. Finish the unchanged 450-group warmed S0 competence read and apply its
   registered gate. STOP_NO_SIGNAL stops the existing study path.
2. If it passes, complete the registered warm A/B pilot ladder and adopt its
   runtime/inventory receipt. Complete both independent enrollment processes
   with the registered zero-churn checks.
3. Freeze and commit the proposed native source. Finish actual tiny CPU gradient
   and checkpoint tests, with independent review of the runner/adapter boundary.
4. Register a distinct, bounded qualification scope and obtain authorization
   for it before loading the native model or applying disposable updates.

The qualification cannot silently inherit the warm or pilot optimizer. It must
use the proposed S1 recipe and a disposable model restored from the exact
deployed warm anchor. No qualification result selects a learning recipe or
reads donor DEV outcomes.

## Required checks

| Check | Required observation |
| --- | --- |
| Sampling | At least 16 actual native trajectories, each bound to the same policy within its boundary, with exact sample seeds and unchanged sampler. |
| Scoring | Native strict scoring agrees with the independent scorer; qualification fixtures and actual task rewards are reported separately. |
| Token likelihood | Every selected completion token has a generation raw log probability and a teacher-forced differentiable likelihood. Compare per token at the registered 1e-4 nat tolerance. |
| Sequence masks | Unequal lengths, a genuine generated EOS, and a length-capped trajectory without EOS. EOS contributes once; prompt and pad positions contribute zero. |
| Accumulation | The fixed denominator is 16. Compare accumulated gradients to an independently constructed reference, with one uniform and one mixed group. |
| Uniform boundary | C advances while Adam U and parameters/moments do not; test both empty and previously initialized Adam state. |
| Optimizer | At least two proved real Adam steps, finite gradients confined to the adopted inventory, and unchanged frozen parameters/buffers. |
| Checkpoint | At least two saves and one restore; resumed parameters, Adam state, and Python/NumPy/Torch CPU/CUDA RNG states agree bitwise with uninterrupted execution. |
| Rollback | Returned work after the chosen checkpoint stays charged and cannot be reused after the fresh restore fence. Prove C greater than U and an interrupted partial step. |
| Donor | Extract relative to the common warm anchor, independently bind the own start, and report deployed-rounding differences. No endpoint-versus-delta ambiguity. |
| Publication | A completion requires returned result publication and a matching invocation-end cost seal; preserve unsealed results on resume. |

Synthetic reward arrangements needed for optimizer/masking mechanics must be
explicitly labeled qualification fixtures. They cannot be reported as real
successes or substituted into production TRAIN scoring. Token arrays supplied
to EOS/cap fixtures must be labeled separately from actual sampled trajectories.
The production qualification schema and coverage checks must preserve this
distinction when the entry point is implemented.

## Measurements before production

Measure per-operation elapsed prices and conservative upper bounds for reads,
writes, identity hash attempts/bytes, and additional host RAM. The operation set
is defined by `runner.IO_KINDS` and `runner.HOST_STAGES`. The launch floor is at
least the adopted S0 floor and one GiB above the measured peak worker RSS,
including checkpoint/donor work. Every ongoing stage also requires one GiB
plus its declared additional transient allocation.

Keep the .78 CUDA allocator fraction, .95 reserved-fraction ceiling, .5 GiB
free VRAM guard, 14 GiB disk floor, two rolling checkpoints, and bounded chunk
IO. Measure hidden work inside reused model/tokenizer/checkpoint helpers; the
analytical model-load identity lower bound is insufficient as a price.

The receipt must bind the exact code bundle, anchor and own start, inventory,
update recipe, sampler/runtime/scorer, TRAIN plan, manifest and token census.
Production budgets include failed and rolled-back physical attempts, metadata
and lease wait time, final checkpoint/donor readback, and result publication.
Do not infer numeric ceilings from unrelated CE throughput.

The retained physical ledger also needs a whole-unit storage reservation.
64 MiB is a minimum metadata allowance, not a demonstrated bound for all
trajectories and operations. The current safe fallback reserves the entire
bound cumulative IO-write allowance as potential metadata, in addition to
28 bytes per trainable parameter for binary artifacts. This can exceed local
disk capacity. A smaller practical reservation requires a separately measured
metadata census and an enforced counter covering every metadata writer;
that tighter accounting remains unimplemented.

Qualification provenance is now checked in a bound scope, separate scope
authorization, native samples, mechanics token/reward fixtures, and a retained
cost ledger/seal. Return events must precede dependent teacher/scorer/restore
work. Gradient-reference tolerances are declared by the authorized scope;
this planning document does not invent them. These metadata checks depend on
a separately executed controlled qualifier and cannot prove execution alone.

## Remaining work

- Implement the separately scoped native qualifier without adding a production
  admission bypass.
- Make its counts and native-versus-fixture provenance auditable in the protocol.
- Execute it only after prerequisites, resource admission, and authorization.
- Build a concrete production packet using its measured limits; obtain the
  separate S1 decision required by the registered study.

Even a passed qualification and a completed donor do not establish skill
acquisition. Registered donor and composition/retention gates remain required.
