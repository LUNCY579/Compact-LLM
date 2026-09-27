# Rineya composable-subspace stack — CSS-001

**Status: implemented and CPU/synthetic-tested. Native Rineya behavior and native training NOT RUN.**

An additive research lab on `research/rineya-composable-subspace-stack-v1`, based on
`research/titan-model-soup-mini-v1` at `816a1e8f4d1ab1c9d1ab847697d962a141521bac`.
Nothing in the source experiment, production model, shared checkpoints, or other grants is changed.

## What is implemented

| Module | Working capability | Evidence boundary |
|---|---|---|
| `core.py` | Double-sided base protection, spectral-consensus shared/private decomposition, HOSVD/HOOI CtM, SVD-after-merge, individual compression, norm controls and geometry | Finite matrix mechanics, not semantic protection |
| `protocol.py` | Frozen plans, explicit doses/tensors, ancestor-ledger disjointness, candidate hashes, single-attempt grants and shared host lease | Operator attestations are not cryptographic authorization or proof of complete ancestry |
| `artifacts.py` | Pinned mini-soup compatibility bridge and one-artifact FP32 safetensors writer; all unselected tensors remain base weights | Tiny writer round trips tested; real endpoint bridge and Windows memory behavior unqualified |
| `routing.py` | CAL-trained task-prototype routing, input-only decisions, fallback, oracle diagnostics, live additive Linear skill-bank hooks | Inspired by LORAUTER, not its full published algorithm |
| `training.py` | Functional JVP qualification, curvature-surrogate tangent teacher, nonlinear hidden-state distillation, matched ordinary-training control, all three skill families through one merger | Working generic PyTorch APIs; no native Qwen3 backend qualification or pretrained optimizer steps |
| `evaluation.py` | Matched per-row/domain retention and capability, parent bootstrap, loss interactions, factorial contrasts, frozen CAL selection and CHECK confirmation | Descriptive intervals, not a formal noninferiority or release claim |
| `native.py` | One-arm local-only FP32 raw prediction collector with reviewed runtime/ancestry/artifact bindings | Shipped plan refuses native execution |
| `demo.py` | Entire synthetic pipeline through artifacts, freeze, selection, CHECK, routing and tiny distillation | Synthetic outcomes are not Rineya results |

## Reproduce without model weights

Run from the repository root, in a separate mechanics environment. Do not upgrade a live research runtime.

```bash
python -m pytest -q experiments/rineya_composable_subspace_stack_v1/tests
python -m experiments.rineya_composable_subspace_stack_v1 init-plan --output css-plan.local.json
python -m experiments.rineya_composable_subspace_stack_v1 validate --plan css-plan.local.json
python -m experiments.rineya_composable_subspace_stack_v1 demo --output /absolute/new/css-demo
```

The output directory must not exist. The demo writes real tiny NPZ artifacts and a frozen
selection record, and reports geometry, synthetic scoring and CHECK acceptance in `report.json`. It never downloads or
loads a pretrained model. `VALIDATION.json` records the actually exercised environment.
`requirements-mechanics.txt` is not a qualified CUDA/Windows/Transformers stack.

## Registered comparison structure

P = protected-base projection; S = shared/private **reweighting**; C = CtM.
The eight `P0S0C0` through `P1S1C1` cells form the initial three-factor design. Three additional
method controls are `merge_then_compress`, `individual_then_merge`, and `full_norm_matched`.
That is **11 unique method arms**, not 11 training campaigns.

Leave-one-out aliases reuse existing cells: full `P1S1C1`, minus P `P0S1C1`, minus S `P1S0C1`,
minus C `P1S1C0`. They must not be counted twice as independent evidence.

With the default two donors, eight further controls are registered: FP32 full `base`, full donor
endpoints, scoped individual donors, scoped individual donors at their mixture doses, and scoped
`delta_sum`. Thus 19 configurations are available, **not an automatic 19-artifact build**.
Freeze a predeclared subset containing base and at least one candidate; missing arms remain
explicitly unscreened. Full donor endpoints and scoped donors answer different questions.

Default donors are Titan-Sparse and Opallo, coefficients 0.5/0.5. The illustrative first scope is
one explicit matrix, `model.layers.16.self_attn.q_proj.weight`, with rank 16. That layer/rank is a
bounded starting configuration, not an empirically selected optimum. Expand the explicit scope
only through a new frozen plan. Embeddings, heads, norms and all unselected weights are copied
from the base in method/scoped-control arms. This is not full-checkpoint soup by another name.

The native bridge admits only the inherited published common-base Supa/Titan/Titan-Sparse/Opallo
exports. Other Rineya family architectures require their own alignment/runtime qualification;
shape similarity is not permission to merge them. Stored BF16 deltas remain precision-stratified:
FP32 subtraction cannot recover information already rounded away.

## Scientific caveats that affect the implementation

Shared/private decomposition with unit gains is an identity. S therefore explicitly retains the
shared component and multiplies the residual by a frozen `private_gain` (default 0.5). A unit-gain
identity test prevents claiming a benefit from decomposition alone. This is a Share-inspired
post-hoc intervention, not the paper's complete evolving continual-learning procedure.

CtM learns its common basis before merging; coordinates use the original, unnormalized deltas.
For a fixed common basis, projection and linear summation commute. The meaningful comparison
is the *choice of basis* versus SVD of the already-merged matrix, not a magical ordering effect.
The implementation includes norm-weighted HOSVD/HOOI but not the paper's factor-space speedup.

Protected overlap is never labeled harmful without a functional experiment. Preserving selected
base singular triples does not guarantee semantic retention, continued top-k ordering, or useful
acquisition. Private residuals are not assumed mutually orthogonal. Low cosine is geometry only.

Read `PAPER_MAPPING.md` and `HOST_HANDOFF.md` before native use. No CI/native execution,
model-quality improvement, lossless retention, paper-exact reproduction or release is implied.
