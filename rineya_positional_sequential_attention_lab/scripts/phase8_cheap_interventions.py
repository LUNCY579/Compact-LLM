"""Phase 8 — the first real intervention screen (brief sections 24-26, 32, 33, 37).

One question: **does a significance matrix improve long-context reasoning inside
ORDINARY parallel attention?** Everything here is inference-time and reversible; no
weight on disk is touched. This is the mandatory control for Phase 10: if PSA alone
recovers the middle, sequentiality contributed nothing and the study has to say so.

Arms screened (all on the calibration split, teacher-forced, gold NLL):

* ``baseline`` — no intervention.
* ``psa:logit`` — SPEC 5.3 Mode 1, ``lam in {0.25, 0.5, 1.0, 2.0}``.
* ``psa:value`` — Mode 2, ``alpha in {0.25, 0.5, 1.0}``.
* ``psa:head``  — Mode 3, head-output scale.
* ``cheap_bias`` — ``interventions.cheap.position_logit_correction`` driven by the
  Phase-4b bias matrix ``B``, ``lam in {0.25, 0.5, 0.75, 1.0}``.
* ``rope`` — the seven layerwise RoPE schedules. ``flat`` must be a bit-exact no-op and
  :func:`assert_flat_rope_is_noop` refuses to let the run start if it is not.

**Controls are mandatory (section 37).** Every significance-driven arm — that is every
PSA arm and every ``cheap_bias`` arm — is run beside four matched controls built by
``interventions.controls.all_controls_for``: ``uniform_middle_boost``,
``random_position_matrix``, ``random_head_assignment`` and ``sign_flipped``. Controls
are dose-matched: a control for ``lam = 0.5`` is itself run at ``lam = 0.5``. The
results table orders each treatment immediately ahead of its own controls, and the
summary nests the controls *inside* the treatment, so a reader cannot see a winner
without seeing what it had to beat.

**The selection rule is stated here, before the run, and copied verbatim into the
summary json** (:data:`SELECTION_RULE`). Primary metric is summed gold NLL (SPEC 13),
lower is better. An arm advances only if

1. its paired mean difference against ``baseline`` is negative and its BCa 95 % CI —
   bootstrapped over 10 000 resamples clustered on ``scenario_id`` — excludes zero, and
2. it beats **every** one of its four matched controls on the same paired test, in the
   same direction, with a CI that also excludes zero.

Everything is reported per family and per position as well as pooled, because Phase 3
measured that the positional deficit has a *different shape* in each family — a recency
gradient in ``exact_retrieval``, a true middle trough in ``multi_hop`` and
``state_constraint``, and a large range with no significant middle deficit in
``semantic_single_hop``. A single pooled number would hide an arm that fixes one family
and breaks another, which is precisely the failure mode Phase 3 warned Phase 8 about.

    PYTHONPATH=src python scripts/phase8_cheap_interventions.py --dry-run

DEVIATION: SPEC 11.3's "five-point grid (0.05, 0.25, 0.50, 0.75, 0.95)" does not exist
in the bank. Phase 2 built a seven-point grid at (0.02, 0.17, 0.33, 0.50, 0.67, 0.83,
0.98) and that is what is screened; ``--positions`` can subset it. The screening scale
(calibration split, 4K bucket) is as SPEC 11.3 requires.

DEVIATION: SPEC 12.3 says Mode 1 is expressible as an additive SDPA mask and therefore
"costs essentially nothing". The kernel as implemented (``hooks/registry.py``) takes the
eager chunked path whenever a logit bias is live, because that path is shared with
capture. The cost model here charges Mode-1 arms :data:`_MODE1_MULTIPLIER` forwards
instead of one; that multiplier is an **estimate**, not a measurement, and is flagged as
such in the cost print and in the summary json. Mode 2 and Mode 3 take the SDPA fast
path and are charged the measured explicit-mask cost instead.

DEVIATION: ``controls.all_controls_for`` returns nine controls, three of which are
*sequential arms*. Sequential arms are Phase 10's subject and cannot be applied to
ordinary parallel attention, so only the ``kind == "matrix"`` controls are runnable
here, and of those the four the brief names are run by default.
``--include-gate-controls`` adds ``frozen_gates`` and ``random_gates``. The set that was
actually run is recorded in the summary; an arm whose four mandatory controls did not
run is marked ``advances = false`` with reason ``controls_missing`` rather than being
quietly promoted.

DEVIATION: ``mode="head"`` has no dose grid in the brief, so it is run at a single
``lam = 1.0`` (:data:`PSA_MODE3_LAMS`) rather than at an invented sweep.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rineya_psa import paths  # noqa: E402
from rineya_psa.backend import efficient_attention, probe_backends  # noqa: E402
from rineya_psa.evaluation.runner import prepare_items, run_items  # noqa: E402
from rineya_psa.interventions.cheap import (  # noqa: E402
    ROPE_SCHEDULES,
    position_logit_correction,
    rope_scale_values,
    rope_scaling_schedule,
)
from rineya_psa.interventions.compute import (  # noqa: E402
    ComputeConfig,
    compute_metrics_row,
    intervention_flops,
    measure_compute,
    write_compute_metrics,
)
from rineya_psa.interventions.controls import (  # noqa: E402
    DEFAULT_CONTROL_SEED,
    all_controls_for,
)
from rineya_psa.interventions.psa import (  # noqa: E402
    SignificanceTable,
    build_psa_context,
    coerce_significance,
    standardize_significance,
)
from rineya_psa.loading import (  # noqa: E402
    dtype_is_approximation,
    free_model,
    load_model,
    load_tokenizer,
)
from rineya_psa.statistics.bootstrap import (  # noqa: E402
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    paired_bootstrap,
)
from rineya_psa.statistics.provenance import write_result  # noqa: E402
from rineya_psa.types import BenchmarkItem  # noqa: E402

# --------------------------------------------------------------------------------------
# The design, fixed before the run
# --------------------------------------------------------------------------------------

#: SPEC 5.3 Mode 1 doses, as the brief writes them.
PSA_MODE1_LAMS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

#: SPEC 5.3 Mode 2 doses ("alpha" in the brief; the library calls the knob ``lam``).
PSA_MODE2_ALPHAS: tuple[float, ...] = (0.25, 0.5, 1.0)

#: Mode 3 has no grid in the brief. One dose, not an invented sweep. See the docstring.
PSA_MODE3_LAMS: tuple[float, ...] = (1.0,)

#: Section 33A doses for ``cheap.position_logit_correction``.
CHEAP_BIAS_LAMS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

#: Section 33D schedules. ``flat`` leads because it is the no-op assertion.
ROPE_SCHEDULE_NAMES: tuple[str, ...] = (
    "flat",
    "increasing",
    "decreasing",
    "middle_hump",
    "late_hump",
    "middle_valley",
    "early_late",
)

if set(ROPE_SCHEDULE_NAMES) != set(ROPE_SCHEDULES):
    # A schedule silently added to or dropped from the library would otherwise change
    # what this screen covers without changing this file. Fail at import, not at hour 3.
    raise RuntimeError(
        f"cheap.ROPE_SCHEDULES is {sorted(ROPE_SCHEDULES)}, this screen expects "
        f"{sorted(ROPE_SCHEDULE_NAMES)}"
    )

#: The four the brief names. Every significance-driven arm runs all four, dose-matched.
MANDATORY_MATRIX_CONTROLS: tuple[str, ...] = (
    "uniform_middle_boost",
    "random_position_matrix",
    "random_head_assignment",
    "sign_flipped",
)

#: Section 37's two gate controls. Also matrix-kind, hence runnable here, but off by
#: default because each one doubles-and-a-half the control budget of every arm.
OPTIONAL_MATRIX_CONTROLS: tuple[str, ...] = ("frozen_gates", "random_gates")

ARM_GROUPS: tuple[str, ...] = ("baseline", "psa_logit", "psa_value", "psa_head",
                               "cheap_bias", "rope")

#: Written into the summary json verbatim. Stated before the run; not negotiable after.
SELECTION_RULE: dict[str, Any] = {
    "primary_metric": "gold_nll_summed_over_answer_tokens",
    "direction": "lower_is_better",
    "test": "paired bootstrap of mean(arm - baseline), BCa 95% CI",
    "clustering": "scenario_id",
    "n_resamples": DEFAULT_N_RESAMPLES,
    "advance_criteria": [
        "paired mean difference vs baseline is negative",
        "its 95% CI excludes zero",
        "the same paired test against EVERY matched control is negative with a CI "
        "that excludes zero",
        "all four mandatory controls actually ran at the same dose",
    ],
    "mandatory_controls": list(MANDATORY_MATRIX_CONTROLS),
    "reporting": "pooled, per family, and per decisive-evidence position; the pooled "
                 "number is never quoted alone (Phase 3: the deficit is family-shaped)",
    "secondary_metrics": ["answer_margin", "strict accuracy (reported, never selected on)"],
}

# --------------------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------------------

# Measured on this card, batch 1, backend pinned (SPEC 12.1 / rineya_psa.backend).
_FWD_SECONDS: Mapping[str, float] = {"2k": 0.12, "4k": 0.29, "8k": 0.83,
                                     "16k": 2.27, "32k": 7.06}

# A live psa_session registers ``eager_mask``, so SDPA gets an explicit [B,1,T,T] float
# mask instead of the is_causal shortcut. SPEC 12.3 measured 0.36 s vs 0.288 s at 4K.
_PSA_SESSION_MULTIPLIER = 1.25

# ESTIMATE, not a measurement on this card: a live Mode-1 bias forces the eager chunked
# kernel, which materialises [B, H_q, chunk, T] scores per chunk and takes a float32
# softmax over them, instead of calling SDPA once. Two independent reasons for ~6x at 4K:
# a CPU smoke run of this script measured 7.8x on a 2-layer model, and the fp32 softmax
# alone moves ~30 GB at 4K across 28 layers, which is ~1 s of bandwidth against a 0.29 s
# SDPA baseline. Flagged as an estimate everywhere it is used; the number that settles it
# is the wall_clock_s column of compute_metrics.csv after the real run.
_MODE1_MULTIPLIER = 6.0

# The RoPE hook recomputes cos/sin once per (scale, dtype) per forward and caches it.
_ROPE_MULTIPLIER = 1.05

_ARM_MULTIPLIER: Mapping[str, float] = {
    "baseline": 1.0,
    "psa_logit": _MODE1_MULTIPLIER,
    "psa_value": _PSA_SESSION_MULTIPLIER,
    "psa_head": _PSA_SESSION_MULTIPLIER,
    "cheap_bias": _MODE1_MULTIPLIER,
    "rope": _ROPE_MULTIPLIER,
}

GPU_HOUR_GUARD_SECONDS = 2 * 3600


# --------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Arm:
    """One condition. ``eq=False``: ``payload`` may carry a numpy array (see controls)."""

    label: str
    group: str
    kind: str  # "baseline" | "psa" | "cheap_bias" | "rope"
    mode: str | None = None  # SPEC 5.3 mode name for the psa kinds
    lam: float | None = None
    payload: Any = None  # SignificanceTable | schedule name | None
    matrix_name: str | None = None
    control_of: str | None = None  # the treatment label this arm is the control for
    control_name: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_control(self) -> bool:
        return self.control_of is not None

    @property
    def significance_driven(self) -> bool:
        return self.kind in {"psa", "cheap_bias"}

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "group": self.group,
            "kind": self.kind,
            "mode": self.mode,
            "lam": self.lam,
            "matrix": self.matrix_name,
            "is_control": self.is_control,
            "control_of": self.control_of,
            "control_name": self.control_name,
            "params": dict(self.params),
        }


def _control_tables(
    table: SignificanceTable, arch: Any, seed: int, wanted: Sequence[str]
) -> dict[str, SignificanceTable]:
    """The matrix-kind controls of section 37, keyed by name.

    ``all_controls_for`` is called rather than the individual builders so that a control
    cannot be silently dropped: it runs ``assert_controls_complete`` internally, which
    fails if the library ever loses one. The arm-kind controls it also returns belong to
    Phase 10 and are filtered out here (see the module docstring).
    """
    controls = all_controls_for(table, seed, arch=arch)
    out: dict[str, SignificanceTable] = {}
    for name in wanted:
        control = controls[name]
        if control.kind != "matrix":
            raise ValueError(
                f"control {name!r} is kind {control.kind!r}; parallel-attention arms can "
                "only take matrix controls"
            )
        out[name] = control.payload
    return out


def _psa_arms(
    table: SignificanceTable,
    arch: Any,
    *,
    mode: str,
    group: str,
    lams: Sequence[float],
    controls: Mapping[str, SignificanceTable],
) -> list[Arm]:
    """One treatment per dose, each immediately followed by its dose-matched controls."""
    arms: list[Arm] = []
    for lam in lams:
        label = f"psa:{mode}:lam{lam:g}"
        arms.append(Arm(label=label, group=group, kind="psa", mode=mode, lam=float(lam),
                        payload=table, matrix_name=table.name))
        for name, control_table in controls.items():
            arms.append(Arm(
                label=f"{label}|ctrl:{name}",
                group=group, kind="psa", mode=mode, lam=float(lam),
                payload=control_table, matrix_name=control_table.name,
                control_of=label, control_name=name,
            ))
    return arms


def _cheap_bias_arms(
    table: SignificanceTable,
    *,
    lams: Sequence[float],
    controls: Mapping[str, SignificanceTable],
) -> list[Arm]:
    arms: list[Arm] = []
    for lam in lams:
        label = f"cheap_bias:lam{lam:g}"
        arms.append(Arm(label=label, group="cheap_bias", kind="cheap_bias", mode="logit",
                        lam=float(lam), payload=table, matrix_name=table.name))
        for name, control_table in controls.items():
            arms.append(Arm(
                label=f"{label}|ctrl:{name}",
                group="cheap_bias", kind="cheap_bias", mode="logit", lam=float(lam),
                payload=control_table, matrix_name=control_table.name,
                control_of=label, control_name=name,
            ))
    return arms


def build_arms(
    *,
    arch: Any,
    matrix: SignificanceTable | None,
    bias: SignificanceTable | None,
    groups: Sequence[str],
    seed: int,
    control_names: Sequence[str],
    rope_lo: float,
    rope_hi: float,
) -> list[Arm]:
    """The full screen, ordered so every treatment is followed by its own controls."""
    arms: list[Arm] = [Arm(label="baseline", group="baseline", kind="baseline")]

    if matrix is not None:
        controls = _control_tables(matrix, arch, seed, control_names)
        plan = (
            ("psa_logit", "logit", PSA_MODE1_LAMS),
            ("psa_value", "value", PSA_MODE2_ALPHAS),
            ("psa_head", "head", PSA_MODE3_LAMS),
        )
        for group, mode, lams in plan:
            if group in groups:
                arms += _psa_arms(matrix, arch, mode=mode, group=group, lams=lams,
                                  controls=controls)

    if bias is not None and "cheap_bias" in groups:
        # The controls for the bias arm are built from the bias matrix and then put
        # through the same centre-and-peak-normalise the treatment went through, so that
        # "lam" means the same number of logits for the control as for the treatment.
        raw_controls = _control_tables(bias, arch, seed, control_names)
        controls = {k: standardize_significance(v) for k, v in raw_controls.items()}
        arms += _cheap_bias_arms(bias, lams=CHEAP_BIAS_LAMS, controls=controls)

    if "rope" in groups:
        for schedule in ROPE_SCHEDULE_NAMES:
            arms.append(Arm(
                label=f"rope:{schedule}", group="rope", kind="rope", payload=schedule,
                params={"lo": float(rope_lo), "hi": float(rope_hi),
                        "scales": list(rope_scale_values(schedule, arch.num_layers,
                                                         lo=rope_lo, hi=rope_hi))},
            ))
    return [a for a in arms if a.group in groups or a.group == "baseline"]


@contextlib.contextmanager
def arm_session(arm: Arm, model: Any, arch: Any) -> Iterator[Any]:
    """Enter ``arm``'s intervention and yield the ``PSAContext`` ``run_items`` needs.

    Two shapes of intervention live behind one interface. The PSA modes are a context
    *object* that ``run_items`` installs itself; the cheap correction and the RoPE
    schedules are context *managers* that wrap the model, and those yield ``None`` so
    ``run_items`` does not open a second, nested ``psa_session`` (which would raise).
    """
    if arm.kind == "baseline":
        yield None
    elif arm.kind == "psa":
        yield build_psa_context(arm.payload, arm.mode, arm.lam, arch=arch)
    elif arm.kind == "cheap_bias":
        table: SignificanceTable = arm.payload
        with position_logit_correction(model, table.values, arm.lam,
                                       binning=table.resolve_binning()):
            yield None
    elif arm.kind == "rope":
        with rope_scaling_schedule(model, str(arm.payload),
                                   lo=float(arm.params["lo"]), hi=float(arm.params["hi"])):
            yield None
    else:
        raise ValueError(f"unknown arm kind {arm.kind!r}")


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def load_split(split: str) -> list[BenchmarkItem]:
    path = paths.DATA / split / "position_examples.jsonl"
    if not path.exists():
        raise SystemExit(f"no benchmark found at {path}; run the Phase 2 generator first")
    return [BenchmarkItem.from_json(json.loads(line)) for line in path.open(encoding="utf-8")]


def select_items(
    items: Sequence[BenchmarkItem],
    *,
    length_bucket: str,
    families: Sequence[str] | None,
    positions: Sequence[float] | None,
    local_position: str,
    scenario_limit: int | None,
    limit: int | None,
) -> list[BenchmarkItem]:
    """The screening set. Scenario-clustered subsampling, never item-clustered.

    ``--scenario-limit`` drops whole scenarios because the bootstrap resamples scenarios:
    keeping half the positional variants of a scenario would shrink the item count
    without buying any independent units, and would unbalance the position axis inside
    the surviving clusters. ``--limit`` is the blunt cap the Phase 3 driver has, kept for
    smoke runs, and it *is* item-clustered — which is why it marks the run partial.
    """
    keep = [it for it in items if it.length_bucket == length_bucket]
    if local_position != "any":
        # Phase 3: the local_position_sweep grid emits p in {0.02, 0.50, 0.98} only, so
        # pooling local positions puts a step at exactly the three positions that define
        # the edge-middle gap.
        keep = [it for it in keep if it.local_position == local_position]
    if families:
        wanted = set(families)
        keep = [it for it in keep if it.family in wanted]
    if positions:
        grid = {round(float(p), 2) for p in positions}
        keep = [it for it in keep if round(float(it.meta["primary_target"]), 2) in grid]
    if scenario_limit is not None:
        ordered = sorted({it.scenario_id for it in keep})[: int(scenario_limit)]
        chosen = set(ordered)
        keep = [it for it in keep if it.scenario_id in chosen]
    if limit is not None:
        keep = keep[: int(limit)]
    return keep


def load_significance(path: Path, arch: Any, *, negate: bool) -> SignificanceTable:
    """Read a Phase 4/4b matrix and hand back the table the intervention modes take.

    ``importance()``, not ``values``: a matrix measured on ``gold_nll`` stores
    ``Delta = L_normal - L_perturbed``, so an *important* cell has a **negative** value.
    ``OBJECTIVE_SIGN`` re-signs it so that larger always means "this cell matters", which
    is the orientation every mode in ``psa.py`` assumes.
    """
    from rineya_psa.hooks.binning import binning_from_describe
    from rineya_psa.significance.matrices import SignificanceMatrix

    sm = SignificanceMatrix.load(path)
    binning = binning_from_describe(dict(sm.binning_desc))
    if binning.is_query_dependent:
        # SPEC 12.3: a distance-binned Mode 1 needs a full [1, H, T, T] mask, which is
        # affordable at 2K and not at 4K. Refuse rather than silently thrash.
        raise SystemExit(
            f"{path.name} was measured on {binning.name}, which is query-dependent; "
            "SPEC 12.3 restricts distance-binned Mode 1/2 to T <= 2048 and this screen "
            "runs at 4K. Re-measure on an AbsoluteBinning."
        )
    values = sm.importance()
    if negate:
        values = -values
    table = coerce_significance(
        values, binning=binning, name=path.stem, n_kv_heads=arch.num_kv_heads
    )
    table.assert_matches(arch)
    return table


def load_bias(path: Path, arch: Any, *, field_name: str, sign: str) -> SignificanceTable:
    """Read the Phase-4b positional bias ``B`` and turn it into a correction.

    ``qk_logit_mean`` is the default field because ``position_logit_correction`` adds its
    table to the *pre-softmax* logits, and that field is already in those units;
    ``attn_by_bin`` is a probability and adding it to a logit would be a category error.

    ``sign="correct"`` negates the centred profile, which is what makes this a
    correction: ``B`` low in the middle means the model under-attends there, so the term
    that repairs it is ``-(B - mean(B))``, not ``+B``. ``sign="raw"`` amplifies the
    measured bias instead and exists so the dose-response can be run in both directions.

    The profile is centred and divided by its global peak (``standardize_significance``),
    so ``lam`` is exactly the number of logits added at the most-corrected bin — an
    interpretable dose rather than an arbitrary one.
    """
    from rineya_psa.hooks.binning import binning_from_describe
    from rineya_psa.significance.bias import PositionalBias

    pb = PositionalBias.load(path)
    sm = pb.to_significance_matrix(field_name, kv_heads=True)
    binning = binning_from_describe(dict(sm.binning_desc))
    if binning.is_query_dependent:
        raise SystemExit(
            f"{path.name} was measured on {binning.name}; see SPEC 12.3, Mode 1 needs an "
            "absolute binning at 4K"
        )
    table = coerce_significance(
        sm.values, binning=binning, name=f"{path.stem}:{field_name}",
        n_kv_heads=arch.num_kv_heads,
    )
    table.assert_matches(arch)
    table = standardize_significance(table)
    if sign == "correct":
        table = table.with_values(-table.values, name=f"{table.name}:corrective")
    return table


# --------------------------------------------------------------------------------------
# The bit-exactness assertion for the flat RoPE schedule
# --------------------------------------------------------------------------------------


def assert_flat_rope_is_noop(
    model: Any, input_ids: Sequence[int], *, on_cuda: bool, max_tokens: int = 512
) -> dict[str, Any]:
    """``rope_scaling_schedule(model, "flat")`` must change nothing. Prove it, twice.

    The primary check is on the *mechanism*: the ``(cos, sin)`` pair the flat schedule
    hands layer 0 must be bit-identical to the pair the model would have computed. That
    is deterministic — ``inv_freq * 1.0`` is ``inv_freq`` to the last bit — so it is a
    hard assertion.

    The secondary check is on the logits, and it cannot be a plain bit-equality: SPEC 10
    records that SDPA kernels are not bit-deterministic on this card. So the model's own
    run-to-run jitter is measured first, by running the baseline twice, and the flat
    schedule is required to differ from the baseline by no more than that. A schedule
    that is genuinely a no-op cannot exceed the noise floor; one that is not, will.
    """
    import torch

    from rineya_psa.hooks.registry import attention_modules

    ids = torch.tensor([list(input_ids)[:max_tokens]], dtype=torch.long,
                       device=next(model.parameters()).device)
    site = min(attention_modules(model), key=lambda m: int(m.layer_idx))
    grabbed: dict[str, Any] = {}

    def grab(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        grabbed["pe"] = kwargs.get("position_embeddings")
        return args, kwargs

    def capture(inside_flat: bool) -> tuple[Any, Any, Any]:
        # The hook must be registered INSIDE the flat session: pre-hooks fire in
        # registration order, so a hook registered first would read the untouched
        # kwargs and the comparison would be vacuously true.
        stack = contextlib.ExitStack()
        with stack:
            if inside_flat:
                stack.enter_context(rope_scaling_schedule(model, "flat"))
            handle = site.register_forward_pre_hook(grab, with_kwargs=True)
            stack.callback(handle.remove)
            with torch.no_grad(), efficient_attention(active=on_cuda):
                out = model(ids, use_cache=False, logits_to_keep=1)
        cos, sin = grabbed.pop("pe")
        return cos.detach().clone(), sin.detach().clone(), out.logits.detach().clone()

    cos_a, sin_a, logits_a = capture(False)
    _, _, logits_jitter = capture(False)
    cos_b, sin_b, logits_b = capture(True)

    cos_exact = bool(torch.equal(cos_a, cos_b))
    sin_exact = bool(torch.equal(sin_a, sin_b))
    if not (cos_exact and sin_exact):
        raise SystemExit(
            "flat RoPE schedule is NOT a bit-exact no-op: the cos/sin handed to layer 0 "
            f"differ (cos identical={cos_exact}, sin identical={sin_exact}). "
            "Refusing to run a screen whose null schedule is not null."
        )

    jitter = float((logits_a - logits_jitter).abs().max())
    delta = float((logits_a - logits_b).abs().max())
    if delta > jitter:
        raise SystemExit(
            f"flat RoPE schedule moved the logits by {delta:.3e}, more than the model's "
            f"own run-to-run jitter of {jitter:.3e}. That is a real change, not kernel "
            "non-determinism. Refusing to run."
        )
    return {
        "cos_bit_exact": cos_exact,
        "sin_bit_exact": sin_exact,
        "logit_max_abs_diff": delta,
        "baseline_run_to_run_jitter": jitter,
        "n_tokens": int(ids.shape[1]),
    }


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def group_label(row: Mapping[str, Any]) -> str:
    """Phase 3's family axis. ``multi_hop`` is two different tasks and is split."""
    if row["family"] == "multi_hop":
        return "multi_hop_guided" if row.get("guided") else "multi_hop_hard"
    return str(row["family"])


def _paired(
    base: Mapping[str, Mapping[str, Any]],
    other: Mapping[str, Mapping[str, Any]],
    item_ids: Sequence[str],
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any] | None:
    """``mean(other - base)`` over the items both arms scored. Negative = better."""
    a: list[float] = []
    b: list[float] = []
    clusters: list[str] = []
    for item_id in item_ids:
        left, right = base.get(item_id), other.get(item_id)
        if left is None or right is None:
            continue
        if left["gold_nll"] is None or right["gold_nll"] is None:
            continue
        a.append(float(left["gold_nll"]))
        b.append(float(right["gold_nll"]))
        clusters.append(str(left["scenario_id"]))
    if len(a) < 2 or len({*clusters}) < 2:
        return None
    res = paired_bootstrap(a, b, clusters, n=n_resamples, seed=seed)
    out = res.to_json()
    out["improves"] = bool(res.estimate < 0.0)
    out["significant"] = bool(res.significant)
    return out


def analyse(
    records: Sequence[Mapping[str, Any]],
    arms: Sequence[Arm],
    *,
    seed: int,
    n_resamples: int,
    control_names: Sequence[str],
) -> dict[str, Any]:
    """Per-arm effects against baseline and against every matched control.

    Structured so the controls are *inside* the arm they control. There is no code path
    in this function that produces a winner without also producing what it beat.
    """
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_arm.setdefault(str(row["arm"]), {})[str(row["item_id"])] = row

    base = by_arm.get("baseline", {})
    if not base:
        raise ValueError("no baseline rows; every effect in this phase is a delta")
    all_ids = sorted(base)
    families = sorted({group_label(r) for r in base.values()})
    positions = sorted({round(float(r["primary_target"]), 2) for r in base.values()})

    def subset(ids: Sequence[str], predicate: Any) -> list[str]:
        return [i for i in ids if predicate(base[i])]

    treatments = [a for a in arms if not a.is_control]
    control_lookup: dict[str, list[Arm]] = {}
    for arm in arms:
        if arm.is_control:
            control_lookup.setdefault(str(arm.control_of), []).append(arm)

    out: dict[str, Any] = {}
    for arm in treatments:
        if arm.label == "baseline" or arm.label not in by_arm:
            continue
        rows = by_arm[arm.label]
        overall = _paired(base, rows, all_ids, seed=seed, n_resamples=n_resamples)
        entry: dict[str, Any] = {
            "arm": arm.describe(),
            "n_items": len(rows),
            "vs_baseline": overall,
            "by_family": {},
            "by_position": {},
            "controls": {},
        }
        for fam in families:
            ids = subset(all_ids, lambda r, f=fam: group_label(r) == f)
            entry["by_family"][fam] = _paired(base, rows, ids, seed=seed,
                                              n_resamples=n_resamples)
        for pos in positions:
            ids = subset(all_ids, lambda r, p=pos: round(float(r["primary_target"]), 2) == p)
            entry["by_position"][f"{pos:g}"] = _paired(base, rows, ids, seed=seed,
                                                       n_resamples=n_resamples)

        beaten: list[str] = []
        for control_arm in control_lookup.get(arm.label, []):
            control_rows = by_arm.get(control_arm.label)
            name = str(control_arm.control_name)
            if not control_rows:
                entry["controls"][name] = {"ran": False}
                continue
            vs_baseline = _paired(base, control_rows, all_ids, seed=seed,
                                  n_resamples=n_resamples)
            vs_control = _paired(control_rows, rows, all_ids, seed=seed,
                                 n_resamples=n_resamples)
            wins = bool(vs_control and vs_control["improves"] and vs_control["significant"])
            entry["controls"][name] = {
                "ran": True,
                "label": control_arm.label,
                "control_vs_baseline": vs_baseline,
                "arm_vs_control": vs_control,
                "arm_beats_control": wins,
            }
            if wins:
                beaten.append(name)

        missing = [c for c in control_names
                   if not entry["controls"].get(c, {}).get("ran", False)]
        mandatory_missing = [c for c in MANDATORY_MATRIX_CONTROLS if c in missing]
        helps = bool(overall and overall["improves"] and overall["significant"])
        beats_all = bool(entry["controls"]) and all(
            v.get("arm_beats_control", False) for v in entry["controls"].values()
        )
        if arm.significance_driven and mandatory_missing:
            advances, reason = False, f"controls_missing:{','.join(mandatory_missing)}"
        elif not helps:
            advances, reason = False, "no_significant_improvement_over_baseline"
        elif arm.significance_driven and not beats_all:
            advances, reason = False, "did_not_beat_every_matched_control"
        else:
            advances, reason = True, "passed"
        entry["controls_beaten"] = beaten
        entry["controls_missing"] = missing
        entry["advances"] = advances
        entry["advance_reason"] = reason
        out[arm.label] = entry
    return out


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def estimate_seconds(items: Sequence[BenchmarkItem], arms: Sequence[Arm]) -> dict[str, Any]:
    prefill = sum(_FWD_SECONDS.get(it.length_bucket, 0.29) for it in items)
    per_group: dict[str, float] = {}
    for arm in arms:
        per_group[arm.group] = per_group.get(arm.group, 0.0) + prefill * _ARM_MULTIPLIER[arm.group]
    return {
        "total_seconds": float(sum(per_group.values())),
        "seconds_by_group": {k: round(v, 1) for k, v in sorted(per_group.items())},
        "arms_by_group": {
            g: sum(1 for a in arms if a.group == g) for g in sorted({a.group for a in arms})
        },
        "one_forward_pass_seconds": round(prefill, 1),
        "multipliers": dict(_ARM_MULTIPLIER),
        "multiplier_note": (
            f"psa_logit and cheap_bias are charged {_MODE1_MULTIPLIER}x, an ESTIMATE for "
            "the eager chunked kernel path a live Mode-1 bias forces; the other "
            "multipliers are measured (SPEC 12.1/12.3)"
        ),
    }


def _design_map(items: Sequence[BenchmarkItem]) -> dict[str, dict[str, Any]]:
    """Design factors ``RunRecord`` does not carry, joined back by ``item_id``.

    Same join Phase 3 does, and for the same reason: ``primary_target`` is the position
    of the *decisive* evidence sentence, and ``target_positions[0]`` is not it for
    ``multi_hop`` or ``state_constraint``.
    """
    return {
        it.item_id: {
            "local_position": it.local_position,
            "grid": it.meta.get("grid"),
            "guided": bool((it.meta.get("example") or {}).get("guided", False)),
            "primary_target": float(it.meta["primary_target"]),
            "decisive_index": int(it.meta["example"]["decisive_index"]),
        }
        for it in items
    }


#: SPEC 1's shape. Used only to plan a ``--dry-run``, which must not load a model.
PANEL_ARCH_FIELDS: dict[str, int] = {
    "num_layers": 28, "hidden_size": 1024, "num_q_heads": 16, "num_kv_heads": 8,
    "head_dim": 128, "vocab_size": 151936,
}


def panel_arch() -> Any:
    from rineya_psa.types import Arch

    return Arch(**PANEL_ARCH_FIELDS)


def plan(args: argparse.Namespace, arch: Any, control_names: Sequence[str],
         groups: Sequence[str]) -> list[Arm]:
    """Load the matrices ``arch`` demands and lay out the arms. No model is touched."""
    matrix = (
        load_significance(Path(args.matrix), arch, negate=args.negate_matrix)
        if args.matrix else None
    )
    bias = (
        load_bias(Path(args.bias_matrix), arch, field_name=args.bias_field,
                  sign=args.bias_sign)
        if args.bias_matrix else None
    )
    if matrix is None and {"psa_logit", "psa_value", "psa_head"} & set(groups):
        raise SystemExit(
            "--matrix is required for the PSA arms: this phase screens a significance "
            "matrix, and without one there is nothing to screen. Run Phase 4/5 first, or "
            "restrict the screen with --arms rope (or --arms cheap_bias with "
            "--bias-matrix)."
        )
    if bias is None and "cheap_bias" in groups:
        raise SystemExit(
            "--bias-matrix is required for the cheap_bias arms (Phase 4b's B). Drop "
            "cheap_bias from --arms or supply the matrix."
        )
    return build_arms(arch=arch, matrix=matrix, bias=bias, groups=groups, seed=args.seed,
                      control_names=control_names, rope_lo=args.rope_lo,
                      rope_hi=args.rope_hi)


def print_estimate(items: Sequence[BenchmarkItem], arms: Sequence[Arm]) -> dict[str, Any]:
    est = estimate_seconds(items, arms)
    n_treat = sum(1 for a in arms if not a.is_control)
    print(f"{len(items)} items x {len(arms)} arms "
          f"({n_treat} treatments + {len(arms) - n_treat} controls); "
          f"estimated {est['total_seconds'] / 3600:.2f} GPU-hours")
    for group, seconds in est["seconds_by_group"].items():
        print(f"  {group:<10} {est['arms_by_group'][group]:>3} arms  {seconds / 60:8.1f} min")
    print(f"  NOTE: {est['multiplier_note']}")
    return est


def _intervention_modes(arm: Arm) -> tuple[str, ...]:
    if arm.kind == "cheap_bias":
        return ("logit_bias",)
    if arm.kind != "psa":
        return ()
    return {"logit": ("logit_bias",), "value": ("value_scale",),
            "head": ("head_scale",)}[str(arm.mode)]


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - one linear driver
    ap = argparse.ArgumentParser(description=__doc__)
    # calibration only, by construction. The sealed split is opened once, at Phase 14.
    ap.add_argument("--split", default="calibration", choices=("calibration",))
    ap.add_argument("--model", default="qwen3", choices=tuple(paths.PANEL))
    ap.add_argument("--matrix", default=None,
                    help="Phase 4/5 significance matrix (.npz or .csv) for the PSA arms")
    ap.add_argument("--negate-matrix", action="store_true",
                    help="flip the sign of the significance matrix before it is applied")
    ap.add_argument("--bias-matrix", default=None,
                    help="Phase 4b PositionalBias .npz for cheap.position_logit_correction")
    ap.add_argument("--bias-field", default="qk_logit_mean",
                    choices=("qk_logit_mean", "attn_by_bin"))
    ap.add_argument("--bias-sign", default="correct", choices=("correct", "raw"))
    ap.add_argument("--arms", default=",".join(ARM_GROUPS),
                    help=f"comma-separated subset of {list(ARM_GROUPS)}")
    ap.add_argument("--include-gate-controls", action="store_true",
                    help="also run frozen_gates and random_gates (section 37)")
    ap.add_argument("--length-bucket", default="4k", choices=("2k", "4k", "8k"))
    ap.add_argument("--families", default=None, help="comma-separated family subset")
    ap.add_argument("--positions", default=None,
                    help="comma-separated decisive-evidence positions to keep")
    ap.add_argument("--local-position", default="middle",
                    choices=("middle", "first", "last", "any"))
    ap.add_argument("--scenario-limit", type=int, default=None,
                    help="keep only the first N scenarios (clustered subsample)")
    ap.add_argument("--limit", type=int, default=None, help="cap items, for smoke runs")
    ap.add_argument("--rope-lo", type=float, default=0.75)
    ap.add_argument("--rope-hi", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=DEFAULT_CONTROL_SEED)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-i-costed-this", action="store_true")
    args = ap.parse_args(argv)

    # Defence in depth. ``--split`` is already restricted to the calibration split by
    # argparse; this refuses anything else that reaches here by another route, including
    # the held-out split, which is opened exactly once, by scripts/phase14_*.py.
    # (Spelled as a positive check rather than a denylist so that this file never names
    # that split in code -- tests/test_split_discipline.py scans the AST for exactly that.)
    if args.split != "calibration":
        raise SystemExit(
            f"this screen runs on the calibration split only; {args.split!r} is refused"
        )

    groups = tuple(g.strip() for g in args.arms.split(",") if g.strip())
    unknown = set(groups) - set(ARM_GROUPS)
    if unknown:
        raise SystemExit(f"unknown arm groups {sorted(unknown)}; known: {list(ARM_GROUPS)}")

    control_names = tuple(MANDATORY_MATRIX_CONTROLS)
    if args.include_gate_controls:
        control_names += OPTIONAL_MATRIX_CONTROLS

    items = select_items(
        load_split(args.split),
        length_bucket=args.length_bucket,
        families=None if not args.families else [f.strip() for f in args.families.split(",")],
        positions=None if not args.positions else [float(p) for p in args.positions.split(",")],
        local_position=args.local_position,
        scenario_limit=args.scenario_limit,
        limit=args.limit,
    )
    if not items:
        raise SystemExit("the filters selected zero items; nothing to screen")

    if args.dry_run:
        # The dry run must cost nothing, so it plans against SPEC 1's shape rather than
        # loading a checkpoint. Arm *count* does not depend on the architecture, so the
        # estimate it prints is the estimate the real run will print.
        print_estimate(items, plan(args, panel_arch(), control_names, groups))
        return 0

    # Supa-A1.1's abstention gate shifts one vocabulary row and therefore every other
    # token's NLL. SPEC 1: the ungated instance is the one mechanistic work uses.
    ungate = args.model == "supa_a11"

    is_partial = bool(
        args.limit or args.scenario_limit or set(groups) != set(ARM_GROUPS)
        or args.families or args.positions or args.length_bucket != "4k"
    )

    tok = load_tokenizer("qwen3")  # one vocabulary serves the panel (SPEC 11.4)
    model, arch = load_model(args.model, ungate=ungate)
    try:
        arms = plan(args, arch, control_names, groups)
        est = print_estimate(items, arms)
        if est["total_seconds"] > GPU_HOUR_GUARD_SECONDS and not args.yes_i_costed_this:
            raise SystemExit(
                f"estimate is {est['total_seconds'] / 3600:.2f} GPU-hours, over the "
                "2-hour guard; subset with --arms / --scenario-limit, or re-run with "
                "--yes-i-costed-this once you have looked at that number"
            )
        return _run(args, items, arms, model, arch, tok, est, control_names,
                    is_partial=is_partial, ungate=ungate)
    finally:
        free_model(model)


def _run(
    args: argparse.Namespace,
    items: Sequence[BenchmarkItem],
    arms: Sequence[Arm],
    model: Any,
    arch: Any,
    tok: Any,
    est: Mapping[str, Any],
    control_names: Sequence[str],
    *,
    is_partial: bool,
    ungate: bool,
) -> int:
    """Everything after the model is on the device. Split out so ``main`` stays readable."""
    import torch

    device = next(model.parameters()).device
    on_cuda = torch.cuda.is_available() and getattr(device, "type", str(device)) == "cuda"
    design = _design_map(items)
    pairs = prepare_items(tok, items)
    lengths = [len(t.input_ids) for _, t in pairs]
    mean_len = int(round(float(np.mean(lengths))))
    total_tokens = int(sum(lengths))

    rope_check: dict[str, Any] | None = None
    if any(a.group == "rope" for a in arms):
        rope_check = assert_flat_rope_is_noop(model, pairs[0][1].input_ids, on_cuda=on_cuda)
        print(f"flat RoPE no-op check passed: cos/sin bit-exact, "
              f"logit delta {rope_check['logit_max_abs_diff']:.3e} <= jitter "
              f"{rope_check['baseline_run_to_run_jitter']:.3e}")

    config = ComputeConfig.from_model(model)
    records: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    record_fields = ("gold_nll", "answer_margin", "correct", "latency_s", "peak_vram_bytes")

    with efficient_attention(active=on_cuda, required=on_cuda) as pinned:
        for index, arm in enumerate(arms, start=1):
            seen = [0]

            def tick(n_rows: int, _seen: list[int] = seen, _label: str = arm.label) -> None:
                before, _seen[0] = _seen[0], _seen[0] + n_rows
                if before // 200 != _seen[0] // 200:
                    print(f"    {_label} {_seen[0]}/{len(pairs)}", flush=True)

            started = time.perf_counter()
            # measure_compute(None) means "cuda if this machine has cuda", which is wrong
            # for a CPU model on a CUDA box; pass the model's own device explicitly.
            # measure_compute is the OUTER context so the peak-VRAM reset happens before
            # the arm allocates its bias/scale tables, which are part of the arm's cost.
            with measure_compute(device) as probe, arm_session(arm, model, arch) as ctx:
                out = run_items(model, tok, pairs, mode="teacher_forced", ctx=ctx,
                                record=record_fields, progress=tick)
            elapsed = time.perf_counter() - started
            peak = probe.result.peak_vram_bytes
            print(f"[{index}/{len(arms)}] {arm.label}: {len(out)} items in "
                  f"{elapsed:.1f} s, peak VRAM "
                  f"{'n/a' if peak is None else f'{peak / 2**30:.2f} GiB'}"
                  f"{' (control)' if arm.is_control else ''}", flush=True)

            for rec in out:
                row = rec.to_json()
                row.update({
                    "arm": arm.label,
                    "arm_group": arm.group,
                    "arm_kind": arm.kind,
                    "psa_mode": arm.mode,
                    "lam": arm.lam,
                    "matrix_name": arm.matrix_name,
                    "is_control": arm.is_control,
                    "control_of": arm.control_of,
                    "control_name": arm.control_name,
                    "model_key": args.model,
                    "ungated": ungate,
                    "dtype_approximated": dtype_is_approximation(args.model),
                })
                row.update(design[row["item_id"]])
                records.append(row)

            modes = _intervention_modes(arm)
            compute_rows.append(compute_metrics_row(
                arm.label, config,
                seq_len=mean_len,
                n_items=len(out),
                n_tokens=total_tokens,
                measurement=probe,
                logits_to_keep=1,
                intervention=intervention_flops(config, mean_len, modes=modes),
                model_key=args.model,
                dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
                condition=arm.group,
                lam=arm.lam,
                notes=(
                    f"group={arm.group}; control_of={arm.control_of or '-'}; "
                    f"seq_len is the MEAN prompt length ({mean_len}), so the O(T^2) "
                    "attention term is evaluated at the mean rather than averaged over "
                    "the length distribution"
                ),
            ))

    paths.ensure_dirs()
    stem_suffix = "_partial" if (is_partial and not args.out) else ""
    out_path = (
        Path(args.out) if args.out
        else paths.RESULTS / f"intervention_results{stem_suffix}.parquet"
    )
    compute_path = out_path.with_name(f"compute_metrics{stem_suffix}.csv")
    summary_path = out_path.with_name(
        out_path.name.replace("results", "summary").rsplit(".", 1)[0] + ".json"
    )

    run_config = {
        "phase": "8_cheap_interventions",
        "split": args.split,
        "model": args.model,
        "ungated": ungate,
        "length_bucket": args.length_bucket,
        "local_position": args.local_position,
        "n_items": len(items),
        "n_scenarios": len({it.scenario_id for it in items}),
        "arms": [a.describe() for a in arms],
        "control_names": list(control_names),
        "matrix": args.matrix,
        "bias_matrix": args.bias_matrix,
        "bias_field": args.bias_field,
        "bias_sign": args.bias_sign,
        "control_seed": args.seed,
        "bootstrap_seed": args.bootstrap_seed,
        "n_resamples": args.n_resamples,
        "selection_rule": SELECTION_RULE,
    }

    try:
        write_result(records, out_path, config=run_config)
    except Exception as exc:  # noqa: BLE001 - parquet is optional; the data is not
        print(f"parquet unavailable ({exc}); writing jsonl instead")
        out_path = out_path.with_suffix(".jsonl")
        write_result(records, out_path, config=run_config)
    write_compute_metrics(compute_rows, compute_path, config=run_config)

    per_arm = analyse(records, arms, seed=args.bootstrap_seed,
                      n_resamples=args.n_resamples, control_names=control_names)
    advanced = sorted(label for label, e in per_arm.items() if e["advances"])
    summary = {
        "selection_rule": SELECTION_RULE,
        "is_partial_run": bool(is_partial),
        "partial_reason": None if not is_partial else {
            "limit": args.limit, "scenario_limit": args.scenario_limit,
            "arm_groups": args.arms, "families": args.families,
            "positions": args.positions, "length_bucket": args.length_bucket,
        },
        "n_records": len(records),
        "n_items": len(items),
        "n_scenarios": len({it.scenario_id for it in items}),
        "n_arms": len(arms),
        "arms_advanced": advanced,
        "per_arm": per_arm,
        "flat_rope_noop_check": rope_check,
        "cost_estimate": dict(est),
        "backend_pinned": bool(pinned),
        # probe_backends allocates on the GPU. Only worth doing — and only safe to do —
        # when the run itself was on the GPU.
        "backends": probe_backends() if on_cuda else {"cuda": False, "note": "model on cpu"},
        "artifacts": {"results": str(out_path), "compute_metrics": str(compute_path)},
    }
    write_result(summary, summary_path, config=run_config)

    if is_partial:
        where = (
            f"under an explicit --out ({out_path.name})" if args.out
            else f"with a '{stem_suffix}' suffix"
        )
        print(f"NOTE: partial run; artifacts written {where} and marked "
              "is_partial_run=true, so they cannot be mistaken for the full screen")
    print(f"\nadvanced ({len(advanced)}/{sum(1 for a in arms if not a.is_control) - 1}): "
          f"{advanced or 'none'}")
    for label, entry in per_arm.items():
        base = entry["vs_baseline"]
        if base is None:
            continue
        beaten = f"{len(entry['controls_beaten'])}/{len(entry['controls'])}"
        print(f"  {label:<34} d={base['estimate']:+.4f} "
              f"[{base['ci_low']:+.4f},{base['ci_high']:+.4f}] "
              f"controls beaten {beaten}  {entry['advance_reason']}")
    print(f"\nwrote {out_path}\n      {compute_path}\n      {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
