"""Warm-start activation schedules and layer-placement search.

Both are **development-only** research tools. Neither may read the Phase-1
confirmatory bank, and neither produces a confirmatory result.

## Warm start

The risk a retrofit runs is that the architecture change erases pretrained
capability before the trajectory pathway has learned anything useful. Because
this conversion is exactly function-preserving, the model does not *start*
damaged — but the first optimizer steps can still damage it if the pretrained
projections are immediately free to move while the new modules are still random.

The registered candidate schedule freezes the pretrained tensors and trains only
the trajectory pathway, then progressively releases:

1. ``trajectory`` — only the new modules train. Pretrained function is frozen and
   cannot be destroyed. The conditioning weights move off zero, so the model
   begins to differ from native for the first time.
2. ``attention`` — additionally release the attention projections of converted
   layers, so the pretrained attention can adapt to the control signal.
3. ``converted_layers`` — additionally release everything inside converted layers.
4. ``all`` — full fine-tuning.

``immediate_full`` (everything trainable from step 0) is the registered
comparison arm. The point is to find out whether staging actually helps, not to
assume it.

## Layer placement

Nothing justifies converting every layer. TGSA might matter most early (building
representations), in the middle (composition), late (decision refinement), or
uniformly. :func:`placement_candidates` enumerates a ladder that samples
positions across depth rather than assuming a favourite region, so a single-layer
sweep covers early/middle/late evenly instead of testing only the layers someone
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal, Sequence

from torch import nn

STAGE_ORDER: tuple[str, ...] = ("trajectory", "attention", "converted_layers", "all")

NATIVE_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True, slots=True)
class PlacementCandidate:
    """One layer-placement arm."""

    name: str
    layers: tuple[int, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "layers": list(self.layers), "rationale": self.rationale}


def placement_candidates(total_layers: int) -> list[PlacementCandidate]:
    """The registered placement ladder for a model of ``total_layers`` layers.

    Progresses single -> few distributed -> contiguous block -> fraction -> all.
    The single-layer arms sample five positions evenly across depth, so the sweep
    cannot accidentally test only early layers and conclude that early is best.
    """

    if total_layers < 2:
        raise ValueError("layer placement needs at least two layers")

    candidates: list[PlacementCandidate] = [
        PlacementCandidate("none", (), "unconverted control; the exact parent model")
    ]

    # Single layers sampled across depth, including the extremes.
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    seen: set[int] = set()
    for fraction in fractions:
        index = min(total_layers - 1, max(0, round(fraction * (total_layers - 1))))
        if index in seen:
            continue
        seen.add(index)
        depth = "early" if fraction < 0.34 else ("middle" if fraction < 0.67 else "late")
        candidates.append(
            PlacementCandidate(
                f"single_l{index}",
                (index,),
                f"one converted layer at {int(fraction * 100)}% depth ({depth})",
            )
        )

    # A few layers distributed evenly.
    for count in (2, 4):
        if count > total_layers:
            continue
        step = max(1, total_layers // count)
        layers = tuple(sorted({min(total_layers - 1, i * step) for i in range(count)}))
        candidates.append(
            PlacementCandidate(
                f"distributed_{len(layers)}",
                layers,
                f"{len(layers)} converted layers spread evenly across depth",
            )
        )

    # Contiguous blocks at three depths.
    block = max(2, total_layers // 4)
    for label, start in (
        ("early", 0),
        ("middle", max(0, (total_layers - block) // 2)),
        ("late", max(0, total_layers - block)),
    ):
        layers = tuple(range(start, min(total_layers, start + block)))
        candidates.append(
            PlacementCandidate(
                f"block_{label}",
                layers,
                f"contiguous block of {len(layers)} layers, {label} in the network",
            )
        )

    # Larger fractions, then everything.
    for fraction, label in ((0.5, "half"), (1.0, "all")):
        count = max(1, round(total_layers * fraction))
        step = max(1, total_layers // count)
        layers = tuple(range(0, total_layers, step))[:count] if fraction < 1.0 else tuple(range(total_layers))
        candidates.append(
            PlacementCandidate(
                f"fraction_{label}",
                layers,
                f"{label} of eligible layers converted",
            )
        )

    # Deduplicate by layer set while keeping the first, most descriptive name.
    unique: dict[tuple[int, ...], PlacementCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.layers, candidate)
    return list(unique.values())


@dataclass(slots=True)
class FreezeSchedule:
    """Which parameter groups are trainable at each stage.

    ``stage_steps`` is the number of optimizer steps spent in each stage, in
    ``STAGE_ORDER``. A schedule of ``{"trajectory": N}`` alone trains only the
    new modules for the whole run.
    """

    stage_steps: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.stage_steps) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"unknown warm-start stages: {sorted(unknown)}")
        if any(value < 0 for value in self.stage_steps.values()):
            raise ValueError("stage step counts must be non-negative")

    @classmethod
    def immediate_full(cls, steps: int) -> "FreezeSchedule":
        """Registered comparison arm: everything trainable from step 0."""

        return cls({"all": steps})

    @classmethod
    def progressive(cls, steps: int) -> "FreezeSchedule":
        """Registered warm start: quarter of the budget per stage."""

        share = max(1, steps // 4)
        return cls({stage: share for stage in STAGE_ORDER})

    @property
    def total_steps(self) -> int:
        return sum(self.stage_steps.values())

    def stage_at(self, step: int) -> str:
        """Which stage a given global step falls in."""

        cursor = 0
        for stage in STAGE_ORDER:
            cursor += self.stage_steps.get(stage, 0)
            if step < cursor:
                return stage
        return STAGE_ORDER[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_steps": dict(self.stage_steps),
            "total_steps": self.total_steps,
            "order": list(STAGE_ORDER),
        }


def is_trajectory_parameter(name: str) -> bool:
    """True for tensors the retrofit introduced, false for pretrained ones."""

    tail = name.split(".")
    return not any(part in NATIVE_PROJECTIONS for part in tail)


def apply_stage(
    model: nn.Module,
    stage: str,
    *,
    converted_layer_names: Iterable[str] = (),
) -> dict[str, int]:
    """Set ``requires_grad`` for ``stage`` and report the resulting counts.

    Returns trainable/frozen parameter counts so a training log can record
    exactly what was released at each transition rather than leaving it implicit.
    """

    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}")
    converted = tuple(converted_layer_names)

    for name, parameter in model.named_parameters():
        inside_converted = (not converted) or any(prefix in name for prefix in converted)
        if stage == "all":
            trainable = True
        elif stage == "trajectory":
            trainable = is_trajectory_parameter(name) and inside_converted
        elif stage == "attention":
            trainable = inside_converted and (
                is_trajectory_parameter(name)
                or any(projection in name for projection in NATIVE_PROJECTIONS)
            )
        else:  # converted_layers
            trainable = inside_converted
        parameter.requires_grad_(trainable)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "stage": stage,  # type: ignore[dict-item]
        "trainable_parameters": trainable_count,
        "frozen_parameters": frozen_count,
        "trainable_tensors": sum(1 for p in model.parameters() if p.requires_grad),
    }


def iter_stage_transitions(schedule: FreezeSchedule) -> Iterator[tuple[int, str]]:
    """Yield ``(global_step, stage)`` at each stage boundary."""

    cursor = 0
    for stage in STAGE_ORDER:
        steps = schedule.stage_steps.get(stage, 0)
        if steps <= 0:
            continue
        yield cursor, stage
        cursor += steps
