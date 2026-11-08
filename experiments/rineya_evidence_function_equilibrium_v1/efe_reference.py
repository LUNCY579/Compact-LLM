"""EFE-001 CPU references. No checkpoint loading, native training, or GPU calls."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Callable, Mapping, Sequence

import numpy as np


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _scalar(value: float, name: str, minimum: float = 0.0) -> None:
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _array(value: object, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def select_positions(
    positions: Sequence[int], *, prompt_length: int, next_position: int,
    budget: int, recent: int, selector: str = "random", protect_prompt: bool = True,
    seed: int = 0, kv_head: int = 0, scores: Mapping[int, float] | None = None,
) -> tuple[int, ...]:
    """Select from CURRENT resident logical positions; never restore evicted tokens.

    budget includes protected prefill and recent generated tokens, per KV head.
    next_position is the next unused absolute token position, not physical length.
    This is an index reference, not a KV tensor compactor or attention kernel.
    """
    for name, value in (("prompt_length", prompt_length), ("next_position", next_position),
                        ("recent", recent), ("seed", seed), ("kv_head", kv_head)):
        _integer(value, name)
    _integer(budget, "budget", 1)
    if not isinstance(protect_prompt, bool) or next_position < prompt_length:
        raise ValueError("invalid protection setting or incomplete prefill")
    if selector not in {"random", "recency", "score"}:
        raise ValueError("selector must be random, recency, or score")
    for value in positions:
        _integer(value, "position")
    resident = set(positions)
    if len(resident) != len(positions) or any(p >= next_position for p in resident):
        raise ValueError("positions must be unique and earlier than next_position")
    required = set(range(prompt_length)) if protect_prompt else set()
    required.update(range(max(prompt_length, next_position - recent), next_position))
    if not required.issubset(resident):
        raise ValueError("protected positions missing; cannot reconstruct evicted KV")
    if len(required) > budget:
        raise ValueError("ineligible: protected prefill/recent tokens exceed total budget")
    eligible = sorted(resident - required)
    count = min(budget - len(required), len(eligible))
    if selector == "random":
        # Independent deterministic stream key per seed, step and shared KV head.
        chosen = random.Random(f"EFE1:{seed}:{next_position}:{kv_head}").sample(eligible, count)
    elif selector == "recency":
        chosen = sorted(eligible, reverse=True)[:count]
    else:
        if scores is None or any(p not in scores or not math.isfinite(scores[p]) for p in eligible):
            raise ValueError("finite scores are required for every eligible resident token")
        chosen = sorted(eligible, key=lambda p: (-scores[p], -p))[:count]
    return tuple(sorted(required | set(chosen)))


def maximum_decode_only_saving(prompt_length: int, generated_length: int) -> float:
    """Ideal token-KV fraction removable; ignores necessary generated KV and overhead."""
    _integer(prompt_length, "prompt_length")
    _integer(generated_length, "generated_length")
    if prompt_length + generated_length == 0:
        raise ValueError("empty workload has no defined saving fraction")
    return generated_length / (prompt_length + generated_length)


def fit_coefficients(
    new_directions: object, new_gradient: object, keep_directions: object, *,
    step_size: float = 1.0, preservation_weight: float = 1.0, ridge: float = 1e-6,
) -> dict:
    """Fit a finite, UNBOUNDED local surrogate using supplied output directions.

    Objective: mean((A c + eta g)^2) + lambda mean((B c)^2) + mu ||c||^2.
    A/B contain supplied Jd columns; this function does not compute native JVPs.
    Callers must apply provenance/split guards before fitting real exports.
    No trust region or adaptive representation growth is implemented here.
    """
    a = _array(new_directions, 2, "new_directions")
    g = _array(new_gradient, 1, "new_gradient")
    b = _array(keep_directions, 2, "keep_directions")
    if min(a.shape) == 0 or g.shape != (a.shape[0],) or b.shape[1] != a.shape[1]:
        raise ValueError("incompatible or empty direction/gradient dimensions")
    for name, value in (("step_size", step_size), ("preservation_weight", preservation_weight),
                        ("ridge", ridge)):
        _scalar(value, name)
    if step_size == 0:
        raise ValueError("step_size must be positive")
    target = -step_size * g
    blocks, targets = [a / math.sqrt(a.shape[0])], [target / math.sqrt(a.shape[0])]
    if b.shape[0] and preservation_weight:
        blocks.append(b * math.sqrt(preservation_weight / b.shape[0]))
        targets.append(np.zeros(b.shape[0]))
    if ridge:
        blocks.append(math.sqrt(ridge) * np.eye(a.shape[1]))
        targets.append(np.zeros(a.shape[1]))
    coefficients = np.linalg.lstsq(np.vstack(blocks), np.concatenate(targets), rcond=None)[0]
    unconstrained, _, rank, _ = np.linalg.lstsq(a, target, rcond=None)
    residual = a @ unconstrained - target
    fitted_error = a @ coefficients - target
    norm = float(np.linalg.norm(target))
    result = {
        "coefficients": coefficients.tolist(), "basis_rank": int(rank),
        "unconstrained_target_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "fitted_target_rmse": float(np.sqrt(np.mean(fitted_error ** 2))),
        "relative_span_residual": float(np.linalg.norm(residual) / norm) if norm else None,
        "preservation_output_rmse": float(np.sqrt(np.mean((b @ coefficients) ** 2))) if b.shape[0] else None,
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "scope": "local_surrogate_only_not_native_efficacy",
    }
    if not np.isfinite(coefficients).all() or any(
        isinstance(v, float) and not math.isfinite(v) for v in result.values()
    ):
        raise ValueError("nonfinite fit; rescale inputs and re-qualify")
    return result


@dataclass(frozen=True)
class EquilibriumResult:
    state: np.ndarray
    iterations: int
    function_evaluations: int
    undamped_residual: float
    damped_residual: float
    numerically_converged: bool


def solve_fixed_point(
    function: Callable[[np.ndarray], np.ndarray], initial: object, *,
    damping: float = 1.0, max_steps: int = 100, atol: float = 1e-8, rtol: float = 1e-6,
) -> EquilibriumResult:
    """Bounded forward-only Picard reference; stopping uses UNDAMPED residual.

    Convergence is numerical, not a quality, uniqueness, or gradient certificate.
    The residual is checked at the returned state, including after the last update.
    """
    z = _array(initial, 1, "initial").copy()
    if z.size == 0:
        raise ValueError("initial state must be nonempty")
    _integer(max_steps, "max_steps", 1)
    for name, value in (("damping", damping), ("atol", atol), ("rtol", rtol)):
        _scalar(value, name)
    if not 0 < damping <= 1 or atol + rtol == 0:
        raise ValueError("require 0 < damping <= 1 and a positive tolerance")
    for step in range(max_steps + 1):
        mapped = _array(function(z.copy()), 1, "function output")
        if mapped.shape != z.shape:
            raise ValueError("function changed the state shape")
        raw = float(np.linalg.norm(mapped - z))
        if not math.isfinite(raw):
            raise ValueError("nonfinite residual")
        converged = raw <= atol + rtol * float(np.linalg.norm(z))
        if converged or step == max_steps:
            return EquilibriumResult(z.copy(), step, step + 1, raw, damping * raw, converged)
        z = (1.0 - damping) * z + damping * mapped
    raise AssertionError("unreachable")


def implicit_adjoint(jacobian: object, output_gradient: object, *, condition_limit: float = 1e8) -> np.ndarray:
    """Small dense qualification of (I-J)^T v=g; NOT an LLM backward implementation."""
    j = _array(jacobian, 2, "jacobian")
    g = _array(output_gradient, 1, "output_gradient")
    _scalar(condition_limit, "condition_limit", 1.0)
    if not j.size or j.shape[0] != j.shape[1] or g.shape != (j.shape[0],):
        raise ValueError("jacobian must be nonempty square and match gradient")
    system = np.eye(j.shape[0]) - j
    condition = float(np.linalg.cond(system))
    if not math.isfinite(condition) or condition > condition_limit:
        raise ValueError("singular or ill-conditioned implicit system")
    answer = np.linalg.solve(system.T, g)
    if not np.isfinite(answer).all():
        raise ValueError("nonfinite implicit solution")
    return answer


def audit_splits(groups: Mapping[str, Sequence[Mapping]]) -> None:
    """Check DECLARED IDs, parent families, hashes and spent status; not raw provenance.

    All three roles must be nonempty. Parent variants can share a role but cannot
    cross roles. Native use must independently hash content and consult quarantine.
    """
    if set(groups) != {"calibration", "preservation", "confirmation"}:
        raise ValueError("require calibration, preservation and confirmation roles")
    seen = {key: {} for key in ("row_id", "parent_id", "content_sha256")}
    for role, rows in groups.items():
        if not rows:
            raise ValueError(f"empty {role} split")
        for row in rows:
            if row.get("spent") is not False:
                raise ValueError("spent or unspecified admissibility cannot be used")
            for key, owners in seen.items():
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"missing {key}")
                if key == "content_sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError("content_sha256 must be 64 lowercase hexadecimal characters")
                if value in owners and (owners[value] != role or key != "parent_id"):
                    raise ValueError(f"duplicate or cross-split overlap in {key}")
                owners[value] = role


def smoke() -> dict:
    """Invented mathematical fixtures only; never read repository datasets or weights."""
    fit = fit_coefficients([[1.0, 0.0], [0.0, 1.0]], [-0.6, 0.2], [[0.0, 0.0]], ridge=0)
    streams = []
    for head in range(2):
        resident = tuple(range(4))
        for position in range(4, 24):
            resident = select_positions((*resident, position), prompt_length=4, next_position=position + 1,
                                        budget=10, recent=2, seed=17, kv_head=head)
        streams.append(list(resident))
    fixed = solve_fixed_point(lambda z: 0.5 * z + np.ones_like(z), [0.0, 0.0])
    return {
        "experiment": "EFE-001", "measurement_kind": "synthetic_cpu_mechanics",
        "native_model_loads": 0, "native_training_steps": 0, "promotion_eligible": False,
        "functional_fixture": fit, "cache_position_fixture": streams,
        "equilibrium_fixture": {
            "state": fixed.state.tolist(), "undamped_residual": fixed.undamped_residual,
            "numerically_converged": fixed.numerically_converged,
            "function_evaluations": fixed.function_evaluations,
        },
        "decode_only_ideal_saving_4096_256": maximum_decode_only_saving(4096, 256),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="print the queued research manifest")
    smoke_parser = commands.add_parser("smoke", help="run synthetic CPU references")
    smoke_parser.add_argument("--out", type=Path, required=True, help="new JSON file; parent must exist")
    commands.add_parser("native", help="report native implementation boundary and exit 2")
    args = parser.parse_args(argv)
    if args.command == "native":
        print(json.dumps({"status": "BLOCKED", "reason": "native adapters are not implemented in EFE-001"}))
        return 2
    try:
        if args.command == "plan":
            print((Path(__file__).with_name("plan.json")).read_text(encoding="utf-8"))
        else:
            encoded = json.dumps(smoke(), indent=2, allow_nan=False) + "\n"
            with args.out.open("x", encoding="utf-8") as file:
                file.write(encoded)
            print(f"Saved synthetic mechanics only: {args.out}")
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
