"""SPEC section 10: the determinism contract, implemented exactly as written.

SPEC 10 says:

    ``torch.manual_seed``, ``numpy.default_rng(seed)``,
    ``torch.use_deterministic_algorithms(False)`` (SDPA kernels are not bit-deterministic
    on this card; we record that rather than pay the speed cost). Generation is
    **greedy, do_sample=False** everywhere.

That is a *reproducible-seeding* contract, not a bit-exactness contract. This module
implements it literally: ``seed_everything`` seeds every RNG the lab draws from and
then calls ``torch.use_deterministic_algorithms(False)`` **explicitly**, so the choice
is an executed decision rather than an unexamined default. ``describe()`` returns the
record of that decision -- what was seeded, what was deliberately left
non-deterministic, and why -- and is embedded in every provenance sidecar
(``statistics/provenance.stamp``).

Do NOT "strengthen" this into ``use_deterministic_algorithms(True)``. On this card that
either raises for SDPA / ``scatter_add`` style kernels or silently falls back to a
much slower path, and the lab's measured compute budget (SPEC section 11) assumes the
fast kernels. Callers that genuinely need bit-exactness can opt in per call site with
``seed_everything(seed, deterministic_algorithms=True)``.

DEVIATION: ``CUBLAS_WORKSPACE_CONFIG`` is deliberately **not** set. It is only
meaningful together with ``use_deterministic_algorithms(True)``, and setting it here
would imply a guarantee this lab does not make. ``describe()["env"]`` records its
actual value so a reader can see it was left alone rather than forgotten.

No function here touches the GPU: ``torch.manual_seed`` defers its CUDA half through
``torch.cuda._lazy_call``, so seeding does not create a CUDA context or allocate on a
card another process owns.
"""

from __future__ import annotations

import importlib
import os
import random
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_SEED",
    "NON_DETERMINISTIC_SOURCES",
    "DeterminismState",
    "describe",
    "seed_everything",
    "state",
]

#: The lab-wide seed. Every script defaults to this unless a config overrides it.
DEFAULT_SEED = 20260828

#: What SPEC 10 knowingly leaves non-bit-deterministic, and why. Recorded, not fixed.
NON_DETERMINISTIC_SOURCES: tuple[dict[str, str], ...] = (
    {
        "source": "torch.nn.functional.scaled_dot_product_attention",
        "scope": "cuda",
        "reason": (
            "The fused SDPA kernels reduce in a work-group-dependent order, so repeated "
            "runs can differ in the last bits. SPEC 10 records this rather than paying "
            "the math-fallback cost."
        ),
    },
    {
        "source": "cublas matmul split-k reductions",
        "scope": "cuda",
        "reason": (
            "Reduction order depends on the selected kernel and therefore on free VRAM "
            "and on what else is resident on the card."
        ),
    },
    {
        "source": "atomics in index_add / scatter_add backward",
        "scope": "cuda",
        "reason": "Float atomics commit in arrival order; only used off the eval path.",
    },
    {
        "source": "PYTHONHASHSEED",
        "scope": "process",
        "reason": (
            "String hash randomisation is fixed before the interpreter starts and "
            "cannot be set from inside it. The lab never orders results by set/dict "
            "iteration, so this affects nothing measured."
        ),
    },
)

_TORCH_UNAVAILABLE = "torch is not importable in this interpreter"


@dataclass(frozen=True)
class DeterminismState:
    """What the last :func:`seed_everything` call actually did."""

    seed: int | None = None
    seeded: tuple[str, ...] = ()
    skipped: dict[str, str] = field(default_factory=dict)
    deterministic_algorithms: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "seeded": list(self.seeded),
            "skipped": dict(self.skipped),
            "deterministic_algorithms": self.deterministic_algorithms,
        }


_STATE = DeterminismState()


def state() -> DeterminismState:
    """The record of the most recent :func:`seed_everything` call in this process."""
    return _STATE


def _torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except Exception:  # noqa: BLE001 - torch-free interpreters are supported
        return None


def seed_everything(
    seed: int = DEFAULT_SEED,
    *,
    deterministic_algorithms: bool = False,
) -> Any:
    """Seed every RNG the lab draws from and record the determinism decision.

    Seeds ``random``, ``numpy``'s legacy global RNG, and ``torch`` (CPU, plus CUDA via
    torch's own lazy hook -- no CUDA context is created). Then calls
    ``torch.use_deterministic_algorithms(deterministic_algorithms)``; the default of
    ``False`` is SPEC 10's contract, stated explicitly rather than inherited.

    Returns the ``numpy.random.Generator`` from ``numpy.default_rng(seed)``, which is the
    object SPEC 10 names and the one callers should thread through instead of reaching
    for module-level ``numpy.random`` functions.
    """
    global _STATE
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must fit in uint32 for numpy/torch parity, got {seed}")

    seeded: list[str] = []
    skipped: dict[str, str] = {}

    random.seed(seed)
    seeded.append("python_random")

    np = importlib.import_module("numpy")
    np.random.seed(seed)
    seeded.append("numpy_legacy_global")
    rng = np.random.default_rng(seed)
    seeded.append("numpy_default_rng")

    torch = _torch()
    applied_deterministic: bool | None = None
    if torch is None:
        skipped["torch"] = _TORCH_UNAVAILABLE
    else:
        # torch.manual_seed also seeds every CUDA device, but through
        # torch.cuda._lazy_call, so it is a no-op until someone else initialises CUDA.
        torch.manual_seed(seed)
        seeded.append("torch_cpu")
        seeded.append("torch_cuda_lazy")
        torch.use_deterministic_algorithms(deterministic_algorithms)
        applied_deterministic = bool(torch.are_deterministic_algorithms_enabled())

    skipped["PYTHONHASHSEED"] = (
        "fixed before interpreter start; cannot be changed from inside the process"
    )
    if not deterministic_algorithms:
        skipped["torch.use_deterministic_algorithms"] = (
            "SPEC 10 leaves this False: SDPA is not bit-deterministic on this card and "
            "the deterministic fallback breaks the SPEC 11 compute budget"
        )
        skipped["CUBLAS_WORKSPACE_CONFIG"] = (
            "only meaningful with deterministic algorithms enabled; deliberately unset"
        )

    _STATE = DeterminismState(
        seed=seed,
        seeded=tuple(seeded),
        skipped=skipped,
        deterministic_algorithms=applied_deterministic,
    )
    return rng


def describe() -> dict[str, Any]:
    """A JSON-serialisable record of the determinism contract and its current state.

    Safe to call without having called :func:`seed_everything`; ``seed`` is then ``None``
    and ``seeded`` is empty, which is itself the fact worth recording in a sidecar.
    Reads only; never seeds, never initialises CUDA.
    """
    torch = _torch()
    payload: dict[str, Any] = {
        "spec_section": "10",
        "contract": (
            "torch.manual_seed + numpy.default_rng(seed) + "
            "torch.use_deterministic_algorithms(False); generation greedy, do_sample=False"
        ),
        "default_seed": DEFAULT_SEED,
        "seed": _STATE.seed,
        "seed_everything_called": _STATE.seed is not None,
        "seeded": list(_STATE.seeded),
        "skipped": dict(_STATE.skipped),
        "generation": {"do_sample": False, "greedy": True, "num_beams": 1},
        "not_deterministic": [dict(item) for item in NON_DETERMINISTIC_SOURCES],
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }

    if torch is None:
        payload["torch"] = None
        payload["deterministic_algorithms"] = {
            "enabled": None,
            "requested_by_spec": False,
            "reason": _TORCH_UNAVAILABLE,
        }
        return payload

    payload["torch"] = {
        "version": str(getattr(torch, "__version__", "unknown")),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    payload["deterministic_algorithms"] = {
        "enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "requested_by_spec": False,
        "reason": (
            "SPEC 10 deliberately does not enable deterministic algorithms: the SDPA "
            "kernels this lab depends on are not bit-deterministic on an RTX 4070 and "
            "the deterministic fallback is unaffordable under the SPEC 11 budget. The "
            "non-determinism is recorded in 'not_deterministic' instead of suppressed."
        ),
    }
    return payload
