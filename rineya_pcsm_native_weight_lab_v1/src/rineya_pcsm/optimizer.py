"""An AdamW optimizer that can write only an immutable exact support."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast, overload

import torch
from torch import nn
from torch.optim import Optimizer

from .application import (
    AppliedTensorPlan,
    AppliedUpdatePlan,
    NativeUpdatePlan,
    TensorUpdatePlan,
)


class ExactSupportAdamW(Optimizer):
    """Sparse-state AdamW with indexed writes to selected native coordinates.

    The optimizer has no dense moment tensors.  Each parameter state contains a
    one-dimensional support index and two fp32 vectors of exactly the same
    length.  Adam normalization is computed first, then the per-coordinate PCSM
    stabilizer ``A`` scales that normalized update.  Decoupled weight decay is
    also computed only for selected values.  A single ``index_copy_`` writes the
    result, so no frozen coordinate is ever assigned by this optimizer.
    """

    _FORMAT_VERSION = 1

    def __init__(
        self,
        planned_parameters: AppliedUpdatePlan | Iterable[AppliedTensorPlan],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        maximize: bool = False,
        plan_hash: str | None = None,
        support_hash: str | None = None,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if eps < 0:
            raise ValueError("epsilon must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must both be in [0, 1)")

        if isinstance(planned_parameters, AppliedUpdatePlan):
            applied = planned_parameters
            entries = applied.entries
            expected_plan_hash = applied.plan_hash
            expected_support_hash = applied.support_hash
        else:
            entries = tuple(planned_parameters)
            hash_plan = NativeUpdatePlan(
                tuple(
                    TensorUpdatePlan(
                        name=entry.name,
                        numel=entry.numel,
                        flat_indices=entry.flat_indices,
                        stabilizer=entry.stabilizer,
                        shape=entry.shape,
                    )
                    for entry in entries
                )
            )
            expected_plan_hash = hash_plan.plan_hash
            expected_support_hash = hash_plan.support_hash
        if not entries:
            raise ValueError(
                "ExactSupportAdamW requires at least one planned parameter"
            )
        if plan_hash is not None and plan_hash != expected_plan_hash:
            raise ValueError("provided plan_hash does not match planned parameters")
        if support_hash is not None and support_hash != expected_support_hash:
            raise ValueError("provided support_hash does not match planned parameters")
        self.plan_hash = expected_plan_hash
        self.support_hash = expected_support_hash

        names: set[str] = set()
        parameter_ids: set[int] = set()
        groups: list[dict[str, Any]] = []
        for entry in entries:
            if entry.name in names:
                raise ValueError(f"duplicate planned parameter name: {entry.name}")
            if id(entry.parameter) in parameter_ids:
                raise ValueError(
                    "the same parameter is present in multiple plan entries"
                )
            names.add(entry.name)
            parameter_ids.add(id(entry.parameter))
            if not entry.parameter.is_floating_point():
                raise TypeError(f"{entry.name}: parameter must have a floating dtype")
            if not entry.parameter.is_contiguous():
                raise ValueError(f"{entry.name}: native parameter must be contiguous")
            if entry.parameter.numel() != entry.numel:
                raise ValueError(f"{entry.name}: applied plan numel is stale")
            groups.append(
                {
                    "params": [entry.parameter],
                    "name": entry.name,
                    "numel": entry.numel,
                    "shape": entry.shape,
                    "stabilizer": entry.stabilizer.detach()
                    .to(dtype=torch.float32, device="cpu")
                    .clone(),
                }
            )

        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "maximize": bool(maximize),
        }
        super().__init__(groups, defaults)

        entries_by_name = {entry.name: entry for entry in entries}
        for group in self.param_groups:
            parameter = cast(nn.Parameter, group["params"][0])
            entry = entries_by_name[cast(str, group["name"])]
            indices = entry.flat_indices.to(
                dtype=torch.int64, device=parameter.device
            ).clone()
            self.state[parameter] = {
                "step": 0,
                "flat_indices": indices,
                "exp_avg": torch.zeros(
                    indices.numel(), dtype=torch.float32, device=parameter.device
                ),
                "exp_avg_sq": torch.zeros(
                    indices.numel(), dtype=torch.float32, device=parameter.device
                ),
            }

    @property
    def selected_coordinate_count(self) -> int:
        """Total selected K over the disjoint named-parameter namespace."""

        return sum(
            int(
                self.state[cast(nn.Parameter, group["params"][0])][
                    "flat_indices"
                ].numel()
            )
            for group in self.param_groups
        )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one exact-support AdamW step."""

        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            parameter = cast(nn.Parameter, group["params"][0])
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError(
                    "ExactSupportAdamW does not support sparse gradients"
                )
            if gradient.shape != parameter.shape:
                raise RuntimeError("gradient shape does not match native parameter")
            if not parameter.is_contiguous():
                raise RuntimeError("native parameter became non-contiguous")

            state = self.state[parameter]
            indices = cast(torch.Tensor, state["flat_indices"])
            exp_avg = cast(torch.Tensor, state["exp_avg"])
            exp_avg_sq = cast(torch.Tensor, state["exp_avg_sq"])
            if indices.device != parameter.device:
                indices = indices.to(device=parameter.device)
                state["flat_indices"] = indices
            if exp_avg.device != parameter.device:
                exp_avg = exp_avg.to(dtype=torch.float32, device=parameter.device)
                exp_avg_sq = exp_avg_sq.to(dtype=torch.float32, device=parameter.device)
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
            if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
                raise RuntimeError("optimizer moments must remain fp32")

            selected_gradient = (
                gradient.detach()
                .reshape(-1)
                .index_select(0, indices)
                .to(dtype=torch.float32)
            )
            if cast(bool, group["maximize"]):
                selected_gradient.neg_()

            beta1, beta2 = cast(tuple[float, float], group["betas"])
            step = cast(int, state["step"]) + 1
            state["step"] = step
            exp_avg.mul_(beta1).add_(selected_gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                selected_gradient, selected_gradient, value=1.0 - beta2
            )

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            normalized = exp_avg / bias_correction1
            denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
            denominator.add_(cast(float, group["eps"]))
            normalized.div_(denominator)

            stabilizer = cast(torch.Tensor, group["stabilizer"]).to(
                dtype=torch.float32, device=parameter.device
            )
            if stabilizer.numel() != indices.numel():
                raise RuntimeError("stabilizer length no longer matches exact support")
            flat_parameter = parameter.view(-1)
            selected_values = flat_parameter.index_select(0, indices).to(
                dtype=torch.float32
            )
            learning_rate = cast(float, group["lr"])
            decay = cast(float, group["weight_decay"])
            if decay:
                normalized.add_(selected_values, alpha=decay)
            # A scales the complete selected-coordinate AdamW direction after
            # moment normalization, including selected-only decoupled decay.
            normalized.mul_(stabilizer)
            selected_values.add_(normalized, alpha=-learning_rate)

            # This is the only native-parameter write in the optimizer.  In
            # particular there is no dense mul_/add_ that can round or decay a
            # frozen BF16 coordinate.
            flat_parameter.index_copy_(
                0, indices, selected_values.to(dtype=parameter.dtype)
            )

        return loss

    def state_dict(self) -> dict[str, Any]:
        """Serialize sparse state plus immutable plan identity metadata."""

        serialized_state: dict[int, dict[str, object]] = {}
        serialized_groups: list[dict[str, object]] = []
        for state_id, group in enumerate(self.param_groups):
            parameter = cast(nn.Parameter, group["params"][0])
            state = self.state[parameter]
            serialized_state[state_id] = {
                "step": cast(int, state["step"]),
                "flat_indices": cast(torch.Tensor, state["flat_indices"])
                .detach()
                .to(device="cpu")
                .clone(),
                "exp_avg": cast(torch.Tensor, state["exp_avg"])
                .detach()
                .to(dtype=torch.float32, device="cpu")
                .clone(),
                "exp_avg_sq": cast(torch.Tensor, state["exp_avg_sq"])
                .detach()
                .to(dtype=torch.float32, device="cpu")
                .clone(),
            }
            serialized_groups.append(
                {
                    "params": [state_id],
                    "name": cast(str, group["name"]),
                    "numel": cast(int, group["numel"]),
                    "shape": list(cast(tuple[int, ...], group["shape"])),
                    "stabilizer": cast(torch.Tensor, group["stabilizer"])
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                    .clone(),
                    "lr": cast(float, group["lr"]),
                    "betas": tuple(cast(tuple[float, float], group["betas"])),
                    "eps": cast(float, group["eps"]),
                    "weight_decay": cast(float, group["weight_decay"]),
                    "maximize": cast(bool, group["maximize"]),
                }
            )
        return {
            "state": serialized_state,
            "param_groups": serialized_groups,
            "pcsm_metadata": {
                "format_version": self._FORMAT_VERSION,
                "plan_hash": self.plan_hash,
                "support_hash": self.support_hash,
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load a checkpoint only when its exact plan/support identity matches."""

        metadata = state_dict.get("pcsm_metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("optimizer checkpoint lacks PCSM metadata")
        if metadata.get("format_version") != self._FORMAT_VERSION:
            raise ValueError("unsupported optimizer checkpoint format")
        if metadata.get("support_hash") != self.support_hash:
            raise ValueError("optimizer checkpoint support_hash mismatch")
        if metadata.get("plan_hash") != self.plan_hash:
            raise ValueError("optimizer checkpoint plan_hash mismatch")

        raw_groups = state_dict.get("param_groups")
        raw_state = state_dict.get("state")
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
            raise TypeError("optimizer checkpoint param_groups are malformed")
        if not isinstance(raw_state, Mapping):
            raise TypeError("optimizer checkpoint state is malformed")

        loaded_by_name: dict[str, Mapping[str, Any]] = {}
        for item in raw_groups:
            if not isinstance(item, Mapping):
                raise TypeError("optimizer checkpoint group is malformed")
            name = item.get("name")
            if not isinstance(name, str):
                raise TypeError("optimizer checkpoint group name is malformed")
            if name in loaded_by_name:
                raise ValueError("optimizer checkpoint has invalid group names")
            loaded_by_name[name] = cast(Mapping[str, Any], item)
        current_names = {cast(str, group["name"]) for group in self.param_groups}
        if set(loaded_by_name) != current_names:
            raise ValueError("optimizer checkpoint parameter names do not match plan")

        pending: list[
            tuple[
                dict[str, Any],
                nn.Parameter,
                int,
                torch.Tensor,
                torch.Tensor,
                float,
                float,
                float,
                tuple[float, float],
                bool,
            ]
        ] = []
        for group in self.param_groups:
            parameter = cast(nn.Parameter, group["params"][0])
            name = cast(str, group["name"])
            loaded_group = loaded_by_name[name]
            if loaded_group.get("numel") != group["numel"]:
                raise ValueError(f"{name}: checkpoint numel mismatch")
            raw_shape = loaded_group.get("shape")
            if not isinstance(raw_shape, Sequence) or isinstance(
                raw_shape, (str, bytes)
            ):
                raise TypeError(f"{name}: checkpoint shape is malformed")
            if tuple(int(size) for size in raw_shape) != cast(
                tuple[int, ...], group["shape"]
            ):
                raise ValueError(f"{name}: checkpoint shape mismatch")

            loaded_stabilizer = loaded_group.get("stabilizer")
            if not isinstance(loaded_stabilizer, torch.Tensor):
                raise TypeError(f"{name}: checkpoint stabilizer is malformed")
            if loaded_stabilizer.dtype != torch.float32:
                raise TypeError(f"{name}: checkpoint stabilizer must be fp32")
            expected_stabilizer = cast(torch.Tensor, group["stabilizer"])
            if not torch.equal(
                loaded_stabilizer.to(dtype=torch.float32, device="cpu"),
                expected_stabilizer,
            ):
                raise ValueError(f"{name}: checkpoint stabilizer mismatch")

            raw_params = loaded_group.get("params")
            if (
                not isinstance(raw_params, Sequence)
                or isinstance(raw_params, (str, bytes))
                or len(raw_params) != 1
            ):
                raise TypeError(f"{name}: checkpoint parameter id is malformed")
            state_id = raw_params[0]
            if not isinstance(state_id, (int, str)) or isinstance(state_id, bool):
                raise TypeError(f"{name}: checkpoint parameter id is malformed")
            loaded_state = raw_state.get(state_id)
            if loaded_state is None and isinstance(state_id, int):
                loaded_state = raw_state.get(str(state_id))
            if not isinstance(loaded_state, Mapping):
                raise TypeError(f"{name}: checkpoint sparse state is missing")

            loaded_indices = loaded_state.get("flat_indices")
            exp_avg = loaded_state.get("exp_avg")
            exp_avg_sq = loaded_state.get("exp_avg_sq")
            step = loaded_state.get("step")
            if not isinstance(loaded_indices, torch.Tensor):
                raise TypeError(f"{name}: checkpoint support is malformed")
            if loaded_indices.dtype != torch.int64 or loaded_indices.ndim != 1:
                raise TypeError(f"{name}: checkpoint support must be flat int64")
            expected_indices = cast(
                torch.Tensor, self.state[parameter]["flat_indices"]
            ).to(device="cpu")
            loaded_indices = loaded_indices.to(dtype=torch.int64, device="cpu")
            if not torch.equal(loaded_indices, expected_indices):
                raise ValueError(f"{name}: checkpoint exact support mismatch")
            if not isinstance(exp_avg, torch.Tensor) or not isinstance(
                exp_avg_sq, torch.Tensor
            ):
                raise TypeError(f"{name}: checkpoint moments are malformed")
            if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
                raise TypeError(f"{name}: checkpoint moments must be fp32")
            if exp_avg.shape != expected_indices.shape or (
                exp_avg_sq.shape != expected_indices.shape
            ):
                raise ValueError(f"{name}: checkpoint sparse moment shape mismatch")
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(f"{name}: checkpoint step is malformed")
            loaded_lr = _loaded_nonnegative_float(loaded_group, "lr")
            loaded_eps = _loaded_nonnegative_float(loaded_group, "eps")
            loaded_decay = _loaded_nonnegative_float(loaded_group, "weight_decay")
            loaded_betas = loaded_group.get("betas")
            if not isinstance(loaded_betas, Sequence) or len(loaded_betas) != 2:
                raise TypeError("checkpoint betas are malformed")
            betas = (float(loaded_betas[0]), float(loaded_betas[1]))
            if not all(0 <= beta < 1 for beta in betas):
                raise ValueError("checkpoint betas are out of range")
            maximize = loaded_group.get("maximize")
            if not isinstance(maximize, bool):
                raise TypeError("checkpoint maximize is malformed")
            pending.append(
                (
                    group,
                    parameter,
                    step,
                    exp_avg.detach()
                    .to(dtype=torch.float32, device=parameter.device)
                    .clone(),
                    exp_avg_sq.detach()
                    .to(dtype=torch.float32, device=parameter.device)
                    .clone(),
                    loaded_lr,
                    loaded_eps,
                    loaded_decay,
                    betas,
                    maximize,
                )
            )

        # Mutate only after every group has passed all validation.
        for (
            group,
            parameter,
            step,
            exp_avg,
            exp_avg_sq,
            loaded_lr,
            loaded_eps,
            loaded_decay,
            betas,
            maximize,
        ) in pending:
            group["lr"] = loaded_lr
            group["eps"] = loaded_eps
            group["weight_decay"] = loaded_decay
            group["betas"] = betas
            group["maximize"] = maximize
            self.state[parameter]["step"] = step
            self.state[parameter]["exp_avg"] = exp_avg
            self.state[parameter]["exp_avg_sq"] = exp_avg_sq


def _loaded_nonnegative_float(group: Mapping[str, Any], key: str) -> float:
    value = group.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"checkpoint {key} is malformed")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"checkpoint {key} is out of range")
    return result
