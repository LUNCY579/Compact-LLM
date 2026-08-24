"""Matched KV-slot and total-persistent-byte accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    layers: int
    kv_heads: int
    head_dim: int
    kv_dtype_bytes: int = 2
    score_dtype_bytes: int = 4
    count_dtype_bytes: int = 4
    id_dtype_bytes: int = 8
    position_dtype_bytes: int = 8
    token_dtype_bytes: int = 8
    active_dtype_bytes: int = 1

    def __post_init__(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("all cache geometry values must be positive")


@dataclass(frozen=True, slots=True)
class BudgetReport:
    slots_per_layer_head: int
    kv_bytes: int
    auxiliary_bytes: int
    controller_bytes: int
    total_persistent_bytes: int
    significance_state: bool
    metadata_state: bool


class BudgetLedger:
    def __init__(self, geometry: CacheGeometry) -> None:
        self.geometry = geometry

    @property
    def rows(self) -> int:
        return self.geometry.layers * self.geometry.kv_heads

    @property
    def kv_bytes_per_slot(self) -> int:
        return self.rows * 2 * self.geometry.head_dim * self.geometry.kv_dtype_bytes

    def auxiliary_bytes_per_slot(self, *, significance: bool, metadata: bool = True) -> int:
        per_row = 0
        if significance:
            per_row += self.geometry.score_dtype_bytes + self.geometry.count_dtype_bytes
        if metadata:
            per_row += (
                self.geometry.id_dtype_bytes
                + self.geometry.position_dtype_bytes
                + self.geometry.token_dtype_bytes
                + self.geometry.active_dtype_bytes
            )
        return self.rows * per_row

    def report(
        self,
        slots_per_layer_head: int,
        *,
        significance: bool,
        metadata: bool = True,
        controller_bytes: int = 0,
    ) -> BudgetReport:
        if slots_per_layer_head < 0 or controller_bytes < 0:
            raise ValueError("slot and controller budgets must be non-negative")
        kv = slots_per_layer_head * self.kv_bytes_per_slot
        auxiliary = slots_per_layer_head * self.auxiliary_bytes_per_slot(
            significance=significance,
            metadata=metadata,
        )
        return BudgetReport(
            slots_per_layer_head=slots_per_layer_head,
            kv_bytes=kv,
            auxiliary_bytes=auxiliary,
            controller_bytes=controller_bytes,
            total_persistent_bytes=kv + auxiliary + controller_bytes,
            significance_state=significance,
            metadata_state=metadata,
        )

    def slots_within_total_bytes(
        self,
        total_bytes: int,
        *,
        significance: bool,
        metadata: bool = True,
        controller_bytes: int = 0,
    ) -> int:
        if total_bytes < 0 or controller_bytes < 0 or controller_bytes > total_bytes:
            raise ValueError("invalid total/controller byte budget")
        per_slot = self.kv_bytes_per_slot + self.auxiliary_bytes_per_slot(
            significance=significance,
            metadata=metadata,
        )
        return (total_bytes - controller_bytes) // per_slot

    def match_total_bytes(
        self,
        baseline_slots: int,
        *,
        baseline_significance: bool,
        candidate_significance: bool,
        metadata: bool = True,
        baseline_controller_bytes: int = 0,
        candidate_controller_bytes: int = 0,
    ) -> tuple[BudgetReport, BudgetReport]:
        baseline = self.report(
            baseline_slots,
            significance=baseline_significance,
            metadata=metadata,
            controller_bytes=baseline_controller_bytes,
        )
        candidate_slots = self.slots_within_total_bytes(
            baseline.total_persistent_bytes,
            significance=candidate_significance,
            metadata=metadata,
            controller_bytes=candidate_controller_bytes,
        )
        candidate = self.report(
            candidate_slots,
            significance=candidate_significance,
            metadata=metadata,
            controller_bytes=candidate_controller_bytes,
        )
        if candidate.total_persistent_bytes > baseline.total_persistent_bytes:
            raise AssertionError("candidate exceeded matched total-byte budget")
        return baseline, candidate
