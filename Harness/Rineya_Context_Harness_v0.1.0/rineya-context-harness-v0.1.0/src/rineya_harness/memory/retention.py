from __future__ import annotations

from dataclasses import dataclass, field

from rineya_harness.domain.enums import MemoryKind, MemoryStatus, MemoryTemperature
from rineya_harness.domain.models import MemoryItem


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Hard policy around model-generated future-utility proposals.

    Active evidence is not permanently protected by default because RSF must be able to
    replace raw passages with compact attributable claims. The state invariant validator
    enforces replacement-before-eviction for active evidence.
    """

    always_retain_kinds: frozenset[MemoryKind] = field(
        default_factory=lambda: frozenset(
            {
                MemoryKind.GLOBAL_INVARIANT,
                MemoryKind.OPEN_GOAL,
                MemoryKind.CONFLICT,
            }
        )
    )
    protect_active_evidence: bool = False

    def protected(self, item: MemoryItem) -> bool:
        if item.pinned or item.temperature is MemoryTemperature.PINNED:
            return True
        if item.status is not MemoryStatus.ACTIVE:
            return False
        if item.kind in self.always_retain_kinds:
            return True
        return item.kind is MemoryKind.ACTIVE_EVIDENCE and self.protect_active_evidence

    def default_evictable(self, item: MemoryItem) -> bool:
        if self.protected(item):
            return False
        return item.kind in {
            MemoryKind.RESOLVED_EVIDENCE,
            MemoryKind.TRANSIENT_OBSERVATION,
            MemoryKind.COMPLETED_CHECKPOINT,
            MemoryKind.SUPERSEDED_STATE,
            MemoryKind.EVENT_HISTORY,
        }
