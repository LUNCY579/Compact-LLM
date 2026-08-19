from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    def __init__(self, value: datetime) -> None:
        if value.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime")
        self.value = value

    def now(self) -> datetime:
        return self.value
