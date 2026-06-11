"""Small result container used by app and engine systems."""

from __future__ import annotations

from dataclasses import dataclass, field

from game.core.events import GameEvent
from game.core.hci import HciResultAnalysis


@dataclass
class Result[T]:
    success: bool
    value: T | None = None
    events: list[GameEvent] = field(default_factory=list)
    error: str | None = None
    hci: HciResultAnalysis | None = None

    @classmethod
    def ok(
        cls,
        value: T | None = None,
        events: list[GameEvent] | None = None,
        *,
        hci: HciResultAnalysis | None = None,
    ) -> Result[T]:
        return cls(success=True, value=value, events=events or [], hci=hci)

    @classmethod
    def fail(
        cls,
        error: str,
        events: list[GameEvent] | None = None,
        *,
        hci: HciResultAnalysis | None = None,
    ) -> Result[T]:
        return cls(success=False, value=None, events=events or [], error=error, hci=hci)
