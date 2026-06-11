"""Internal Textual screen model types."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from game.app.views import ScreenAction


@dataclass(frozen=True)
class ScreenDescriptor:
    state_id: str
    activate: Callable[[str], None]
    back: Callable[[], None]
    locked: bool = False


@dataclass(frozen=True)
class TuiScreenModel:
    state_id: str
    title: str
    body: str
    actions: Sequence[ScreenAction]
    message: str = ""
    detail: str = ""
    log: str = ""
    activate: Callable[[str], None] | None = None
    back: Callable[[], None] | None = None
