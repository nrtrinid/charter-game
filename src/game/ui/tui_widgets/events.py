"""Combat event parsing helpers shared by animation and combat widgets."""

from __future__ import annotations

from typing import Any


def _event_kind(event: Any) -> str:
    event_type = getattr(event, "event_type", "")
    return str(getattr(event_type, "value", event_type))


def _is_effort_change_event(event: Any) -> bool:
    return (
        _event_kind(event) == "status_changed"
        and str(getattr(event, "status", "")).lower() == "effort"
    )


def _is_combat_effect_event(event: Any) -> bool:
    return _event_kind(event) == "combat_effect"


def _combat_effect_label(event: Any) -> str:
    return str(getattr(event, "label", "")).strip()


def _combat_effect_emphasis(event: Any) -> str:
    return str(getattr(event, "emphasis", "")).lower()


def _combat_effect_style(event: Any) -> str:
    emphasis = _combat_effect_emphasis(event)
    if emphasis == "good":
        return "bold green"
    if emphasis == "bad":
        return "bold bright_blue"
    if emphasis == "critical":
        return "bold red"
    return "bold yellow"