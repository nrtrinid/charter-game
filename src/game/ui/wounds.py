"""Display helpers for mortal wound state."""

from __future__ import annotations

from game.combat.death import MORTAL_WOUNDS_TO_DIE


def mortal_wound_count(value: int) -> str:
    return f"{value}/{MORTAL_WOUNDS_TO_DIE}"


def mortal_wound_badge(value: int, *, markup_safe: bool = False) -> str:
    wound_count = max(0, min(value, MORTAL_WOUNDS_TO_DIE))
    open_count = MORTAL_WOUNDS_TO_DIE - wound_count
    badge = f"[{'x' * wound_count}{'o' * open_count}]"
    if markup_safe:
        return badge.replace("[", "\\[")
    return badge
