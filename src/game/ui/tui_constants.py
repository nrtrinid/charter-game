"""Shared TUI constants and route/AI label helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from game.app.views import ScreenActionRisk
from game.combat.enemy_decision import (
    PRODUCTION_ENEMY_AI_MODE_DESCRIPTIONS,
    PRODUCTION_ENEMY_AI_MODE_LABELS,
    SUPPORTED_PRODUCTION_ENEMY_AI_MODES,
    production_enemy_movement_mode,
    production_enemy_wait_mode,
)

DEFAULT_SAVE_PATH = Path("saves/company.json")
BREACH_PENDING_FLAG = "opening_breach_pending"
DEFAULT_COMPANY_NAME = "Haven Charter"
BEAT_ANIMATION_LAST_FRAME = 4
BEAT_ANIMATION_START_FRAME = -1
BEAT_IDLE_CYCLE = 16
TURN_FLASH_LAST_FRAME = 1
UNSAFE_DEFAULT_RISKS = {
    ScreenActionRisk.COSTLY,
    ScreenActionRisk.RISKY,
    ScreenActionRisk.IRREVERSIBLE,
    "costly",
    "risky",
    "irreversible",
}
GLOBAL_SHORTCUT_TEXT = "Shortcuts: [M] Map  [P] Pack  [C] Company  [?] Help"
GLOBAL_SHORTCUT_SCREENS = {
    "regional_place",
    "regional_interact",
    "regional_map",
    "world_map",
    "dungeon",
    "dungeon_interact",
    "dungeon_map",
    "pack",
    "company_summary",
    "help",
}


def _route_direction_label(exit_node: Any) -> str:
    direction = str(getattr(exit_node, "direction", "")).strip()
    return direction.title() if direction else "Listed Exit"


def _route_summary_line(exit_node: Any) -> str:
    pieces = [_route_direction_label(exit_node)]
    tag = _route_exception_tag(exit_node)
    if tag:
        pieces.append(tag)
    return " - ".join(piece for piece in pieces if piece)


def _route_exception_tag(exit_node: Any) -> str:
    if bool(getattr(exit_node, "cleared", False)):
        return ""
    node_type = str(getattr(exit_node, "node_type", ""))
    if node_type not in {"boss", "breach", "maze"}:
        return ""
    return node_type.replace("_", " ").title()


def _route_warning_line(exit_node: Any) -> str:
    if bool(getattr(exit_node, "cleared", False)):
        return ""
    node_type = str(getattr(exit_node, "node_type", ""))
    if node_type == "boss":
        return "Warning: serious danger ahead."
    if node_type == "breach":
        return "Warning: breach threshold ahead."
    if node_type == "maze":
        return "Warning: Maze route ahead."
    return ""


def _enemy_ai_mode_label(mode: str) -> str:
    return PRODUCTION_ENEMY_AI_MODE_LABELS.get(mode, mode.replace("_", " ").title())


def _next_enemy_ai_mode(mode: str) -> str:
    modes = SUPPORTED_PRODUCTION_ENEMY_AI_MODES
    try:
        index = modes.index(mode)
    except ValueError:
        return modes[0]
    return modes[(index + 1) % len(modes)]


def _enemy_ai_mode_text(current_mode: str) -> str:
    lines = [
        "Enemy AI",
        f"Current: {_enemy_ai_mode_label(current_mode)}",
        (
            "Timing: wait "
            f"{_enemy_timing_label(production_enemy_wait_mode(current_mode))}, "
            f"move {_enemy_timing_label(production_enemy_movement_mode(current_mode))}"
        ),
        "",
    ]
    for mode in SUPPORTED_PRODUCTION_ENEMY_AI_MODES:
        marker = "*" if mode == current_mode else "-"
        lines.append(
            f"{marker} {_enemy_ai_mode_label(mode)}: "
            f"{PRODUCTION_ENEMY_AI_MODE_DESCRIPTIONS[mode]}"
        )
    return "\n".join(lines)


def _enemy_timing_label(mode: str) -> str:
    return mode.replace("_", " ").title()


def _enemy_ai_controls_text(*, ai_mode: str) -> str:
    return _enemy_ai_mode_text(ai_mode)
