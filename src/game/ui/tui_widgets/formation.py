"""Formation layout helpers shared by town and combat widgets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from game.ui.tui_widgets.constants import FORMATION_INWARD_SLOTS


def formation_slot_faces_inward(slot_key: str) -> bool:
    return slot_key in FORMATION_INWARD_SLOTS


def _formation_preview_text(
    before: Sequence[tuple[str, str]],
    after: Sequence[tuple[str, str]],
) -> str:
    if not before or not after:
        return ""
    before_map = dict(before)
    after_map = dict(after)
    rows = (
        ("FRONT_LEFT", "FRONT_RIGHT"),
        ("BACK_LEFT", "BACK_RIGHT"),
    )
    before_lines = [_formation_preview_row(before_map, row) for row in rows]
    after_lines = [_formation_preview_row(after_map, row) for row in rows]
    return "\n".join(
        f"{left}  ->  {right}" for left, right in zip(before_lines, after_lines, strict=True)
    )


def _formation_preview_row(names_by_slot: Mapping[str, str], row: tuple[str, str]) -> str:
    return " ".join(f"[{_preview_name(names_by_slot.get(slot, 'empty'))}]" for slot in row)


def _preview_name(name: str) -> str:
    if name == "empty":
        return "    "
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[-1][0]}".upper().ljust(4)
    return name[:4].title().ljust(4)


def _slot_name(slot: str) -> str:
    return slot.replace("_", " ").title()