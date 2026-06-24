"""Shared player-facing HCI text helpers for terminal frontends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game.app.views import FormationSlotView, FormationView, HeroListEntry, ScreenAction
from game.core.events import GameEvent
from game.core.hci import HciResultAnalysis, tactical_brief_lines
from game.ui.wounds import mortal_wound_badge

META_SEPARATOR = "  |  "
PARTY_WATCH_ROWS: tuple[tuple[str, str], ...] = (
    ("BACK_LEFT", "FRONT_LEFT"),
    ("BACK_RIGHT", "FRONT_RIGHT"),
)
PARTY_WATCH_COLUMN_WIDTH = 22


def label_text(value: object) -> str:
    return str(value).replace("_", " ").title()


def format_formation_slot(slot: str) -> str:
    """Player-facing formation slot label, e.g. BACK_LEFT -> Back Left."""
    return label_text(slot)


def _hp_from_vitals_line(vitals_line: str) -> str:
    hp_part = vitals_line.split(",", maxsplit=1)[0].strip()
    return hp_part.removesuffix(" HP").strip()


def format_party_watch_slot(slot_view: FormationSlotView) -> str:
    """Compact party-watch cell, e.g. Orren 9/11 [xoo]."""
    if slot_view.hero_name == "empty":
        return "—"
    name = slot_view.hero_name.split()[0]
    hp = _hp_from_vitals_line(slot_view.vitals_line)
    if slot_view.mortal_wounds:
        return f"{name} {hp} {mortal_wound_badge(slot_view.mortal_wounds)}"
    return f"{name} {hp}"


def format_party_watch(
    view: FormationView,
    *,
    column_width: int = PARTY_WATCH_COLUMN_WIDTH,
) -> str:
    """Render the compact 2x2 party watch block for location scenes."""
    slots_by_label = {slot.slot_label: slot for slot in view.slots}
    if not any(slot.hero_id is not None for slot in view.slots):
        return ""
    lines = ["Party Watch"]
    for left_slot, right_slot in PARTY_WATCH_ROWS:
        left = format_party_watch_slot(slots_by_label[left_slot]).ljust(column_width)
        right = format_party_watch_slot(slots_by_label[right_slot])
        lines.append(f"{left}{right}")
    return "\n".join(lines)


def format_move_destination_label(
    to_slot: str,
    occupant_name: str,
    *,
    number: str = "",
) -> str:
    """Destination-first combat movement label."""
    slot_label = format_formation_slot(to_slot)
    if occupant_name == "empty":
        return f"{slot_label} — open slot"
    return f"{slot_label} — swap with {occupant_name}"


def format_compact_roster_row(
    hero: HeroListEntry,
    *,
    memorial: bool = False,
) -> str:
    """Scan-critical roster row without memory, gear, or quirk noise."""
    if memorial:
        wounds = mortal_wound_badge(hero.mortal_wounds) if hero.mortal_wounds else "none"
        return format_meta_line(
            hero.name,
            label_text(hero.class_id),
            f"Wounds {wounds}",
        )
    slot_label = format_formation_slot(hero.slot) if hero.slot else "Reserve"
    parts: list[str] = [
        slot_label,
        hero.name,
        label_text(hero.class_id),
        f"{hero.hp}/{hero.max_hp} HP",
        f"{hero.effort}/{hero.max_effort} Effort",
    ]
    if hero.morale and hero.morale.lower() != "steady":
        parts.append(f"Morale {hero.morale}")
    if hero.strain and hero.strain.lower() != "steady":
        parts.append(f"Strain {hero.strain}")
    if hero.mortal_wounds:
        parts.append(f"Wounds {mortal_wound_badge(hero.mortal_wounds)}")
    abnormal = tuple(
        status for status in hero.statuses if status.lower() != "ready"
    )
    if abnormal:
        parts.append(", ".join(abnormal))
    return format_meta_line(*parts)


def format_formation_board_cell(slot_view: Any) -> tuple[str, str, str]:
    """Return (name_line, detail_line, state_line) for a formation board cell."""
    if slot_view.hero_name == "empty":
        return ("empty", "", "")
    name_line = slot_view.hero_name
    class_name = getattr(slot_view, "class_name", "") or label_text(
        getattr(slot_view, "class_id", "")
    )
    detail_line = format_meta_line(
        class_name,
        getattr(slot_view, "vitals_line", "") or getattr(slot_view, "condition", ""),
    )
    state_line = getattr(slot_view, "protection_line", "") or ""
    abnormal = getattr(slot_view, "abnormal_status", "")
    if abnormal:
        state_line = format_meta_line(state_line, abnormal) if state_line else abnormal
    return (name_line, detail_line, state_line)


def format_formation_lane_summary(
    slots: Sequence[tuple[str, str]],
    *,
    slot_order: Sequence[str] | None = None,
) -> str:
    """Compact lane summary, e.g. Back Left Orren / Back Right Senn."""
    names_by_slot = dict(slots)
    order = slot_order or tuple(slot for slot, _name in slots)
    parts = [
        f"{format_formation_slot(slot)} {names_by_slot.get(slot, 'empty')}"
        for slot in order
    ]
    return " / ".join(parts)


def risk_label(action: ScreenAction) -> str:
    return label_text(action.risk)


def kind_label(action: ScreenAction) -> str:
    return label_text(action.kind)


def unavailable_message(action: ScreenAction, *, fallback: str = "No route is available.") -> str:
    reason = action.unavailable_reason or action.description or fallback
    return f"{action.label} is unavailable. {reason}"


def action_dock_detail(action: ScreenAction) -> str:
    if not action.enabled and action.unavailable_reason:
        return action.unavailable_reason
    return action.description or action.preview or action.result_hint


def event_messages_text(events: Sequence[GameEvent], *, limit: int = 8) -> str:
    visible_events = _dedupe_combat_effect_messages(events)
    return "\n".join(event.message for event in visible_events[-limit:])


def _dedupe_combat_effect_messages(events: Sequence[GameEvent]) -> list[GameEvent]:
    typed_keys = {
        _combat_effect_key(event)
        for event in events
        if _event_kind(event) == "combat_effect"
    }
    visible: list[GameEvent] = []
    for event in events:
        if _event_kind(event) == "status_changed" and _status_key(event) in typed_keys:
            continue
        visible.append(event)
    return visible


def _event_kind(event: GameEvent) -> str:
    event_type = getattr(event, "event_type", "")
    return str(getattr(event_type, "value", event_type))


def _combat_effect_key(event: GameEvent) -> tuple[str, str]:
    actor_id = str(getattr(event, "actor_id", ""))
    resource = str(getattr(event, "resource", ""))
    tag = str(getattr(event, "tag", ""))
    if resource:
        return actor_id, resource
    if tag:
        return actor_id, tag
    return actor_id, str(getattr(event, "effect_type", ""))


def _status_key(event: GameEvent) -> tuple[str, str]:
    return str(getattr(event, "actor_id", "")), str(getattr(event, "status", ""))


def hci_summary_lines(hci: HciResultAnalysis) -> list[str]:
    return list(hci.summary) if hci.summary else tactical_brief_lines(hci)


def result_log_text(
    events: Sequence[GameEvent],
    hci: HciResultAnalysis | None,
    *,
    max_lines: int = 14,
) -> str:
    if hci is not None:
        lines = hci_summary_lines(hci)
        if lines:
            return "\n".join(lines[:max_lines])
    return event_messages_text(events)


def format_meta_line(*parts: object) -> str:
    """Join compact metadata with consistent spacing for monospaced panes."""
    return META_SEPARATOR.join(
        text for part in parts if (text := str(part).strip())
    )


def primary_hotkey(action: ScreenAction) -> str:
    return next((alias for alias in action.aliases if len(alias) == 1), "")


def generic_action_detail(
    action: ScreenAction,
    *,
    safe_default: bool,
) -> str:
    hotkey = primary_hotkey(action)
    lines = ["Stage Focus", "", action.label]
    kind = kind_label(action)
    if kind and str(kind).lower() not in {"general", ""}:
        lines.append(kind)
    if hotkey:
        lines.append(f"Hotkey: {hotkey}")
    if action.default and safe_default:
        lines.append("Default command")
    elif action.default:
        lines.append("Marked default, but Enter will avoid it because it is not safe.")
    if not action.enabled:
        lines.append("Unavailable")
        if action.unavailable_reason:
            lines.append(action.unavailable_reason)
    if action.cost:
        lines.append(f"Cost: {action.cost}")
    if action.preview:
        lines.extend(("", "Preview", action.preview))
    if action.description:
        lines.extend(("", "Detail", action.description))
    if action.result_hint:
        lines.extend(("", "Expected Result", action.result_hint))
    if action.confirm:
        lines.extend(("", "Confirmation", action.confirm))
    return "\n".join(lines)


def format_scene_body(
    title: str,
    sections: Sequence[tuple[str, Sequence[str]]],
    *,
    lead: str = "",
    hint: str = "",
) -> str:
    """Compact screen body with optional lead line and first-visit hint."""
    lines = [title]
    if lead:
        lines.extend(("", lead))
    for heading, entries in sections:
        clean_entries = [entry for entry in entries if entry]
        if not clean_entries:
            continue
        lines.extend(("", heading, *clean_entries))
    if hint:
        lines.extend(("", hint))
    return "\n".join(lines)


def format_fixed_table(
    headers: tuple[str, ...],
    rows: Sequence[tuple[str, ...]],
    *,
    widths: tuple[int, ...] | None = None,
) -> str:
    if not rows:
        return "none"
    if widths is None:
        widths = tuple(
            max(len(headers[index]), *(len(str(row[index])) for row in rows))
            for index in range(len(headers))
        )
    lines = [
        "".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "".join("-" * widths[index] for index in range(len(headers))),
    ]
    for row in rows:
        lines.append(
            "".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        )
    return "\n".join(lines)


def format_quantity_rows(items: dict[str, int]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (label_text(item_id), f"x{quantity}")
        for item_id, quantity in sorted(items.items())
        if quantity > 0
    )


def format_inventory_section(
    title: str,
    rows: Sequence[tuple[str, ...]],
    *,
    headers: tuple[str, ...] = ("Name", "Qty"),
) -> str:
    if not rows:
        return f"{title}\nnone"
    return f"{title}\n{format_fixed_table(headers, rows)}"


def format_supply_stock_rows(actions: Sequence[ScreenAction]) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for action in actions:
        if action.value == "back":
            continue
        state = "ready" if action.enabled else "short coin"
        cost = action.cost or "-"
        rows.append((action.label, cost, state))
    return tuple(rows)


def format_gear_stock_rows(items: Sequence[Any]) -> tuple[tuple[str, str, str, str], ...]:
    visible = [
        item
        for item in items
        if getattr(item, "owned_count", 0) > 0 or getattr(item, "equipped_count", 0) > 0
    ]
    return tuple(
        (
            str(getattr(item, "name", "")),
            str(getattr(item, "owned_count", 0)),
            str(getattr(item, "available_count", 0)),
            str(getattr(item, "equipped_count", 0)),
        )
        for item in visible
    )


def format_gear_shop_rows(items: Sequence[Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(getattr(item, "name", "")),
            str(getattr(item, "state", "")),
            "reward"
            if getattr(item, "cost", None) is None
            else str(getattr(item, "cost", "")),
        )
        for item in items
    )


def format_equipped_kit_rows(heroes: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            str(getattr(hero, "name", "")),
            str(getattr(hero, "equipped_gear_name", "") or "-"),
        )
        for hero in heroes
    )
