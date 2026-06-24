"""Textual widgets used by the fullscreen frontend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text
from textual.widgets import Static

from game.app.views import (
    CombatView,
    DungeonView,
    ExpeditionReportView,
    FormationView,
    GearInventoryView,
    RelicBrokerView,
    ScreenAction,
    ShellStatusView,
    SupplyShopView,
    TownDashboardView,
)
from game.ui.hci_text import (
    format_compact_roster_row,
    format_equipped_kit_rows,
    format_fixed_table,
    format_formation_board_cell,
    format_formation_slot,
    format_gear_shop_rows,
    format_gear_stock_rows,
    format_quantity_rows,
    format_scene_body,
    format_supply_stock_rows,
)
from game.ui.screens import EventBeat
from game.ui.wounds import mortal_wound_badge

GRID_ROWS: tuple[tuple[str, str], ...] = (
    ("BACK_LEFT", "BACK_RIGHT"),
    ("FRONT_LEFT", "FRONT_RIGHT"),
)
CELL_WIDTH = 31
COMBAT_CELL_WIDTH = 14
COMBAT_FIELD_LAYOUT = "mini"
MINI_SLOT_WIDTH = 18
MINI_ART_WIDTH = 6
MINI_MIDDLE_WIDTH = 10
MINI_SIDE_GAP = "  "
MINI_VISIBLE_STATUS_TAGS = 2
PARTY_COMBAT_ROWS: tuple[tuple[str, str], ...] = (
    ("BACK_LEFT", "FRONT_LEFT"),
    ("BACK_RIGHT", "FRONT_RIGHT"),
)
ENEMY_COMBAT_ROWS: tuple[tuple[str, str], ...] = (
    ("FRONT_LEFT", "BACK_LEFT"),
    ("FRONT_RIGHT", "BACK_RIGHT"),
)
META_SEPARATOR = "  |  "
COMMAND_LABEL_WIDTH = 32
COMMAND_KEY_WIDTH = 6
DOCK_WIDE_MIN_WIDTH = 86
DOCK_COMMAND_COLUMN_WIDTH = 42
DOCK_GAP = "  |  "
MAP_NODE_WIDTH = 4
MAP_NODE_CENTER = (MAP_NODE_WIDTH - 1) // 2
MAP_X_STRIDE = 7
MAP_Y_STRIDE = 2
MAP_VIEWPORT_WIDTH = 31
MAP_VIEWPORT_HEIGHT = 9
FORMATION_INWARD_SLOTS = frozenset({"FRONT_LEFT", "FRONT_RIGHT"})
ACTION_ANIMATION_LAST_FRAME = 4
IDLE_ANIMATION_CYCLE = 4
ANTICIPATION_FRAME = 0
SOURCE_ACTION_FRAME = 1
DAMAGE_REACTION_FRAME = 3
PORTRAIT_EFFECT_LANE_COUNT = 2
BEAT_PORTRAIT_WIDTH = 32
BEAT_CONNECTOR_WIDTH = 8
BEAT_ART_SIDE_PADDING = 4


def _compact_art_lines(
    art_lines: Sequence[str],
    *,
    max_lines: int,
    max_width: int,
) -> list[str]:
    return [line[:max_width].rstrip() for line in art_lines[:max_lines]]


def _authored_animation_frames(actor: Any, cue: str) -> tuple[tuple[str, ...], ...]:
    return tuple((getattr(actor, "art_frames", {}) or {}).get(cue, ()))


def _authored_animation_holds(actor: Any, cue: str) -> tuple[int, ...]:
    frames = _authored_animation_frames(actor, cue)
    raw_holds = tuple((getattr(actor, "art_frame_holds", {}) or {}).get(cue, ()))
    holds = tuple(_animation_hold_ticks(hold) for hold in raw_holds[: len(frames)])
    if len(holds) < len(frames):
        holds = (*holds, *(1 for _ in range(len(frames) - len(holds))))
    return holds


def _animation_hold_ticks(hold: Any) -> int:
    if isinstance(hold, int):
        hold_units = hold
    elif isinstance(hold, str):
        hold_units = int(hold)
    else:
        hold_units = 1
    hold_units = max(1, hold_units)
    return max(1, (hold_units + 1) // 2)


def _held_frame_index(actor: Any, cue: str, animation_frame: int) -> int | None:
    if animation_frame < 0:
        return None
    elapsed = 0
    for index, hold in enumerate(_authored_animation_holds(actor, cue)):
        elapsed += hold
        if animation_frame < elapsed:
            return index
    return None


def _animation_art_lines(actor: Any, cue: str, frame: int) -> tuple[str, ...]:
    base_lines = tuple(getattr(actor, "art_lines", ()) or ())
    if not cue:
        return base_lines

    authored = _authored_animation_frames(actor, cue)
    if authored:
        if cue == "idle":
            return tuple(authored[frame % len(authored)])
        if cue in {"attack", "cast"}:
            held_index = _held_frame_index(actor, cue, frame)
            if held_index is not None:
                return tuple(authored[held_index])
            return base_lines
        sequence = (base_lines, *authored, base_lines)
        return tuple(sequence[min(max(frame, 0), len(sequence) - 1)])

    return _procedural_animation_art_lines(actor, cue, frame, base_lines)


def _portrait_animation_art_lines(
    actor: Any,
    cue: str,
    animation_frame: int,
    *,
    hp: int | None = None,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
) -> tuple[str, ...]:
    if cue in {"attack", "cast"} and _authored_animation_frames(actor, cue):
        if 0 <= animation_frame < _authored_action_frame_count(actor, cue):
            return _animation_art_lines(actor, cue, animation_frame)
        if animation_frame > action_end_frame:
            return _animation_art_lines(
                actor,
                "idle",
                _beat_idle_pose_frame(
                    actor,
                    animation_frame,
                    hp=hp,
                    post_action=True,
                    action_end_frame=action_end_frame,
                ),
            )
        return _animation_art_lines(actor, "", 0)

    staged_cue, staged_frame = _staged_animation_cue(cue, animation_frame, impact_frame)
    if staged_cue:
        return _animation_art_lines(actor, staged_cue, staged_frame)
    if cue and animation_frame > action_end_frame:
        return _animation_art_lines(
            actor,
            "idle",
            _beat_idle_pose_frame(
                actor,
                animation_frame,
                hp=hp,
                post_action=True,
                action_end_frame=action_end_frame,
            ),
        )
    if cue:
        return _animation_art_lines(actor, "", 0)
    return _animation_art_lines(
        actor,
        "idle",
        _beat_idle_pose_frame(actor, animation_frame, hp=hp),
    )


def _staged_animation_cue(
    cue: str,
    animation_frame: int,
    impact_frame: int = DAMAGE_REACTION_FRAME,
) -> tuple[str, int]:
    if cue == "hurt":
        return ("hurt", 1) if animation_frame == impact_frame else ("", 0)
    if cue in {"attack", "cast"} and animation_frame == ANTICIPATION_FRAME:
        return "anticipate", 1
    if cue in {"attack", "cast"}:
        return (cue, 1) if animation_frame == SOURCE_ACTION_FRAME else ("", 0)
    return "", 0


def _idle_pose_frame(actor: Any, animation_frame: int) -> int:
    phase = (animation_frame + _actor_idle_offset(actor)) % IDLE_ANIMATION_CYCLE
    return 1 if phase in {1, 2} else 0


def _beat_idle_pose_frame(
    actor: Any,
    animation_frame: int,
    *,
    hp: int | None = None,
    post_action: bool = False,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
) -> int:
    base_frame = animation_frame
    if post_action:
        base_frame = max(0, animation_frame - action_end_frame - 2)
    return _idle_pose_frame(actor, base_frame // _idle_frame_hold(actor, hp=hp))


def _idle_frame_hold(actor: Any, *, hp: int | None = None) -> int:
    max_hp = max(1, int(getattr(actor, "max_hp", 1)))
    current_hp = int(getattr(actor, "hp", max_hp) if hp is None else hp)
    ratio = max(0.0, min(1.0, current_hp / max_hp))
    if ratio <= 0.25:
        return 4
    if ratio <= 0.5:
        return 3
    return 2


def _actor_idle_offset(actor: Any) -> int:
    actor_id = str(getattr(actor, "actor_id", getattr(actor, "name", "")))
    return sum(ord(character) for character in actor_id) % IDLE_ANIMATION_CYCLE


def _authored_action_frame_count(actor: Any, cue: str) -> int:
    if cue not in {"attack", "cast"}:
        return 0
    return sum(_authored_animation_holds(actor, cue))


def _action_impact_frame(actor: Any | None, cue: str) -> int:
    if actor is None or cue not in {"attack", "cast"}:
        return DAMAGE_REACTION_FRAME
    frame_count = _authored_action_frame_count(actor, cue)
    if frame_count <= 0:
        return DAMAGE_REACTION_FRAME
    authored_frames = _authored_animation_frames(actor, cue)
    authored_index = int(
        (getattr(actor, "art_frame_impacts", {}) or {}).get(
            cue,
            max(0, len(authored_frames) - 1),
        )
    )
    authored_index = max(0, min(len(authored_frames) - 1, authored_index))
    return sum(_authored_animation_holds(actor, cue)[:authored_index])


def _action_end_frame(actor: Any | None, cue: str) -> int:
    if actor is None or cue not in {"attack", "cast"}:
        return ACTION_ANIMATION_LAST_FRAME
    frame_count = _authored_action_frame_count(actor, cue)
    if frame_count <= 0:
        return ACTION_ANIMATION_LAST_FRAME
    return frame_count


def _procedural_animation_art_lines(
    actor: Any,
    cue: str,
    frame: int,
    base_lines: tuple[str, ...],
) -> tuple[str, ...]:
    if not base_lines:
        return base_lines
    if cue == "idle":
        return _procedural_idle_art_lines(actor, base_lines) if frame % 2 else base_lines
    if cue == "anticipate":
        return _offset_art_lines(base_lines, _team_direction(actor) * -1)
    if frame <= 0 or frame >= ACTION_ANIMATION_LAST_FRAME:
        return base_lines

    distance = 2 if frame == 1 else 1
    if cue == "attack":
        return _offset_art_lines(base_lines, _team_direction(actor) * distance)
    if cue == "cast":
        return _marked_art_lines(base_lines, "*")
    if cue == "hurt":
        return _marked_art_lines(base_lines, "!")
    return base_lines


def _team_direction(actor: Any) -> int:
    return 1 if getattr(actor, "team", "") == "hero" else -1


def _offset_art_lines(lines: Sequence[str], offset: int) -> tuple[str, ...]:
    if offset == 0:
        return tuple(lines)
    if offset > 0:
        return tuple(f"{' ' * offset}{line}" for line in lines)
    trim = abs(offset)
    return tuple(line[trim:] if len(line) > trim else "" for line in lines)


def _procedural_idle_art_lines(actor: Any, lines: Sequence[str]) -> tuple[str, ...]:
    if _uses_humanoid_breathing(actor):
        return _breathing_art_lines(lines)
    return _offset_art_lines(lines, 1)


def _uses_humanoid_breathing(actor: Any) -> bool:
    if getattr(actor, "team", "") == "hero":
        return True
    actor_text = " ".join(
        str(getattr(actor, attribute, "")).lower()
        for attribute in ("actor_id", "name", "class_id")
    )
    return any(token in actor_text for token in ("acolyte", "bone", "soldier"))


def _breathing_art_lines(lines: Sequence[str]) -> tuple[str, ...]:
    if len(lines) < 3:
        return tuple(lines)

    breathed = list(lines)
    breathed = list(_head_bob_art_lines(breathed))
    breathed[1] = _shoulder_breath_line(breathed[1])
    return tuple(breathed)


def _head_bob_art_lines(lines: Sequence[str]) -> tuple[str, ...]:
    head = _single_glyph_head(lines[0])
    if head is None:
        return tuple(lines)
    shoulder_left = next(
        (index for index, character in enumerate(lines[1]) if character != " "),
        None,
    )
    if shoulder_left is None:
        return tuple(lines)

    shoulder_right = len(lines[1]) - 1
    while shoulder_right >= 0 and lines[1][shoulder_right] == " ":
        shoulder_right -= 1
    if shoulder_right <= shoulder_left + 1:
        return tuple(lines)

    bobbed = list(lines)
    bobbed[0] = " " * len(lines[0])
    characters = list(lines[1])
    center = (shoulder_left + shoulder_right) // 2
    characters[center] = head
    bobbed[1] = "".join(characters)
    return tuple(bobbed)


def _single_glyph_head(line: str) -> str | None:
    stripped = line.strip()
    if len(stripped) != 1:
        return None
    if stripped in {"/", "\\", "|", "_"}:
        return None
    return stripped


def _shoulder_breath_line(line: str) -> str:
    left = next((index for index, character in enumerate(line) if character != " "), None)
    if left is None:
        return line

    right = len(line) - 1
    while right >= 0 and line[right] == " ":
        right -= 1

    if right <= left:
        return line

    characters = list(line)
    changed = False
    if characters[left] == "/":
        characters[left] = "|"
        changed = True
    if characters[right] == "\\":
        characters[right] = "|"
        changed = True
    return "".join(characters) if changed else line


def formation_slot_faces_inward(slot_key: str) -> bool:
    """Front-column formation slots face inward toward party center."""
    return slot_key in FORMATION_INWARD_SLOTS


def _portrait_display_art_lines(actor: Any, lines: Sequence[str]) -> tuple[str, ...]:
    padding = " " * BEAT_ART_SIDE_PADDING
    return tuple(f"{padding}{line}{padding}" for line in lines)


def _marked_art_lines(lines: Sequence[str], marker: str) -> tuple[str, ...]:
    marked: list[str] = []
    for line in lines:
        if not line:
            marked.append(line)
        elif len(line) == 1:
            marked.append(marker)
        else:
            marked.append(f"{marker}{line[1:-1]}{marker}")
    return tuple(marked)


def _animation_cues_for_events(
    events: Sequence[Any],
    target_intents: Mapping[str, str],
) -> dict[str, str]:
    cues: dict[str, str] = {}
    source_cue = "cast" if _has_cast_intent(target_intents) else "attack"
    for event in events:
        kind = _event_kind(event)
        if kind == "skill_used":
            cues[str(getattr(event, "actor_id", ""))] = source_cue
        elif kind == "enemy_intent":
            cues[str(getattr(event, "enemy_id", ""))] = source_cue
        elif kind == "reaction_used":
            cues[str(getattr(event, "actor_id", ""))] = "cast"
            cues[str(getattr(event, "enemy_id", ""))] = "hurt"
        elif kind == "damage":
            cues.setdefault(str(getattr(event, "source_id", "")), source_cue)
            cues[str(getattr(event, "target_id", ""))] = "hurt"
        elif kind == "healing":
            cues[str(getattr(event, "source_id", ""))] = "cast"
            cues[str(getattr(event, "target_id", ""))] = "heal"
        elif kind == "miss":
            cues.setdefault(str(getattr(event, "actor_id", "")), source_cue)
        elif kind == "combat_effect":
            actor_id = str(getattr(event, "actor_id", ""))
            cues[actor_id] = "heal" if _combat_effect_emphasis(event) == "good" else "hurt"
        elif kind in {"downed", "death", "status_changed"}:
            cues[str(getattr(event, "actor_id", ""))] = "hurt"
    cues.pop("", None)
    return cues


def _beat_hp_overrides(
    events: Sequence[Any],
    animation_frame: int,
    *actors: Any | None,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    deferred_events: Sequence[Any] = (),
) -> dict[str, int]:
    damage_by_actor: dict[str, int] = {}
    hp_before_by_actor: dict[str, int] = {}
    for event in events:
        if _event_kind(event) != "damage":
            continue
        target_id = str(getattr(event, "target_id", ""))
        amount = int(getattr(event, "amount", 0))
        if target_id and amount > 0:
            damage_by_actor[target_id] = damage_by_actor.get(target_id, 0) + amount
            hp_before = getattr(event, "hp_before", None)
            if hp_before is not None:
                hp_before_by_actor.setdefault(target_id, int(hp_before))
    deferred_damage_by_actor: dict[str, int] = {}
    deferred_hp_before_by_actor: dict[str, int] = {}
    for event in deferred_events:
        if _event_kind(event) != "damage":
            continue
        target_id = str(getattr(event, "target_id", ""))
        amount = int(getattr(event, "amount", 0))
        if target_id and amount > 0:
            deferred_damage_by_actor[target_id] = (
                deferred_damage_by_actor.get(target_id, 0) + amount
            )
            hp_before = getattr(event, "hp_before", None)
            if hp_before is not None:
                deferred_hp_before_by_actor.setdefault(target_id, int(hp_before))
    overrides: dict[str, int] = {}
    for actor in actors:
        if actor is None:
            continue
        actor_id = str(getattr(actor, "actor_id", ""))
        current_damage = damage_by_actor.get(actor_id, 0)
        deferred_damage = deferred_damage_by_actor.get(actor_id, 0)
        damage = current_damage + deferred_damage
        if damage:
            current_hp = int(getattr(actor, "hp", 0))
            max_hp = int(getattr(actor, "max_hp", current_hp))
            if animation_frame < impact_frame:
                hp_before = hp_before_by_actor.get(
                    actor_id,
                    deferred_hp_before_by_actor.get(actor_id),
                )
                if hp_before is None:
                    hp_before = current_hp + damage
                overrides[actor_id] = min(max_hp, max(0, hp_before))
            elif deferred_damage:
                hp_before = deferred_hp_before_by_actor.get(actor_id)
                if hp_before is None:
                    hp_before = current_hp + deferred_damage
                overrides[actor_id] = min(max_hp, max(0, hp_before))
            else:
                overrides[actor_id] = current_hp
    return overrides


def _beat_motion_offsets(
    events: Sequence[Any],
    animation_frame: int,
    *actors: Any | None,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
) -> dict[str, int]:
    if animation_frame < impact_frame or animation_frame > action_end_frame:
        return {}

    terminal_actor_ids = {
        str(getattr(event, "actor_id", ""))
        for event in events
        if _event_kind(event) in {"downed", "death"}
    }
    damage_by_actor: dict[str, int] = {}
    for event in events:
        if _event_kind(event) != "damage":
            continue
        target_id = str(getattr(event, "target_id", ""))
        amount = int(getattr(event, "amount", 0))
        if target_id and amount > 0:
            damage_by_actor[target_id] = damage_by_actor.get(target_id, 0) + amount

    offsets: dict[str, int] = {}
    for actor in actors:
        if actor is None:
            continue
        actor_id = str(getattr(actor, "actor_id", ""))
        amount = damage_by_actor.get(actor_id, 0)
        if amount <= 0:
            continue
        if animation_frame >= action_end_frame and actor_id in terminal_actor_ids:
            continue
        distance = _knockback_distance(amount)
        if animation_frame >= action_end_frame:
            distance = max(0, distance - 2)
        if distance == 0:
            continue
        offsets[actor_id] = _knockback_direction(actor) * distance
    return offsets


def _knockback_distance(amount: int) -> int:
    if amount >= 9:
        return 4
    if amount >= 6:
        return 3
    if amount >= 3:
        return 2
    return 1


def _knockback_direction(actor: Any) -> int:
    return -_team_direction(actor)


def _beat_callouts(
    events: Sequence[Any],
    animation_frame: int,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
) -> dict[str, tuple[str, str]]:
    if animation_frame < impact_frame or animation_frame > action_end_frame:
        return {}
    terminal_actor_ids = {
        str(getattr(event, "actor_id", ""))
        for event in events
        if _event_kind(event) in {"downed", "death"}
    }
    callouts: dict[str, tuple[str, str]] = {}
    for event in events:
        kind = _event_kind(event)
        if kind == "miss":
            target_id = str(getattr(event, "target_id", ""))
            if target_id and target_id not in terminal_actor_ids:
                callouts[target_id] = ("MISS", "bold yellow")
        elif _is_effort_change_event(event):
            actor_id = str(getattr(event, "actor_id", ""))
            if actor_id and actor_id not in terminal_actor_ids:
                callouts[actor_id] = ("EF -1", "bold bright_blue")
        elif _is_combat_effect_event(event):
            actor_id = str(getattr(event, "actor_id", ""))
            label = _combat_effect_label(event)
            if actor_id and label and actor_id not in terminal_actor_ids:
                callouts[actor_id] = (label, _combat_effect_style(event))
    return callouts


def _beat_status_overrides(
    events: Sequence[Any],
    animation_frame: int,
    *actors: Any | None,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
    deferred_events: Sequence[Any] = (),
) -> dict[str, tuple[str, ...]]:
    if animation_frame >= action_end_frame and not deferred_events:
        return {}

    added_statuses: dict[str, set[str]] = {}
    removed_statuses: dict[str, set[str]] = {}
    status_events = list(events)
    if animation_frame >= action_end_frame:
        status_events = []
    status_events.extend(deferred_events)
    for event in status_events:
        kind = _event_kind(event)
        actor_id = str(getattr(event, "actor_id", ""))
        if not actor_id:
            continue
        if kind == "downed":
            added_statuses.setdefault(actor_id, set()).add("downed")
        elif kind == "death":
            added_statuses.setdefault(actor_id, set()).add("dead")
        elif kind == "status_changed":
            status = str(getattr(event, "status", ""))
            if not status:
                continue
            if bool(getattr(event, "added", False)):
                added_statuses.setdefault(actor_id, set()).add(status)
            else:
                removed_statuses.setdefault(actor_id, set()).add(status)

    overrides: dict[str, tuple[str, ...]] = {}
    for actor in actors:
        if actor is None:
            continue
        actor_id = str(getattr(actor, "actor_id", ""))
        added = added_statuses.get(actor_id, set())
        removed = removed_statuses.get(actor_id, set())
        if not added and not removed:
            continue
        statuses = [
            status for status in tuple(getattr(actor, "statuses", ())) if status not in added
        ]
        for status in sorted(removed):
            if status not in statuses:
                statuses.append(status)
        overrides[actor_id] = tuple(statuses)
    return overrides


def _beat_pulse_styles(
    events: Sequence[Any],
    animation_frame: int,
    cues: Mapping[str, str],
    impact_frame: int = DAMAGE_REACTION_FRAME,
) -> dict[str, str]:
    styles: dict[str, str] = {}
    if animation_frame == ANTICIPATION_FRAME:
        for actor_id, cue in cues.items():
            if cue == "cast":
                styles[actor_id] = "bright_cyan"
    if animation_frame == SOURCE_ACTION_FRAME:
        for actor_id, cue in cues.items():
            if cue == "cast":
                styles[actor_id] = "bold cyan"
    if animation_frame == impact_frame:
        for event in events:
            kind = _event_kind(event)
            if kind == "damage":
                styles[str(getattr(event, "target_id", ""))] = "bold red"
            elif kind == "healing":
                styles[str(getattr(event, "target_id", ""))] = "bold green"
            elif kind == "combat_effect":
                styles[str(getattr(event, "actor_id", ""))] = _combat_effect_style(event)
    styles.pop("", None)
    return styles


def _has_cast_intent(target_intents: Mapping[str, str]) -> bool:
    return any(intent in {"heal", "debuff"} for intent in target_intents.values())


def format_meta_line(*parts: object) -> str:
    """Join compact metadata with consistent spacing for monospaced panes."""
    return META_SEPARATOR.join(
        text for part in parts if (text := str(part).strip())
    )


def primary_hotkey(action: ScreenAction) -> str:
    return next((alias for alias in action.aliases if len(alias) == 1), "")


def _dock_label(action: ScreenAction) -> str:
    risk = str(action.risk)
    is_combat = str(action.kind) == "combat"
    if not action.enabled:
        return f"x {action.label}"
    prefix = "!" if _show_dock_warning(action, risk=risk, is_combat=is_combat) else " "
    return f"{prefix}{action.label}"


def _show_dock_warning(action: ScreenAction, *, risk: str, is_combat: bool) -> bool:
    if is_combat:
        return False
    if str(action.kind) == "travel":
        return bool(action.route_warning)
    return risk in {"risky", "irreversible"} or bool(action.confirm)


def _is_boss_route_action(action: ScreenAction) -> bool:
    detail = f"{action.description} {action.preview} {action.result_hint}".lower()
    return "boss" in detail.split()


class StatusHeader(Static):
    """Top status strip for company and save-slot state."""

    def update_status(self, status: ShellStatusView) -> None:
        self.update(self.render_text(status))

    @staticmethod
    def render_text(status: ShellStatusView) -> str:
        company = status.company_name or "No company"
        return "\n".join(
            (
                format_meta_line("The Charter", company, status.location),
                format_meta_line(
                    f"Rep: {status.reputation}",
                    f"Coin: {status.coin}",
                    f"Party: {status.active_count} active / {status.reserve_count} reserve",
                    f"Condition: {status.wounded_count} wounded / {status.downed_count} downed",
                ),
                format_meta_line(
                    f"Save: {status.save_status}",
                    f"Breaches: {status.breaches}",
                ),
            )
        )


class BodyPane(Static):
    """Main screen body without repeated pane chrome."""

    def update_screen(self, _title: str, body: str) -> None:
        self.update(body)


class DetailPane(Static):
    """Focused command or contextual detail pane."""

    def update_detail(self, text: str) -> None:
        self.update(text)


class LogPane(Static):
    """Compact recent log pane."""

    def update_log(self, text: Any) -> None:
        self.update(text or "No recent log.")


class CommandDock(Static):
    """Bottom command dock with focus, numbers, and hotkeys."""

    def update_actions(
        self,
        actions: Sequence[ScreenAction],
        focused_index: int,
        help_text: str = "",
        shortcut_text: str = "",
    ) -> None:
        self.update(
            self.render_text(
                actions,
                focused_index,
                help_text=help_text,
                shortcut_text=shortcut_text,
                width=self.size.width,
            )
        )

    @staticmethod
    def render_text(
        actions: Sequence[ScreenAction],
        focused_index: int,
        *,
        help_text: str = "",
        shortcut_text: str = "",
        width: int = 0,
    ) -> str:
        shortcut_line = _shortcut_line(shortcut_text)
        if not actions:
            lines = ["No commands available."]
            if shortcut_line:
                lines.extend(("", shortcut_line))
            return "\n".join(lines)
        command_lines = _flat_command_lines(actions, focused_index)
        help_lines = _dock_help_lines(help_text)
        if help_lines and width >= DOCK_WIDE_MIN_WIDTH:
            dock_text = _wide_dock_text(command_lines, help_lines, width=width)
            if shortcut_line:
                return f"{dock_text}\n\n{shortcut_line}"
            return dock_text
        lines = list(command_lines)
        if shortcut_line:
            lines.extend(("", shortcut_line))
        if help_lines:
            lines.extend(("", "Focus", *help_lines))
        return "\n".join(lines)


def _flat_command_lines(actions: Sequence[ScreenAction], focused_index: int) -> list[str]:
    command_lines = ["Commands"]
    for index, action in enumerate(actions):
        command_lines.append(_dock_command_line(action, focused=index == focused_index))
    return command_lines


def _dock_command_line(action: ScreenAction, *, focused: bool) -> str:
    focus = ">" if focused else " "
    label = _dock_label(action)
    return f"{focus} {action.number:>2}  {label[:COMMAND_LABEL_WIDTH]:<{COMMAND_LABEL_WIDTH}}"


def _shortcut_line(shortcut_text: str) -> str:
    if not shortcut_text:
        return ""
    return shortcut_text.replace("[", "\\[")


class ExpeditionProgressStrip(Static):
    """Compact room/node progress for expedition playback."""

    def update_progress(self, beats: Sequence[EventBeat], playback_index: int) -> None:
        self.update(self.render_text(beats, playback_index))

    @staticmethod
    def render_text(beats: Sequence[EventBeat], playback_index: int) -> str:
        if not beats:
            return "Expedition Progress\n(no route)"
        lines = ["Expedition Progress"]
        for index, beat in enumerate(beats):
            if index < playback_index:
                marker = "[x]"
            elif index == playback_index:
                marker = "[>]"
            else:
                marker = "[ ]"
            kind = "combat" if beat.combat else beat.title.lower()
            lines.append(f"{marker} {index + 1:02}. {beat.title} ({kind})")
        return "\n".join(lines)


class FormationBoard(Static):
    """Reusable 2x2 party formation board."""

    def update_formation(self, view: FormationView, *, focus_slot: str = "") -> None:
        self.update(self.render_text(view, focus_slot=focus_slot))

    @staticmethod
    def render_text(view: FormationView, *, focus_slot: str = "") -> str:
        cells = {}
        for slot in view.slots:
            name_line, detail_line, state_line = format_formation_board_cell(slot)
            cells[slot.slot_label] = _grid_cell(
                name_line,
                detail_line,
                state_line,
                marker=">" if slot.slot_label == focus_slot else " ",
            )
        return "Party Formation\n" + _grid_text(cells)

    @staticmethod
    def render_mini_text(
        portrait_actors_by_slot: Mapping[str, Any],
        *,
        focus_slot: str = "",
        focus_hero_id: str = "",
        idle_frame: int = 0,
        inward_facing: bool = False,
    ) -> str:
        slot_labels = {
            "BACK_LEFT": format_formation_slot("BACK_LEFT"),
            "BACK_RIGHT": format_formation_slot("BACK_RIGHT"),
            "FRONT_LEFT": format_formation_slot("FRONT_LEFT"),
            "FRONT_RIGHT": format_formation_slot("FRONT_RIGHT"),
        }
        lines = ["Party Formation"]
        for row in PARTY_COMBAT_ROWS:
            lines.append(
                format_meta_line(
                    format_formation_slot(row[0]),
                    format_formation_slot(row[1]),
                )
            )
            lines.extend(
                _mini_side_rows(
                    portrait_actors_by_slot,
                    rows=(row,),
                    focus_id=focus_hero_id,
                    legal_ids=set(),
                    highlight_ids={focus_hero_id} if focus_hero_id else set(),
                    intents={},
                    source_ids=set(),
                    slot_annotations=slot_labels,
                    idle_frame=idle_frame,
                    turn_flash_actor_id="",
                    turn_flash_frame=0,
                    inward_facing=inward_facing,
                )
            )
        return "\n".join(lines)


class TownDashboardPanel(Static):
    """Haven town dashboard summary."""

    def update_dashboard(
        self,
        view: TownDashboardView,
        hero_lines: str,
        reserve_lines: str,
    ) -> None:
        self.update(self.render_text(view, hero_lines=hero_lines, reserve_lines=reserve_lines))

    @staticmethod
    def render_text(
        view: TownDashboardView,
        *,
        hero_lines: str,
        reserve_lines: str,
    ) -> str:
        resource_line = format_meta_line(
            f"Reputation: {view.reputation}",
            f"Coin: {view.coin}",
            f"Roster cap: {view.roster_cap}",
        )
        roster_line = format_meta_line(
            f"Active: {view.active_count}",
            f"Reserves: {view.reserve_count}",
        )
        condition_line = format_meta_line(
            f"Wounded: {view.wounded_count}",
            f"Downed: {view.downed_count}",
            f"Memorial: {view.deceased_count}",
        )
        contract_lines = "\n".join(
            _contract_summary_line(entry) for entry in view.contract_board
        )
        upgrade_lines = "\n".join(
            _upgrade_summary_line(entry) for entry in view.upgrades
        )
        return (
            "Haven Town\n"
            f"{view.company_name} at {view.location}\n\n"
            "Company Status\n"
            f"{resource_line}\n"
            f"{roster_line}\n"
            f"{condition_line}\n\n"
            "Current Objective\n"
            f"{_lines_or_none(_objective_lines(view.objective))}\n\n"
            "Posted Contracts\n"
            f"{contract_lines or 'none'}\n\n"
            "Company Upgrades\n"
            f"{upgrade_lines or 'none'}\n\n"
            "Active Party\n"
            f"{hero_lines}\n\n"
            "Reserves\n"
            f"{reserve_lines}"
        )


class YardPanel(Static):
    """Compact company yard summary."""

    @staticmethod
    def render_text(
        view: TownDashboardView,
        *,
        formation_text: str = "",
        hint: str = "",
    ) -> str:
        roster_lines = tuple(
            format_compact_roster_row(hero) for hero in (*view.active_party, *view.reserves)
        )
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Status",
                (
                    format_meta_line(
                        f"Active {view.active_count}",
                        f"Reserves {view.reserve_count}",
                        f"Cap {view.roster_cap}",
                    ),
                ),
            ),
        ]
        if formation_text:
            sections.append(("Formation", (formation_text,)))
        sections.append(
            (
                "Roster",
                roster_lines or ("none",),
            )
        )
        return format_scene_body("Company Yard", tuple(sections), hint=hint)


class PackPanel(Static):
    """Carried supplies, items, gear, and equipped kits."""

    @staticmethod
    def render_text(
        supplies: dict[str, int],
        inventory: dict[str, int],
        gear: GearInventoryView,
        *,
        hint: str = "",
    ) -> str:
        sections = [
            (
                "Supplies",
                (format_fixed_table(("Name", "Qty"), format_quantity_rows(supplies)),)
                if supplies
                else ("none",),
            ),
            (
                "Items",
                (format_fixed_table(("Name", "Qty"), format_quantity_rows(inventory)),)
                if inventory
                else ("none",),
            ),
            (
                "Gear",
                (
                    format_fixed_table(
                        ("Kit", "Own", "Free", "Eq"),
                        format_gear_stock_rows(gear.items),
                    ),
                )
                if format_gear_stock_rows(gear.items)
                else ("none",),
            ),
            (
                "Equipped",
                (
                    format_fixed_table(
                        ("Hero", "Kit"),
                        format_equipped_kit_rows(gear.heroes),
                    ),
                )
                if gear.heroes
                else ("none",),
            ),
        ]
        if not gear.can_manage and gear.manage_reason:
            sections.append(("Note", (gear.manage_reason,)))
        return format_scene_body("Pack", tuple(sections), hint=hint)


class GearLockerPanel(Static):
    """Armory stock and equipped kits."""

    @staticmethod
    def render_text(view: GearInventoryView, *, hint: str = "") -> str:
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Status",
                (
                    format_meta_line(
                        f"Reputation {view.reputation}",
                        f"Coin {view.coin}",
                    ),
                    "Purchases: "
                    + ("available" if view.can_purchase else view.purchase_reason),
                ),
            ),
            (
                "Company Gear",
                (
                    format_fixed_table(
                        ("Kit", "State", "Cost"),
                        format_gear_shop_rows(view.items),
                    ),
                )
                if view.items
                else ("none",),
            ),
            (
                "Equipped",
                (
                    format_fixed_table(
                        ("Hero", "Kit"),
                        format_equipped_kit_rows(view.heroes),
                    ),
                )
                if view.heroes
                else ("none",),
            ),
        ]
        return format_scene_body("Armory", tuple(sections), hint=hint)


class RelicBrokerPanel(Static):
    """Relic clerk buy/file list."""

    @staticmethod
    def render_text(view: RelicBrokerView, *, hint: str = "") -> str:
        inventory = ", ".join(
            f"{item_id} x{quantity}" for item_id, quantity in view.inventory
        )
        offer_rows = [
            (
                action.label.removeprefix("Sell ").removeprefix("File "),
                action.description or action.preview or "",
                "ready" if action.enabled else "blocked",
            )
            for action in view.actions
            if action.value != "back"
        ]
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Ledger",
                (
                    f"Coin {view.coin}",
                    f"Inventory {inventory or 'none'}",
                ),
            ),
            (
                "Offers",
                (
                    format_fixed_table(("Item", "Terms", "State"), offer_rows),
                )
                if offer_rows
                else ("No sellable or fileable relics are catalogued.",),
            ),
        ]
        return format_scene_body("Relic Clerk", tuple(sections), hint=hint)


class SupplyShopPanel(Static):
    """Quartermaster stock list."""

    @staticmethod
    def render_text(view: SupplyShopView, *, hint: str = "") -> str:
        stock_rows = format_supply_stock_rows(view.actions)
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Budget",
                (
                    f"Coin {view.coin}",
                    "Purchase quantity: 1",
                ),
            ),
            (
                "Stock",
                (
                    format_fixed_table(("Supply", "Cost", "State"), stock_rows),
                )
                if stock_rows
                else ("none",),
            ),
        ]
        return format_scene_body("Quartermaster", tuple(sections), hint=hint)


class CompanyPanel(Static):
    """Slim charter company overview."""

    @staticmethod
    def render_text(
        company_name: str,
        objective_line: str,
        formation_text: str,
        roster_lines: Sequence[str],
        *,
        hint: str = "",
    ) -> str:
        sections: list[tuple[str, Sequence[str]]] = [
            ("Objective", (objective_line,)),
            ("Formation", (formation_text,)),
            (
                "Characters",
                tuple(roster_lines) or ("none",),
            ),
        ]
        return format_scene_body(f"Company — {company_name}", tuple(sections), hint=hint)


class DungeonMapPanel(Static):
    """Known dungeon node graph summary."""

    def update_map(self, view: DungeonView) -> None:
        self.update(self.render_minimap(view))

    @staticmethod
    def render_minimap(
        view: DungeonView,
        *,
        highlighted_node_id: str = "",
        actions: Sequence[ScreenAction] | None = None,
    ) -> Text:
        text = Text(
            DungeonMapPanel.render_minimap_text(
                view,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
        )
        _highlight_minimap_nodes(
            text,
            view,
            highlighted_node_id=highlighted_node_id,
            actions=actions,
        )
        return text

    @staticmethod
    def render_minimap_text(
        view: DungeonView,
        *,
        actions: Sequence[ScreenAction] | None = None,
        highlighted_node_id: str = "",
    ) -> str:
        lines = [
            "Mini Map",
            _short_map_name(view.current_room.name),
            "",
        ]
        lines.extend(
            _minimap_lines(
                view,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
        )
        lines.extend(("", "Legend: @ you   o known   ? unknown   ! quest"))
        return "\n".join(lines)

    @staticmethod
    def render_text(
        view: DungeonView,
        *,
        actions: Sequence[ScreenAction] | None = None,
        title: str = "Company Map",
        survey_label: str = "Survey",
        legend_line: str = "@ current  |  o visited  |  ? known  |  number = route command",
    ) -> str:
        lines = [
            title,
            view.current_map_id.replace("_", " ").title(),
            "",
            survey_label,
        ]
        map_lines = _full_map_lines(view)
        lines.extend(map_lines if map_lines else ["No explored map yet."])
        lines.extend(
            (
                "",
                "Legend",
                legend_line,
                "",
                "Inventory",
                _map_inventory_line(view),
                "",
                "Known Places",
            )
        )
        for node in view.map_nodes:
            lines.extend(_map_node_detail_lines(view, node, actions=actions))
        if view.exits:
            lines.extend(("", "Current Routes"))
            for exit_node in view.exits:
                state = "visited" if exit_node.visited else "unexplored"
                if exit_node.map_id != view.current_map_id:
                    state = f"{state}, enters {exit_node.map_id.replace('_', ' ')}"
                number = _action_number_for_value(
                    view,
                    exit_node.node_id,
                    actions=actions,
                )
                prefix = f"{number}. " if number else "- "
                route_detail = format_meta_line(
                    exit_node.name,
                    state,
                    _node_inventory_brief(exit_node),
                )
                lines.append(f"{prefix}{route_detail}")
        return "\n".join(lines)

    @staticmethod
    def render_legacy_text(view: DungeonView) -> str:
        lines = [
            "Company Map",
            view.current_map_id.replace("_", " ").title(),
            "",
            "Known Places",
        ]
        for node in view.map_nodes:
            safe = "safe return" if node.safe_return else ""
            lines.append(f"{_map_status_label(node)}: {node.name}")
            node_detail = format_meta_line(
                node.node_type.replace("_", " "),
                node.status,
                safe,
            )
            lines.append(f"  {node_detail}")
        connection_lines = _full_map_lines(view)
        if connection_lines:
            lines.extend(("", "Connections", *connection_lines))
        if view.exits:
            lines.extend(("", "Current Routes"))
            for exit_node in view.exits:
                state = "visited" if exit_node.visited else "unexplored"
                if exit_node.map_id != view.current_map_id:
                    state = f"{state}, enters {exit_node.map_id.replace('_', ' ')}"
                number = _action_number_for_value(view, exit_node.node_id)
                prefix = f"{number}. " if number else "- "
                lines.append(f"{prefix}{format_meta_line(exit_node.name, state)}")
        return "\n".join(lines)


class DungeonRoomPanel(Static):
    """Current dungeon room panel."""

    def update_room(self, view: DungeonView) -> None:
        self.update(self.render_text(view))

    @staticmethod
    def render_text(view: DungeonView) -> str:
        room = view.current_room
        art_lines = _compact_art_lines(room.art_lines, max_lines=10, max_width=72)
        lines = [
            *art_lines,
            "" if art_lines else None,
            room.name,
            "",
            room.text,
        ]
        impact_lines = [
            line
            for line in (room.scene_state, room.route_hint, room.party_hint)
            if line.strip()
        ][:3]
        if impact_lines:
            lines.extend(("", *impact_lines))
        if view.maze_off_spine_hint.strip():
            lines.extend(("", view.maze_off_spine_hint))
        interactable_hint = _interactable_hint(view.room_actions)
        if interactable_hint:
            lines.extend(("", interactable_hint))
        return "\n".join(line for line in lines if line is not None)


def _interactable_hint(room_actions: Sequence[Any]) -> str:
    targets = tuple(
        dict.fromkeys(
            _interactable_target(str(action.label))
            for action in room_actions
            if str(getattr(action, "state", "")) != "completed"
        )
    )
    targets = tuple(target for target in targets if target)
    if not targets:
        return ""
    if len(targets) == 1:
        return "\n".join(
            (
                f"Focus: {targets[0]}",
                "Action: Interact to handle it.",
            )
        )
    return "\n".join(
        (
            "Notable: " + ", ".join(targets[:3]),
            "Action: Interact to choose one.",
        )
    )


def _interactable_target(label: str) -> str:
    words = label.split()
    while words and words[0].lower() in {
        "brace",
        "clear",
        "force",
        "recover",
        "search",
        "unlock",
        "use",
    }:
        words = words[1:]
    return " ".join(words).strip()


class ExpeditionReportPanel(Static):
    """Filed company record summary."""

    def update_report(self, view: ExpeditionReportView) -> None:
        self.update(self.render_text(view))

    @staticmethod
    def render_text(view: ExpeditionReportView) -> str:
        return "\n".join(
            (
                "Filed Company Record",
                f"Outcome: {view.outcome.replace('_', ' ').title()}",
                f"Route: {view.expedition_id}",
                f"Dungeon: {view.dungeon_id or 'none'}",
                "",
                "Record Brief",
                _lines_or_none(_report_brief_lines(view)),
                "",
                "What Changed",
                _lines_or_none(view.what_changed),
                "",
                "Next Objective",
                _lines_or_none(_objective_lines(view.objective)),
                "",
                "Rooms Entered",
                _lines_or_none(view.rooms_entered),
                "",
                "Encounters Resolved",
                _lines_or_none(view.encounters_resolved),
                "",
                "Room Actions",
                _lines_or_none(view.room_actions),
                "",
                "Rewards",
                _reward_lines(view),
                "",
                "Operational Delta",
                f"Reputation: {view.reputation_start}->{view.reputation_end} "
                f"({_signed(view.reputation_delta)})",
                f"Coin: {view.coin_start}->{view.coin_end} "
                f"({_signed(view.coin_delta)})",
                "Supplies",
                _delta_lines(view.supply_deltas),
                "Inventory",
                _delta_lines(view.inventory_deltas),
                "Gear",
                _delta_lines(view.gear_deltas),
                "",
                "Hero Outcomes",
                _lines_or_none(view.hero_outcomes),
                "",
                "Notable Moments",
                _lines_or_none(view.notable_moments),
                "",
                "Company Condition",
                f"Wounded: {view.wounded_count}",
                f"Downed: {view.downed_count}",
                f"Memorial: {view.deceased_count}",
            )
        )


class CombatPanel(Static):
    """Combat command surface with formation boards and target previews."""

    def update_combat(
        self,
        view: CombatView,
        *,
        phase: str,
        focused_action: ScreenAction | None,
    ) -> None:
        self.update(self.render_text(view, phase=phase, focused_action=focused_action))

    @staticmethod
    def render_text(
        view: CombatView,
        *,
        phase: str,
        focused_action: ScreenAction | None,
        idle_frame: int = 0,
        turn_flash_actor_id: str = "",
        turn_flash_frame: int = 0,
    ) -> str:
        reaction_intent = view.pending_enemy_intent if phase == "reaction" else None
        healing_targets = phase == "target" and any(
            target.intent == "heal" for target in view.targets
        )
        party_focus_id = _focused_actor_id(focused_action) if healing_targets else ""
        enemy_focus_id = (
            _focused_actor_id(focused_action)
            if phase == "target" and not healing_targets
            else ""
        )
        if reaction_intent is not None:
            party_focus_id = reaction_intent.target_id
            enemy_focus_id = ""
        move_context = _move_board_context(view, focused_action) if phase == "move" else None
        lines = [
            "Combat Command",
            format_meta_line(
                view.encounter_name,
                f"Round {view.round_number}",
                f"Cohesion {view.cohesion}",
            ),
            _turn_order_rail(view),
            _turn_line(view),
            "",
            _combat_duel_board(
                view.party,
                view.enemies,
                party_focus_id=party_focus_id,
                party_legal_ids=(
                    move_context[1]
                    if move_context is not None
                    else {reaction_intent.target_id}
                    if reaction_intent is not None
                    else {target.target_id for target in view.targets}
                    if healing_targets
                    else set()
                ),
                party_highlight_ids=(
                    move_context[2]
                    if move_context is not None
                    else {party_focus_id}
                    if party_focus_id
                    else set()
                ),
                party_intents=(
                    {reaction_intent.target_id: "debuff"}
                    if reaction_intent is not None
                    else {target.target_id: target.intent for target in view.targets}
                    if healing_targets
                    else {}
                ),
                enemy_focus_id=enemy_focus_id,
                enemy_legal_ids=(
                    {target.target_id for target in view.targets}
                    if phase == "target" and not healing_targets
                    else set()
                ),
                enemy_highlight_ids={enemy_focus_id} if enemy_focus_id else set(),
                enemy_intents=(
                    {target.target_id: target.intent for target in view.targets}
                    if phase == "target" and not healing_targets
                    else {}
                ),
                party_source_ids=(
                    move_context[3]
                    if move_context is not None
                    else set()
                ),
                enemy_source_ids={reaction_intent.enemy_id}
                if reaction_intent is not None
                else set(),
                party_slot_annotations=move_context[0] if move_context is not None else {},
                idle_frame=idle_frame,
                turn_flash_actor_id=turn_flash_actor_id,
                turn_flash_frame=turn_flash_frame,
            ),
            "",
        ]
        if phase == "reaction":
            intent = view.pending_enemy_intent
            if intent is not None:
                lines.extend(
                    (
                        "Enemy Intent",
                        format_meta_line(
                            intent.enemy_name,
                            intent.label,
                            f"Target {intent.target_name}",
                            f"Threat {intent.threat_level}",
                            intent.obvious_effect,
                        ),
                        "",
                        "Class Reactions",
                    )
                )
            for option in view.reaction_options:
                cost = f"Cost {option.cost}" if option.cost else "No cost"
                lines.append(
                    f"{option.action.number}. "
                    f"{format_meta_line(option.action.label, cost, option.summary)}"
                )
        return "\n".join(lines)

    @staticmethod
    def render_enemy_turn(
        view: CombatView | None,
        events: Sequence[Any],
        *,
        source_actor_ids: set[str],
        target_intents: Mapping[str, str],
        animation_frame: int = ACTION_ANIMATION_LAST_FRAME,
        animation_cues: Mapping[str, str] | None = None,
        deferred_events: Sequence[Any] = (),
    ) -> str:
        return CombatPanel.render_combat_beat(
            view,
            events,
            title="Danger" if _has_danger_event(events) else "Enemy Response",
            source_actor_ids=source_actor_ids,
            target_intents=target_intents,
            animation_frame=animation_frame,
            animation_cues=animation_cues,
            deferred_events=deferred_events,
        )

    @staticmethod
    def beat_animation_last_frame(
        view: CombatView | None,
        events: Sequence[Any],
        *,
        source_actor_ids: set[str],
        target_intents: Mapping[str, str],
        animation_cues: Mapping[str, str] | None = None,
    ) -> int:
        if view is None:
            return ACTION_ANIMATION_LAST_FRAME
        cues = (
            _animation_cues_for_events(events, target_intents)
            if animation_cues is None
            else animation_cues
        )
        source_actor, _target_actor = _beat_actor_pair(
            view,
            events,
            source_actor_ids=source_actor_ids,
            target_actor_ids=set(target_intents),
        )
        source_cue = cues.get(getattr(source_actor, "actor_id", ""), "")
        return _action_end_frame(source_actor, source_cue)

    @staticmethod
    def render_combat_beat(
        view: CombatView | None,
        events: Sequence[Any],
        *,
        title: str,
        source_actor_ids: set[str],
        target_intents: Mapping[str, str],
        animation_frame: int = ACTION_ANIMATION_LAST_FRAME,
        animation_cues: Mapping[str, str] | None = None,
        deferred_events: Sequence[Any] = (),
    ) -> str:
        if view is None:
            return "\n".join((title, "", _lines_or_none(_fallback_beat_lines(events))))

        cues = (
            _animation_cues_for_events(events, target_intents)
            if animation_cues is None
            else animation_cues
        )
        source_actor, target_actor = _beat_actor_pair(
            view,
            events,
            source_actor_ids=source_actor_ids,
            target_actor_ids=set(target_intents),
        )
        lines = [
            title,
            format_meta_line(
                view.encounter_name,
                f"Round {view.round_number}",
                f"Cohesion {view.cohesion}",
            ),
        ]
        left_actor, right_actor, connector = _beat_portrait_layout(source_actor, target_actor)
        source_cue = cues.get(getattr(source_actor, "actor_id", ""), "")
        impact_frame = _action_impact_frame(source_actor, source_cue)
        action_end_frame = _action_end_frame(source_actor, source_cue)
        portrait_lines = _portrait_versus_lines(
            left_actor,
            right_actor,
            connector=connector,
            animation_cues=cues,
            animation_frame=animation_frame,
            impact_frame=impact_frame,
            action_end_frame=action_end_frame,
            hp_overrides=_beat_hp_overrides(
                events,
                animation_frame,
                source_actor,
                target_actor,
                impact_frame=impact_frame,
                deferred_events=deferred_events,
            ),
            motion_offsets=_beat_motion_offsets(
                events,
                animation_frame,
                source_actor,
                target_actor,
                impact_frame=impact_frame,
                action_end_frame=action_end_frame,
            ),
            callouts=_beat_callouts(events, animation_frame, impact_frame, action_end_frame),
            pulse_styles=_beat_pulse_styles(events, animation_frame, cues, impact_frame),
            status_overrides=_beat_status_overrides(
                events,
                animation_frame,
                source_actor,
                target_actor,
                action_end_frame=action_end_frame,
                deferred_events=deferred_events,
            ),
        )
        if portrait_lines:
            lines.extend(("", *portrait_lines))

        result_lines = _combat_beat_result_lines(
            events,
            animation_frame=animation_frame,
            impact_frame=impact_frame,
            action_end_frame=action_end_frame,
        )
        if result_lines:
            lines.extend(("", *result_lines))
        footer_lines = _combat_beat_footer_lines(events)
        if footer_lines:
            lines.extend(("", *footer_lines))
        return "\n".join(lines)

    @staticmethod
    def command_help_text(view: CombatView, phase: str, action: ScreenAction) -> str:
        if phase == "command":
            availability = "available" if action.enabled else "unavailable"
            purpose = action.description or "Choose how the current hero spends this turn."
            lines = [format_meta_line(action.label, availability), purpose]
            if action.preview:
                lines.append(action.preview)
            if action.result_hint:
                lines.append(action.result_hint)
            if not action.enabled and action.unavailable_reason:
                lines.append(action.unavailable_reason)
            return "\n".join(lines)

        if phase == "skill":
            skill = next(
                (option for option in view.skills if option.action.value == action.value),
                None,
            )
            if skill is None:
                return action.description or action.label
            effect_label = "healing" if skill.intent == "heal" else "damage"
            lines = [
                format_meta_line(
                    skill.name,
                    f"EF: {skill.effort_cost}",
                    skill.attack_type,
                    skill.usable_from_label,
                    f"{skill.damage_label} {effect_label}",
                )
            ]
            if skill.flavor_text:
                lines.append(skill.flavor_text)
            lines.append(f"Effect: {skill.effect_text}")
            lines.append(f"{skill.target_count} legal target(s).")
            if not action.enabled and action.unavailable_reason:
                lines.append(action.unavailable_reason)
            return "\n".join(lines)

        if phase == "move":
            move = next(
                (option for option in view.moves if option.action.value == action.value),
                None,
            )
            if move is None:
                return action.description or action.label
            return "\n".join(
                _move_focus_lines(
                    move,
                    action.result_hint or "Enter moves the acting hero and ends their turn.",
                )
            )

        if phase == "reaction":
            option = next(
                (
                    candidate
                    for candidate in view.reaction_options
                    if candidate.action.value == action.value
                ),
                None,
            )
            if option is None:
                return action.description or action.label
            cost = f"Cost {option.cost}" if option.cost else "No effort cost"
            return "\n".join(
                (
                    format_meta_line(option.action.label, cost),
                    option.summary,
                    option.action.result_hint,
                )
            )

        target = next(
            (option for option in view.targets if option.action.value == action.value),
            None,
        )
        if target is None:
            return action.description or action.label
        effect_label = "healing" if target.intent == "heal" else "damage"
        return "\n".join(
            (
                format_meta_line(
                    target.name,
                    f"{target.hit_chance}% hit",
                    f"{target.damage_label} {effect_label}",
                ),
                target.legality_reason,
                action.result_hint,
            )
        )

    @staticmethod
    def detail_text(
        view: CombatView,
        phase: str,
        action: ScreenAction,
        *,
        idle_frame: int = 0,
    ) -> str:
        if phase == "reaction":
            intent = view.pending_enemy_intent
            option = next(
                (
                    candidate
                    for candidate in view.reaction_options
                    if candidate.action.value == action.value
                ),
                None,
            )
            lines = ["Reaction Focus", ""]
            actor = (
                _actor_by_id(view, option.actor_id)
                if option is not None and option.actor_id is not None
                else _actor_by_id(view, intent.enemy_id)
                if intent is not None
                else None
            )
            lines.extend((*_portrait_detail_lines(actor, idle_frame=idle_frame), ""))
            if intent is not None:
                lines.extend(
                    (
                        f"{intent.enemy_name}: {intent.label}",
                        f"Target: {intent.target_name}",
                    )
                )
            if option is None:
                lines.append(action.label)
            else:
                lines.extend(
                    (
                        f"Ready: {option.action.label}",
                        f"Reaction: {option.kind.replace('_', ' ')}",
                    )
                )
            return "\n".join(lines)

        if phase == "command":
            lines = ["Command Focus", ""]
            lines.extend((*_portrait_detail_lines(view.current_actor, idle_frame=idle_frame), ""))
            lines.extend(
                (
                    action.label,
                    "Available" if action.enabled else "Unavailable",
                )
            )
            if action.cost:
                lines.append(f"Cost: {action.cost}")
            if action.preview:
                lines.extend(("", "Preview", action.preview))
            if action.result_hint:
                lines.extend(("", "Consequence", action.result_hint))
            if not action.enabled and action.unavailable_reason:
                lines.extend(("", action.unavailable_reason))
            return "\n".join(line for line in lines if line is not None)

        if phase == "skill":
            skill = next(
                (option for option in view.skills if option.action.value == action.value),
                None,
            )
            if skill is None:
                return action.label
            lines = ["Skill Focus", ""]
            lines.extend((*_portrait_detail_lines(view.current_actor, idle_frame=idle_frame), ""))
            lines.extend(
                (
                    f"Preparing: {skill.name}",
                    f"Intent: {skill.intent}",
                    f"Usable: {skill.usable_from_label}",
                )
            )
            if skill.action.cost:
                lines.append(f"Cost: {skill.action.cost}")
            if not skill.action.enabled and skill.action.unavailable_reason:
                lines.extend(("", f"Unavailable: {skill.action.unavailable_reason}"))
            if skill.action.preview:
                lines.extend(("", skill.action.preview))
            if skill.action.result_hint:
                lines.append(skill.action.result_hint)
            return "\n".join(line for line in lines if line is not None)

        if phase == "move":
            move = next(
                (option for option in view.moves if option.action.value == action.value),
                None,
            )
            if move is None:
                return action.label
            lines = ["Move Preview", ""]
            lines.extend((*_portrait_detail_lines(view.current_actor, idle_frame=idle_frame), ""))
            lines.extend(_move_focus_lines(move, move.action.result_hint))
            return "\n".join(line for line in lines if line is not None)

        target = next(
            (option for option in view.targets if option.action.value == action.value),
            None,
        )
        if target is None:
            return action.label
        status = ", ".join(target.statuses)
        portrait_lines = _portrait_detail_lines(
            _actor_by_id(view, target.target_id),
            idle_frame=idle_frame,
        )
        lines = ["Target Focus", ""]
        if portrait_lines:
            lines.extend((*portrait_lines, ""))
        lines.extend(
            (
                target.name,
                f"Slot: {target.slot}",
                f"HP: {target.hp}/{target.max_hp}",
                f"Status: {status}",
                f"Preview: {target.action.result_hint}",
                f"Why: {target.legality_reason}",
            )
        )
        return "\n".join(lines)


def _dock_help_lines(text: str) -> list[str]:
    return [line[:88].rstrip() for line in text.splitlines() if line.strip()][:4]


def _move_board_context(
    view: CombatView,
    focused_action: ScreenAction | None,
) -> tuple[dict[str, str], set[str], set[str], set[str]]:
    annotations: dict[str, str] = {}
    legal_ids: set[str] = set()
    highlight_ids: set[str] = set()
    source_ids: set[str] = set()
    if view.current_actor is not None:
        source_ids.add(view.current_actor.actor_id)
    focused_slot = focused_action.value if focused_action is not None else ""
    party_by_slot = {actor.slot: actor for actor in view.party}
    for move in view.moves:
        annotations[move.to_slot] = f"[{move.action.number}]"
        occupant = party_by_slot.get(move.to_slot)
        if occupant is not None:
            legal_ids.add(occupant.actor_id)
            if move.to_slot == focused_slot:
                highlight_ids.add(occupant.actor_id)
    return annotations, legal_ids, highlight_ids, source_ids


def _move_focus_lines(move: Any, result_hint: str = "") -> tuple[str, ...]:
    lines: list[str] = [
        f"Actor: {move.actor_name}",
        f"Current: {format_formation_slot(move.from_slot)}",
        f"Destination: {format_formation_slot(move.to_slot)}",
    ]
    if move.occupant_name != "empty":
        lines.append(f"Swap target: {move.occupant_name}")
    if result_hint:
        lines.extend(("", f"Result: {result_hint}"))
    preview = _formation_preview_text(move.before_formation, move.after_formation)
    if preview:
        lines.extend(("", "Formation Preview", preview))
    return tuple(lines)


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
        f"{left}  ->  {right}"
        for left, right in zip(before_lines, after_lines, strict=True)
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


def _wide_dock_text(command_lines: list[str], help_lines: list[str], *, width: int) -> str:
    left_width = min(DOCK_COMMAND_COLUMN_WIDTH, max(34, width // 2 - len(DOCK_GAP)))
    right_width = max(24, width - left_width - len(DOCK_GAP))
    left = [_fit_dock_line(line, left_width) for line in command_lines]
    right = ["Hint", *(_fit_dock_line(line, right_width) for line in help_lines)]
    height = max(len(left), len(right))
    left.extend([" " * left_width] * (height - len(left)))
    right.extend([""] * (height - len(right)))
    return "\n".join(
        f"{left_line}{DOCK_GAP}{right_line}"
        for left_line, right_line in zip(left, right, strict=True)
    )


def _fit_dock_line(line: str, width: int) -> str:
    return line[:width].ljust(width)


def portrait_detail_lines(
    *actors: Any | None,
    idle_frame: int = 0,
    mirror_facing: bool = False,
) -> list[str]:
    return _portrait_detail_lines(*actors, idle_frame=idle_frame, mirror_facing=mirror_facing)


def _portrait_detail_lines(
    *actors: Any | None,
    idle_frame: int = 0,
    mirror_facing: bool = False,
) -> list[str]:
    visible = []
    for actor in actors:
        if actor is None:
            continue
        art_lines = getattr(actor, "art_lines", None)
        mini_lines = getattr(actor, "mini_lines", None)
        if art_lines or mini_lines:
            visible.append(actor)
    if not visible:
        return []
    lines = ["Portraits" if len(visible) > 1 else "Portrait"]
    for index, actor in enumerate(visible):
        if index:
            lines.append("->")
        lines.append(actor.name)
        badges = _portrait_badges(actor, pulse=idle_frame % IDLE_ANIMATION_CYCLE)
        if badges:
            lines.append(_card_line(badges, 24, style=_portrait_badge_style(actor)).rstrip())
        animation_lines = _animation_art_lines(
            actor,
            "idle",
            _idle_pose_frame(actor, idle_frame),
        )
        display_lines = _portrait_display_art_lines(actor, animation_lines)
        if not display_lines:
            display_lines = _mini_art_lines(actor, idle_frame)
        art_lines = _compact_art_lines(display_lines, max_lines=6, max_width=24)
        lines.extend(_card_line(line, 24, literal=True).rstrip() for line in art_lines)
        lines.append(f"HP {actor.hp}/{actor.max_hp}  Effort {actor.effort}/{actor.max_effort}")
        lines.extend(_actor_state_detail_lines(actor))
    return lines


def _actor_state_detail_lines(actor: Any) -> list[str]:
    lines: list[str] = []
    morale = _display_state(getattr(actor, "morale", ""))
    strain = _display_state(getattr(actor, "strain", ""))
    if morale and (morale != "Steady" or bool(getattr(actor, "acting", False))):
        lines.append(f"Morale {morale}")
    if strain and strain != "Steady":
        lines.append(f"Strain {strain}")

    effects = _actor_effect_lines(actor)
    if effects:
        lines.append("Effects:")
        lines.extend(f"- {effect}" for effect in effects)
    return lines


def _actor_effect_lines(actor: Any) -> list[str]:
    effects: list[str] = []
    statuses = {_normalized_state(value) for value in getattr(actor, "statuses", ())}
    tags = {_normalized_state(value) for value in getattr(actor, "tags", ())}
    marks = {_normalized_state(value) for value in getattr(actor, "strain_marks", ())}
    strain = _normalized_state(getattr(actor, "strain", ""))
    mortal_wounds = int(getattr(actor, "mortal_wounds", 0))

    if "DEAD" in statuses:
        effects.append("Dead: removed from combat")
    elif "DOWNED" in statuses:
        effects.append("Downed: cannot act or protect")
    if mortal_wounds > 0:
        effects.append(f"Mortal Wounds {mortal_wounds}: death at 3")

    if strain == "SPENT":
        effects.append("Spent: counts as Winded, Drained, and Frayed")
        marks.difference_update({"WINDED", "DRAINED", "FRAYED"})

    mark_effects = {
        "WINDED": "Winded: cannot move",
        "DRAINED": "Drained: -1 Effort at combat start",
        "BATTERED": "Battered: -1 Defense",
        "FRAYED": "Frayed: morale pressure",
    }
    tag_effects = {
        "STUNNED": "Stunned: cannot act or protect",
        "KNOCKED_DOWN": "Knocked Down: cannot act or protect",
        "FROZEN": "Frozen: cannot act or move",
        "GUARDED": "Guarded: protecting an ally",
        "MARKED": "Marked: easier to target",
    }
    quirk_effects = {
        "HOLD_THE_LINE": "Hold the Line: +1 Defense in front row",
        "OPPORTUNIST": "Opportunist: bonuses against marked targets",
        "GENTLE_HANDS": "Gentle Hands: support can lift morale",
        "GRAVE_CALM": "Grave Calm: resists morale loss",
        "BLOOD_HOT": "Blood Hot: attack pressure",
        "ICE_NERVES": "Ice Nerves: resists frozen/shocked pressure",
    }

    effects.extend(mark_effects[key] for key in mark_effects if key in marks)
    effects.extend(tag_effects[key] for key in tag_effects if key in tags)
    quirk_values = {_normalized_state(getattr(actor, "personal_quirk", ""))}
    quirk_values.update(_normalized_state(value) for value in getattr(actor, "quirks", ()))
    effects.extend(quirk_effects[key] for key in quirk_effects if key in quirk_values)
    return _dedupe_text(effects)


def _display_state(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    return text.title() if text else ""


def _normalized_state(value: Any) -> str:
    return str(value or "").replace("-", "_").replace(" ", "_").upper()


def _dedupe_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _portrait_card_lines(
    actor: Any | None,
    *,
    width: int = BEAT_PORTRAIT_WIDTH,
    animation_cue: str = "",
    animation_frame: int = 0,
    hp: int | None = None,
    motion_offset: int = 0,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
    callout: tuple[str, str] | None = None,
    pulse_style: str = "",
    statuses: Sequence[str] | None = None,
) -> list[str]:
    if actor is None:
        return [" " * width]
    status_values = tuple(getattr(actor, "statuses", ())) if statuses is None else tuple(statuses)
    display_hp = actor.hp if hp is None else hp
    raw_art_lines = _portrait_animation_art_lines(
        actor,
        animation_cue,
        animation_frame,
        hp=display_hp,
        impact_frame=impact_frame,
        action_end_frame=action_end_frame,
    )
    if "dead" in status_values and animation_frame >= action_end_frame:
        raw_art_lines = _death_art_lines()
    raw_art_lines = _portrait_display_art_lines(
        actor,
        raw_art_lines,
    )
    if "dead" not in status_values and motion_offset:
        raw_art_lines = _offset_art_lines(raw_art_lines, motion_offset)
    art_lines = _compact_art_lines(
        raw_art_lines,
        max_lines=6,
        max_width=width,
    )
    if not art_lines:
        art_lines = [_actor_sprite(actor)]
    badges = _portrait_badges(
        actor,
        statuses=status_values,
        pulse=animation_frame % IDLE_ANIMATION_CYCLE,
    )
    lines = [_card_line(actor.name, width)]
    lines.append(
        _card_line(
            badges,
            width,
            style=_portrait_badge_style(actor, statuses=status_values) if badges else "",
        )
    )
    lines.extend(_portrait_effect_lines(callout, animation_frame, width=width))
    for line in art_lines:
        lines.append(_card_line(line, width, style=pulse_style, literal=True))
    vitals = f"HP {display_hp}/{actor.max_hp}"
    lines.append(_card_line(vitals.center(width), width))
    return lines


def _death_art_lines() -> tuple[str, ...]:
    return (
        "        ",
        "   .'.  ",
        "  .:::. ",
        " .:::::.",
        "  ~~~~~ ",
    )


def _portrait_effect_lines(
    callout: tuple[str, str] | None,
    animation_frame: int,
    *,
    width: int,
) -> list[str]:
    lanes = [_card_line("", width) for _ in range(PORTRAIT_EFFECT_LANE_COUNT)]
    if callout is None:
        return lanes
    text, style = callout
    lane_index = 0 if animation_frame >= ACTION_ANIMATION_LAST_FRAME else 1
    lanes[lane_index] = _card_line(text, width, style=style)
    return lanes


def _card_line(
    text: str,
    width: int,
    *,
    style: str = "",
    literal: bool = False,
) -> str:
    visible = text[:width]
    padding = " " * max(0, width - len(visible))
    if literal and visible.endswith("\\") and padding:
        visible = f"{visible} "
        padding = padding[1:]
    if literal:
        visible = _markup_safe_visible(visible)
    if not style:
        return f"{visible}{padding}"
    if not literal:
        visible = _markup_safe_visible(visible)
    return f"[{style}]{visible}[/]{padding}"


def _markup_safe_visible(text: str) -> str:
    return text.replace("[", "\\[")


def _portrait_badges(
    actor: Any,
    *,
    statuses: Sequence[str] | None = None,
    pulse: int = 0,
) -> str:
    status_values = tuple(getattr(actor, "statuses", ())) if statuses is None else tuple(statuses)
    badges = [status.upper() for status in status_values if status != "ready"]
    return " ".join(badges[:3])


def _portrait_badge_style(actor: Any, *, statuses: Sequence[str] | None = None) -> str:
    status_values = set(getattr(actor, "statuses", ()) if statuses is None else statuses)
    if "dead" in status_values:
        return "bold red"
    if "downed" in status_values:
        return "bold magenta"
    if status_values - {"ready"}:
        return "bold yellow"
    return "dim"


def _portrait_versus_lines(
    left_actor: Any | None,
    right_actor: Any | None,
    *,
    connector: str = "->",
    animation_cues: Mapping[str, str] | None = None,
    animation_frame: int = 0,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
    hp_overrides: Mapping[str, int] | None = None,
    callouts: Mapping[str, tuple[str, str]] | None = None,
    pulse_styles: Mapping[str, str] | None = None,
    motion_offsets: Mapping[str, int] | None = None,
    status_overrides: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    if left_actor is None and right_actor is None:
        return []
    cues = animation_cues or {}
    hp_values = hp_overrides or {}
    callout_values = callouts or {}
    pulse_values = pulse_styles or {}
    motion_values = motion_offsets or {}
    status_values = status_overrides or {}
    left = _portrait_card_lines(
        left_actor,
        animation_cue=cues.get(getattr(left_actor, "actor_id", ""), ""),
        animation_frame=animation_frame,
        hp=hp_values.get(getattr(left_actor, "actor_id", "")),
        motion_offset=motion_values.get(getattr(left_actor, "actor_id", ""), 0),
        impact_frame=impact_frame,
        action_end_frame=action_end_frame,
        callout=callout_values.get(getattr(left_actor, "actor_id", "")),
        pulse_style=pulse_values.get(getattr(left_actor, "actor_id", ""), ""),
        statuses=status_values.get(getattr(left_actor, "actor_id", "")),
    )
    right = _portrait_card_lines(
        right_actor,
        animation_cue=cues.get(getattr(right_actor, "actor_id", ""), ""),
        animation_frame=animation_frame,
        hp=hp_values.get(getattr(right_actor, "actor_id", "")),
        motion_offset=motion_values.get(getattr(right_actor, "actor_id", ""), 0),
        impact_frame=impact_frame,
        action_end_frame=action_end_frame,
        callout=callout_values.get(getattr(right_actor, "actor_id", "")),
        pulse_style=pulse_values.get(getattr(right_actor, "actor_id", ""), ""),
        statuses=status_values.get(getattr(right_actor, "actor_id", "")),
    )
    height = max(len(left), len(right))
    left.extend([" " * BEAT_PORTRAIT_WIDTH] * (height - len(left)))
    right.extend([" " * BEAT_PORTRAIT_WIDTH] * (height - len(right)))
    connector_row = min(2, max(0, height - 1))
    return [
        (
            f"{left_line}"
            f"{_centered_connector(connector, index == connector_row)}"
            f"{right_line}"
        )
        for index, (left_line, right_line) in enumerate(zip(left, right, strict=True))
    ]


def _centered_connector(connector: str, visible: bool) -> str:
    label = connector if visible else ""
    return label.center(BEAT_CONNECTOR_WIDTH)


def _beat_portrait_layout(
    source_actor: Any | None,
    target_actor: Any | None,
) -> tuple[Any | None, Any | None, str]:
    if not _opposing_combat_teams(source_actor, target_actor):
        return source_actor, target_actor, "->"
    if getattr(source_actor, "team", "") == "enemy":
        return target_actor, source_actor, "VS"
    return source_actor, target_actor, "VS"


def _opposing_combat_teams(left_actor: Any | None, right_actor: Any | None) -> bool:
    left_team = getattr(left_actor, "team", "")
    right_team = getattr(right_actor, "team", "")
    return {left_team, right_team} == {"hero", "enemy"}


def _beat_actor_pair(
    view: CombatView,
    events: Sequence[Any],
    *,
    source_actor_ids: set[str],
    target_actor_ids: set[str],
) -> tuple[Any | None, Any | None]:
    source_id = _beat_source_actor_id(events) or next(iter(sorted(source_actor_ids)), "")
    target_id = _beat_target_actor_id(events) or next(iter(sorted(target_actor_ids)), "")
    return _actor_by_id(view, source_id), _actor_by_id(view, target_id)


def _beat_source_actor_id(events: Sequence[Any]) -> str:
    for event in events:
        if _event_kind(event) == "skill_used":
            return str(getattr(event, "actor_id", ""))
    for event in events:
        if _event_kind(event) == "enemy_intent":
            return str(getattr(event, "enemy_id", ""))
    for event in events:
        if hasattr(event, "source_id"):
            return str(getattr(event, "source_id", ""))
    for event in events:
        if hasattr(event, "actor_id"):
            return str(getattr(event, "actor_id", ""))
    return ""


def _beat_target_actor_id(events: Sequence[Any]) -> str:
    for event in events:
        target_id = str(getattr(event, "target_id", "") or "")
        if target_id:
            return target_id
    for event in events:
        actor_id = str(getattr(event, "actor_id", "") or "")
        if actor_id and _event_kind(event) in {
            "combat_effect",
            "downed",
            "death",
            "status_changed",
        }:
            return actor_id
    return ""


def _combat_beat_result_lines(
    events: Sequence[Any],
    *,
    animation_frame: int = ACTION_ANIMATION_LAST_FRAME,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
) -> list[str]:
    action_lines: list[str] = []
    effect_events: list[Any] = []
    typed_effect_keys = {
        _combat_effect_duplicate_key(event)
        for event in events
        if _is_combat_effect_event(event)
    }
    for event in events:
        kind = _event_kind(event)
        if kind in _footer_event_kinds():
            continue
        if kind == "status_changed" and _status_duplicate_key(event) in typed_effect_keys:
            continue
        message = _event_sentence(event)
        if kind == "move":
            action_lines.append(_move_event_line(event))
            continue
        if kind in {
            "skill_used",
            "enemy_intent",
            "reaction_used",
            "reaction_skipped",
            "move",
            "turn_delayed",
            "turn_passed",
        }:
            action_lines.append(_styled_action_line(kind, message))
        elif message:
            effect_events.append(event)

    visible_effect_events = _visible_beat_effect_events(
        effect_events,
        animation_frame,
        impact_frame=impact_frame,
        action_end_frame=action_end_frame,
    )
    primary_is_typed = _primary_effect_is_typed(visible_effect_events)
    effect_line = _combined_effect_line(
        visible_effect_events,
    )
    status_line = _combat_beat_status_line(
        visible_effect_events,
        include_typed_effects=not primary_is_typed,
    )
    result_line_count = int(bool(effect_line)) + int(bool(status_line))
    action_limit = 1 if result_line_count >= 2 else 2
    lines = _dedupe_lines(action_lines)[:action_limit]
    if effect_line:
        lines.append(effect_line)
    if status_line:
        lines.append(status_line)
    return lines[:3]


def _visible_beat_effect_events(
    effect_events: Sequence[Any],
    animation_frame: int,
    *,
    impact_frame: int = DAMAGE_REACTION_FRAME,
    action_end_frame: int = ACTION_ANIMATION_LAST_FRAME,
) -> list[Any]:
    if animation_frame < impact_frame:
        return []
    if animation_frame < action_end_frame:
        return [
            event
            for event in effect_events
            if _event_kind(event) in {"damage", "healing", "miss"}
            or _is_combat_effect_event(event)
            or _is_effort_change_event(event)
        ]
    return list(effect_events)


def _combat_beat_footer_lines(events: Sequence[Any]) -> list[str]:
    combat_end = next(
        (event for event in events if _event_kind(event) == "combat_ended"),
        None,
    )
    if combat_end is not None:
        victor = str(getattr(combat_end, "victor", ""))
        return ["[dim]Victory.[/]"] if victor == "heroes" else ["[dim]Defeat.[/]"]

    messages = [
        _event_sentence(event)
        for event in events
        if _event_kind(event) in _footer_event_kinds()
    ]
    if not messages:
        return []
    return [f"[dim]{'  |  '.join(messages[:3])}[/]"]


def _fallback_beat_lines(events: Sequence[Any]) -> list[str]:
    result_lines = _combat_beat_result_lines(events)
    footer_lines = _combat_beat_footer_lines(events)
    return [*result_lines, *footer_lines] or [event.message for event in events]


def _styled_action_line(kind: str, message: str) -> str:
    if kind == "enemy_intent":
        return f"[bold yellow]{message}[/]"
    if kind == "reaction_used":
        return f"[bold cyan]{message}[/]"
    if kind == "reaction_skipped":
        return f"[dim]{message}[/]"
    if kind in {"move", "turn_delayed", "turn_passed"}:
        return message
    return f"[bold]{message}[/]"


def _move_event_line(event: Any) -> str:
    message = _event_sentence(event)
    from_slot = _slot_name(str(getattr(event, "from_slot", "")))
    to_slot = _slot_name(str(getattr(event, "to_slot", "")))
    if from_slot and to_slot and "->" not in message and "<->" not in message:
        return f"[bold cyan]Formation[/] {message}  ({from_slot} -> {to_slot})"
    return f"[bold cyan]Formation[/] {message}"


def _combined_effect_line(effect_events: Sequence[Any]) -> str:
    if not effect_events:
        return ""

    death_event = next((event for event in effect_events if _event_kind(event) == "death"), None)
    if death_event is not None:
        return f"[bold red]{_death_summary(death_event)}[/]"

    downed_event = next((event for event in effect_events if _event_kind(event) == "downed"), None)
    if downed_event is not None:
        return f"[bold magenta]{_event_sentence(downed_event)}[/]"

    effort_event = next(
        (event for event in effect_events if _is_effort_change_event(event)),
        None,
    )
    typed_effects = [
        event
        for event in sorted(effect_events, key=_combat_effect_sort_key)
        if _is_combat_effect_event(event)
    ]
    typed_effect = next(iter(typed_effects), None)
    typed_effect_line = _typed_effect_result_line(typed_effects)
    damage_event = next(
        (event for event in effect_events if _event_kind(event) == "damage"),
        None,
    )
    if typed_effect is not None and (
        damage_event is None or int(getattr(damage_event, "amount", 0)) <= 0
    ):
        return f"[{_combat_effect_style(typed_effect)}]{typed_effect_line}[/]"
    if effort_event is not None and (
        damage_event is None or int(getattr(damage_event, "amount", 0)) <= 0
    ):
        return f"[bold bright_blue]{_event_sentence(effort_event)}[/]"

    for kind, style in (
        ("damage", "bold red"),
        ("healing", "bold green"),
        ("combat_effect", ""),
        ("miss", "bold yellow"),
        ("status_changed", "yellow"),
    ):
        event = next(
            (candidate for candidate in effect_events if _event_kind(candidate) == kind),
            None,
        )
        if event is not None:
            if kind == "combat_effect":
                style = _combat_effect_style(event)
            return f"[{style}]{_event_sentence(event)}[/]"

    return f"[dim]{_event_sentence(effect_events[0])}[/]"


def _death_summary(event: Any) -> str:
    message = _event_sentence(event)
    subject, separator, _reason = message.partition(" dies:")
    if separator and subject:
        reason = _reason.strip().rstrip(".")
        if "damage" in reason:
            damage_clause = reason.split(",", 1)[0].strip()
            if damage_clause:
                return f"{subject} dies: {damage_clause}."
        return f"{subject} dies."
    return message


def _primary_effect_is_typed(effect_events: Sequence[Any]) -> bool:
    if not any(_is_combat_effect_event(event) for event in effect_events):
        return False
    return not any(
        _event_kind(event) in {"damage", "healing", "miss", "downed", "death"}
        for event in effect_events
    )


def _combat_beat_status_line(
    effect_events: Sequence[Any],
    *,
    include_typed_effects: bool,
) -> str:
    items: list[str] = []
    typed_effects = [
        event
        for event in sorted(effect_events, key=_combat_effect_sort_key)
        if _is_combat_effect_event(event)
    ]
    typed_replacements = {
        _typed_effect_status_replacement(event) for event in typed_effects
    }
    typed_replacements.discard(("", ""))
    if include_typed_effects:
        items.extend(_combat_effect_clause(event) for event in typed_effects)
    for event in effect_events:
        if _event_kind(event) != "status_changed":
            continue
        key = _status_duplicate_key(event)
        status = key[1].lower()
        if key in typed_replacements or status in {"dead", "downed", "effort"}:
            continue
        message = _event_sentence(event)
        if message:
            items.append(message)
    items = _dedupe_lines(items)
    if not items:
        return ""
    return f"[dim]Effects: {'  |  '.join(items[:3])}[/]"


def _typed_effect_result_line(events: Sequence[Any]) -> str:
    if not events:
        return ""
    first, *rest = events
    pieces = [_event_sentence(first)]
    pieces.extend(_combat_effect_clause(event) for event in rest[:1])
    return "  |  ".join(piece for piece in pieces if piece)


def _combat_effect_clause(event: Any) -> str:
    label = _combat_effect_label(event)
    source_kind = str(getattr(event, "source_kind", ""))
    source_id = str(getattr(event, "source_id", ""))
    if source_kind == "quirk" and source_id:
        return f"{_display_state(source_id)}: {label}"
    return label or _event_sentence(event)


def _typed_effect_status_replacement(event: Any) -> tuple[str, str]:
    actor_id = str(getattr(event, "actor_id", ""))
    tag = str(getattr(event, "tag", ""))
    if tag:
        return actor_id, tag
    source_kind = str(getattr(event, "source_kind", ""))
    source_id = str(getattr(event, "source_id", ""))
    if source_kind == "tag" and source_id:
        return actor_id, source_id
    return "", ""


def _is_effort_change_event(event: Any) -> bool:
    return _event_kind(event) == "status_changed" and str(
        getattr(event, "status", "")
    ).lower() == "effort"


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


def _combat_effect_sort_key(event: Any) -> tuple[int, str]:
    effect_type = str(getattr(event, "effect_type", ""))
    emphasis = _combat_effect_emphasis(event)
    rank = {
        ("resource", "bad"): 0,
        ("resource", "good"): 1,
        ("mitigation", "good"): 2,
        ("tag", "bad"): 3,
        ("tag", "good"): 4,
    }.get((effect_type, emphasis), 5)
    return rank, _event_sentence(event)


def _combat_effect_duplicate_key(event: Any) -> tuple[str, str]:
    actor_id = str(getattr(event, "actor_id", ""))
    resource = str(getattr(event, "resource", ""))
    tag = str(getattr(event, "tag", ""))
    if resource:
        return actor_id, resource
    if tag:
        return actor_id, tag
    return actor_id, str(getattr(event, "effect_type", ""))


def _status_duplicate_key(event: Any) -> tuple[str, str]:
    return str(getattr(event, "actor_id", "")), str(getattr(event, "status", ""))


def _dedupe_lines(lines: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for line in lines:
        if line and line not in deduped:
            deduped.append(line)
    return deduped


def _event_sentence(event: Any) -> str:
    message = str(getattr(event, "message", "")).strip()
    if not message:
        return ""
    return message if message.endswith((".", "!", "?")) else f"{message}."


def _event_clause(event: Any) -> str:
    return _event_sentence(event).rstrip(".!?")


def _footer_event_kinds() -> set[str]:
    return {
        "round_started",
        "round_ended",
        "order_changed",
        "encounter_ended",
        "combat_ended",
        "combat_retreat_declared",
        "combat_retreated",
        "expedition",
        "loot_gained",
        "breach_discovered",
        "expedition_returned",
    }


def _has_danger_event(events: Sequence[Any]) -> bool:
    return any(_event_kind(event) == "enemy_intent" for event in events)


def _event_kind(event: Any) -> str:
    event_type = getattr(event, "event_type", "")
    return str(getattr(event_type, "value", event_type))


def _actor_by_id(view: CombatView, actor_id: str) -> Any | None:
    return next(
        (actor for actor in (*view.party, *view.enemies) if actor.actor_id == actor_id),
        None,
    )


def _combat_board(
    title: str,
    actors: Sequence[Any],
    *,
    focused_actor_id: str,
    legal_actor_ids: set[str],
    highlighted_actor_ids: set[str] | None = None,
    target_intents: Mapping[str, str],
    source_actor_ids: set[str] | None = None,
) -> str:
    source_actor_ids = source_actor_ids or set()
    highlighted_actor_ids = highlighted_actor_ids or set()
    cells = _combat_cells(
        actors,
        focused_actor_id=focused_actor_id,
        legal_actor_ids=legal_actor_ids,
        highlighted_actor_ids=highlighted_actor_ids,
        target_intents=target_intents,
        source_actor_ids=source_actor_ids,
        width=CELL_WIDTH,
    )
    return f"{title}\n{_grid_text(cells)}"


def _combat_duel_board(
    party: Sequence[Any],
    enemies: Sequence[Any],
    *,
    party_focus_id: str,
    party_legal_ids: set[str],
    party_highlight_ids: set[str],
    party_intents: Mapping[str, str],
    enemy_focus_id: str,
    enemy_legal_ids: set[str],
    enemy_highlight_ids: set[str],
    enemy_intents: Mapping[str, str],
    party_source_ids: set[str] | None = None,
    enemy_source_ids: set[str] | None = None,
    party_slot_annotations: Mapping[str, str] | None = None,
    idle_frame: int = 0,
    turn_flash_actor_id: str = "",
    turn_flash_frame: int = 0,
) -> str:
    if COMBAT_FIELD_LAYOUT == "mini":
        return _mini_combat_duel_board(
            party,
            enemies,
            party_focus_id=party_focus_id,
            party_legal_ids=party_legal_ids,
            party_highlight_ids=party_highlight_ids,
            party_intents=party_intents,
            enemy_focus_id=enemy_focus_id,
            enemy_legal_ids=enemy_legal_ids,
            enemy_highlight_ids=enemy_highlight_ids,
            enemy_intents=enemy_intents,
            party_source_ids=party_source_ids or set(),
            enemy_source_ids=enemy_source_ids or set(),
            party_slot_annotations=party_slot_annotations or {},
            idle_frame=idle_frame,
            turn_flash_actor_id=turn_flash_actor_id,
            turn_flash_frame=turn_flash_frame,
        )
    return _glyph_combat_duel_board(
        party,
        enemies,
        party_focus_id=party_focus_id,
        party_legal_ids=party_legal_ids,
        party_highlight_ids=party_highlight_ids,
        party_intents=party_intents,
        enemy_focus_id=enemy_focus_id,
        enemy_legal_ids=enemy_legal_ids,
        enemy_highlight_ids=enemy_highlight_ids,
        enemy_intents=enemy_intents,
        party_source_ids=party_source_ids,
        enemy_source_ids=enemy_source_ids,
        idle_frame=idle_frame,
        turn_flash_actor_id=turn_flash_actor_id,
        turn_flash_frame=turn_flash_frame,
    )


def _glyph_combat_duel_board(
    party: Sequence[Any],
    enemies: Sequence[Any],
    *,
    party_focus_id: str,
    party_legal_ids: set[str],
    party_highlight_ids: set[str],
    party_intents: Mapping[str, str],
    enemy_focus_id: str,
    enemy_legal_ids: set[str],
    enemy_highlight_ids: set[str],
    enemy_intents: Mapping[str, str],
    party_source_ids: set[str] | None = None,
    enemy_source_ids: set[str] | None = None,
    idle_frame: int = 0,
    turn_flash_actor_id: str = "",
    turn_flash_frame: int = 0,
) -> str:
    party_cells = _combat_cells(
        party,
        focused_actor_id=party_focus_id,
        legal_actor_ids=party_legal_ids,
        highlighted_actor_ids=party_highlight_ids,
        target_intents=party_intents,
        source_actor_ids=party_source_ids or set(),
        width=COMBAT_CELL_WIDTH,
        idle_frame=idle_frame,
        turn_flash_actor_id=turn_flash_actor_id,
        turn_flash_frame=turn_flash_frame,
    )
    enemy_cells = _combat_cells(
        enemies,
        focused_actor_id=enemy_focus_id,
        legal_actor_ids=enemy_legal_ids,
        highlighted_actor_ids=enemy_highlight_ids,
        target_intents=enemy_intents,
        source_actor_ids=enemy_source_ids or set(),
        width=COMBAT_CELL_WIDTH,
        idle_frame=idle_frame,
        turn_flash_actor_id=turn_flash_actor_id,
        turn_flash_frame=turn_flash_frame,
    )
    party_lines = _grid_text(
        party_cells,
        rows=PARTY_COMBAT_ROWS,
        width=COMBAT_CELL_WIDTH,
        show_labels=False,
    ).splitlines()
    enemy_lines = _grid_text(
        enemy_cells,
        rows=ENEMY_COMBAT_ROWS,
        width=COMBAT_CELL_WIDTH,
        show_labels=False,
    ).splitlines()
    left_title = "Party"
    right_title = "Enemies"
    gap = "  VS  "
    board_width = 1 + 2 * (COMBAT_CELL_WIDTH + 3)
    lines = [
        f"{left_title:<{board_width}}{gap}{right_title}",
    ]
    for party_line, enemy_line in zip(party_lines, enemy_lines, strict=True):
        lines.append(f"{party_line}{gap}{enemy_line}")
    return "\n".join(lines)


def _mini_combat_duel_board(
    party: Sequence[Any],
    enemies: Sequence[Any],
    *,
    party_focus_id: str,
    party_legal_ids: set[str],
    party_highlight_ids: set[str],
    party_intents: Mapping[str, str],
    enemy_focus_id: str,
    enemy_legal_ids: set[str],
    enemy_highlight_ids: set[str],
    enemy_intents: Mapping[str, str],
    party_source_ids: set[str],
    enemy_source_ids: set[str],
    party_slot_annotations: Mapping[str, str] | None = None,
    idle_frame: int = 0,
    turn_flash_actor_id: str = "",
    turn_flash_frame: int = 0,
) -> str:
    party_by_slot = {actor.slot: actor for actor in party}
    enemy_by_slot = {actor.slot: actor for actor in enemies}
    party_rows = _mini_side_rows(
        party_by_slot,
        rows=PARTY_COMBAT_ROWS,
        focus_id=party_focus_id,
        legal_ids=party_legal_ids,
        highlight_ids=party_highlight_ids,
        intents=party_intents,
        source_ids=party_source_ids,
        slot_annotations=party_slot_annotations or {},
        idle_frame=idle_frame,
        turn_flash_actor_id=turn_flash_actor_id,
        turn_flash_frame=turn_flash_frame,
    )
    enemy_rows = _mini_side_rows(
        enemy_by_slot,
        rows=ENEMY_COMBAT_ROWS,
        focus_id=enemy_focus_id,
        legal_ids=enemy_legal_ids,
        highlight_ids=enemy_highlight_ids,
        intents=enemy_intents,
        source_ids=enemy_source_ids,
        idle_frame=idle_frame,
        turn_flash_actor_id=turn_flash_actor_id,
        turn_flash_frame=turn_flash_frame,
    )
    side_width = (MINI_SLOT_WIDTH * 2) + len(MINI_SIDE_GAP)
    middle_blank = " " * MINI_MIDDLE_WIDTH
    lines = [
        f"{'PARTY':^{side_width}}{middle_blank}{'ENEMIES':^{side_width}}",
        f"{'':{side_width}}{'VS':^{MINI_MIDDLE_WIDTH}}{'':{side_width}}",
    ]
    for party_line, enemy_line in zip(party_rows, enemy_rows, strict=True):
        lines.append(f"{party_line}{middle_blank}{enemy_line}")
    return "\n".join(lines)


def _mini_side_rows(
    actors_by_slot: Mapping[str, Any],
    *,
    rows: Sequence[tuple[str, str]],
    focus_id: str,
    legal_ids: set[str],
    highlight_ids: set[str],
    intents: Mapping[str, str],
    source_ids: set[str],
    slot_annotations: Mapping[str, str] | None = None,
    idle_frame: int,
    turn_flash_actor_id: str,
    turn_flash_frame: int,
    inward_facing: bool = False,
) -> list[str]:
    annotations = slot_annotations or {}
    rendered_rows: list[str] = []
    for row in rows:
        left = _mini_slot_lines(
            actors_by_slot.get(row[0]),
            slot_key=row[0],
            slot_annotation=annotations.get(row[0], ""),
            focus_id=focus_id,
            legal_ids=legal_ids,
            highlight_ids=highlight_ids,
            intents=intents,
            source_ids=source_ids,
            idle_frame=idle_frame,
            turn_flash_actor_id=turn_flash_actor_id,
            turn_flash_frame=turn_flash_frame,
            mirror_facing=inward_facing and formation_slot_faces_inward(row[0]),
        )
        right = _mini_slot_lines(
            actors_by_slot.get(row[1]),
            slot_key=row[1],
            slot_annotation=annotations.get(row[1], ""),
            focus_id=focus_id,
            legal_ids=legal_ids,
            highlight_ids=highlight_ids,
            intents=intents,
            source_ids=source_ids,
            idle_frame=idle_frame,
            turn_flash_actor_id=turn_flash_actor_id,
            turn_flash_frame=turn_flash_frame,
            mirror_facing=inward_facing and formation_slot_faces_inward(row[1]),
        )
        rendered_rows.extend(
            f"{left_line}{MINI_SIDE_GAP}{right_line}"
            for left_line, right_line in zip(left, right, strict=True)
        )
        rendered_rows.append(" " * ((MINI_SLOT_WIDTH * 2) + len(MINI_SIDE_GAP)))
    if rendered_rows:
        rendered_rows.pop()
    return rendered_rows


def _mini_slot_lines(
    actor: Any | None,
    *,
    slot_key: str = "",
    slot_annotation: str = "",
    focus_id: str,
    legal_ids: set[str],
    highlight_ids: set[str],
    intents: Mapping[str, str],
    source_ids: set[str],
    idle_frame: int,
    turn_flash_actor_id: str,
    turn_flash_frame: int,
    mirror_facing: bool = False,
) -> list[str]:
    if actor is None:
        if slot_annotation:
            return [
                _mini_cell_line(slot_annotation),
                _mini_cell_line("open slot"),
                _mini_cell_line(""),
                _mini_cell_line(""),
                _mini_cell_line(""),
                _mini_cell_line(""),
                _mini_cell_line(""),
            ]
        return [" " * MINI_SLOT_WIDTH for _ in range(7)]
    actor_id = str(getattr(actor, "actor_id", ""))
    intent = intents.get(actor_id, "")
    marker = _mini_marker(
        actor,
        focused=actor_id == focus_id,
        legal=actor_id in legal_ids,
        highlighted=actor_id in highlight_ids,
        source=actor_id in source_ids,
        intent=intent,
    )
    if slot_annotation:
        marker = f"{slot_annotation}{marker}".strip()
    name = _mini_actor_name(actor)
    art_lines = _mini_art_lines(actor, idle_frame, mirror_facing=mirror_facing)
    hp = _mini_hp_line(actor)
    tags = _mini_status_tags(actor, intent)
    nudge = _mini_slot_nudge(actor)
    name_style = _mini_name_style(
        intent,
        focused=actor_id == focus_id,
        highlighted=actor_id in highlight_ids,
        source=actor_id in source_ids,
        acting=bool(getattr(actor, "acting", False)),
        turn_flash=actor_id == turn_flash_actor_id and turn_flash_frame == 0,
    )
    lines = [
        _mini_cell_line(marker, indent=nudge),
        _mini_cell_line(name, style=name_style, indent=nudge),
        *(_mini_cell_line(line, literal=True, indent=nudge) for line in art_lines),
        _mini_cell_line(hp, style=_mini_hp_style(actor), indent=nudge),
        _mini_cell_line(tags, style=_mini_status_style(tags, intent), indent=nudge),
    ]
    return lines


def _mini_slot_nudge(actor: Any) -> int:
    slot = str(getattr(actor, "slot", ""))
    team = str(getattr(actor, "team", ""))
    seed = f"{team}:{slot}"
    return sum(ord(character) for character in seed) % 3


def _mini_hp_line(actor: Any) -> str:
    return f"HP {int(getattr(actor, 'hp', 0))}/{int(getattr(actor, 'max_hp', 0))}"


def _mini_hp_style(actor: Any) -> str:
    hp = int(getattr(actor, "hp", 0))
    max_hp = max(1, int(getattr(actor, "max_hp", 1)))
    ratio = hp / max_hp
    if hp <= 0:
        return "bold red"
    if ratio <= 0.35:
        return "red"
    if ratio <= 0.65:
        return "yellow"
    return "dim"


def _mini_name_style(
    intent: str,
    *,
    focused: bool,
    highlighted: bool,
    source: bool,
    acting: bool,
    turn_flash: bool,
) -> str:
    if turn_flash:
        return "bold black on bright_cyan"
    if acting:
        return "bold cyan"
    if source:
        return "bold magenta"
    if focused or highlighted:
        if intent == "heal":
            return "bold bright_green"
        if intent in {"attack", "debuff"}:
            return "bold yellow"
        return "bold white"
    return ""


def _mini_actor_name(actor: Any) -> str:
    return str(getattr(actor, "name", "") or getattr(actor, "display_name", "") or "Unknown")


def _mini_marker(
    actor: Any,
    *,
    focused: bool,
    legal: bool,
    highlighted: bool,
    source: bool,
    intent: str,
) -> str:
    if bool(getattr(actor, "acting", False)):
        return "ACTING v"
    if focused or highlighted:
        return "TARGET v" if intent and intent != "heal" else "FOCUS v"
    if source:
        return "SOURCE"
    if legal:
        return "TARGET"
    return ""


def _mini_art_lines(
    actor: Any,
    idle_frame: int,
    *,
    mirror_facing: bool = False,
) -> tuple[str, str, str]:
    frames = tuple((getattr(actor, "mini_frames", {}) or {}).get("idle", ()))
    if frames:
        frame = frames[_idle_pose_frame(actor, idle_frame) % len(frames)]
        lines = tuple(frame)
    else:
        lines = tuple(getattr(actor, "mini_lines", ()) or ())
        if not lines:
            lines = _derive_mini_art_from_actor(actor)
    return _normalize_mini_lines(lines)


def _derive_mini_art_from_actor(actor: Any) -> tuple[str, ...]:
    lines = tuple(getattr(actor, "art_lines", ()) or ())
    if len(lines) >= 3:
        return (lines[0].strip(), lines[len(lines) // 2].strip(), lines[-1].strip())
    glyph = str(getattr(actor, "glyph", "") or _actor_grid_glyph(actor))
    return (glyph, "|", "/ \\")


def _normalize_mini_lines(lines: Sequence[str]) -> tuple[str, str, str]:
    normalized = [str(line).rstrip() for line in lines[:3]]
    while len(normalized) < 3:
        normalized.append("")
    block_width = min(
        MINI_ART_WIDTH,
        max((len(line) for line in normalized[:3]), default=0),
    )
    left_padding = max(0, (MINI_ART_WIDTH - block_width) // 2)
    right_padding = max(0, MINI_ART_WIDTH - block_width - left_padding)
    centered = [
        f"{' ' * left_padding}{line[:block_width].ljust(block_width)}{' ' * right_padding}"
        for line in normalized[:3]
    ]
    return (centered[0], centered[1], centered[2])


def _mini_status_tags(actor: Any, intent: str) -> str:
    tags: list[str] = []
    mortal_wounds = int(getattr(actor, "mortal_wounds", 0))
    if mortal_wounds > 0:
        tags.append(mortal_wound_badge(mortal_wounds))
    for status in tuple(getattr(actor, "statuses", ())):
        label = _mini_status_label(str(status))
        if label and label not in tags:
            tags.append(label)
    if _normalized_state(getattr(actor, "strain", "")) == "SPENT" and "SPENT" not in tags:
        tags.append("SPENT")
    for tag in tuple(getattr(actor, "tags", ())):
        label = _mini_status_label(str(tag))
        if label and label not in tags:
            tags.append(label)
    return " ".join(_mini_badge_text(tag) for tag in tags[:MINI_VISIBLE_STATUS_TAGS])


def _mini_badge_text(tag: str) -> str:
    if tag.startswith("\\["):
        return tag
    if tag.startswith("[") and tag.endswith("]"):
        return tag
    return f"[{tag}]"


def _mini_status_label(value: str) -> str:
    normalized = value.upper().replace(" ", "_")
    aliases = {
        "GUARDED": "GUARD",
        "GUARD": "GUARD",
        "WARD": "WARD",
        "WARDED": "WARD",
        "BLEEDING": "BLEED",
        "BLEED": "BLEED",
        "STUNNED": "STUN",
        "STUN": "STUN",
        "KNOCKED_DOWN": "KDOWN",
        "DOWNED": "DOWN",
        "DEAD": "DEAD",
        "SPENT": "SPENT",
        "MARKED": "MARK",
        "WET": "WET",
        "FROZEN": "FROZ",
        "BURNING": "BURN",
        "SHOCKED": "SHOCK",
    }
    return aliases.get(normalized, normalized[:5] if normalized != "READY" else "")


def _mini_status_style(tags: str, intent: str) -> str:
    if not tags:
        return ""
    if "DOWN" in tags:
        return "bold red"
    if "[x" in tags:
        return "yellow"
    return _soft_intent_style(intent) if intent else "dim"


def _mini_cell_line(
    text: str,
    *,
    style: str = "",
    literal: bool = False,
    indent: int = 0,
) -> str:
    safe_indent = max(0, indent)
    if len(text) + safe_indent > MINI_SLOT_WIDTH:
        safe_indent = 0
    visible = f"{' ' * safe_indent}{text}"[:MINI_SLOT_WIDTH]
    padding = " " * max(0, MINI_SLOT_WIDTH - len(visible))
    if literal:
        visible = _markup_safe_visible(visible)
    if not style:
        return f"{visible}{padding}"
    if not literal:
        visible = _markup_safe_visible(visible)
    return f"[{style}]{visible}[/]{padding}"


def _combat_cells(
    actors: Sequence[Any],
    *,
    focused_actor_id: str,
    legal_actor_ids: set[str],
    highlighted_actor_ids: set[str],
    target_intents: Mapping[str, str],
    source_actor_ids: set[str],
    width: int,
    idle_frame: int = 0,
    turn_flash_actor_id: str = "",
    turn_flash_frame: int = 0,
) -> dict[str, tuple[str, str, str, str]]:
    actor_by_slot = {actor.slot: actor for actor in actors}
    cells: dict[str, tuple[str, str, str, str]] = {}
    for row in GRID_ROWS:
        for slot in row:
            actor = actor_by_slot.get(slot)
            if actor is None:
                cells[slot] = _grid_cell("empty", "empty", marker=" ", width=width)
                continue
            marker = _actor_marker(
                actor.actor_id,
                acting=actor.acting,
                focused_actor_id=focused_actor_id,
                legal_actor_ids=legal_actor_ids,
                source_actor_ids=source_actor_ids,
                statuses=actor.statuses,
            )
            intent = target_intents.get(actor.actor_id, "")
            status = _combat_status_line(
                actor.statuses,
                int(getattr(actor, "mortal_wounds", 0)),
                intent,
                focused=actor.actor_id == focused_actor_id,
                highlighted=actor.actor_id in highlighted_actor_ids,
                source=actor.actor_id in source_actor_ids,
            )
            turn_flash = (
                actor.actor_id == turn_flash_actor_id
                and turn_flash_frame == 0
            )
            cells[slot] = _grid_cell(
                _combat_cell_figure(
                    actor,
                    focused=actor.actor_id == focused_actor_id,
                    acting=actor.acting,
                    idle_frame=idle_frame,
                ),
                f"HP {actor.hp}/{actor.max_hp}  EF {actor.effort}",
                status,
                marker=marker,
                width=width,
                style="bold cyan"
                if turn_flash
                else _combat_cell_style(
                    intent,
                    focused=actor.actor_id == focused_actor_id,
                    highlighted=actor.actor_id in highlighted_actor_ids,
                    source=actor.actor_id in source_actor_ids,
                    acting=actor.acting,
                ),
            )
    return cells


def _lines_or_none(values: Sequence[str]) -> str:
    if not values:
        return "none"
    return "\n".join(f"- {value}" for value in values)


def _objective_lines(objective: Any) -> tuple[str, ...]:
    lines = [
        f"{objective.title} [{objective.status}]",
        objective.summary,
    ]
    if getattr(objective, "progress", ""):
        lines.append(f"Progress: {objective.progress}")
    lines.extend(
        [
            f"Next: {objective.next_step}",
            f"Chapter: {objective.chapter_status}",
        ]
    )
    lines.extend(
        f"{step.state.title()}: {step.name}"
        for step in objective.steps
    )
    return tuple(line for line in lines if line)


def _contract_summary_line(entry: Any) -> str:
    reward = _contract_reward_summary(entry)
    line = f"- {entry.name}: {entry.state.title()} ({reward})"
    if entry.unavailable_reason:
        line += f" - {entry.unavailable_reason}"
    return line


def _contract_reward_summary(entry: Any) -> str:
    pieces: list[str] = []
    if getattr(entry, "reward_reputation", 0):
        pieces.append(f"+{entry.reward_reputation} rep")
    if getattr(entry, "coin_reward", 0):
        pieces.append(f"+{entry.coin_reward} Coin")
    return ", ".join(pieces) or "no payout"


def _upgrade_summary_line(entry: Any) -> str:
    line = f"- {entry.name}: {entry.state.title()} (cost {entry.cost})"
    if entry.effect_summary:
        line += f" - {entry.effect_summary}"
    if entry.unavailable_reason and entry.state != "installed":
        line += f" - {entry.unavailable_reason}"
    return line


def _report_brief_lines(view: ExpeditionReportView) -> tuple[str, ...]:
    lines = [
        f"Outcome: {view.outcome.replace('_', ' ').title()}.",
        (
            f"Reputation: {view.reputation_start}->{view.reputation_end} "
            f"({_signed(view.reputation_delta)})."
        ),
        (
            f"Coin: {view.coin_start}->{view.coin_end} "
            f"({_signed(view.coin_delta)})."
        ),
    ]
    if view.wounded_count or view.downed_count or view.deceased_count:
        lines.append(
            "Condition: "
            f"{view.wounded_count} wounded, "
            f"{view.downed_count} downed, "
            f"{view.deceased_count} memorialized."
        )
    if view.supplies_spent:
        spent = ", ".join(
            f"{supply_id} x{quantity}" for supply_id, quantity in view.supplies_spent
        )
        lines.append(f"Supplies spent: {spent}.")
    if view.loot:
        loot = ", ".join(f"{item_id} x{quantity}" for item_id, quantity in view.loot)
        lines.append(f"Loot secured: {loot}.")
    if getattr(view, "gear", ()):
        gear = ", ".join(f"{gear_id} x{quantity}" for gear_id, quantity in view.gear)
        lines.append(f"Gear secured: {gear}.")
    if view.breaches_discovered:
        breaches = ", ".join(view.breaches_discovered)
        lines.append(f"Breach discovered: {breaches}.")
    lines.extend(view.hero_outcomes[:2])
    lines.extend(view.notable_moments[:3])
    return tuple(lines)


def _reward_lines(view: ExpeditionReportView) -> str:
    lines = [f"Reputation: {view.reputation_gained}", f"Coin: {view.coin_gained}"]
    lines.extend(f"Loot: {item_id} x{quantity}" for item_id, quantity in view.loot)
    lines.extend(f"Supplies: {supply_id} x{quantity}" for supply_id, quantity in view.supplies)
    lines.extend(f"Gear: {gear_id} x{quantity}" for gear_id, quantity in view.gear)
    lines.extend(
        f"Spent: {supply_id} x{quantity}"
        for supply_id, quantity in view.supplies_spent
    )
    lines.extend(f"Breach: {breach_id}" for breach_id in view.breaches_discovered)
    return "\n".join(lines) if lines else "none"


def _delta_lines(values: Sequence[tuple[str, int, int, int]]) -> str:
    if not values:
        return "none"
    return "\n".join(
        f"- {item_id}: {start}->{end} ({_signed(delta)})"
        for item_id, start, end, delta in values
    )


def _signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _action_number_for_value(
    view: DungeonView,
    value: str,
    *,
    actions: Sequence[ScreenAction] | None = None,
) -> str:
    available_actions = view.actions if actions is None else actions
    action = next(
        (candidate for candidate in available_actions if candidate.value == value),
        None,
    )
    return action.number if action is not None else ""


def _map_inventory_line(view: DungeonView) -> str:
    inventory = _quantity_line(view.inventory)
    supplies = _quantity_line(view.supplies)
    return format_meta_line(
        f"inventory: {inventory}" if inventory else "inventory: none",
        f"supplies: {supplies}" if supplies else "",
    )


def _map_node_detail_lines(
    view: DungeonView,
    node: Any,
    *,
    actions: Sequence[ScreenAction] | None = None,
) -> list[str]:
    marker = _minimap_node_label(view, node, actions=actions)
    safe = "safe return" if node.safe_return else ""
    heading = f"{marker} {node.name}"
    detail = format_meta_line(
        _map_status_label(node),
        node.node_type.replace("_", " "),
        node.direction,
        safe,
    )
    lines = [heading, f"  {detail}"]
    if node.memory_summary:
        lines.append(f"  memory: {node.memory_summary}")
    inventory = _node_inventory_brief(node)
    if inventory:
        lines.append(f"  inventory: {inventory}")
    for action_line in node.action_summaries[:3]:
        lines.append(f"  action: {action_line}")
    if len(node.action_summaries) > 3:
        lines.append(f"  action: +{len(node.action_summaries) - 3} more")
    for note in node.memory_notes[:2]:
        lines.append(f"  note: {note}")
    return lines


def _node_inventory_brief(node: Any) -> str:
    pieces = [
        f"loot {_quantity_line(node.inventory_rewards)}"
        if node.inventory_rewards
        else "",
        f"supplies {_quantity_line(node.supply_rewards)}"
        if node.supply_rewards
        else "",
        f"reputation +{node.reputation_reward}" if node.reputation_reward else "",
        f"Coin +{node.coin_reward}" if node.coin_reward else "",
        f"needs {_quantity_line(node.inventory_requirements)}"
        if node.inventory_requirements
        else "",
        f"costs {_quantity_line(node.supply_costs)}" if node.supply_costs else "",
    ]
    return format_meta_line(*pieces)


def _quantity_line(values: Sequence[tuple[str, int]]) -> str:
    return ", ".join(
        f"{_label_identifier(item_id)} x{quantity}"
        for item_id, quantity in values
        if quantity
    )


def _label_identifier(value: str) -> str:
    return value.replace("_", " ")


def _minimap_lines(
    view: DungeonView,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    return _coordinate_minimap_lines(
        view,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    ) or _minimap_branch_lines(
        view,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )


def _highlight_minimap_nodes(
    text: Text,
    view: DungeonView,
    *,
    highlighted_node_id: str,
    actions: Sequence[ScreenAction] | None = None,
) -> None:
    map_start = text.plain.find("\n\n")
    if map_start == -1:
        return
    map_start += 2
    for node in view.map_nodes:
        if node.current:
            style = "bold cyan"
            labels = _minimap_current_node_labels(view, node, actions=actions)
        elif node.node_id == highlighted_node_id:
            style = "bold yellow"
            labels = _minimap_highlighted_node_labels(
                view,
                node,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
        else:
            continue
        for label in labels:
            if not label:
                continue
            start = map_start
            while True:
                index = text.plain.find(label, start)
                if index == -1:
                    break
                text.stylize(style, index, index + len(label))
                start = index + len(label)


def _minimap_current_node_labels(
    view: DungeonView,
    node: Any,
    *,
    actions: Sequence[ScreenAction] | None = None,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                _minimap_node_label(view, node, actions=actions),
                _minimap_node_label(
                    view,
                    node,
                    include_number=False,
                    actions=actions,
                ),
            )
        )
    )


def _minimap_highlighted_node_labels(
    view: DungeonView,
    node: Any,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> tuple[str, ...]:
    numbered_label = _minimap_node_label(
        view,
        node,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )
    if _action_number_for_value(view, node.node_id, actions=actions):
        return (numbered_label,)
    return (
        _minimap_node_label(
            view,
            node,
            include_number=False,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        ),
    )


def _coordinate_minimap_lines(
    view: DungeonView,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    spatial_nodes = _spatial_map_nodes(view)
    if not spatial_nodes:
        return []
    current = next((node for node in spatial_nodes if node.current), None)
    if current is None:
        return _coordinate_full_map_lines(view)

    center_row = MAP_VIEWPORT_HEIGHT // 2
    center_col = (MAP_VIEWPORT_WIDTH - MAP_NODE_WIDTH) // 2
    canvas = [
        [" " for _ in range(MAP_VIEWPORT_WIDTH)] for _ in range(MAP_VIEWPORT_HEIGHT)
    ]
    nodes_by_id = {node.node_id: node for node in spatial_nodes}
    drawn_edges: set[tuple[str, str]] = set()
    anchor_x, anchor_y = _minimap_viewport_anchor(current, nodes_by_id)

    def position_for(node: Any) -> tuple[int, int]:
        return (
            center_row + round((node.map_y - anchor_y) * MAP_Y_STRIDE),
            center_col + round((node.map_x - anchor_x) * MAP_X_STRIDE),
        )

    for node in spatial_nodes:
        for exit_node_id in node.exit_node_ids:
            exit_node = nodes_by_id.get(exit_node_id)
            if exit_node is None or not _map_nodes_share_drawable_edge(node, exit_node):
                continue
            edge_key = _map_edge_key(node, exit_node)
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            from_row, from_col = position_for(node)
            to_row, to_col = position_for(exit_node)
            _draw_map_edge_between(canvas, from_row, from_col, to_row, to_col)

    overflow_top = overflow_bottom = overflow_left = overflow_right = False
    for node in spatial_nodes:
        row, col = position_for(node)
        overflow_top = overflow_top or row < 0
        overflow_bottom = overflow_bottom or row >= MAP_VIEWPORT_HEIGHT
        overflow_left = overflow_left or col < 0
        overflow_right = overflow_right or col + MAP_NODE_WIDTH > MAP_VIEWPORT_WIDTH
        _draw_map_label_at(
            canvas,
            view,
            node,
            row=row,
            col=col,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )

    _draw_map_overflow_markers(
        canvas,
        top=overflow_top,
        bottom=overflow_bottom,
        left=overflow_left,
        right=overflow_right,
    )
    return [line.rstrip() for line in ("".join(row) for row in canvas)]


def _coordinate_full_map_lines(view: DungeonView) -> list[str]:
    spatial_nodes = _spatial_map_nodes(view)
    if not spatial_nodes:
        return []

    min_x = min(node.map_x for node in spatial_nodes if node.map_x is not None)
    max_x = max(node.map_x for node in spatial_nodes if node.map_x is not None)
    min_y = min(node.map_y for node in spatial_nodes if node.map_y is not None)
    max_y = max(node.map_y for node in spatial_nodes if node.map_y is not None)
    width = (max_x - min_x) * MAP_X_STRIDE + MAP_NODE_WIDTH
    height = (max_y - min_y) * MAP_Y_STRIDE + 1
    canvas = [[" " for _ in range(width)] for _ in range(height)]
    nodes_by_id = {node.node_id: node for node in spatial_nodes}
    drawn_edges: set[tuple[str, str]] = set()

    for node in spatial_nodes:
        for exit_node_id in node.exit_node_ids:
            exit_node = nodes_by_id.get(exit_node_id)
            if exit_node is None or not _map_nodes_share_drawable_edge(node, exit_node):
                continue
            edge_key = _map_edge_key(node, exit_node)
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            _draw_map_edge(canvas, node, exit_node, min_x=min_x, min_y=min_y)

    for node in spatial_nodes:
        _draw_map_label(canvas, view, node, min_x=min_x, min_y=min_y)

    return [line.rstrip() for line in ("".join(row) for row in canvas)]


def _minimap_viewport_anchor(
    current: Any,
    nodes_by_id: dict[str, Any],
) -> tuple[float, float]:
    focus_nodes = [current]
    for exit_node_id in current.exit_node_ids:
        exit_node = nodes_by_id.get(exit_node_id)
        if (
            exit_node is not None
            and exit_node.map_x is not None
            and exit_node.map_y is not None
        ):
            focus_nodes.append(exit_node)
    xs = [node.map_x for node in focus_nodes if node.map_x is not None]
    ys = [node.map_y for node in focus_nodes if node.map_y is not None]
    if not xs or not ys:
        return float(current.map_x or 0), float(current.map_y or 0)
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _map_nodes_share_drawable_edge(from_node: Any, to_node: Any) -> bool:
    if (
        from_node.map_x is None
        or from_node.map_y is None
        or to_node.map_x is None
        or to_node.map_y is None
    ):
        return False
    return not (
        from_node.map_x == to_node.map_x and from_node.map_y == to_node.map_y
    )


def _full_map_lines(view: DungeonView) -> list[str]:
    return _coordinate_full_map_lines(view) or _minimap_branch_lines(view)


def _spatial_map_nodes(view: DungeonView) -> list[Any]:
    return [
        node for node in view.map_nodes if node.map_x is not None and node.map_y is not None
    ]


def _draw_map_edge(
    canvas: list[list[str]],
    from_node: Any,
    to_node: Any,
    *,
    min_x: int,
    min_y: int,
) -> None:
    from_row, from_col = _map_canvas_position(from_node, min_x=min_x, min_y=min_y)
    to_row, to_col = _map_canvas_position(to_node, min_x=min_x, min_y=min_y)
    _draw_map_edge_between(canvas, from_row, from_col, to_row, to_col)


def _draw_map_edge_between(
    canvas: list[list[str]],
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
) -> None:
    from_center_col = from_col + MAP_NODE_CENTER
    to_center_col = to_col + MAP_NODE_CENTER
    if from_row == to_row:
        _draw_map_horizontal_connector(canvas, from_row, from_center_col, to_center_col)
        return
    if from_center_col == to_center_col:
        _draw_map_vertical_connector(canvas, from_center_col, from_row, to_row)
        return
    _draw_map_horizontal_connector(canvas, from_row, from_center_col, to_center_col)
    _draw_map_vertical_connector(canvas, to_center_col, from_row, to_row)


def _draw_map_horizontal_connector(
    canvas: list[list[str]],
    row: int,
    from_center_col: int,
    to_center_col: int,
) -> None:
    if from_center_col == to_center_col:
        return
    left_center = min(from_center_col, to_center_col)
    right_center = max(from_center_col, to_center_col)
    start = left_center - MAP_NODE_CENTER + MAP_NODE_WIDTH
    end = right_center - MAP_NODE_CENTER
    for col in range(start, end):
        _put_map_char(canvas, row, col, "-")


def _draw_map_vertical_connector(
    canvas: list[list[str]],
    center_col: int,
    from_row: int,
    to_row: int,
) -> None:
    start = min(from_row, to_row) + 1
    end = max(from_row, to_row)
    for row in range(start, end):
        _put_map_char(canvas, row, center_col, "|")


def _draw_map_label(
    canvas: list[list[str]],
    view: DungeonView,
    node: Any,
    *,
    min_x: int,
    min_y: int,
) -> None:
    row, col = _map_canvas_position(node, min_x=min_x, min_y=min_y)
    _draw_map_label_at(canvas, view, node, row=row, col=col)


def _draw_map_label_at(
    canvas: list[list[str]],
    view: DungeonView,
    node: Any,
    *,
    row: int,
    col: int,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> None:
    node_label = _minimap_node_label(
        view,
        node,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )
    label = f"{node_label:^{MAP_NODE_WIDTH}}"[:MAP_NODE_WIDTH]
    for offset, character in enumerate(label):
        _put_map_char(canvas, row, col + offset, character)


def _draw_map_overflow_markers(
    canvas: list[list[str]],
    *,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
) -> None:
    if not canvas:
        return
    middle_row = len(canvas) // 2
    if top:
        _put_map_char(canvas, 0, len(canvas[0]) - 1, "^")
    if bottom:
        _put_map_char(canvas, len(canvas) - 1, len(canvas[0]) - 1, "v")
    if left:
        _put_map_char(canvas, middle_row, 0, "<")
    if right:
        _put_map_char(canvas, middle_row, len(canvas[0]) - 1, ">")


def _map_edge_key(from_node: Any, to_node: Any) -> tuple[str, str]:
    return (
        min(from_node.node_id, to_node.node_id),
        max(from_node.node_id, to_node.node_id),
    )


def _map_canvas_position(node: Any, *, min_x: int, min_y: int) -> tuple[int, int]:
    return (
        (node.map_y - min_y) * MAP_Y_STRIDE,
        (node.map_x - min_x) * MAP_X_STRIDE,
    )


def _put_map_char(
    canvas: list[list[str]],
    row: int,
    col: int,
    character: str,
) -> None:
    if row < 0 or col < 0 or row >= len(canvas) or col >= len(canvas[row]):
        return
    existing = canvas[row][col]
    if existing in {"-", "|"} and character in {"-", "|"} and existing != character:
        canvas[row][col] = "+"
    else:
        canvas[row][col] = character


def _minimap_branch_lines(
    view: DungeonView,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    nodes_by_id = {node.node_id: node for node in view.map_nodes}
    current = next((node for node in view.map_nodes if node.current), None)
    if current is None:
        return ["No explored map yet."]

    exits = [
        nodes_by_id[node_id]
        for node_id in current.exit_node_ids
        if node_id in nodes_by_id
    ]
    north, west, east, south, extra = _compass_exits(exits)
    current_label = _minimap_node_label(
        view,
        current,
        include_number=False,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )

    lines: list[str] = []
    if north is not None:
        north_label = _minimap_node_label(
            view,
            north,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        lines.extend(
            (
                f"        {north_label}",
                "         |",
            )
        )

    left_label = (
        _minimap_node_label(
            view,
            west,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        if west is not None
        else ""
    )
    right_label = (
        _minimap_node_label(
            view,
            east,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        if east is not None
        else ""
    )
    if west is not None or east is not None:
        lines.append(f"{left_label:<3} --  {current_label}  --  {right_label}")
    else:
        lines.append(f"         {current_label}")

    if south is not None:
        south_label = _minimap_node_label(
            view,
            south,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        lines.extend(
            (
                "         |",
                f"        {south_label}",
            )
        )

    visited_neighbors = [
        exit_node
        for exit_node in exits
        if exit_node.visited and not exit_node.current
    ]
    for neighbor in visited_neighbors:
        neighbor_exits = [
            nodes_by_id[node_id]
            for node_id in neighbor.exit_node_ids
            if node_id in nodes_by_id and node_id != current.node_id
        ]
        if neighbor_exits:
            lines.extend(
                (
                    "",
                    *_visited_branch_lines(
                        view,
                        neighbor,
                        neighbor_exits,
                        actions=actions,
                        highlighted_node_id=highlighted_node_id,
                    ),
                )
            )

    if extra:
        lines.extend(("", "More paths"))
        for node in extra:
            label = _minimap_node_label(
                view,
                node,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
            lines.append(f"  {label}")
    return lines


def _compass_exits(
    nodes: list[Any],
) -> tuple[Any | None, Any | None, Any | None, Any | None, list[Any]]:
    slots: list[Any | None] = [None, None, None, None]
    extra: list[Any] = []
    for index, node in enumerate(nodes):
        if index < len(slots):
            slots[index] = node
        else:
            extra.append(node)
    return slots[0], slots[1], slots[2], slots[3], extra


def _visited_branch_lines(
    view: DungeonView,
    node: Any,
    exits: list[Any],
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    node_label = _minimap_node_label(
        view,
        node,
        include_number=False,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )
    lines = [node_label]
    for index, exit_node in enumerate(exits[:4]):
        branch = "`-" if index == min(len(exits), 4) - 1 else "|-"
        exit_label = _minimap_node_label(
            view,
            exit_node,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        lines.append(f"  {branch} {exit_label}")
    if len(exits) > 4:
        lines.append(f"  `- +{len(exits) - 4} more")
    return lines


def _minimap_node_label(
    view: DungeonView,
    node: Any,
    *,
    include_number: bool = True,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> str:
    if node.current:
        marker = "@"
    elif node.visited:
        marker = "o"
    elif _is_quest_marker_node(node) and not node.cleared:
        marker = "!"
    else:
        marker = "?"
    number = (
        _action_number_for_value(view, node.node_id, actions=actions)
        if include_number
        else ""
    )
    focus = ">" if node.node_id == highlighted_node_id and number else ""
    return f"{focus}{number}{marker}"


def _is_quest_marker_node(node: Any) -> bool:
    return bool(getattr(node, "quest_marker", False))


def _map_status_label(node: Any) -> str:
    if node.current:
        return "Current"
    if node.cleared:
        return "Cleared"
    if node.visited:
        return "Visited"
    return "Unexplored"


def _short_map_name(name: str) -> str:
    replacements = {
        "Haven East Gate": "Gate",
        "Old Road": "Road",
        "Abandoned Toll Post": "Toll Post",
        "Hunter's Trail": "Trail",
        "Bramble Shrine": "Shrine",
        "Shallow Cave Entrance": "Cave Entrance",
        "Cave Mouth": "Mouth",
        "Forked Descent": "Fork",
        "Fungus-Lit Gallery": "Gallery",
        "Old Works Cache": "Cache",
        "Black Stone Sinkhole": "Sinkhole",
        "Dry Creek Bed": "Dry Creek",
        "Black Stone Gate": "Gate",
        "Maze-Touched Lair": "Lair",
    }
    return replacements.get(name, name)


def _actor_marker(
    actor_id: str,
    *,
    acting: bool,
    focused_actor_id: str,
    legal_actor_ids: set[str],
    source_actor_ids: set[str],
    statuses: Sequence[str],
) -> str:
    if actor_id == focused_actor_id:
        return "@"
    if actor_id in source_actor_ids:
        return "!"
    if acting:
        return "*"
    if "dead" in statuses:
        return "x"
    if actor_id in legal_actor_ids:
        return "."
    return " "


def _grid_cell(
    name: str,
    detail: str,
    state: str = "",
    *,
    marker: str,
    width: int = CELL_WIDTH,
    style: str = "",
) -> tuple[str, str, str, str]:
    return (
        f"{marker} {name}"[:width],
        detail[:width],
        state[:width],
        style,
    )


def _grid_text(
    cells: dict[str, tuple[str, str, str, str]],
    *,
    rows: tuple[tuple[str, str], ...] = GRID_ROWS,
    width: int = CELL_WIDTH,
    show_labels: bool = True,
) -> str:
    border = "+" + "+".join("-" * (width + 2) for _ in range(2)) + "+"
    lines = [border]
    for row in rows:
        row_cells = [
            cells.get(slot, _grid_cell("empty", slot, marker=" ", width=width))
            for slot in row
        ]
        if show_labels:
            labels = [format_formation_slot(slot) for slot in row]
            lines.append("|" + "|".join(f" {label:<{width}} " for label in labels) + "|")
        for line_index in range(3):
            values = (
                _styled_cell_value(row_cells[0][line_index], row_cells[0][3], width),
                _styled_cell_value(row_cells[1][line_index], row_cells[1][3], width),
            )
            lines.append(
                "|"
                + "|".join(f" {value} " for value in values)
                + "|"
            )
        lines.append(border)
    return "\n".join(lines)


def _styled_cell_value(value: str, style: str, width: int = CELL_WIDTH) -> str:
    visible = value[:width]
    padding = " " * max(0, width - len(visible))
    if not style:
        return f"{visible}{padding}"
    return f"[{style}]{visible}[/]{padding}"


def _combat_cell_style(
    intent: str,
    *,
    focused: bool,
    highlighted: bool,
    source: bool,
    acting: bool = False,
) -> str:
    if source:
        return "bold magenta"
    if focused:
        return _intent_style(intent)
    if acting:
        return "bold bright_cyan"
    if highlighted:
        return _soft_intent_style(intent)
    return ""


def _intent_style(intent: str) -> str:
    if intent == "attack":
        return "bold black on yellow"
    if intent == "debuff":
        return "bold black on yellow"
    if intent == "heal":
        return "bold white on green"
    return ""


def _soft_intent_style(intent: str) -> str:
    if intent == "attack":
        return "bright_yellow"
    if intent == "debuff":
        return "yellow"
    if intent == "heal":
        return "bright_green"
    return ""


def _intent_label(intent: str) -> str:
    if intent == "attack":
        return "TARGET"
    if intent == "debuff":
        return "CTRL"
    if intent == "heal":
        return "HEAL"
    return ""


def _combat_status_line(
    statuses: Sequence[str],
    mortal_wounds: int,
    intent: str,
    *,
    focused: bool,
    highlighted: bool,
    source: bool,
) -> str:
    intent_label = _intent_label(intent)
    if source:
        return "ATTACKING"
    if focused and intent_label:
        return f"{intent_label}"
    if highlighted and intent_label:
        return intent_label
    active_statuses = [status for status in statuses if status != "ready"]
    active_statuses.append(mortal_wound_badge(mortal_wounds, markup_safe=True))
    return ", ".join(active_statuses)


def _combat_cell_figure(
    actor: Any,
    *,
    focused: bool,
    acting: bool,
    idle_frame: int = 0,
) -> str:
    sprite = _actor_grid_glyph(actor)
    if focused:
        return f"> {sprite} <"
    if acting:
        return f"{sprite} turn"
    return sprite


def _actor_grid_glyph(actor: Any) -> str:
    statuses = set(actor.statuses)
    if "dead" in statuses:
        return "x"
    if "downed" in statuses:
        return "_"

    glyph = str(getattr(actor, "glyph", "") or "").strip()
    if glyph:
        return glyph[0]

    class_id = str(getattr(actor, "class_id", "") or "").lower()
    maze_glyphs = {
        "glass_splinter": "<",
        "pattern_ward": "#",
        "breach_stalker": "S",
        "maze_leech": "~",
        "maze_acolyte": "@",
        "cave_maw_brute": "M",
    }
    if class_id in maze_glyphs:
        return maze_glyphs[class_id]

    actor_id = actor.actor_id.lower()
    name = actor.name.lower()
    if actor.team == "hero":
        if "watchman" in actor_id:
            return "#"
        if "cutpurse" in actor_id:
            return "o"
        if "field_surgeon" in actor_id:
            return "+"
        if "scribe" in actor_id:
            return "*"
        return "@"

    if "bone" in actor_id or "bone" in name:
        return "0"
    if "skulker" in actor_id or "skulker" in name:
        return "^"
    if "brute" in actor_id or "maw" in actor_id or "brute" in name or "maw" in name:
        return "M"
    if "leech" in actor_id or "leech" in name:
        return "~"
    if "acolyte" in actor_id or "acolyte" in name:
        return "@"
    if "maze" in actor_id or "maze" in name:
        return "m"
    return "!"


def _actor_sprite(actor: Any) -> str:
    statuses = set(actor.statuses)
    if "dead" in statuses:
        return "<x>"
    if "downed" in statuses:
        return "<_>"

    glyph = str(getattr(actor, "glyph", "") or "").strip()
    if glyph:
        return f"<{glyph[0]}>"

    class_id = str(getattr(actor, "class_id", "") or "").lower()
    maze_glyphs = {
        "glass_splinter": "<",
        "pattern_ward": "#",
        "breach_stalker": "S",
        "maze_leech": "~",
        "maze_acolyte": "@",
        "cave_maw_brute": "M",
    }
    if class_id in maze_glyphs:
        return f"<{maze_glyphs[class_id]}>"

    actor_id = actor.actor_id.lower()
    name = actor.name.lower()
    if actor.team == "hero":
        if "watchman" in actor_id:
            return "<#>"
        if "cutpurse" in actor_id:
            return "<o>"
        if "field_surgeon" in actor_id:
            return "<+>"
        if "scribe" in actor_id:
            return "<*>"
        return "<@>"

    if "bone" in actor_id or "bone" in name:
        return "<0>"
    if "skulker" in actor_id or "skulker" in name:
        return "<^>"
    if "brute" in actor_id or "maw" in actor_id or "brute" in name or "maw" in name:
        return "<M>"
    if "leech" in actor_id or "leech" in name:
        return "<~>"
    if "acolyte" in actor_id or "acolyte" in name:
        return "<@>"
    if "maze" in actor_id or "maze" in name:
        return "<m>"
    return "<!>"


def _turn_order_rail(view: CombatView) -> str:
    if not view.turn_order:
        return "Turns  unknown"
    chips = [_turn_order_chip(entry) for entry in view.turn_order[:8]]
    if len(view.turn_order) > 8:
        chips.append("[dim]...[/]")
    return f"Turns  {' > '.join(chips)}"


def _turn_order_chip(entry: Any) -> str:
    label = _short_order_name(entry.name, team=entry.team)
    statuses = set(entry.statuses)
    if "dead" in statuses:
        return f"[dim strike]{label}[/]"
    if "downed" in statuses:
        return f"[bold white on dark_magenta]{label}[/]"
    if entry.active:
        if entry.team == "hero":
            return f"[bold black on cyan]{label}[/]"
        return f"[bold white on red]{label}[/]"
    if entry.acted:
        return f"[dim]{label}[/]"
    return f"[cyan]{label}[/]" if entry.team == "hero" else f"[red]{label}[/]"


def _short_order_name(name: str, *, team: str = "") -> str:
    parts = name.replace("-", " ").split()
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:10]
    if team == "enemy" and any(part.lower() == "maw" for part in parts):
        return "Maw"
    if team == "enemy":
        return parts[-1][:10]
    if parts[0].lower() in {"the", "a", "an"}:
        return parts[1][:10]
    return parts[0][:10]


def _turn_line(view: CombatView) -> str:
    if view.pending_enemy_intent is not None:
        intent = view.pending_enemy_intent
        return format_meta_line(
            "Enemy Intent",
            intent.enemy_name,
            intent.label,
            f"Threat {intent.threat_level}",
        )
    actor = view.current_actor
    if actor is None:
        return "Turn: resolving"
    return format_meta_line(
        "Turn",
        actor.name,
        getattr(actor, "display_name", "") or _actor_grid_glyph(actor),
    )


def _pressure_lines(view: CombatView, events: Sequence[Any]) -> list[str]:
    names = {actor.actor_id: actor.name for actor in (*view.party, *view.enemies)}
    lines: list[str] = []
    current_source_id = ""
    for event in events:
        source_id = ""
        target_id = ""
        if hasattr(event, "skill_id"):
            source_id = str(getattr(event, "actor_id", ""))
            target_id = str(getattr(event, "target_id", "") or "")
            current_source_id = source_id
        elif hasattr(event, "source_id") and hasattr(event, "target_id"):
            source_id = str(getattr(event, "source_id", ""))
            target_id = str(getattr(event, "target_id", ""))
            current_source_id = source_id
        elif hasattr(event, "target_id"):
            source_id = current_source_id
            target_id = str(getattr(event, "target_id", ""))
        elif hasattr(event, "actor_id") and current_source_id:
            source_id = current_source_id
            target_id = str(getattr(event, "actor_id", ""))
        if not source_id or not target_id:
            continue
        line = f"{names.get(source_id, source_id)} -> {names.get(target_id, target_id)}"
        if line not in lines:
            lines.append(line)
    return lines


def _names_for_actor_ids(view: CombatView, actor_ids: set[str]) -> str:
    if not actor_ids:
        return "none"
    names = {actor.actor_id: actor.name for actor in (*view.party, *view.enemies)}
    return ", ".join(names.get(actor_id, actor_id) for actor_id in sorted(actor_ids))


def _focused_actor_id(action: ScreenAction | None) -> str:
    return action.value if action is not None else ""


def _selected_skill_name(view: CombatView) -> str:
    if view.selected_skill_id is None:
        return "selected skill"
    for skill in view.skills:
        if skill.skill_id == view.selected_skill_id:
            return skill.name
    return view.selected_skill_id
