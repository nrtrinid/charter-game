"""Animation and portrait art helpers for TUI widgets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from game.ui.tui_widgets.constants import (
    ACTION_ANIMATION_LAST_FRAME,
    ANTICIPATION_FRAME,
    BEAT_ART_SIDE_PADDING,
    DAMAGE_REACTION_FRAME,
    IDLE_ANIMATION_CYCLE,
    SOURCE_ACTION_FRAME,
)
from game.ui.tui_widgets.events import (
    _combat_effect_emphasis,
    _combat_effect_label,
    _combat_effect_style,
    _event_kind,
    _is_combat_effect_event,
    _is_effort_change_event,
)


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
        str(getattr(actor, attribute, "")).lower() for attribute in ("actor_id", "name", "class_id")
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
