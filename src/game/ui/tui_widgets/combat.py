"""Combat panel and beat-rendering helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from textual.widgets import Static

from game.app.views import (
    CombatView,
    ScreenAction,
)
from game.ui.hci_text import (
    format_formation_slot,
)
from game.ui.tui_widgets.animation import (
    _action_end_frame,
    _action_impact_frame,
    _animation_art_lines,
    _animation_cues_for_events,
    _beat_callouts,
    _beat_hp_overrides,
    _beat_motion_offsets,
    _beat_pulse_styles,
    _beat_status_overrides,
    _compact_art_lines,
    _idle_pose_frame,
    _offset_art_lines,
    _portrait_animation_art_lines,
    _portrait_display_art_lines,
)
from game.ui.tui_widgets.constants import (
    ACTION_ANIMATION_LAST_FRAME,
    BEAT_CONNECTOR_WIDTH,
    BEAT_PORTRAIT_WIDTH,
    CELL_WIDTH,
    COMBAT_CELL_WIDTH,
    COMBAT_FIELD_LAYOUT,
    DAMAGE_REACTION_FRAME,
    ENEMY_COMBAT_ROWS,
    GRID_ROWS,
    IDLE_ANIMATION_CYCLE,
    MINI_ART_WIDTH,
    MINI_MIDDLE_WIDTH,
    MINI_SIDE_GAP,
    MINI_SLOT_WIDTH,
    MINI_VISIBLE_STATUS_TAGS,
    PARTY_COMBAT_ROWS,
    PORTRAIT_EFFECT_LANE_COUNT,
)
from game.ui.tui_widgets.events import (
    _combat_effect_emphasis,
    _combat_effect_label,
    _combat_effect_style,
    _event_kind,
    _is_combat_effect_event,
    _is_effort_change_event,
)
from game.ui.tui_widgets.formation import (
    _formation_preview_text,
    _slot_name,
    formation_slot_faces_inward,
)
from game.ui.tui_widgets.shell import format_meta_line
from game.ui.wounds import mortal_wound_badge


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
            _focused_actor_id(focused_action) if phase == "target" and not healing_targets else ""
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
                party_source_ids=(move_context[3] if move_context is not None else set()),
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
        (f"{left_line}{_centered_connector(connector, index == connector_row)}{right_line}")
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
        _combat_effect_duplicate_key(event) for event in events if _is_combat_effect_event(event)
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
        _event_sentence(event) for event in events if _event_kind(event) in _footer_event_kinds()
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
    typed_replacements = {_typed_effect_status_replacement(event) for event in typed_effects}
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
            turn_flash = actor.actor_id == turn_flash_actor_id and turn_flash_frame == 0
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
            cells.get(slot, _grid_cell("empty", slot, marker=" ", width=width)) for slot in row
        ]
        if show_labels:
            labels = [format_formation_slot(slot) for slot in row]
            lines.append("|" + "|".join(f" {label:<{width}} " for label in labels) + "|")
        for line_index in range(3):
            values = (
                _styled_cell_value(row_cells[0][line_index], row_cells[0][3], width),
                _styled_cell_value(row_cells[1][line_index], row_cells[1][3], width),
            )
            lines.append("|" + "|".join(f" {value} " for value in values) + "|")
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
