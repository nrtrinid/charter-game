"""App-facing view models for terminal rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from game.app.actions import (
    ActionProvider,
    ScreenAction,
    ScreenActionKind,
    ScreenActionRisk,
    join_detail,
)
from game.app.manual_combat import (
    ManualCombatSession,
    can_delay_hero,
    heal_amount_range_for_skill,
    legal_move_slots,
    legal_reaction_options,
    legal_skill_ids,
    legal_target_ids,
    skill_target_ids,
    skill_unavailable_reason,
    visible_skill_ids,
)
from game.app.views.art import (
    _art_display_name,
    _art_frame_holds,
    _art_frame_impacts,
    _art_frames,
    _art_glyph,
    _art_lines,
    _art_mini_frames,
    _art_mini_lines,
    _combatant_art_asset,
    _derive_mini_lines,
    _hero_art_asset,
)
from game.app.views.hero import _trait_label
from game.app.views.shared import (
    _life_state_labels,
    _skill_description,
    _skill_intent,
    _slot_display,
)
from game.campaign.company import (
    HeroState,
)
from game.campaign.gear import effective_hero_stats
from game.combat.combat_state import Combatant, Team
from game.combat.damage_range import format_damage_label
from game.combat.formation import (
    FormationSlot,
)
from game.combat.preview import preview_attack
from game.combat.targeting import skill_position_label
from game.content.definitions import GameDefinitions
from game.core.events import GameEvent


@dataclass(frozen=True)
class CombatActorView:
    actor_id: str
    name: str
    team: str
    slot: str
    hp: int
    max_hp: int
    effort: int
    max_effort: int
    mortal_wounds: int
    morale: str
    strain: str
    tags: tuple[str, ...]
    life_state: str
    statuses: tuple[str, ...] = ()
    personal_quirk: str = ""
    quirks: tuple[str, ...] = ()
    strain_marks: tuple[str, ...] = ()
    acting: bool = False
    class_id: str = ""
    display_name: str = ""
    glyph: str = ""
    mini_lines: tuple[str, ...] = ()
    mini_frames: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    art_lines: tuple[str, ...] = ()
    art_frames: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    art_frame_holds: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    art_frame_impacts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.statuses:
            object.__setattr__(self, "statuses", _life_state_labels(self.life_state))

    @property
    def fatigue(self) -> str:
        return self.strain

    @property
    def conditions(self) -> tuple[str, ...]:
        return self.strain_marks

@dataclass(frozen=True)
class CombatSkillOption:
    action: ScreenAction
    skill_id: str
    name: str
    effort_cost: int
    attack_type: str
    usable_from: str
    usable_from_label: str
    flavor_text: str
    effect_text: str
    unavailable_reason: str
    intent: str
    damage_estimate: int
    damage_label: str
    target_count: int

@dataclass(frozen=True)
class CombatTargetOption:
    action: ScreenAction
    target_id: str
    name: str
    slot: str
    hp: int
    max_hp: int
    life_state: str
    hit_chance: int
    damage_estimate: int
    damage_label: str
    legality_reason: str
    intent: str

    @property
    def statuses(self) -> tuple[str, ...]:
        return _life_state_labels(self.life_state)

@dataclass(frozen=True)
class CombatMoveOption:
    action: ScreenAction
    from_slot: str
    to_slot: str
    actor_name: str
    occupant_name: str
    description: str
    before_formation: tuple[tuple[str, str], ...] = ()
    after_formation: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True)
class CombatEnemyIntentView:
    enemy_id: str
    enemy_name: str
    skill_id: str
    skill_name: str
    label: str
    target_id: str
    target_name: str
    threat_level: str
    obvious_effect: str
    debug_hit_chance: int | None = None
    debug_damage_estimate: int | None = None
    debug_damage_label: str = ""

@dataclass(frozen=True)
class CombatReactionOption:
    action: ScreenAction
    reaction_id: str | None
    kind: str
    actor_id: str | None
    actor_name: str
    cost: int
    summary: str

@dataclass(frozen=True)
class CombatTurnOrderEntry:
    actor_id: str
    name: str
    team: str
    life_state: str
    active: bool = False
    acted: bool = False

    @property
    def statuses(self) -> tuple[str, ...]:
        return _life_state_labels(self.life_state)

@dataclass(frozen=True)
class CombatView:
    encounter_id: str
    encounter_name: str
    round_number: int
    cohesion: str
    current_actor: CombatActorView | None
    selected_skill_id: str | None
    party: tuple[CombatActorView, ...]
    enemies: tuple[CombatActorView, ...]
    commands: tuple[ScreenAction, ...] = ()
    skills: tuple[CombatSkillOption, ...] = ()
    targets: tuple[CombatTargetOption, ...] = ()
    moves: tuple[CombatMoveOption, ...] = ()
    pending_enemy_intent: CombatEnemyIntentView | None = None
    reaction_options: tuple[CombatReactionOption, ...] = ()
    recent_events: tuple[GameEvent, ...] = ()
    turn_order: tuple[CombatTurnOrderEntry, ...] = ()

def build_combat_view(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    *,
    retreat_available: bool = False,
    debug_combat_preview: bool = False,
) -> CombatView:
    current_actor = session.pending_hero()
    selected_skill_id = session.selected_skill_id
    skill_ids = visible_skill_ids(session, definitions)
    legal_skill_id_set = set(legal_skill_ids(session, definitions))
    move_slots = legal_move_slots(session)
    commands = ActionProvider.combat_commands(
        has_skills=bool(skill_ids),
        has_usable_skills=bool(legal_skill_id_set),
        has_moves=bool(move_slots),
        can_delay=can_delay_hero(session),
        can_act=current_actor is not None,
        retreat_available=retreat_available,
        actor_name=current_actor.name if current_actor is not None else "",
    )
    skills: list[CombatSkillOption] = []
    for index, skill_id in enumerate(skill_ids, start=1):
        skill = definitions.skills[skill_id]
        enabled = skill_id in legal_skill_id_set
        unavailable_reason = (
            ""
            if enabled
            else skill_unavailable_reason(
                session,
                definitions,
                skill_id,
            )
        )
        target_ids = (
            legal_target_ids(session, definitions, skill_id)
            if enabled
            else skill_target_ids(session, definitions, skill_id)
        )
        intent = _skill_intent(skill.tags)
        damage_estimate = 0
        damage_label = "0"
        if current_actor is not None and target_ids and enabled:
            if intent == "heal":
                heal_min, heal_max = heal_amount_range_for_skill(
                    session,
                    definitions,
                    skill_id,
                )
                damage_estimate = heal_max
                damage_label = f"{heal_min}-{heal_max}" if heal_min != heal_max else str(heal_max)
            else:
                preview = preview_attack(
                    session.state,
                    current_actor.actor_id,
                    skill,
                    target_ids[0],
                )
                damage_estimate = preview.damage
                damage_label = preview.damage_label
        skill_effect_description = _skill_description(
            intent,
            skill.attack_type.value,
            damage_label,
            len(target_ids),
        )
        skills.append(
            CombatSkillOption(
                action=ScreenAction(
                    number=str(index),
                    label=skill.name,
                    value=skill.id,
                    aliases=(skill.id,),
                    enabled=enabled,
                    default=enabled and not any(option.action.enabled for option in skills),
                    description=skill_effect_description,
                    kind=ScreenActionKind.COMBAT,
                    risk=ScreenActionRisk.COSTLY if skill.effort_cost else ScreenActionRisk.LOW,
                    cost=f"{skill.effort_cost} Effort" if skill.effort_cost else "",
                    unavailable_reason=unavailable_reason,
                    preview=join_detail(
                        (
                            f"{current_actor.name if current_actor is not None else 'Actor'} "
                            f"prepares {skill.name}."
                        ),
                        f"Usable: {skill_position_label(skill)}.",
                        skill.description,
                        skill_effect_description,
                    ),
                    result_hint=join_detail(
                        "Choose a legal target next." if enabled else unavailable_reason,
                        skill_effect_description,
                    ),
                ),
                skill_id=skill.id,
                name=skill.name,
                effort_cost=skill.effort_cost,
                attack_type=skill.attack_type.value,
                usable_from=skill.usable_from.value,
                usable_from_label=skill_position_label(skill, compact=True),
                flavor_text=skill.description,
                effect_text=skill.effect_text or skill_effect_description,
                unavailable_reason=unavailable_reason,
                intent=intent,
                damage_estimate=damage_estimate,
                damage_label=damage_label,
                target_count=len(target_ids),
            )
        )

    targets: list[CombatTargetOption] = []
    if current_actor is not None and selected_skill_id is not None:
        target_ids = sorted(
            legal_target_ids(session, definitions, selected_skill_id),
            key=lambda target_id: _target_sort_key(session, target_id),
        )
        skill = definitions.skills[selected_skill_id]
        intent = _skill_intent(skill.tags)
        single_target = len(target_ids) == 1
        for index, target_id in enumerate(target_ids, start=1):
            target = session.state.actor(target_id)
            if intent == "heal":
                hit_chance = 100
                heal_min, heal_max = heal_amount_range_for_skill(
                    session,
                    definitions,
                    selected_skill_id,
                    target_id,
                )
                amount = heal_max
                amount_label = f"{heal_min}-{heal_max}" if heal_min != heal_max else str(heal_max)
                legality_reason = "living ally"
                effect_label = format_damage_label(heal_min, heal_max, "heal")
            else:
                preview = preview_attack(session.state, current_actor.actor_id, skill, target_id)
                hit_chance = preview.hit_chance
                amount = preview.damage
                amount_label = preview.damage_label
                legality_reason = preview.legality_reason
                effect_label = format_damage_label(preview.damage_min, preview.damage_max)
            slot = session.state.formation_for(target.team).slot_of(target_id)
            targets.append(
                CombatTargetOption(
                    action=ScreenAction(
                        number=str(index),
                        label=target.name,
                        value=target_id,
                        aliases=(target_id,),
                        default=index == 1 or single_target,
                        description=f"{hit_chance}% hit, {effect_label}",
                        kind=ScreenActionKind.COMBAT,
                        risk=ScreenActionRisk.LOW,
                        preview=join_detail(
                            f"{target.name}: HP {target.hp}/{target.max_hp}",
                            f"Legal: {legality_reason}",
                            f"Projected: {hit_chance}% hit, {effect_label}",
                        ),
                        result_hint=(
                            f"Enter commits {skill.name} on {target.name}: "
                            f"{hit_chance}% hit, {effect_label}."
                        ),
                    ),
                    target_id=target_id,
                    name=target.name,
                    slot=slot.value if slot is not None else "-",
                    hp=target.hp,
                    max_hp=target.max_hp,
                    life_state=target.life_state.value,
                    hit_chance=hit_chance,
                    damage_estimate=amount,
                    damage_label=amount_label,
                    legality_reason=legality_reason,
                    intent=intent,
                )
            )

    moves: list[CombatMoveOption] = []
    from_slot = "-"
    if current_actor is not None:
        actor_slot = session.state.party_formation.slot_of(current_actor.actor_id)
        if actor_slot is not None:
            from_slot = actor_slot.value
    for index, to_slot in enumerate(move_slots, start=1):
        occupant_id = session.state.party_formation.actor_at(to_slot)
        occupant = session.state.actor(occupant_id) if occupant_id is not None else None
        actor_name = current_actor.name if current_actor is not None else "Actor"
        occupant_name = occupant.name if occupant is not None else "empty"
        moving_to = _slot_display(to_slot.value)
        moving_from = _slot_display(from_slot)
        if occupant is None:
            label = f"{moving_to} — open slot"
            description = f"{actor_name} shifts from {moving_from} to {moving_to}."
            result_hint = "Turn ends and protection changes immediately."
        else:
            label = f"{moving_to} — swap with {occupant_name}"
            description = (
                f"{actor_name} at {moving_from} swaps with {occupant_name} at {moving_to}."
            )
            result_hint = "Turn ends and protection changes immediately."
        before_formation = _formation_preview_slots(session.state.party_formation, session)
        after_formation = _move_preview_slots(
            before_formation,
            from_slot=from_slot,
            to_slot=to_slot.value,
        )
        moves.append(
            CombatMoveOption(
                action=ScreenAction(
                    number=str(index),
                    label=label,
                    value=to_slot.value,
                    aliases=(to_slot.value.lower(), to_slot.value.replace("_", " ").lower()),
                    default=index == 1,
                    description=f"{from_slot} -> {to_slot.value}",
                    kind=ScreenActionKind.COMBAT,
                    risk=ScreenActionRisk.LOW,
                    preview=join_detail(
                        description,
                        "Before -> after preview available in focus detail.",
                    ),
                    result_hint=result_hint,
                ),
                from_slot=from_slot,
                to_slot=to_slot.value,
                actor_name=actor_name,
                occupant_name=occupant_name,
                description=description,
                before_formation=before_formation,
                after_formation=after_formation,
            )
        )

    return CombatView(
        encounter_id=session.encounter_id,
        encounter_name=session.encounter_name,
        round_number=session.state.round_number,
        cohesion=session.state.derive_cohesion().name.title(),
        current_actor=_combatant_view(session, definitions, current_actor, acting=True)
        if current_actor is not None
        else None,
        selected_skill_id=selected_skill_id,
        party=tuple(
            _combatant_view(session, definitions, combatant, acting=current_actor == combatant)
            for combatant in _slot_ordered_combatants(session, Team.HERO)
        ),
        enemies=tuple(
            _combatant_view(session, definitions, combatant, acting=current_actor == combatant)
            for combatant in _slot_ordered_combatants(session, Team.ENEMY)
        ),
        commands=commands,
        skills=tuple(skills),
        targets=tuple(targets),
        moves=tuple(moves),
        pending_enemy_intent=_enemy_intent_view(
            session,
            debug_combat_preview=debug_combat_preview,
        ),
        reaction_options=_reaction_options(session),
        recent_events=tuple(session.recent_events[-8:]),
        turn_order=_turn_order_entries(session),
    )

def _slot_ordered_combatants(
    session: ManualCombatSession,
    team: Team,
) -> tuple[Combatant, ...]:
    formation = session.state.formation_for(team)
    side = session.state.side_for(team)
    ordered: list[Combatant] = []
    for slot in FormationSlot:
        actor_id = formation.actor_at(slot)
        if actor_id is not None and actor_id in side:
            ordered.append(side[actor_id])
    for combatant in side.values():
        if combatant not in ordered:
            ordered.append(combatant)
    return tuple(ordered)

def _formation_preview_slots(
    formation: Any,
    session: ManualCombatSession,
) -> tuple[tuple[str, str], ...]:
    slots: list[tuple[str, str]] = []
    for slot in FormationSlot:
        actor_id = formation.actor_at(slot)
        actor_name = session.state.actor(actor_id).name if actor_id is not None else "empty"
        slots.append((slot.value, actor_name))
    return tuple(slots)

def _move_preview_slots(
    slots: tuple[tuple[str, str], ...],
    *,
    from_slot: str,
    to_slot: str,
) -> tuple[tuple[str, str], ...]:
    names_by_slot = dict(slots)
    from_name = names_by_slot.get(from_slot, "empty")
    to_name = names_by_slot.get(to_slot, "empty")
    names_by_slot[from_slot] = to_name
    names_by_slot[to_slot] = from_name
    return tuple((slot, names_by_slot.get(slot, "empty")) for slot, _name in slots)

def _enemy_intent_view(
    session: ManualCombatSession,
    *,
    debug_combat_preview: bool,
) -> CombatEnemyIntentView | None:
    intent = session.pending_enemy_intent
    if intent is None:
        return None
    return CombatEnemyIntentView(
        enemy_id=intent.enemy_id,
        enemy_name=intent.enemy_name,
        skill_id=intent.skill_id,
        skill_name=intent.skill_name,
        label=intent.label,
        target_id=intent.target_id,
        target_name=intent.target_name,
        threat_level=intent.threat_level,
        obvious_effect=intent.obvious_effect,
        debug_hit_chance=intent.hit_chance if debug_combat_preview else None,
        debug_damage_estimate=intent.damage_estimate if debug_combat_preview else None,
        debug_damage_label=intent.damage_label if debug_combat_preview else "",
    )

def _turn_order_entries(session: ManualCombatSession) -> tuple[CombatTurnOrderEntry, ...]:
    entries: list[CombatTurnOrderEntry] = []
    for index, initiative_entry in enumerate(session.initiative):
        actor = session.state.actor(initiative_entry.actor_id)
        if actor is None:
            continue
        entries.append(
            CombatTurnOrderEntry(
                actor_id=actor.actor_id,
                name=actor.name,
                team=actor.team.value,
                life_state=actor.life_state.value,
                active=not session.ended and index == session.turn_index,
                acted=index < session.turn_index,
            )
        )
    return tuple(entries)

def _reaction_options(session: ManualCombatSession) -> tuple[CombatReactionOption, ...]:
    if session.pending_enemy_intent is None:
        return ()
    options: list[CombatReactionOption] = []
    for index, reaction in enumerate(legal_reaction_options(session), start=1):
        options.append(
            CombatReactionOption(
                action=ScreenAction(
                    str(index),
                    _reaction_label(reaction.kind, reaction.actor_name),
                    reaction.reaction_id,
                    (reaction.kind, reaction.actor_name.lower().replace(" ", "_")),
                    description=reaction.summary,
                    kind=ScreenActionKind.COMBAT,
                    risk=ScreenActionRisk.COSTLY if reaction.cost else ScreenActionRisk.LOW,
                    cost=f"{reaction.cost} Effort" if reaction.cost else "",
                    preview=join_detail(
                        f"{reaction.actor_name} reacts.",
                        reaction.summary,
                    ),
                    result_hint="Spend the listed Effort to interrupt or soften the enemy action.",
                ),
                reaction_id=reaction.reaction_id,
                kind=reaction.kind,
                actor_id=reaction.actor_id,
                actor_name=reaction.actor_name,
                cost=reaction.cost,
                summary=reaction.summary,
            )
        )
    options.append(
        CombatReactionOption(
            action=ScreenAction(
                str(len(options) + 1),
                "Skip Reaction",
                "skip",
                ("s", "none", "wait"),
                default=not options,
                description="Let the enemy action resolve normally.",
                kind=ScreenActionKind.COMBAT,
                risk=ScreenActionRisk.RISKY,
                preview="Skip protection and let the enemy action resolve normally.",
                result_hint="The threatened target takes the full pending action if it lands.",
            ),
            reaction_id=None,
            kind="skip",
            actor_id=None,
            actor_name="",
            cost=0,
            summary="Let the enemy action resolve normally.",
        )
    )
    return tuple(options)

def _reaction_label(kind: str, actor_name: str) -> str:
    labels = {
        "watchman_intercede": "Keep Watch",
        "cutpurse_evade": "Slip Away",
        "field_surgeon_stabilize": "Field Dress",
        "scribe_disrupt": "Annotate",
    }
    return f"{labels.get(kind, kind.replace('_', ' ').title())}: {actor_name}"

def _target_sort_key(
    session: ManualCombatSession,
    target_id: str,
) -> tuple[int, str]:
    target = session.state.actor(target_id)
    slot = session.state.formation_for(target.team).slot_of(target_id)
    slot_priority = {
        FormationSlot.FRONT_LEFT: 0,
        FormationSlot.FRONT_RIGHT: 1,
        FormationSlot.BACK_LEFT: 2,
        FormationSlot.BACK_RIGHT: 3,
    }
    if slot is None:
        return 99, target.name
    return slot_priority.get(slot, 99), target.name

def _combatant_view(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    combatant: Combatant,
    *,
    acting: bool = False,
) -> CombatActorView:
    slot = session.state.formation_for(combatant.team).slot_of(combatant.actor_id)
    art_asset = _combatant_art_asset(definitions, combatant)
    return CombatActorView(
        actor_id=combatant.actor_id,
        name=combatant.name,
        team=combatant.team.value,
        slot=slot.value if slot is not None else "-",
        hp=combatant.hp,
        max_hp=combatant.max_hp,
        effort=combatant.effort,
        max_effort=combatant.max_effort,
        mortal_wounds=combatant.mortal_wounds,
        morale=combatant.morale.name.title(),
        strain=combatant.strain.name.title(),
        tags=tuple(sorted(tag.name.title() for tag in combatant.tags)),
        life_state=combatant.life_state.value,
        personal_quirk=_trait_label(combatant.personal_quirk),
        quirks=tuple(combatant.quirks),
        strain_marks=tuple(
            _trait_label(mark.value)
            for mark in sorted(combatant.strain_marks, key=lambda mark: mark.value)
        ),
        acting=acting,
        class_id=combatant.class_id,
        display_name=_art_display_name(art_asset),
        glyph=_art_glyph(art_asset),
        mini_lines=_art_mini_lines(art_asset),
        mini_frames=_art_mini_frames(art_asset),
        art_lines=_art_lines(art_asset),
        art_frames=_art_frames(art_asset),
        art_frame_holds=_art_frame_holds(art_asset),
        art_frame_impacts=_art_frame_impacts(art_asset),
    )


def build_hero_portrait_view(
    hero: HeroState,
    definitions: GameDefinitions,
    *,
    slot: str = "",
) -> CombatActorView:
    stats = effective_hero_stats(hero, definitions)
    art_asset = _hero_art_asset(definitions, hero.hero_id, hero.class_id)
    slot_label = slot or hero.formation_slot.value
    return CombatActorView(
        actor_id=hero.hero_id,
        name=hero.name,
        team="hero",
        slot=slot_label,
        hp=hero.hp,
        max_hp=stats.max_hp,
        effort=hero.effort,
        max_effort=stats.max_effort,
        mortal_wounds=hero.mortal_wounds,
        morale=hero.morale.name.title(),
        strain=hero.strain.name.title(),
        tags=(),
        life_state=hero.life_state.value,
        class_id=hero.class_id,
        display_name=_art_display_name(art_asset),
        glyph=_art_glyph(art_asset),
        mini_lines=_art_mini_lines(art_asset),
        mini_frames=_art_mini_frames(art_asset),
        art_lines=_art_lines(art_asset) or _derive_mini_lines(_art_mini_lines(art_asset)),
        art_frames=_art_frames(art_asset),
        art_frame_holds=_art_frame_holds(art_asset),
        art_frame_impacts=_art_frame_impacts(art_asset),
        strain_marks=tuple(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda mark: mark.value)
        ),
    )

