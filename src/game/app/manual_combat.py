"""Manual combat-lite session helpers for app/UI coordination."""

from __future__ import annotations

from dataclasses import dataclass, field

from game.combat.actions import SkillResolutionModifiers, use_skill
from game.combat.combat_state import (
    Combatant,
    CombatState,
    StrainMark,
    Tag,
    Team,
    tick_marked_turn,
)
from game.combat.damage import heal_combatant
from game.combat.damage_range import (
    combatant_damage_range,
    roll_skill_base_damage,
    skill_base_damage_max,
)
from game.combat.effects import effort_delta_event
from game.combat.enemy_actions import (
    enemy_has_choice_after_swap,
    enemy_proactive_move,
    enemy_recovery_events,
    enemy_recovery_move,
    enemy_recovery_slots,
    enemy_wait_reason,
)
from game.combat.enemy_decision import (
    EnemyDecisionRuntimeContext,
    choose_enemy_skill_and_target,
    production_enemy_decision_policy,
    production_enemy_movement_mode,
    production_enemy_wait_mode,
)
from game.combat.formation import FormationSlot, are_adjacent
from game.combat.morale import raise_morale
from game.combat.preview import preview_attack
from game.combat.targeting import legal_targets, skill_position_unavailable_reason
from game.combat.traits import (
    PERSONAL_GENTLE_HANDS,
    QUIRK_FIELD_MEDIC,
    QUIRK_STEADY_VOICE,
    has_strain_mark,
)
from game.combat.turn_order import InitiativeEntry, roll_initiative
from game.content.definitions import GameDefinitions
from game.core.events import (
    CombatEndedEvent,
    CombatRetreatDeclaredEvent,
    EncounterEndedEvent,
    EncounterStartedEvent,
    EnemyIntentEvent,
    GameEvent,
    HealingEvent,
    MemorySignalEvent,
    MoveEvent,
    ReactionSkippedEvent,
    ReactionUsedEvent,
    RoundEndedEvent,
    RoundStartedEvent,
    SkillUsedEvent,
    StatusChangedEvent,
    TurnDelayedEvent,
    TurnPassedEvent,
)
from game.core.rng import GameRng
from game.data.schemas import SkillDefinition


@dataclass(frozen=True)
class EnemyIntent:
    enemy_id: str
    enemy_name: str
    skill_id: str
    skill_name: str
    label: str
    target_id: str
    target_name: str
    threat_level: str
    obvious_effect: str
    hit_chance: int
    damage_estimate: int
    damage_label: str


@dataclass(frozen=True)
class CombatReaction:
    reaction_id: str
    kind: str
    actor_id: str
    actor_name: str
    cost: int
    summary: str


@dataclass
class ManualCombatSession:
    encounter_id: str
    encounter_name: str
    state: CombatState
    initiative: list[InitiativeEntry] = field(default_factory=list)
    turn_index: int = 0
    selected_skill_id: str | None = None
    selected_target_id: str | None = None
    pending_enemy_intent: EnemyIntent | None = None
    delayed_actor_ids: set[str] = field(default_factory=set)
    retreat_pending: bool = False
    retreat_actor_id: str | None = None
    outcome: str | None = None
    recent_events: list[GameEvent] = field(default_factory=list)
    event_log: list[GameEvent] = field(default_factory=list)
    ended: bool = False
    enemy_ai_mode: str = "learned_static"
    enemy_wait_mode: str = "package_only"
    enemy_movement_mode: str = "package_only"

    def current_actor(self) -> Combatant | None:
        if self.ended or self.turn_index >= len(self.initiative):
            return None
        return self.state.actor(self.initiative[self.turn_index].actor_id)

    def pending_hero(self) -> Combatant | None:
        if self.pending_enemy_intent is not None:
            return None
        actor = self.current_actor()
        if actor is None or actor.team != Team.HERO or not actor.can_act():
            return None
        return actor


def start_manual_session(
    encounter_id: str,
    encounter_name: str,
    state: CombatState,
    definitions: GameDefinitions,
    rng: GameRng,
    enemy_ai_mode: str = "learned_static",
) -> tuple[ManualCombatSession, list[GameEvent]]:
    session = ManualCombatSession(
        encounter_id=encounter_id,
        encounter_name=encounter_name,
        state=state,
        enemy_ai_mode=enemy_ai_mode,
        enemy_wait_mode=production_enemy_wait_mode(enemy_ai_mode),
        enemy_movement_mode=production_enemy_movement_mode(enemy_ai_mode),
    )
    events: list[GameEvent] = [
        EncounterStartedEvent(
            message=f"{encounter_name} begins.",
            encounter_id=encounter_id,
            encounter_name=encounter_name,
            actor_ids=sorted(state.all_combatants()),
        )
    ]
    events.extend(_start_round(session, rng))
    events.extend(auto_advance_to_hero(session, definitions, rng))
    session.recent_events = events[-8:]
    session.event_log = list(events)
    return session, events


def legal_skill_ids(session: ManualCombatSession, definitions: GameDefinitions) -> list[str]:
    actor = session.pending_hero()
    if actor is None:
        return []
    skill_ids: list[str] = []
    for skill_id in visible_skill_ids(session, definitions):
        if skill_unavailable_reason(session, definitions, skill_id):
            continue
        skill_ids.append(skill_id)
    return skill_ids


def visible_skill_ids(session: ManualCombatSession, definitions: GameDefinitions) -> list[str]:
    actor = session.pending_hero()
    if actor is None:
        return []
    return [
        skill_id
        for skill_id in actor.skills
        if actor.effort >= definitions.skills[skill_id].effort_cost
    ]


def skill_unavailable_reason(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    skill_id: str,
) -> str:
    actor = session.pending_hero()
    if actor is None or skill_id not in actor.skills:
        return "No hero can use this skill right now."
    skill = definitions.skills[skill_id]
    if actor.effort < skill.effort_cost:
        return f"Needs {skill.effort_cost} Effort."
    position_reason = skill_position_unavailable_reason(session.state, actor.actor_id, skill)
    if position_reason:
        return position_reason
    if not skill_target_ids(session, definitions, skill_id):
        return "No legal targets."
    return ""


def skill_target_ids(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    skill_id: str,
) -> list[str]:
    actor = session.pending_hero()
    if actor is None or skill_id not in actor.skills:
        return []
    skill = definitions.skills[skill_id]
    if _is_support_skill(skill):
        return _legal_ally_targets(session.state)
    return sorted(legal_targets(session.state, actor.actor_id, skill.attack_type))


def legal_target_ids(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    skill_id: str,
) -> list[str]:
    actor = session.pending_hero()
    if actor is None or skill_id not in legal_skill_ids(session, definitions):
        return []
    return skill_target_ids(session, definitions, skill_id)


def legal_move_slots(session: ManualCombatSession) -> list[FormationSlot]:
    actor = session.pending_hero()
    if actor is None:
        return []
    if Tag.FROZEN in actor.tags or has_strain_mark(actor, StrainMark.WINDED):
        return []
    actor_slot = session.state.party_formation.slot_of(actor.actor_id)
    if actor_slot is None:
        return []
    return [
        slot
        for slot in FormationSlot
        if slot != actor_slot and are_adjacent(actor_slot, slot)
    ]


def can_delay_hero(session: ManualCombatSession) -> bool:
    actor = session.pending_hero()
    if actor is None or actor.actor_id in session.delayed_actor_ids:
        return False
    for entry in session.initiative[session.turn_index + 1 :]:
        if session.state.actor(entry.actor_id).can_act():
            return True
    return False


def legal_reaction_options(session: ManualCombatSession) -> list[CombatReaction]:
    intent = session.pending_enemy_intent
    if intent is None:
        return []
    target = session.state.actor(intent.target_id)
    options: list[CombatReaction] = []
    for hero in sorted(session.state.heroes.values(), key=lambda candidate: candidate.name):
        if not _can_react(hero):
            continue
        class_id = hero.class_id.lower()
        if class_id == "watchman" and target.team == Team.HERO and hero.actor_id != target.actor_id:
            options.append(
                CombatReaction(
                    reaction_id=f"watchman_intercede:{hero.actor_id}",
                    kind="watchman_intercede",
                    actor_id=hero.actor_id,
                    actor_name=hero.name,
                    cost=1,
                    summary=f"{hero.name} takes the incoming action instead.",
                )
            )
        elif class_id == "cutpurse" and hero.actor_id == target.actor_id:
            options.append(
                CombatReaction(
                    reaction_id=f"cutpurse_evade:{hero.actor_id}",
                    kind="cutpurse_evade",
                    actor_id=hero.actor_id,
                    actor_name=hero.name,
                    cost=1,
                    summary=f"{hero.name} cuts the damage roughly in half.",
                )
            )
        elif class_id == "field_surgeon" and target.team == Team.HERO:
            options.append(
                CombatReaction(
                    reaction_id=f"field_surgeon_stabilize:{hero.actor_id}",
                    kind="field_surgeon_stabilize",
                    actor_id=hero.actor_id,
                    actor_name=hero.name,
                    cost=1,
                    summary=f"{hero.name} prevents collapse and reduces the hit.",
                )
            )
        elif class_id == "scribe":
            options.append(
                CombatReaction(
                    reaction_id=f"scribe_disrupt:{hero.actor_id}",
                    kind="scribe_disrupt",
                    actor_id=hero.actor_id,
                    actor_name=hero.name,
                    cost=1,
                    summary=f"{hero.name} disrupts the action's accuracy.",
                )
            )
    return options


def heal_amount_for_skill(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    skill_id: str,
    target_id: str | None = None,
) -> int:
    return heal_amount_range_for_skill(session, definitions, skill_id, target_id)[1]


def heal_amount_range_for_skill(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    skill_id: str,
    target_id: str | None = None,
) -> tuple[int, int]:
    actor = session.pending_hero()
    if actor is None:
        return 0, 0
    skill = definitions.skills[skill_id]
    amount_min, amount_max = combatant_damage_range(skill, actor)
    amount_min = max(1, amount_min)
    amount_max = max(1, amount_max)
    if target_id is None:
        return amount_min, amount_max
    target = session.state.actor(target_id)
    if "brink_heal" in skill.tags and (
        target.is_downed() or target.hp <= target.max_hp // 2
    ):
        amount_min += 2
        amount_max += 2
    if _field_medic_bonus_available(session, actor.actor_id):
        amount_min += 1
        amount_max += 1
    missing_hp = target.max_hp - target.hp
    return max(0, min(amount_min, missing_hp)), max(0, min(amount_max, missing_hp))


def roll_heal_amount_for_skill(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
    skill_id: str,
    target_id: str,
) -> int:
    actor = session.pending_hero()
    if actor is None:
        return 0
    skill = definitions.skills[skill_id]
    amount = max(1, roll_skill_base_damage(skill, rng) + actor.damage)
    target = session.state.actor(target_id)
    if "brink_heal" in skill.tags and (
        target.is_downed() or target.hp <= target.max_hp // 2
    ):
        amount += 2
    if _field_medic_bonus_available(session, actor.actor_id):
        amount += 1
    return max(0, min(amount, target.max_hp - target.hp))


def resolve_hero_action(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
    skill_id: str,
    target_id: str,
) -> list[GameEvent]:
    return resolve_hero_skill(session, definitions, rng, skill_id, target_id)


def resolve_hero_skill(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
    skill_id: str,
    target_id: str,
) -> list[GameEvent]:
    actor = session.pending_hero()
    if actor is None:
        return []
    if skill_id not in legal_skill_ids(session, definitions):
        return []
    if target_id not in legal_target_ids(session, definitions, skill_id):
        return []

    events: list[GameEvent] = []
    skill = definitions.skills[skill_id]
    if _is_treatment_skill(skill.tags):
        target = session.state.actor(target_id)
        actor.effort -= skill.effort_cost
        events.append(
            SkillUsedEvent(
                message=f"{actor.name} uses {skill.name} on {target.name}.",
                actor_id=actor.actor_id,
                skill_id=skill.id,
                target_id=target_id,
            )
        )
        amount = roll_heal_amount_for_skill(session, definitions, rng, skill_id, target_id)
        if QUIRK_FIELD_MEDIC in actor.quirks:
            session.state.quirk_once_per_combat.add(_field_medic_key(actor.actor_id))
        events.append(
            HealingEvent(
                message=f"{target.name} recovers {amount} HP.",
                source_id=actor.actor_id,
                target_id=target_id,
                amount=amount,
            )
        )
        events.append(
            MemorySignalEvent(
                message=f"{actor.name} treated {target.name}.",
                hero_id=actor.actor_id,
                family_id="field_treatment",
                tags=("support", "healing"),
                source_summary=f"{actor.name} used {skill.name} on {target.name}.",
            )
        )
        events.extend(heal_combatant(session.state, target_id, amount))
        if actor.personal_quirk == PERSONAL_GENTLE_HANDS and raise_morale(target):
            events.append(
                StatusChangedEvent(
                    message=f"{actor.name}'s Gentle Hands steady {target.name}.",
                    actor_id=target_id,
                    status=target.morale.name.lower(),
                    added=True,
                )
            )
            if (
                actor.actor_id != target_id
                and QUIRK_STEADY_VOICE in actor.quirks
                and actor.effort < actor.max_effort
            ):
                effort_before = actor.effort
                actor.effort += 1
                events.append(
                    effort_delta_event(
                        actor_id=actor.actor_id,
                        actor_name=actor.name,
                        delta=actor.effort - effort_before,
                        before=effort_before,
                        after=actor.effort,
                        source_kind="quirk",
                        source_id=QUIRK_STEADY_VOICE,
                        source_label="Steady Voice",
                    )
                )
    else:
        result = use_skill(session.state, actor.actor_id, skill, target_id, rng)
        events.extend(result.events)
    return _finish_hero_turn(session, definitions, rng, events)


def resolve_hero_move(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
    to_slot: FormationSlot,
) -> list[GameEvent]:
    actor = session.pending_hero()
    if actor is None:
        return []
    from_slot = session.state.party_formation.slot_of(actor.actor_id)
    if from_slot is None or to_slot not in legal_move_slots(session):
        return []

    swapped_actor_id = session.state.party_formation.actor_at(to_slot)
    if not session.state.party_formation.swap_slots(from_slot, to_slot):
        return []
    actor.formation_slot = to_slot
    if swapped_actor_id is not None:
        session.state.actor(swapped_actor_id).formation_slot = from_slot
    if swapped_actor_id is None:
        message = f"{actor.name} shifts: {from_slot.value} -> {to_slot.value}."
    else:
        swapped_actor = session.state.actor(swapped_actor_id)
        message = (
            f"{actor.name} swaps with {swapped_actor.name}: "
            f"{from_slot.value} <-> {to_slot.value}."
        )

    events: list[GameEvent] = [
        MoveEvent(
            message=message,
            actor_id=actor.actor_id,
            from_slot=from_slot.value,
            to_slot=to_slot.value,
        )
    ]
    return _finish_hero_turn(session, definitions, rng, events)


def resolve_hero_pass(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
) -> list[GameEvent]:
    actor = session.pending_hero()
    if actor is None:
        return []
    events: list[GameEvent] = [
        TurnPassedEvent(
            message=f"{actor.name} holds position.",
            actor_id=actor.actor_id,
            encounter_id=session.encounter_id,
        )
    ]
    return _finish_hero_turn(session, definitions, rng, events)


def resolve_hero_retreat(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
) -> list[GameEvent]:
    actor = session.pending_hero()
    if actor is None or session.retreat_pending:
        return []
    session.retreat_pending = True
    session.retreat_actor_id = actor.actor_id
    events: list[GameEvent] = [
        CombatRetreatDeclaredEvent(
            message=(
                f"{actor.name} calls for a fighting withdrawal. "
                "The party will escape when the round ends."
            ),
            actor_id=actor.actor_id,
            encounter_id=session.encounter_id,
        )
    ]
    return _finish_hero_turn(session, definitions, rng, events)


def resolve_hero_delay(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
) -> list[GameEvent]:
    actor = session.pending_hero()
    if actor is None or not can_delay_hero(session):
        return []

    entry = session.initiative.pop(session.turn_index)
    session.initiative.append(entry)
    session.delayed_actor_ids.add(actor.actor_id)
    session.selected_skill_id = None
    session.selected_target_id = None

    events: list[GameEvent] = [
        TurnDelayedEvent(
            message=f"{actor.name} delays their turn.",
            actor_id=actor.actor_id,
            encounter_id=session.encounter_id,
        )
    ]
    if _combat_finished(session):
        events.extend(_finish_encounter(session))
    else:
        events.extend(auto_advance_to_hero(session, definitions, rng))
    session.event_log.extend(events)
    session.recent_events = events[-8:]
    return events


def resolve_enemy_reaction(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
    reaction_id: str | None,
) -> list[GameEvent]:
    intent = session.pending_enemy_intent
    if intent is None:
        return []

    events: list[GameEvent] = []
    modifiers = SkillResolutionModifiers()
    target_id = intent.target_id
    ignore_target_legality = False

    if reaction_id is None:
        events.append(
            ReactionSkippedEvent(
                message=f"No reaction is used against {intent.label}.",
                enemy_id=intent.enemy_id,
                skill_id=intent.skill_id,
                target_id=intent.target_id,
            )
        )
    else:
        reaction = next(
            (
                option
                for option in legal_reaction_options(session)
                if option.reaction_id == reaction_id
            ),
            None,
        )
        if reaction is None:
            return []
        reactor = session.state.actor(reaction.actor_id)
        reactor.effort -= reaction.cost
        events.append(
            ReactionUsedEvent(
                message=f"{reactor.name} uses {reaction.kind.replace('_', ' ')}.",
                reaction_id=reaction.reaction_id,
                reaction_kind=reaction.kind,
                actor_id=reactor.actor_id,
                actor_name=reactor.name,
                enemy_id=intent.enemy_id,
                skill_id=intent.skill_id,
                target_id=intent.target_id,
            )
        )
        if reaction.kind == "watchman_intercede":
            target_id = reactor.actor_id
            ignore_target_legality = True
        elif reaction.kind == "cutpurse_evade":
            modifiers = SkillResolutionModifiers(
                damage_multiplier_numerator=1,
                damage_multiplier_denominator=2,
            )
        elif reaction.kind == "field_surgeon_stabilize":
            modifiers = SkillResolutionModifiers(
                damage_delta=-2,
                prevent_downed=True,
            )
        elif reaction.kind == "scribe_disrupt":
            modifiers = SkillResolutionModifiers(hit_modifier=-25)

    session.pending_enemy_intent = None
    result = use_skill(
        session.state,
        intent.enemy_id,
        definitions.skills[intent.skill_id],
        target_id,
        rng,
        modifiers=modifiers,
        ignore_target_legality=ignore_target_legality,
    )
    events.extend(result.events)
    events.extend(_finish_actor_activation(session.state.actor(intent.enemy_id)))
    if any(isinstance(event, CombatEndedEvent) for event in events) or _combat_finished(session):
        events.extend(_finish_encounter(session))
        session.event_log.extend(events)
        session.recent_events = events[-8:]
        return events

    session.turn_index += 1
    events.extend(auto_advance_to_hero(session, definitions, rng))
    session.event_log.extend(events)
    session.recent_events = events[-8:]
    return events


def auto_advance_to_hero(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
    *,
    enemy_ai_mode: str | None = None,
) -> list[GameEvent]:
    events: list[GameEvent] = []
    if session.pending_enemy_intent is not None:
        return events
    while not session.ended:
        if _combat_finished(session):
            events.extend(_finish_encounter(session))
            break
        if session.turn_index >= len(session.initiative):
            events.append(
                RoundEndedEvent(
                    message=f"Round {session.state.round_number} ends.",
                    encounter_id=session.encounter_id,
                    round_number=session.state.round_number,
                )
            )
            if session.retreat_pending:
                session.ended = True
                session.outcome = "retreat"
                break
            session.state.round_number += 1
            events.extend(_start_round(session, rng))
            continue

        actor = session.current_actor()
        if actor is None or not actor.can_act():
            session.turn_index += 1
            continue
        if actor.team == Team.HERO and session.retreat_pending:
            events.append(
                TurnPassedEvent(
                    message=f"{actor.name} covers the withdrawal.",
                    actor_id=actor.actor_id,
                    encounter_id=session.encounter_id,
                )
            )
            events.extend(_finish_actor_activation(actor))
            session.turn_index += 1
            continue
        if actor.team == Team.HERO:
            break

        runtime_context = EnemyDecisionRuntimeContext(
            initiative_actor_ids=tuple(entry.actor_id for entry in session.initiative),
            current_turn_index=session.turn_index,
        )
        wait_reason = enemy_wait_reason(
            session.state,
            definitions,
            actor,
            runtime_context,
            session.enemy_wait_mode,
            session.delayed_actor_ids,
        )
        if wait_reason is not None and session.turn_index + 1 < len(session.initiative):
            delayed_entry = session.initiative.pop(session.turn_index)
            session.initiative.append(delayed_entry)
            session.delayed_actor_ids.add(actor.actor_id)
            events.append(
                TurnDelayedEvent(
                    message=f"{actor.name} waits for {wait_reason}.",
                    actor_id=actor.actor_id,
                    encounter_id=session.encounter_id,
                )
            )
            continue

        movement_events = enemy_proactive_move(
            session.state,
            definitions,
            actor,
            session.enemy_movement_mode,
            runtime_context,
        )
        if movement_events:
            events.extend(movement_events)
            events.extend(_finish_actor_activation(actor))
            session.turn_index += 1
            continue

        skill_and_target = choose_enemy_skill_and_target(
            session.state,
            definitions,
            actor.actor_id,
            runtime_context,
            policy=production_enemy_decision_policy(enemy_ai_mode or session.enemy_ai_mode),
        )
        if skill_and_target is not None:
            skill_id, target_id = skill_and_target
            skill = definitions.skills[skill_id]
            if skill.reaction_window:
                events.append(
                    _enemy_special_event(
                        session,
                        definitions,
                        actor.actor_id,
                        skill_id,
                        target_id,
                    )
                )
            result = use_skill(
                session.state,
                actor.actor_id,
                skill,
                target_id,
                rng,
            )
            events.extend(result.events)
        else:
            events.extend(
                enemy_recovery_events(
                    session.state,
                    definitions,
                    actor,
                    session.enemy_movement_mode,
                )
            )
        events.extend(_finish_actor_activation(actor))
        session.turn_index += 1

    if events:
        session.recent_events = events[-8:]
    return events


def _start_round(session: ManualCombatSession, rng: GameRng) -> list[GameEvent]:
    session.initiative = roll_initiative(session.state, rng)
    session.turn_index = 0
    session.delayed_actor_ids.clear()
    return [
        RoundStartedEvent(
            message=f"Round {session.state.round_number} begins.",
            encounter_id=session.encounter_id,
            round_number=session.state.round_number,
            actor_ids=[entry.actor_id for entry in session.initiative],
        )
    ]


def _combat_finished(session: ManualCombatSession) -> bool:
    return session.state.is_victory() or session.state.is_defeat()


def _finish_encounter(session: ManualCombatSession) -> list[GameEvent]:
    session.ended = True
    victor = "heroes" if session.state.is_victory() else "enemies"
    outcome = "victory" if victor == "heroes" else "defeat"
    session.outcome = outcome
    return [
        EncounterEndedEvent(
            message=f"{session.encounter_name} ends in {outcome}.",
            encounter_id=session.encounter_id,
            victor=victor,
        )
    ]


def _finish_hero_turn(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    rng: GameRng,
    events: list[GameEvent],
) -> list[GameEvent]:
    actor = session.current_actor()
    if actor is not None:
        events.extend(_finish_actor_activation(actor))
    session.selected_skill_id = None
    session.selected_target_id = None
    if any(isinstance(event, CombatEndedEvent) for event in events) or _combat_finished(session):
        events.extend(_finish_encounter(session))
        session.event_log.extend(events)
        session.recent_events = events[-8:]
        return events
    session.turn_index += 1
    events.extend(auto_advance_to_hero(session, definitions, rng))
    session.event_log.extend(events)
    session.recent_events = events[-8:]
    return events


def _finish_actor_activation(actor: Combatant) -> list[GameEvent]:
    if not tick_marked_turn(actor):
        return []
    return [
        StatusChangedEvent(
            message=f"{actor.name} is no longer Marked.",
            actor_id=actor.actor_id,
            status=Tag.MARKED.name.lower(),
            added=False,
        )
    ]


def _enemy_position_recovery_move(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    actor: Combatant,
) -> list[GameEvent]:
    return list(enemy_recovery_move(session.state, definitions, actor))


def _enemy_recovery_slots(from_slot: FormationSlot) -> list[FormationSlot]:
    return enemy_recovery_slots(from_slot)


def _enemy_has_choice_after_swap(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    from_slot: FormationSlot,
    to_slot: FormationSlot,
) -> bool:
    return enemy_has_choice_after_swap(state, definitions, actor, from_slot, to_slot)


def _declare_enemy_intent(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    enemy_id: str,
    skill_id: str,
    target_id: str,
) -> EnemyIntentEvent:
    enemy = session.state.actor(enemy_id)
    target = session.state.actor(target_id)
    skill = definitions.skills[skill_id]
    preview = preview_attack(session.state, enemy_id, skill, target_id)
    label = skill.intent_label or skill.name
    obvious_effect = skill.obvious_effect or "Incoming enemy action"
    threat_level = skill.threat_level or "normal"
    intent = EnemyIntent(
        enemy_id=enemy_id,
        enemy_name=enemy.name,
        skill_id=skill_id,
        skill_name=skill.name,
        label=label,
        target_id=target_id,
        target_name=target.name,
        threat_level=threat_level,
        obvious_effect=obvious_effect,
        hit_chance=preview.hit_chance,
        damage_estimate=preview.damage,
        damage_label=preview.damage_label,
    )
    session.pending_enemy_intent = intent
    return EnemyIntentEvent(
        message=(
            f"{enemy.name} prepares {label} against {target.name} "
            f"({threat_level}: {obvious_effect})."
        ),
        enemy_id=enemy_id,
        enemy_name=enemy.name,
        skill_id=skill_id,
        skill_name=skill.name,
        label=label,
        target_id=target_id,
        target_name=target.name,
        threat_level=threat_level,
        obvious_effect=obvious_effect,
        hit_chance=preview.hit_chance,
        damage_estimate=preview.damage,
        damage_label=preview.damage_label,
    )


def _enemy_special_event(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    enemy_id: str,
    skill_id: str,
    target_id: str,
) -> EnemyIntentEvent:
    enemy = session.state.actor(enemy_id)
    target = session.state.actor(target_id)
    skill = definitions.skills[skill_id]
    preview = preview_attack(session.state, enemy_id, skill, target_id)
    label = skill.intent_label or skill.name
    obvious_effect = skill.obvious_effect or "Special enemy action"
    threat_level = skill.threat_level or "high"
    return EnemyIntentEvent(
        message=f"{enemy.name} rears back. {label}.",
        enemy_id=enemy_id,
        enemy_name=enemy.name,
        skill_id=skill_id,
        skill_name=skill.name,
        label=label,
        target_id=target_id,
        target_name=target.name,
        threat_level=threat_level,
        obvious_effect=obvious_effect,
        hit_chance=preview.hit_chance,
        damage_estimate=preview.damage,
        damage_label=preview.damage_label,
    )


def _can_react(hero: Combatant) -> bool:
    return hero.team == Team.HERO and hero.can_act() and hero.effort >= 1


def _is_treatment_skill(tags: list[str]) -> bool:
    return "treatment" in tags or "heal" in tags


def _is_support_skill(skill: SkillDefinition) -> bool:
    tags = skill.tags
    return _is_treatment_skill(tags) or "rally" in tags or (
        "guard" in tags and skill_base_damage_max(skill) <= 0
    )


def _legal_ally_targets(state: CombatState) -> list[str]:
    return sorted(
        hero.actor_id
        for hero in state.heroes.values()
        if hero.is_alive()
    )


def _field_medic_bonus_available(session: ManualCombatSession, actor_id: str) -> bool:
    actor = session.state.actor(actor_id)
    return (
        QUIRK_FIELD_MEDIC in actor.quirks
        and _field_medic_key(actor_id) not in session.state.quirk_once_per_combat
    )


def _field_medic_key(actor_id: str) -> str:
    return f"field_medic:{actor_id}:first_treatment"
