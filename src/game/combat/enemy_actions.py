"""Shared enemy non-skill action helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from game.combat.combat_state import Combatant, CombatState, StrainMark, Tag
from game.combat.damage_range import skill_base_damage_max
from game.combat.enemy_decision import (
    EnemyDecisionRuntimeContext,
    _has_not_acted_yet,
    choose_enemy_skill_and_target,
    explain_enemy_decision,
)
from game.combat.formation import (
    FormationSlot,
    are_adjacent,
    front_slot_for,
    is_front,
    lane_of,
)
from game.combat.targeting import can_use_skill_from_position, legal_targets
from game.combat.traits import has_strain_mark
from game.content.definitions import GameDefinitions
from game.core.events import GameEvent, MoveEvent

SUPPORTED_ENEMY_WAIT_MODES = ("none", "package_only", "unrestricted")
SUPPORTED_ENEMY_MOVEMENT_MODES = (
    "none",
    "recovery_only",
    "package_only",
    "unrestricted",
)
_GOOD_PAYOFF_THRESHOLD = 8
_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()


def enemy_recovery_move(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
) -> tuple[GameEvent, ...]:
    if Tag.FROZEN in actor.tags or has_strain_mark(actor, StrainMark.WINDED):
        return ()
    formation = state.formation_for(actor.team)
    from_slot = formation.slot_of(actor.actor_id)
    if from_slot is None:
        return ()

    for to_slot in enemy_recovery_slots(from_slot):
        if not are_adjacent(from_slot, to_slot):
            continue
        if not enemy_has_choice_after_swap(state, definitions, actor, from_slot, to_slot):
            continue
        swapped_actor_id = formation.actor_at(to_slot)
        if not formation.swap_slots(from_slot, to_slot):
            continue
        actor.formation_slot = to_slot
        if swapped_actor_id is not None:
            state.actor(swapped_actor_id).formation_slot = from_slot
        if swapped_actor_id is None:
            message = f"{actor.name} repositions: {from_slot.value} -> {to_slot.value}."
        else:
            swapped_actor = state.actor(swapped_actor_id)
            message = (
                f"{actor.name} repositions around {swapped_actor.name}: "
                f"{from_slot.value} <-> {to_slot.value}."
            )
        return (
            MoveEvent(
                message=message,
                actor_id=actor.actor_id,
                from_slot=from_slot.value,
                to_slot=to_slot.value,
            ),
        )
    return ()


def enemy_wait_reason(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
    mode: str,
    waited_actor_ids: set[str],
) -> str | None:
    if mode == "none" or actor.actor_id in waited_actor_ids:
        return None
    if enemy_has_legal_marked_payoff(state, definitions, actor):
        return None
    later_allies = _later_enemy_allies(state, runtime_context, actor.actor_id)
    if not later_allies:
        return None
    if _enemy_can_mark(actor, definitions):
        return None
    if not _enemy_has_skill_tag(actor, definitions, {"vulnerable_bonus", "exploit_vulnerable"}):
        return None
    pending_markers = tuple(
        ally
        for ally in later_allies
        if _enemy_can_mark(ally, definitions)
        and _has_not_acted_yet(runtime_context, ally.actor_id)
        and _enemy_can_create_mark_now(state, definitions, ally)
    )
    if not pending_markers:
        return None
    current_features = _current_candidate_features(
        state,
        definitions,
        actor,
        runtime_context,
    )
    if _current_best_action_is_good(state, definitions, actor, current_features):
        return None
    if pending_markers and _plausible_wait_payoff(
        state,
        definitions,
        actor,
        runtime_context,
        pending_markers,
        current_features,
    ):
        return "mark setup"
    has_marked_hero = any(Tag.MARKED in hero.tags for hero in state.heroes.values())
    if mode == "unrestricted" and not has_marked_hero:
        wait_features = _extract_wait_timing_features(
            state,
            definitions,
            actor,
            runtime_context,
        )
        if wait_features.get("wait_no_current_good_action", 0) > 0:
            return "future payoff"
        if wait_features.get("wait_expected_value_delta", 0) > 0:
            return "future payoff"
    return None


def enemy_proactive_move(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    mode: str,
    runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    failed_move_actor_ids: set[str] | None = None,
) -> tuple[GameEvent, ...]:
    if mode not in {"package_only", "unrestricted"}:
        return ()
    if actor.actor_id in (failed_move_actor_ids or set()):
        return ()
    if enemy_has_legal_marked_payoff(state, definitions, actor):
        return ()
    actor_slot = state.enemy_formation.slot_of(actor.actor_id)
    if actor_slot is None:
        return ()
    marked_hero_ids = tuple(
        hero.actor_id
        for hero in state.heroes.values()
        if hero.is_alive() and Tag.MARKED in hero.tags
    )
    if not marked_hero_ids:
        return ()

    current_features = _current_candidate_features(
        state,
        definitions,
        actor,
        runtime_context,
    )
    current_action_good = _current_best_action_is_good(
        state,
        definitions,
        actor,
        current_features,
    )
    had_offensive_before = _enemy_has_offensive_choice(state, definitions, actor)

    occupied_slots = (
        slot for slot, occupant_id in state.enemy_formation.slots.items() if occupant_id is not None
    )
    for to_slot in sorted(occupied_slots, key=lambda slot: slot.value):
        if to_slot == actor_slot or not are_adjacent(actor_slot, to_slot):
            continue
        occupant_id = state.enemy_formation.actor_at(to_slot)
        if occupant_id is None:
            continue
        _swap_enemy_slots(state, actor.actor_id, occupant_id, actor_slot, to_slot)
        accepted = any(
            _enemy_can_pay_off_mark(state, definitions, actor, hero_id)
            for hero_id in marked_hero_ids
        )
        if not accepted:
            _swap_enemy_slots(state, actor.actor_id, occupant_id, to_slot, actor_slot)
            continue
        move_features = _move_features_after_swap(
            state,
            definitions,
            actor,
            runtime_context,
            had_offensive_before=had_offensive_before,
            current_features=current_features,
        )
        unlocks = move_features.get("move_unlocks_future_skill", 0) > 0
        into_lane = move_features.get("move_into_marked_lane", 0) > 0
        value_delta = move_features.get("move_expected_value_delta", 0) > 0
        poor_current = not current_action_good
        if not (unlocks or into_lane or value_delta or poor_current):
            _swap_enemy_slots(state, actor.actor_id, occupant_id, to_slot, actor_slot)
            continue
        if current_action_good and not (unlocks or into_lane or value_delta):
            _swap_enemy_slots(state, actor.actor_id, occupant_id, to_slot, actor_slot)
            continue
        return (
            MoveEvent(
                message=(
                    f"{actor.name} shifts from {actor_slot.value} "
                    f"to {to_slot.value}."
                ),
                actor_id=actor.actor_id,
                from_slot=actor_slot.value,
                to_slot=to_slot.value,
            ),
        )
    return ()


def enemy_recovery_events(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    mode: str,
) -> tuple[GameEvent, ...]:
    if mode == "none":
        return ()
    return tuple(enemy_recovery_move(state, definitions, actor))


def extract_wait_timing_features(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
) -> dict[str, int]:
    return _extract_wait_timing_features(state, definitions, actor, runtime_context)


def extract_move_timing_features(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
    *,
    had_offensive_before: bool | None = None,
) -> dict[str, int]:
    if had_offensive_before is None:
        had_offensive_before = _enemy_has_offensive_choice(state, definitions, actor)
    current_features = _current_candidate_features(
        state,
        definitions,
        actor,
        runtime_context,
    )
    actor_slot = state.enemy_formation.slot_of(actor.actor_id)
    if actor_slot is None:
        return _empty_move_features(current_features, had_offensive_before)
    marked_hero_ids = tuple(
        hero.actor_id
        for hero in state.heroes.values()
        if hero.is_alive() and Tag.MARKED in hero.tags
    )
    if not marked_hero_ids:
        return _empty_move_features(current_features, had_offensive_before)

    best_features: dict[str, int] | None = None
    best_priority = (-1, -1, -1)
    occupied_slots = (
        slot for slot, occupant_id in state.enemy_formation.slots.items() if occupant_id is not None
    )
    for to_slot in sorted(occupied_slots, key=lambda slot: slot.value):
        if to_slot == actor_slot or not are_adjacent(actor_slot, to_slot):
            continue
        occupant_id = state.enemy_formation.actor_at(to_slot)
        if occupant_id is None:
            continue
        _swap_enemy_slots(state, actor.actor_id, occupant_id, actor_slot, to_slot)
        if not any(
            _enemy_can_pay_off_mark(state, definitions, actor, hero_id)
            for hero_id in marked_hero_ids
        ):
            _swap_enemy_slots(state, actor.actor_id, occupant_id, to_slot, actor_slot)
            continue
        move_features = _move_features_after_swap(
            state,
            definitions,
            actor,
            runtime_context,
            had_offensive_before=had_offensive_before,
            current_features=current_features,
        )
        priority = (
            move_features.get("move_unlocks_future_skill", 0),
            move_features.get("move_into_marked_lane", 0),
            move_features.get("move_expected_value_delta", 0),
        )
        if priority > best_priority:
            best_priority = priority
            best_features = move_features
        _swap_enemy_slots(state, actor.actor_id, occupant_id, to_slot, actor_slot)
    if best_features is not None:
        return best_features
    return _empty_move_features(current_features, had_offensive_before)


def enemy_recovery_slots(from_slot: FormationSlot) -> list[FormationSlot]:
    same_lane_front = front_slot_for(from_slot)
    return sorted(
        (slot for slot in FormationSlot if slot != from_slot),
        key=lambda slot: (
            0 if slot == same_lane_front else 1,
            0 if is_front(slot) else 1,
            0 if lane_of(slot) == lane_of(from_slot) else 1,
            slot.value,
        ),
    )


def enemy_has_choice_after_swap(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    from_slot: FormationSlot,
    to_slot: FormationSlot,
) -> bool:
    formation = state.formation_for(actor.team)
    swapped_actor_id = formation.actor_at(to_slot)
    if not formation.swap_slots(from_slot, to_slot):
        return False
    actor.formation_slot = to_slot
    if swapped_actor_id is not None:
        state.actor(swapped_actor_id).formation_slot = from_slot
    try:
        return choose_enemy_skill_and_target(state, definitions, actor.actor_id) is not None
    finally:
        formation.swap_slots(to_slot, from_slot, require_adjacent=False)
        actor.formation_slot = from_slot
        if swapped_actor_id is not None:
            state.actor(swapped_actor_id).formation_slot = to_slot


def enemy_has_legal_marked_payoff(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
) -> bool:
    return any(
        _enemy_can_pay_off_mark(state, definitions, enemy, hero.actor_id)
        for hero in state.heroes.values()
        if hero.is_alive() and Tag.MARKED in hero.tags
    )


def _extract_wait_timing_features(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
) -> dict[str, int]:
    current_features = _current_candidate_features(
        state,
        definitions,
        actor,
        runtime_context,
    )
    current_attack = _current_best_attack_value(current_features)
    current_payoff = _current_best_payoff_value(current_features)
    current_good = _current_best_action_is_good(
        state,
        definitions,
        actor,
        current_features,
    )
    later_allies = _later_enemy_allies(state, runtime_context, actor.actor_id)
    pending_markers = tuple(
        ally
        for ally in later_allies
        if _enemy_can_mark(ally, definitions)
        and _has_not_acted_yet(runtime_context, ally.actor_id)
    )
    expected_payoff = _wait_expected_payoff_value(
        state,
        definitions,
        actor,
        runtime_context,
        pending_markers,
    )
    return {
        "current_best_attack_value": current_attack,
        "current_best_payoff_value": current_payoff,
        "current_best_action_is_good": int(current_good),
        "wait_expected_payoff_value": expected_payoff,
        "wait_expected_value_delta": max(0, expected_payoff - current_payoff),
        "wait_when_current_action_good": int(current_good),
        "wait_no_current_good_action": int(not current_good),
    }


def _empty_move_features(
    current_features: dict[str, int],
    had_offensive_before: bool,
) -> dict[str, int]:
    current_payoff = _current_best_payoff_value(current_features)
    current_good = _current_best_payoff_value(current_features) >= _GOOD_PAYOFF_THRESHOLD
    return {
        "current_best_attack_value": _current_best_attack_value(current_features),
        "current_best_payoff_value": current_payoff,
        "current_best_action_is_good": int(current_good),
        "move_expected_payoff_value": current_payoff,
        "move_expected_value_delta": 0,
        "move_when_current_action_good": int(current_good),
        "move_unlocks_future_skill": 0,
        "move_into_marked_lane": 0,
        "move_toward_payoff_target": 0,
    }


def _move_features_after_swap(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
    *,
    had_offensive_before: bool,
    current_features: dict[str, int],
) -> dict[str, int]:
    current_payoff = _current_best_payoff_value(current_features)
    current_good = _current_best_action_is_good(
        state,
        definitions,
        actor,
        current_features,
    )
    post_features = _current_candidate_features(
        state,
        definitions,
        actor,
        runtime_context,
    )
    post_payoff = _current_best_payoff_value(post_features)
    has_offensive_after = _enemy_has_offensive_choice(state, definitions, actor)
    into_lane = _enemy_in_marked_lane(state, actor)
    toward_payoff = int(
        enemy_has_legal_marked_payoff(state, definitions, actor)
        or post_payoff > current_payoff
    )
    return {
        "current_best_attack_value": _current_best_attack_value(current_features),
        "current_best_payoff_value": current_payoff,
        "current_best_action_is_good": int(current_good),
        "move_expected_payoff_value": post_payoff,
        "move_expected_value_delta": max(0, post_payoff - current_payoff),
        "move_when_current_action_good": int(current_good),
        "move_unlocks_future_skill": int(not had_offensive_before and has_offensive_after),
        "move_into_marked_lane": int(into_lane),
        "move_toward_payoff_target": toward_payoff,
    }


def _current_candidate_features(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
) -> dict[str, int]:
    trace = explain_enemy_decision(
        state,
        definitions,
        actor.actor_id,
        runtime_context,
    )
    if trace is None or trace.chosen is None:
        return {}
    return dict(trace.chosen.features)


def _current_best_attack_value(features: dict[str, int]) -> int:
    return features.get("damage_pressure", 0) + features.get("bandit_marked_attack", 0)


def _current_best_payoff_value(features: dict[str, int]) -> int:
    return (
        features.get("bandit_marked_payoff", 0)
        + features.get("vulnerable_payoff", 0)
        + features.get("maw_bite_payoff", 0)
    )


def _current_best_action_is_good(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    features: dict[str, int],
) -> bool:
    if enemy_has_legal_marked_payoff(state, definitions, actor):
        return True
    marked_payoff_features = (
        features.get("bandit_marked_payoff", 0) + features.get("maw_bite_payoff", 0)
    )
    return marked_payoff_features >= _GOOD_PAYOFF_THRESHOLD


def _plausible_wait_payoff(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
    pending_markers: tuple[Combatant, ...],
    current_features: dict[str, int],
) -> bool:
    expected = _wait_expected_payoff_value(
        state,
        definitions,
        actor,
        runtime_context,
        pending_markers,
    )
    current_payoff = _current_best_payoff_value(current_features)
    if expected > current_payoff:
        return True
    mark_target_id = _best_simulated_mark_target(
        state,
        definitions,
        pending_markers,
    )
    return (
        mark_target_id is not None
        and _enemy_can_pay_off_mark(state, definitions, actor, mark_target_id)
    )


def _wait_expected_payoff_value(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    runtime_context: EnemyDecisionRuntimeContext,
    pending_markers: tuple[Combatant, ...],
) -> int:
    mark_target_id = _best_simulated_mark_target(state, definitions, pending_markers)
    if mark_target_id is None:
        return 0
    with _simulated_mark(state, mark_target_id):
        post_features = _current_candidate_features(
            state,
            definitions,
            actor,
            runtime_context,
        )
    return _current_best_payoff_value(post_features)


def _best_simulated_mark_target(
    state: CombatState,
    definitions: GameDefinitions,
    pending_markers: tuple[Combatant, ...],
) -> str | None:
    best_target_id: str | None = None
    best_priority = (-1, -1, "")
    for marker in pending_markers:
        for skill_id in marker.skills:
            skill = definitions.skills[skill_id]
            if not {"mark", "mark_target"}.intersection(skill.tags):
                continue
            if marker.effort < skill.effort_cost:
                continue
            if not can_use_skill_from_position(state, marker.actor_id, skill):
                continue
            for target_id in legal_targets(state, marker.actor_id, skill.attack_type):
                target = state.actor(target_id)
                if Tag.MARKED in target.tags:
                    continue
                priority = (
                    int(target.hp <= target.max_hp // 2),
                    -target.hp,
                    target_id,
                )
                if priority > best_priority:
                    best_priority = priority
                    best_target_id = target_id
    return best_target_id


@contextmanager
def _simulated_mark(state: CombatState, target_id: str) -> Iterator[None]:
    target = state.actor(target_id)
    was_marked = Tag.MARKED in target.tags
    target.tags.add(Tag.MARKED)
    try:
        yield
    finally:
        if not was_marked:
            target.tags.discard(Tag.MARKED)


def _enemy_has_offensive_choice(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
) -> bool:
    for skill_id in actor.skills:
        skill = definitions.skills[skill_id]
        if actor.effort < skill.effort_cost:
            continue
        if skill_base_damage_max(skill) <= 0:
            continue
        if not can_use_skill_from_position(state, actor.actor_id, skill):
            continue
        if legal_targets(state, actor.actor_id, skill.attack_type):
            return True
    return False


def _enemy_in_marked_lane(state: CombatState, actor: Combatant) -> bool:
    actor_slot = state.enemy_formation.slot_of(actor.actor_id)
    if actor_slot is None:
        return False
    actor_lane = lane_of(actor_slot)
    return any(
        lane_of(slot) == actor_lane
        for hero in state.heroes.values()
        if hero.is_alive() and Tag.MARKED in hero.tags
        for slot in [state.party_formation.slot_of(hero.actor_id)]
        if slot is not None
    )


def _later_enemy_allies(
    state: CombatState,
    runtime_context: EnemyDecisionRuntimeContext,
    actor_id: str,
) -> tuple[Combatant, ...]:
    later_ids = runtime_context.initiative_actor_ids[runtime_context.current_turn_index + 1 :]
    return tuple(
        state.enemies[enemy_id]
        for enemy_id in later_ids
        if enemy_id != actor_id and enemy_id in state.enemies and state.enemies[enemy_id].can_act()
    )


def _enemy_can_create_mark_now(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
) -> bool:
    for skill_id in enemy.skills:
        skill = definitions.skills[skill_id]
        if not {"mark", "mark_target"}.intersection(skill.tags):
            continue
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        legal = legal_targets(state, enemy.actor_id, skill.attack_type)
        if any(Tag.MARKED not in state.actor(target_id).tags for target_id in legal):
            return True
    return False


def _enemy_can_mark(enemy: Combatant, definitions: GameDefinitions) -> bool:
    return _enemy_has_skill_tag(enemy, definitions, {"mark", "mark_target"})


def _enemy_has_skill_tag(
    enemy: Combatant,
    definitions: GameDefinitions,
    tags: set[str],
) -> bool:
    return any(tags.intersection(definitions.skills[skill_id].tags) for skill_id in enemy.skills)


def _enemy_can_pay_off_mark(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    hero_id: str,
) -> bool:
    for skill_id in enemy.skills:
        skill = definitions.skills[skill_id]
        if not {"vulnerable_bonus", "exploit_vulnerable", "basic", "boss_special"}.intersection(
            skill.tags
        ):
            continue
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        if hero_id in legal_targets(state, enemy.actor_id, skill.attack_type):
            return True
    return False


def _swap_enemy_slots(
    state: CombatState,
    actor_id: str,
    occupant_id: str,
    actor_slot: FormationSlot,
    occupant_slot: FormationSlot,
) -> None:
    state.enemy_formation.remove(actor_id)
    state.enemy_formation.remove(occupant_id)
    state.enemy_formation.place(actor_id, occupant_slot)
    state.enemy_formation.place(occupant_id, actor_slot)
    state.actor(actor_id).formation_slot = occupant_slot
    state.actor(occupant_id).formation_slot = actor_slot


__all__ = [
    "SUPPORTED_ENEMY_MOVEMENT_MODES",
    "SUPPORTED_ENEMY_WAIT_MODES",
    "enemy_has_legal_marked_payoff",
    "enemy_has_choice_after_swap",
    "enemy_proactive_move",
    "enemy_recovery_events",
    "enemy_recovery_move",
    "enemy_recovery_slots",
    "enemy_wait_reason",
    "extract_move_timing_features",
    "extract_wait_timing_features",
]
