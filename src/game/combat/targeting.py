"""Target legality for combat skills."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from game.combat.combat_state import CombatState
from game.combat.formation import FormationSlot, is_back, is_front, lane_of

RANGED_COVER_PENALTY = -2


class AttackType(StrEnum):
    MELEE = "melee"
    REACH = "reach"
    RANGED = "ranged"
    MAGIC = "magic"


class SkillUsableFrom(StrEnum):
    ANY_POSITION = "any_position"
    FRONT_ONLY = "front_only"


class PositionalSkillLike(Protocol):
    usable_from: SkillUsableFrom


def can_use_skill_from_position(
    state: CombatState,
    actor_id: str,
    skill: PositionalSkillLike,
) -> bool:
    usable_from = getattr(skill, "usable_from", SkillUsableFrom.ANY_POSITION)
    if usable_from in {SkillUsableFrom.ANY_POSITION, SkillUsableFrom.ANY_POSITION.value}:
        return True
    if usable_from not in {SkillUsableFrom.FRONT_ONLY, SkillUsableFrom.FRONT_ONLY.value}:
        return True

    actor = state.actor(actor_id)
    actor_slot = state.formation_for(actor.team).slot_of(actor_id)
    return actor_slot is not None and is_front(actor_slot)


def skill_position_label(skill: PositionalSkillLike, *, compact: bool = False) -> str:
    usable_from = getattr(skill, "usable_from", SkillUsableFrom.ANY_POSITION)
    if usable_from in {SkillUsableFrom.FRONT_ONLY, SkillUsableFrom.FRONT_ONLY.value}:
        return "Front" if compact else "Front row"
    return "Any" if compact else "Any row"


def skill_position_unavailable_reason(
    state: CombatState,
    actor_id: str,
    skill: PositionalSkillLike,
) -> str:
    usable_from = getattr(skill, "usable_from", SkillUsableFrom.ANY_POSITION)
    if usable_from in {SkillUsableFrom.FRONT_ONLY, SkillUsableFrom.FRONT_ONLY.value}:
        if not can_use_skill_from_position(state, actor_id, skill):
            return "Requires front row."
    return ""


def legal_targets(state: CombatState, actor_id: str, attack_type: AttackType) -> list[str]:
    return [
        target.actor_id
        for target in state.opposing_side_for(state.actor(actor_id).team).values()
        if can_target(state, actor_id, target.actor_id, attack_type)
    ]


def can_target(
    state: CombatState,
    actor_id: str,
    target_id: str,
    attack_type: AttackType,
) -> bool:
    actor = state.actor(actor_id)
    target = state.actor(target_id)
    if actor.team == target.team or not target.is_alive():
        return False

    target_formation = state.formation_for(target.team)
    target_slot = target_formation.slot_of(target_id)
    if target_slot is None:
        return attack_type in {AttackType.RANGED, AttackType.MAGIC}

    if attack_type == AttackType.MAGIC:
        return True

    if attack_type == AttackType.RANGED:
        return True

    if attack_type == AttackType.MELEE:
        return is_front(target_slot) or target_formation.is_exposed(
            target_id,
            state.side_for(target.team),
        )

    if attack_type == AttackType.REACH:
        if is_front(target_slot):
            return True
        return _same_lane(
            actor_slot=state.formation_for(actor.team).slot_of(actor_id),
            target_slot=target_slot,
        )

    return False


def cover_penalty(state: CombatState, target_id: str, attack_type: AttackType) -> int:
    if attack_type != AttackType.RANGED:
        return 0
    target = state.actor(target_id)
    target_formation = state.formation_for(target.team)
    target_slot = target_formation.slot_of(target_id)
    if target_slot is not None and is_back(target_slot):
        if target_formation.is_protected(target_id, state.side_for(target.team)):
            return RANGED_COVER_PENALTY
    return 0


def _same_lane(actor_slot: FormationSlot | None, target_slot: FormationSlot) -> bool:
    return actor_slot is not None and lane_of(actor_slot) == lane_of(target_slot)
