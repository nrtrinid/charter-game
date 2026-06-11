"""Non-mutating combat previews for UI-facing app views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from game.combat.combat_state import CombatState
from game.combat.damage_range import format_damage_range, projected_damage_range
from game.combat.formation import is_front, lane_of
from game.combat.targeting import AttackType, can_target, cover_penalty


class SkillPreviewLike(Protocol):
    id: str
    attack_type: AttackType
    accuracy: int
    damage: int
    tags: list[str]


@dataclass(frozen=True)
class AttackPreview:
    hit_chance: int
    damage_min: int
    damage_max: int
    legality_reason: str

    @property
    def damage(self) -> int:
        return self.damage_max

    @property
    def damage_label(self) -> str:
        return format_damage_range(self.damage_min, self.damage_max)


def preview_attack(
    state: CombatState,
    actor_id: str,
    skill: SkillPreviewLike,
    target_id: str,
) -> AttackPreview:
    actor = state.actor(actor_id)
    target = state.actor(target_id)
    hit_chance = _clamp_percent(
        skill.accuracy
        + actor.accuracy
        - target.defense
        + cover_penalty(state, target_id, skill.attack_type)
    )
    damage_min, damage_max = projected_damage_range(
        state,
        skill,
        actor,
        target,
        set(skill.tags),
    )
    return AttackPreview(
        hit_chance=hit_chance,
        damage_min=damage_min,
        damage_max=damage_max,
        legality_reason=targeting_reason(state, actor_id, target_id, skill.attack_type),
    )


def targeting_reason(
    state: CombatState,
    actor_id: str,
    target_id: str,
    attack_type: AttackType,
) -> str:
    if not can_target(state, actor_id, target_id, attack_type):
        return "illegal"

    actor = state.actor(actor_id)
    target = state.actor(target_id)
    target_formation = state.formation_for(target.team)
    target_slot = target_formation.slot_of(target_id)

    if attack_type == AttackType.MAGIC:
        return "magic"
    if attack_type == AttackType.RANGED:
        return "ranged"
    if target_slot is None:
        return "unformed"
    if is_front(target_slot):
        return "frontline"
    if attack_type == AttackType.MELEE:
        return "exposed"
    if attack_type == AttackType.REACH:
        actor_slot = state.formation_for(actor.team).slot_of(actor_id)
        if actor_slot is not None and lane_of(actor_slot) == lane_of(target_slot):
            return "same lane"
    return "legal"


def _clamp_percent(value: int) -> int:
    return max(0, min(100, value))
