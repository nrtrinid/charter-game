"""Helpers for fixed and ranged skill damage."""

from __future__ import annotations

from typing import Protocol

from game.combat.combat_state import Combatant, CombatState, MoraleState, Tag
from game.combat.formation import is_back
from game.combat.traits import (
    PERSONAL_OPPORTUNIST,
    QUIRK_DESPERATE_FOCUS,
    QUIRK_GRIM_FINISH,
    QUIRK_PREDATOR,
    has_trait,
)
from game.core.rng import GameRng


class SkillDamageLike(Protocol):
    damage: int


def skill_base_damage_min(skill: SkillDamageLike) -> int:
    value = getattr(skill, "damage_min", None)
    return skill.damage if value is None else int(value)


def skill_base_damage_max(skill: SkillDamageLike) -> int:
    value = getattr(skill, "damage_max", None)
    return skill.damage if value is None else int(value)


def roll_skill_base_damage(skill: SkillDamageLike, rng: GameRng) -> int:
    low, high = skill_base_damage_min(skill), skill_base_damage_max(skill)
    if low == high:
        return low
    return rng.randint(low, high)


def combatant_damage_range(skill: SkillDamageLike, actor: Combatant) -> tuple[int, int]:
    low = max(0, skill_base_damage_min(skill) + actor.damage)
    high = max(0, skill_base_damage_max(skill) + actor.damage)
    return low, high


def combatant_damage_max(skill: SkillDamageLike, actor: Combatant) -> int:
    return combatant_damage_range(skill, actor)[1]


def is_vulnerable_target(target: Combatant) -> bool:
    return Tag.MARKED in target.tags or target.hp < target.max_hp


def is_exposed_backliner(state: CombatState, target_id: str) -> bool:
    target = state.actor(target_id)
    formation = state.formation_for(target.team)
    slot = formation.slot_of(target_id)
    return slot is not None and is_back(slot) and formation.is_exposed(
        target_id,
        state.side_for(target.team),
    )


def skill_damage_bonus(
    state: CombatState,
    actor: Combatant,
    target: Combatant,
    skill_tags: set[str] | frozenset[str],
) -> int:
    bonus = 0
    if has_trait(actor, PERSONAL_OPPORTUNIST) and Tag.MARKED in target.tags:
        bonus += 1
    if QUIRK_PREDATOR in actor.quirks:
        bonus += 1 if Tag.MARKED in target.tags else -1
    if QUIRK_DESPERATE_FOCUS in actor.quirks and actor.morale == MoraleState.SHAKEN:
        bonus += 1
    if QUIRK_GRIM_FINISH in actor.quirks and target.hp <= max(1, target.max_hp // 2):
        bonus += 1
    if "vulnerable_bonus" in skill_tags and is_vulnerable_target(target):
        bonus += 2
    if "exposed_bonus" in skill_tags and is_exposed_backliner(state, target.actor_id):
        bonus += 2
    if "basic" in skill_tags and Tag.MARKED in target.tags:
        bonus += 1
    if "shock" in skill_tags and Tag.WET in target.tags:
        bonus += 2
    return bonus


def projected_damage_range(
    state: CombatState,
    skill: SkillDamageLike,
    actor: Combatant,
    target: Combatant,
    skill_tags: set[str] | frozenset[str],
) -> tuple[int, int]:
    bonus = skill_damage_bonus(state, actor, target, skill_tags)
    low, high = combatant_damage_range(skill, actor)
    return max(0, low + bonus), max(0, high + bonus)


def format_damage_range(low: int, high: int) -> str:
    if low == high:
        return str(low)
    return f"{low}-{high}"


def format_damage_label(low: int, high: int, noun: str = "damage") -> str:
    return f"{format_damage_range(low, high)} {noun}"
