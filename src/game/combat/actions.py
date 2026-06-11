"""Skill execution and attack resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from game.combat.combat_state import (
    CohesionState,
    Combatant,
    CombatState,
    MoraleState,
    Tag,
    Team,
    apply_marked,
)
from game.combat.damage import apply_damage, heal_combatant
from game.combat.damage_range import (
    roll_skill_base_damage,
    skill_base_damage_max,
    skill_damage_bonus,
)
from game.combat.effects import effort_delta_event
from game.combat.formation import back_slot_for, front_slot_for, is_back, is_front
from game.combat.morale import raise_morale
from game.combat.targeting import (
    AttackType,
    SkillUsableFrom,
    can_target,
    can_use_skill_from_position,
    cover_penalty,
    skill_position_unavailable_reason,
)
from game.combat.traits import (
    QUIRK_BATTLE_RHYTHM,
    QUIRK_BLOOD_HOT,
    QUIRK_CLEAN_KILL,
    QUIRK_CLOSER,
    QUIRK_HARD_LESSON,
    QUIRK_ICE_NERVES,
    QUIRK_NO_WASTE,
    QUIRK_RED_WORK,
    QUIRK_STEADY_HAND,
    QUIRK_STEADY_VOICE,
)
from game.core.events import (
    CombatEndedEvent,
    GameEvent,
    HealingEvent,
    MemorySignalEvent,
    MissEvent,
    MoveEvent,
    SkillUsedEvent,
    StatusChangedEvent,
)
from game.core.result import Result
from game.core.rng import GameRng


class SkillLike(Protocol):
    id: str
    name: str
    effort_cost: int
    attack_type: AttackType
    usable_from: SkillUsableFrom
    accuracy: int
    damage: int
    tags: list[str]


@dataclass(frozen=True)
class SkillResolutionModifiers:
    hit_modifier: int = 0
    damage_delta: int = 0
    damage_multiplier_numerator: int = 1
    damage_multiplier_denominator: int = 1
    prevent_downed: bool = False


def use_skill(
    state: CombatState,
    actor_id: str,
    skill: SkillLike,
    target_id: str,
    rng: GameRng,
    modifiers: SkillResolutionModifiers | None = None,
    ignore_target_legality: bool = False,
) -> Result[CombatState]:
    modifiers = modifiers or SkillResolutionModifiers()
    actor = state.actor(actor_id)
    if not actor.can_act():
        return Result.fail(f"{actor.name} cannot act.")
    if actor.effort < skill.effort_cost:
        return Result.fail(f"{actor.name} does not have enough Effort.")
    if not can_use_skill_from_position(state, actor_id, skill):
        reason = skill_position_unavailable_reason(state, actor_id, skill)
        return Result.fail(reason or f"{skill.name} cannot be used from this position.")

    target = state.actor(target_id)
    skill_tags = set(skill.tags)
    if "rally" in skill_tags:
        if target.team != Team.HERO:
            return Result.fail("Rally can only target an ally.")
        actor.effort -= skill.effort_cost
        rally_events: list[GameEvent] = [
            SkillUsedEvent(
                message=f"{actor.name} uses {skill.name} on {target.name}.",
                actor_id=actor_id,
                skill_id=skill.id,
                target_id=target_id,
            ),
            MemorySignalEvent(
                message=f"{actor.name} rallies {target.name}.",
                hero_id=actor_id,
                family_id="morale_rally",
                tags=("support", "morale"),
                source_summary=f"{actor.name} used {skill.name} on {target.name}.",
            )
        ]
        max_state = MoraleState.INSPIRED if "inspire" in skill_tags else MoraleState.STEADY
        if raise_morale(target, max_state=max_state):
            rally_events.append(
                StatusChangedEvent(
                    message=f"{target.name}'s Morale rises to {target.morale.name.title()}.",
                    actor_id=target_id,
                    status=target.morale.name.lower(),
                    added=True,
                )
            )
            if (
                actor_id != target_id
                and QUIRK_STEADY_VOICE in actor.quirks
                and actor.effort < actor.max_effort
            ):
                effort_before = actor.effort
                actor.effort += 1
                rally_events.append(
                    effort_delta_event(
                        actor_id=actor_id,
                        actor_name=actor.name,
                        delta=actor.effort - effort_before,
                        before=effort_before,
                        after=actor.effort,
                        source_kind="quirk",
                        source_id=QUIRK_STEADY_VOICE,
                        source_label="Steady Voice",
                    )
                )
        return Result.ok(state, rally_events)

    if "guard" in skill_tags and skill_base_damage_max(skill) <= 0:
        if target.team != actor.team:
            return Result.fail("Guard can only target an ally.")
        actor.effort -= skill.effort_cost
        target.tags.add(Tag.GUARDED)
        guard_events: list[GameEvent] = [
            SkillUsedEvent(
                message=f"{actor.name} uses {skill.name} on {target.name}.",
                actor_id=actor_id,
                skill_id=skill.id,
                target_id=target_id,
            ),
            StatusChangedEvent(
                message=f"{target.name} is Guarded.",
                actor_id=target_id,
                status=Tag.GUARDED.name.lower(),
                added=True,
            ),
            MemorySignalEvent(
                message=f"{actor.name} guarded {target.name}.",
                hero_id=actor_id,
                family_id="frontline_guard",
                tags=("guard", "support"),
                source_summary=f"{actor.name} used {skill.name} on {target.name}.",
            ),
        ]
        return Result.ok(
            state,
            guard_events,
        )

    if not ignore_target_legality and not can_target(state, actor_id, target_id, skill.attack_type):
        return Result.fail(f"{target_id} is not a legal target for {skill.name}.")

    actor.effort -= skill.effort_cost
    events: list[GameEvent] = [
        SkillUsedEvent(
            message=f"{actor.name} uses {skill.name} on {target.name}.",
            actor_id=actor_id,
            skill_id=skill.id,
            target_id=target_id,
        )
    ]

    hit_chance = (
        skill.accuracy
        + actor.accuracy
        - target.defense
        + cover_penalty(
            state,
            target_id,
            skill.attack_type,
        )
        + modifiers.hit_modifier
    )
    if not rng.chance(hit_chance):
        events.append(
            MissEvent(
                message=f"{actor.name} misses {target.name}.",
                actor_id=actor_id,
                target_id=target_id,
            )
        )
        return Result.ok(state, events)

    target_was_alive = target.is_alive()
    target_was_marked = Tag.MARKED in target.tags
    damage = max(0, roll_skill_base_damage(skill, rng) + actor.damage)
    damage += skill_damage_bonus(state, actor, target, skill_tags)
    if modifiers.damage_multiplier_denominator <= 0:
        return Result.fail("Damage multiplier denominator must be positive.")
    damage = (damage * modifiers.damage_multiplier_numerator) // (
        modifiers.damage_multiplier_denominator
    )
    damage = max(0, damage + modifiers.damage_delta)
    if (
        modifiers.prevent_downed
        and target.team == Team.HERO
        and target.is_alive()
        and not target.is_downed()
        and target.hp > 0
        and damage >= target.hp
    ):
        damage = max(0, target.hp - 1)

    target_hp_before = target.hp
    events.extend(apply_damage(state, actor_id, target_id, damage))
    if target_was_alive and not target.is_alive() and target.team == Team.ENEMY:
        family_id = "marked_execution" if target_was_marked else "killing_blow"
        display = "Marked execution" if target_was_marked else "Killing blow"
        if actor.team == Team.HERO:
            kill_tags = (
                ("kill", "combat", "marked")
                if target_was_marked
                else _killing_blow_memory_tags(
                    state,
                    actor,
                    target,
                    target_hp_before=target_hp_before,
                    skill_tags=skill_tags,
                    effort_cost=skill.effort_cost,
                )
            )
            source_summary = f"{actor.name} killed {target.name}."
            if not target_was_marked and state.is_victory():
                source_summary = f"{actor.name} ended the fight by killing {target.name}."
            events.append(
                MemorySignalEvent(
                    message=f"{display}: {actor.name} dropped {target.name}.",
                    hero_id=actor_id,
                    family_id=family_id,
                    tags=kill_tags,
                    source_summary=source_summary,
                )
            )
            if (
                target_was_marked
                and QUIRK_CLEAN_KILL in actor.quirks
                and raise_morale(actor, max_state=MoraleState.STEADY)
            ):
                events.append(
                    StatusChangedEvent(
                        message=(
                            f"{actor.name}'s Clean Kill steadies them "
                            f"to {actor.morale.name.title()}."
                        ),
                        actor_id=actor_id,
                        status=actor.morale.name.lower(),
                        added=True,
                    )
                )
            events.extend(
                _hero_enemy_kill_quirk_events(
                    state,
                    actor,
                    actor_id,
                    skill_tags=skill_tags,
                    effort_cost=skill.effort_cost,
                )
            )
    if "effort_drain" in skill_tags and target.is_alive() and target.effort > 0:
        effort_before = target.effort
        target.effort = max(0, target.effort - 1)
        events.append(
            effort_delta_event(
                actor_id=target_id,
                actor_name=target.name,
                delta=target.effort - effort_before,
                before=effort_before,
                after=target.effort,
                source_kind="skill",
                source_id=skill.id,
            )
        )
        events.append(
            StatusChangedEvent(
                message=(
                    f"{target.name} loses 1 Effort: "
                    f"{effort_before} -> {target.effort} Effort."
                ),
                actor_id=target_id,
                status="effort",
                added=False,
            )
        )
        heal_amount = min(1, actor.max_hp - actor.hp)
        if heal_amount > 0:
            events.append(
                HealingEvent(
                    message=f"{actor.name} feeds on the stolen Effort.",
                    source_id=actor_id,
                    target_id=actor_id,
                    amount=heal_amount,
                )
            )
            events.extend(heal_combatant(state, actor_id, heal_amount))
    if "drag_forward" in skill_tags and target.is_alive():
        was_marked = Tag.MARKED in target.tags
        events.extend(_drag_forward(state, target_id))
        apply_marked(target)
        if not was_marked:
            events.append(
                StatusChangedEvent(
                    message=f"{target.name} is Marked by the drag.",
                    actor_id=target_id,
                    status=Tag.MARKED.name.lower(),
                    added=True,
                )
            )
    if "shove_back" in skill_tags and target.is_alive():
        events.extend(_shove_back(state, target_id))
    if "soak" in skill_tags and target.is_alive():
        if Tag.BURNING in target.tags:
            target.tags.remove(Tag.BURNING)
            events.append(
                StatusChangedEvent(
                    message=f"{target.name}'s Burning is doused.",
                    actor_id=target_id,
                    status=Tag.BURNING.name.lower(),
                    added=False,
                )
            )
        target.tags.add(Tag.WET)
        events.append(
            StatusChangedEvent(
                message=f"{target.name} is Wet.",
                actor_id=target_id,
                status=Tag.WET.name.lower(),
                added=True,
            )
        )
    if "frost" in skill_tags and target.is_alive():
        if Tag.WET in target.tags:
            target.tags.remove(Tag.WET)
            if QUIRK_ICE_NERVES in target.quirks:
                apply_marked(target)
                events.append(
                    StatusChangedEvent(
                        message=f"{target.name}'s Ice Nerves turn the freeze into focus.",
                        actor_id=target_id,
                        status=Tag.MARKED.name.lower(),
                        added=True,
                    )
                )
            else:
                target.tags.add(Tag.FROZEN)
            if Tag.BURNING in target.tags:
                target.tags.remove(Tag.BURNING)
                events.append(
                    StatusChangedEvent(
                        message=f"{target.name}'s Burning is smothered by frost.",
                        actor_id=target_id,
                        status=Tag.BURNING.name.lower(),
                        added=False,
                    )
                )
            if QUIRK_ICE_NERVES not in target.quirks:
                events.append(
                    StatusChangedEvent(
                        message=f"{target.name} freezes solid.",
                        actor_id=target_id,
                        status=Tag.FROZEN.name.lower(),
                        added=True,
                    )
                )
                if target.team == Team.HERO:
                    events.append(
                        MemorySignalEvent(
                            message=f"{target.name} endured freezing shock.",
                            hero_id=target_id,
                            family_id="frost_shock",
                            tags=("frozen", "combat"),
                            source_summary=f"{target.name} froze solid.",
                        )
                    )
        else:
            if Tag.BURNING in target.tags:
                target.tags.remove(Tag.BURNING)
                events.append(
                    StatusChangedEvent(
                        message=f"{target.name}'s Burning is smothered by frost.",
                        actor_id=target_id,
                        status=Tag.BURNING.name.lower(),
                        added=False,
                    )
                )
            apply_marked(target)
            events.append(
                StatusChangedEvent(
                    message=f"{target.name} is Marked by frost.",
                    actor_id=target_id,
                    status=Tag.MARKED.name.lower(),
                    added=True,
            )
        )
    if "mark" in skill_tags and target.is_alive():
        was_marked = Tag.MARKED in target.tags
        apply_marked(target)
        if not was_marked:
            events.append(
                StatusChangedEvent(
                    message=f"{target.name} is Marked.",
                    actor_id=target_id,
                    status=Tag.MARKED.name.lower(),
                    added=True,
                )
            )
    if "shock" in skill_tags and target.is_alive() and Tag.WET in target.tags:
        target.tags.add(Tag.SHOCKED)
        events.append(
            StatusChangedEvent(
                message=f"{target.name} is Shocked.",
                actor_id=target_id,
                status=Tag.SHOCKED.name.lower(),
                added=True,
            )
        )
        if target.team == Team.HERO:
            events.append(
                MemorySignalEvent(
                    message=f"{target.name} endured shock.",
                    hero_id=target_id,
                    family_id="frost_shock",
                    tags=("shock", "combat"),
                    source_summary=f"{target.name} was Shocked.",
                )
            )
    if state.is_victory():
        events.append(CombatEndedEvent(message="The company wins the fight.", victor="heroes"))
    elif state.is_defeat():
        events.append(
            CombatEndedEvent(message="The company can no longer fight.", victor="enemies")
        )
    return Result.ok(state, events)


def _killing_blow_memory_tags(
    state: CombatState,
    actor: Combatant,
    target: Combatant,
    *,
    target_hp_before: int,
    skill_tags: set[str],
    effort_cost: int,
) -> tuple[str, ...]:
    tags = ["kill", "combat"]
    if state.is_victory():
        tags.append("final_kill")
    if effort_cost == 0 or "basic" in skill_tags:
        tags.append("basic")
    if actor.morale in {MoraleState.STEADY, MoraleState.INSPIRED}:
        tags.append("steady")
    if actor.morale in {MoraleState.SHAKEN, MoraleState.BROKEN}:
        tags.append("shaken")
    if state.derive_cohesion() == CohesionState.FRACTURED:
        tags.append("fractured")
    if actor.hp < actor.max_hp or actor.strain_marks:
        tags.append("wounded")
    if target_hp_before <= max(1, target.max_hp // 2):
        tags.append("low_hp")
    if effort_cost > 0:
        tags.append("effort_kill")
    if "boss" in target.class_id.lower():
        tags.append("boss")
    return tuple(tags)


def _hero_enemy_kill_quirk_events(
    state: CombatState,
    actor: Combatant,
    actor_id: str,
    *,
    skill_tags: set[str],
    effort_cost: int,
) -> list[GameEvent]:
    events: list[GameEvent] = []
    if QUIRK_BLOOD_HOT in actor.quirks and actor.effort < actor.max_effort:
        effort_before = actor.effort
        actor.effort += 1
        events.append(
            effort_delta_event(
                actor_id=actor_id,
                actor_name=actor.name,
                delta=actor.effort - effort_before,
                before=effort_before,
                after=actor.effort,
                source_kind="quirk",
                source_id=QUIRK_BLOOD_HOT,
                source_label="Blood Hot",
            )
        )

    rhythm_key = f"battle_rhythm:{actor_id}:kill"
    if (
        QUIRK_BATTLE_RHYTHM in actor.quirks
        and rhythm_key not in state.quirk_once_per_combat
    ):
        state.quirk_once_per_combat.add(rhythm_key)
        actor.tags.add(Tag.GUARDED)
        events.append(
            StatusChangedEvent(
                message=f"{actor.name}'s Battle Rhythm leaves them Guarded.",
                actor_id=actor_id,
                status=Tag.GUARDED.name.lower(),
                added=True,
            )
        )

    if state.is_victory() and QUIRK_CLOSER in actor.quirks and raise_morale(
        actor,
        max_state=MoraleState.STEADY,
    ):
        events.append(
            StatusChangedEvent(
                message=(
                    f"{actor.name}'s Closer steadies them to "
                    f"{actor.morale.name.title()}."
                ),
                actor_id=actor_id,
                status=actor.morale.name.lower(),
                added=True,
            )
        )

    is_basic = effort_cost == 0 or "basic" in skill_tags
    waste_key = f"no_waste:{actor_id}:kill"
    if (
        is_basic
        and QUIRK_NO_WASTE in actor.quirks
        and waste_key not in state.quirk_once_per_combat
        and actor.effort < actor.max_effort
    ):
        state.quirk_once_per_combat.add(waste_key)
        effort_before = actor.effort
        actor.effort += 1
        events.append(
            effort_delta_event(
                actor_id=actor_id,
                actor_name=actor.name,
                delta=actor.effort - effort_before,
                before=effort_before,
                after=actor.effort,
                source_kind="quirk",
                source_id=QUIRK_NO_WASTE,
                source_label="No Waste",
            )
        )

    steady_key = f"steady_hand:{actor_id}:kill"
    if (
        QUIRK_STEADY_HAND in actor.quirks
        and actor.morale in {MoraleState.STEADY, MoraleState.INSPIRED}
        and steady_key not in state.quirk_once_per_combat
        and raise_morale(actor, max_state=MoraleState.INSPIRED)
    ):
        state.quirk_once_per_combat.add(steady_key)
        events.append(
            StatusChangedEvent(
                message=(
                    f"{actor.name}'s Steady Hand lifts them to "
                    f"{actor.morale.name.title()}."
                ),
                actor_id=actor_id,
                status=actor.morale.name.lower(),
                added=True,
            )
        )

    if QUIRK_RED_WORK in actor.quirks and (
        actor.morale in {MoraleState.SHAKEN, MoraleState.BROKEN}
        or state.derive_cohesion() == CohesionState.FRACTURED
    ):
        if raise_morale(actor, max_state=MoraleState.STEADY):
            events.append(
                StatusChangedEvent(
                    message=(
                        f"{actor.name}'s Red Work steadies them to "
                        f"{actor.morale.name.title()}."
                    ),
                    actor_id=actor_id,
                    status=actor.morale.name.lower(),
                    added=True,
                )
            )

    wounded = actor.hp < actor.max_hp or bool(actor.strain_marks)
    lesson_key = f"hard_lesson:{actor_id}:kill"
    if (
        wounded
        and QUIRK_HARD_LESSON in actor.quirks
        and lesson_key not in state.quirk_once_per_combat
    ):
        state.quirk_once_per_combat.add(lesson_key)
        if raise_morale(actor, max_state=MoraleState.STEADY):
            events.append(
                StatusChangedEvent(
                    message=(
                        f"{actor.name}'s Hard Lesson steadies them to "
                        f"{actor.morale.name.title()}."
                    ),
                    actor_id=actor_id,
                    status=actor.morale.name.lower(),
                    added=True,
                )
            )
        elif actor.effort < actor.max_effort:
            effort_before = actor.effort
            actor.effort += 1
            events.append(
                effort_delta_event(
                    actor_id=actor_id,
                    actor_name=actor.name,
                    delta=actor.effort - effort_before,
                    before=effort_before,
                    after=actor.effort,
                    source_kind="quirk",
                    source_id=QUIRK_HARD_LESSON,
                    source_label="Hard Lesson",
                )
            )

    return events


def _drag_forward(state: CombatState, target_id: str) -> list[GameEvent]:
    target = state.actor(target_id)
    formation = state.formation_for(target.team)
    from_slot = formation.slot_of(target_id)
    if from_slot is None or not is_back(from_slot):
        return []

    to_slot = front_slot_for(from_slot)
    swapped_actor_id = formation.actor_at(to_slot)
    if not formation.swap_slots(from_slot, to_slot):
        return []

    target.formation_slot = to_slot
    if swapped_actor_id is not None:
        state.actor(swapped_actor_id).formation_slot = from_slot

    return [
        MoveEvent(
            message=f"{target.name} is dragged forward: {from_slot.value} -> {to_slot.value}.",
            actor_id=target_id,
            from_slot=from_slot.value,
            to_slot=to_slot.value,
        )
    ]


def _shove_back(state: CombatState, target_id: str) -> list[GameEvent]:
    target = state.actor(target_id)
    formation = state.formation_for(target.team)
    from_slot = formation.slot_of(target_id)
    if from_slot is None or not is_front(from_slot):
        return []

    to_slot = back_slot_for(from_slot)
    swapped_actor_id = formation.actor_at(to_slot)
    if not formation.swap_slots(from_slot, to_slot):
        return []

    target.formation_slot = to_slot
    if swapped_actor_id is not None:
        state.actor(swapped_actor_id).formation_slot = from_slot

    return [
        MoveEvent(
            message=f"{target.name} is knocked back: {from_slot.value} -> {to_slot.value}.",
            actor_id=target_id,
            from_slot=from_slot.value,
            to_slot=to_slot.value,
        )
    ]
