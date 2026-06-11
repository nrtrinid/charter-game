"""Damage and healing resolution."""

from __future__ import annotations

from game.combat.combat_state import CombatState, LifeState, Tag, Team
from game.combat.death import add_mortal_wound, mark_dead
from game.combat.effects import effort_delta_event, mitigation_effect_event
from game.combat.morale import apply_hero_downed_morale_loss
from game.combat.traits import QUIRK_KEEPS_COUNT
from game.core.events import (
    DamageEvent,
    DownedEvent,
    GameEvent,
    MemorySignalEvent,
    StatusChangedEvent,
)

GUARDED_DAMAGE_REDUCTION = 3


def apply_damage(
    state: CombatState,
    source_id: str,
    target_id: str,
    amount: int,
) -> list[GameEvent]:
    source = state.actor(source_id)
    target = state.actor(target_id)
    if not target.is_alive():
        return []

    damage = max(0, amount)
    hp_before = target.hp
    events: list[GameEvent] = []
    if damage > 0 and Tag.GUARDED in target.tags:
        target.tags.remove(Tag.GUARDED)
        blocked = min(damage, GUARDED_DAMAGE_REDUCTION)
        damage = max(0, damage - GUARDED_DAMAGE_REDUCTION)
        events.append(
            StatusChangedEvent(
                message=f"{target.name}'s Guard absorbs the blow.",
                actor_id=target_id,
                status=Tag.GUARDED.name.lower(),
                added=False,
            )
        )
        events.append(
            mitigation_effect_event(
                actor_id=target_id,
                actor_name=target.name,
                amount=blocked,
                source_kind="tag",
                source_id=Tag.GUARDED.name.lower(),
            )
        )
    events.append(
        DamageEvent(
            message=f"{source.name} deals {damage} damage to {target.name}.",
            source_id=source_id,
            target_id=target_id,
            amount=damage,
            hp_before=hp_before,
        )
    )
    if damage == 0:
        return events

    if target.team == Team.HERO and target.is_downed():
        events.extend(add_mortal_wound(state, target))
        return events

    target.hp = max(0, target.hp - damage)
    if target.hp > 0:
        return events

    if target.team == Team.HERO:
        target.life_state = LifeState.DOWNED
        events.append(DownedEvent(message=f"{target.name} is Downed.", actor_id=target_id))
        events.append(
            StatusChangedEvent(
                message=f"{target.name} cannot act or protect their lane.",
                actor_id=target_id,
                status=LifeState.DOWNED.value,
                added=True,
            )
        )
        events.extend(apply_hero_downed_morale_loss(state, target_id))
        events.extend(_apply_ally_downed_witnesses(state, target_id))
        return events

    events.extend(mark_dead(state, target, f"took {damage} damage, reduced to 0 HP"))
    return events


def _apply_ally_downed_witnesses(state: CombatState, downed_hero_id: str) -> list[GameEvent]:
    downed = state.heroes[downed_hero_id]
    events: list[GameEvent] = []
    for witness in sorted(state.heroes.values(), key=lambda hero: hero.actor_id):
        if witness.actor_id == downed_hero_id or not witness.can_act():
            continue
        memory_key = f"ally_downed_witnessed:{witness.actor_id}:{downed_hero_id}"
        if memory_key not in state.quirk_once_per_combat:
            state.quirk_once_per_combat.add(memory_key)
            events.append(
                MemorySignalEvent(
                    message="",
                    hero_id=witness.actor_id,
                    family_id="ally_downed_witnessed",
                    tags=("downed", "ally", "morale"),
                    source_summary=f"{witness.name} watched {downed.name} go down.",
                )
            )
        quirk_key = f"keeps_count:{witness.actor_id}:ally_downed"
        if (
            QUIRK_KEEPS_COUNT in witness.quirks
            and quirk_key not in state.quirk_once_per_combat
        ):
            state.quirk_once_per_combat.add(quirk_key)
            if witness.effort < witness.max_effort:
                effort_before = witness.effort
                witness.effort += 1
                events.append(
                    effort_delta_event(
                        actor_id=witness.actor_id,
                        actor_name=witness.name,
                        delta=witness.effort - effort_before,
                        before=effort_before,
                        after=witness.effort,
                        source_kind="quirk",
                        source_id=QUIRK_KEEPS_COUNT,
                        source_label="Keeps Count",
                    )
                )
    return events


def heal_combatant(state: CombatState, target_id: str, amount: int) -> list[GameEvent]:
    target = state.actor(target_id)
    if not target.is_alive() or amount <= 0:
        return []
    was_downed = target.is_downed()
    target.hp = min(target.max_hp, target.hp + amount)
    if was_downed and target.hp > 0:
        target.life_state = LifeState.ALIVE
        return [
            StatusChangedEvent(
                message=f"{target.name} is no longer Downed.",
                actor_id=target_id,
                status=LifeState.DOWNED.value,
                added=False,
            )
        ]
    return []
