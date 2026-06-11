"""Downed and death helpers."""

from __future__ import annotations

from game.combat.combat_state import Combatant, CombatState, LifeState, Team
from game.combat.morale import apply_hero_downed_morale_loss
from game.core.events import DeathEvent, GameEvent, StatusChangedEvent

MORTAL_WOUNDS_TO_DIE = 3


def is_downed(combatant: Combatant) -> bool:
    return combatant.life_state == LifeState.DOWNED


def is_dead(combatant: Combatant) -> bool:
    return combatant.life_state == LifeState.DEAD


def can_act(combatant: Combatant) -> bool:
    return combatant.can_act()


def mark_dead(state: CombatState, combatant: Combatant, reason: str) -> list[GameEvent]:
    if is_dead(combatant):
        return []
    combatant.hp = 0
    combatant.life_state = LifeState.DEAD
    events: list[GameEvent] = [
        DeathEvent(message=f"{combatant.name} dies: {reason}.", actor_id=combatant.actor_id),
        StatusChangedEvent(
            message=f"{combatant.name} is dead.",
            actor_id=combatant.actor_id,
            status=LifeState.DEAD.value,
            added=True,
        ),
    ]
    if combatant.team == Team.HERO:
        events.extend(apply_hero_downed_morale_loss(state, combatant.actor_id))
    return events


def add_mortal_wound(state: CombatState, combatant: Combatant) -> list[GameEvent]:
    combatant.mortal_wounds += 1
    events: list[GameEvent] = [
        StatusChangedEvent(
            message=(
                f"{combatant.name} gains Mortal Wound "
                f"{combatant.mortal_wounds}/{MORTAL_WOUNDS_TO_DIE}."
            ),
            actor_id=combatant.actor_id,
            status="mortal_wound",
            added=True,
        )
    ]
    if combatant.mortal_wounds >= MORTAL_WOUNDS_TO_DIE:
        events.extend(mark_dead(state, combatant, "too many Mortal Wounds"))
    return events
