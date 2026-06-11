"""Individual Morale and derived party Cohesion helpers."""

from __future__ import annotations

from game.combat.combat_state import Combatant, CombatState, MoraleState
from game.core.events import GameEvent, StatusChangedEvent


def raise_morale(combatant: Combatant, *, max_state: MoraleState = MoraleState.STEADY) -> bool:
    new_value = min(combatant.morale.value + 1, max_state.value)
    if new_value == combatant.morale.value:
        return False
    combatant.morale = MoraleState(new_value)
    return True


def lower_morale(combatant: Combatant, *, min_state: MoraleState = MoraleState.BROKEN) -> bool:
    if combatant.personal_quirk == "grave_calm":
        min_state = MoraleState(max(min_state.value, MoraleState.STEADY.value))
    new_value = max(combatant.morale.value - 1, min_state.value)
    if new_value == combatant.morale.value:
        return False
    combatant.morale = MoraleState(new_value)
    return True


def break_morale(combatant: Combatant) -> bool:
    if combatant.morale == MoraleState.BROKEN:
        return False
    combatant.morale = MoraleState.BROKEN
    return True


def apply_hero_downed_morale_loss(state: CombatState, hero_id: str) -> list[GameEvent]:
    hero = state.heroes[hero_id]
    if not break_morale(hero):
        return []
    return [
        StatusChangedEvent(
            message=f"{hero.name}'s Morale breaks.",
            actor_id=hero_id,
            status=hero.morale.name.lower(),
            added=True,
        )
    ]


def apply_horror(state: CombatState, amount: int = 1) -> list[GameEvent]:
    events: list[GameEvent] = []
    if amount <= 0:
        return events
    for hero in sorted(state.heroes.values(), key=lambda candidate: candidate.actor_id):
        if not hero.is_alive():
            continue
        for _ in range(amount):
            if not lower_morale(hero):
                break
        events.append(
            StatusChangedEvent(
                message=f"{hero.name}'s Morale is {hero.morale.name.title()}.",
                actor_id=hero.actor_id,
                status=hero.morale.name.lower(),
                added=True,
            )
        )
    return events


def is_panic_risk(state: CombatState) -> bool:
    return state.derive_cohesion().name == "FRACTURED"
