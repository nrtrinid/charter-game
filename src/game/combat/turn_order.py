"""Initiative and turn ordering."""

from __future__ import annotations

from dataclasses import dataclass

from game.combat.combat_state import Combatant, CombatState, FatigueState, MoraleState
from game.core.rng import GameRng


@dataclass(frozen=True)
class InitiativeEntry:
    actor_id: str
    initiative: int


MORALE_INITIATIVE_MODIFIERS: dict[MoraleState, int] = {
    MoraleState.INSPIRED: 1,
    MoraleState.STEADY: 0,
    MoraleState.SHAKEN: -1,
    MoraleState.BROKEN: -2,
}

FATIGUE_INITIATIVE_MODIFIERS: dict[FatigueState, int] = {
    FatigueState.ENERGIZED: 1,
    FatigueState.STEADY: 0,
    FatigueState.TIRED: -1,
    FatigueState.EXHAUSTED: -2,
}


def morale_initiative_modifier(morale: MoraleState) -> int:
    return MORALE_INITIATIVE_MODIFIERS[morale]


def fatigue_initiative_modifier(fatigue: FatigueState) -> int:
    return FATIGUE_INITIATIVE_MODIFIERS[fatigue]


def initiative_modifier_for_actor(combatant: Combatant) -> int:
    return morale_initiative_modifier(combatant.morale)


def roll_initiative_for_actor(combatant: Combatant, rng: GameRng) -> int:
    return combatant.speed + rng.randint(1, 8) + initiative_modifier_for_actor(combatant)


def roll_initiative(state: CombatState, rng: GameRng) -> list[InitiativeEntry]:
    entries = [
        InitiativeEntry(
            actor_id=combatant.actor_id,
            initiative=roll_initiative_for_actor(combatant, rng),
        )
        for combatant in state.active_combatants()
    ]
    return sorted(entries, key=lambda entry: (-entry.initiative, entry.actor_id))
