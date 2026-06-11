"""Shallow Cave combat setup."""

from __future__ import annotations

from game.campaign.company import CompanyState
from game.campaign.roster import party_combatants
from game.combat.combat_state import Combatant, CombatState, Team
from game.combat.formation import Formation
from game.combat.traits import apply_combat_start_traits
from game.content.definitions import GameDefinitions


def create_shallow_cave_combat(company: CompanyState, definitions: GameDefinitions) -> CombatState:
    return create_encounter_combat(company, definitions, "shallow_cave")


def create_cave_boss_combat(company: CompanyState, definitions: GameDefinitions) -> CombatState:
    return create_encounter_combat(company, definitions, "cave_mini_boss")


def create_encounter_combat(
    company: CompanyState,
    definitions: GameDefinitions,
    encounter_id: str,
) -> CombatState:
    heroes, party_formation = party_combatants(company, definitions)
    enemy_combatants: dict[str, Combatant] = {}
    enemy_formation = Formation.empty()
    encounter = definitions.encounters[encounter_id]
    for encounter_enemy in encounter.enemies:
        definition = definitions.enemies[encounter_enemy.enemy_id]
        enemy = Combatant(
            actor_id=encounter_enemy.actor_id,
            name=definition.name,
            team=Team.ENEMY,
            max_hp=definition.max_hp,
            hp=definition.max_hp,
            speed=definition.speed,
            accuracy=definition.accuracy,
            defense=definition.defense,
            damage=definition.damage,
            max_effort=definition.max_effort,
            effort=definition.max_effort,
            skills=list(definition.skills),
            formation_slot=encounter_enemy.formation_slot,
            class_id=definition.id,
        )
        enemy_combatants[encounter_enemy.actor_id] = enemy
        enemy_formation.place(encounter_enemy.actor_id, encounter_enemy.formation_slot)
    state = CombatState(
        heroes=heroes,
        enemies=enemy_combatants,
        party_formation=party_formation,
        enemy_formation=enemy_formation,
    )
    apply_combat_start_traits(state)
    return state
