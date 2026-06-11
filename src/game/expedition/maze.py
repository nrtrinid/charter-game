"""Pandora's Maze v0.1 deterministic stub."""

from __future__ import annotations

from game.campaign.company import CompanyState
from game.campaign.roster import sync_company_from_combat
from game.combat.morale import apply_horror
from game.content.definitions import GameDefinitions
from game.core.events import GameEvent
from game.core.rng import GameRng
from game.expedition.cave import create_encounter_combat
from game.expedition.travel import (
    apply_node_rewards,
    event_for_node,
    opening_nodes,
    run_combat_to_end,
)


def run_maze_depth1(
    company: CompanyState,
    definitions: GameDefinitions,
    rng: GameRng,
    *,
    enemy_ai_mode: str = "learned_static",
    enemy_wait_mode: str = "package_only",
    enemy_movement_mode: str = "package_only",
) -> list[GameEvent]:
    nodes = opening_nodes(definitions)
    maze_node = nodes["maze_depth_1"]
    events: list[GameEvent] = [
        event_for_node(maze_node),
        event_for_node(nodes["maze_depth_1_room_1"]),
    ]
    combat = create_encounter_combat(company, definitions, maze_node.encounter or "maze_depth_1")
    events.extend(apply_horror(combat, maze_node.horror_morale_loss))
    events.extend(
        run_combat_to_end(
            combat,
            definitions,
            rng,
            encounter_id=maze_node.encounter or "maze_depth_1",
            encounter_name=maze_node.name,
            enemy_ai_mode=enemy_ai_mode,
            enemy_wait_mode=enemy_wait_mode,
            enemy_movement_mode=enemy_movement_mode,
        )
    )
    sync_company_from_combat(company, combat.heroes)
    events.extend(apply_node_rewards(company, maze_node, definitions))
    events.append(event_for_node(nodes["maze_depth_1_room_3"]))
    events.append(event_for_node(nodes["maze_retreat"]))
    return events
