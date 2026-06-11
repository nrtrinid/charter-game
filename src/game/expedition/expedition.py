"""Opening expedition vertical slice."""

from __future__ import annotations

from game.campaign.company import CompanyState
from game.campaign.roster import sync_company_from_combat
from game.content.definitions import GameDefinitions
from game.core.events import ExpeditionReturnedEvent, GameEvent
from game.core.rng import GameRng
from game.expedition.cave import create_cave_boss_combat, create_shallow_cave_combat
from game.expedition.maze import run_maze_depth1
from game.expedition.travel import (
    apply_node_rewards,
    event_for_node,
    node_event,
    opening_nodes,
    run_combat_to_end,
    set_company_location,
    set_company_node_location,
    spend_ration,
    unlock_known_route_for_node,
)

OPENING_EXPEDITION_ID = "opening"
SHALLOW_CAVE_BREACH_ID = "shallow_cave_breach"
OPENING_BREACH_PENDING_FLAG = "opening_breach_pending"


def run_opening_route_to_breach(
    company: CompanyState,
    definitions: GameDefinitions,
    rng: GameRng,
    *,
    enemy_ai_mode: str = "learned_static",
    enemy_wait_mode: str = "package_only",
    enemy_movement_mode: str = "package_only",
) -> list[GameEvent]:
    events: list[GameEvent] = []
    nodes = opening_nodes(definitions)

    spend_ration(company.supplies)
    for node_id in (
        "town_gate",
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
        "fungus_chamber",
        "old_works_cache",
    ):
        node = nodes[node_id]
        set_company_node_location(company, node)
        events.append(event_for_node(node))
        events.extend(apply_node_rewards(company, node, definitions))
        events.extend(unlock_known_route_for_node(company, node))

    cave_combat = create_shallow_cave_combat(company, definitions)
    events.extend(
        run_combat_to_end(
            cave_combat,
            definitions,
            rng,
            encounter_id="shallow_cave",
            encounter_name=nodes["old_works_cache"].name,
            enemy_ai_mode=enemy_ai_mode,
            enemy_wait_mode=enemy_wait_mode,
            enemy_movement_mode=enemy_movement_mode,
        )
    )
    sync_company_from_combat(company, cave_combat.heroes)
    if cave_combat.is_defeat():
        company.expedition_history.append("opening_failed_shallow_cave")
        set_company_location(company, "haven", "Haven Town")
        return events

    if company.inventory.get("cave_key", 0) == 0:
        company.inventory["cave_key"] = 1
        events.append(
            node_event(
                "old_works_cache",
                "A brass cave key drops from the cache wrappings.",
            )
        )

    for node_id in ("fungus_chamber", "stone_gate", "maze_touched_lair"):
        node = nodes[node_id]
        set_company_node_location(company, node)
        events.append(event_for_node(node))
        events.extend(apply_node_rewards(company, node, definitions))

    boss_combat = create_cave_boss_combat(company, definitions)
    events.extend(
        run_combat_to_end(
            boss_combat,
            definitions,
            rng,
            encounter_id="cave_mini_boss",
            encounter_name=nodes["maze_touched_lair"].name,
            enemy_ai_mode=enemy_ai_mode,
            enemy_wait_mode=enemy_wait_mode,
            enemy_movement_mode=enemy_movement_mode,
        )
    )
    sync_company_from_combat(company, boss_combat.heroes)
    if boss_combat.is_defeat():
        company.expedition_history.append("opening_failed_cave_boss")
        set_company_location(company, "haven", "Haven Town")
        return events

    events.append(event_for_node(nodes["cave_mini_boss"]))
    events.extend(apply_node_rewards(company, nodes["cave_mini_boss"], definitions))
    events.append(event_for_node(nodes["maze_breach"]))
    events.extend(apply_node_rewards(company, nodes["maze_breach"], definitions))
    company.flags[OPENING_BREACH_PENDING_FLAG] = True
    set_company_node_location(company, nodes["maze_breach"])

    return events


def return_to_haven_from_breach(
    company: CompanyState,
    definitions: GameDefinitions,
) -> list[GameEvent]:
    if not company.flags.get(OPENING_BREACH_PENDING_FLAG, False):
        return [node_event("haven_return", "The company is already back under Haven's charter.")]

    set_company_location(company, "haven", "Haven Town")
    company.flags[OPENING_BREACH_PENDING_FLAG] = False
    if "returned_to_haven" not in company.expedition_history:
        company.expedition_history.append("returned_to_haven")
    return [
        event_for_node(opening_nodes(definitions)["haven_return"]),
        ExpeditionReturnedEvent(
            message="The company returns to Haven.",
            expedition_id=OPENING_EXPEDITION_ID,
            location="Haven Town",
        ),
    ]


def descend_from_breach(
    company: CompanyState,
    definitions: GameDefinitions,
    rng: GameRng,
    *,
    enemy_ai_mode: str = "learned_static",
    enemy_wait_mode: str = "package_only",
    enemy_movement_mode: str = "package_only",
) -> list[GameEvent]:
    if not company.flags.get(OPENING_BREACH_PENDING_FLAG, False):
        return [node_event("maze_breach", "No active breach expedition is waiting.")]

    events = run_maze_depth1(
        company,
        definitions,
        rng,
        enemy_ai_mode=enemy_ai_mode,
        enemy_wait_mode=enemy_wait_mode,
        enemy_movement_mode=enemy_movement_mode,
    )
    events.extend(return_to_haven_from_breach(company, definitions))
    return events


def run_opening_route(
    company: CompanyState,
    definitions: GameDefinitions,
    rng: GameRng,
    *,
    enter_maze: bool = False,
    stop_at_breach: bool = False,
    enemy_ai_mode: str = "learned_static",
    enemy_wait_mode: str = "package_only",
    enemy_movement_mode: str = "package_only",
) -> list[GameEvent]:
    events = run_opening_route_to_breach(
        company,
        definitions,
        rng,
        enemy_ai_mode=enemy_ai_mode,
        enemy_wait_mode=enemy_wait_mode,
        enemy_movement_mode=enemy_movement_mode,
    )
    if stop_at_breach:
        return events
    if enter_maze:
        events.extend(
            descend_from_breach(
                company,
                definitions,
                rng,
                enemy_ai_mode=enemy_ai_mode,
                enemy_wait_mode=enemy_wait_mode,
                enemy_movement_mode=enemy_movement_mode,
            )
        )
        return events
    events.extend(return_to_haven_from_breach(company, definitions))
    return events
