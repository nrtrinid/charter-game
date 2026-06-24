"""Shared deterministic expedition helpers."""

from __future__ import annotations

from game.campaign.company import (
    CompanyState,
    DungeonMemoryState,
    contract_record,
    record_world_node_cleared,
    record_world_node_discovered,
    record_world_rumor_consumed,
    record_world_shortcut,
    record_world_visit,
)
from game.campaign.economy import add_coin
from game.campaign.reputation import add_reputation
from game.campaign.rewards import grant_contract_rewards
from game.combat.actions import use_skill
from game.combat.combat_state import Combatant, CombatState, Team
from game.combat.enemy_actions import (
    enemy_has_choice_after_swap,
    enemy_proactive_move,
    enemy_recovery_events,
    enemy_recovery_move,
    enemy_recovery_slots,
    enemy_wait_reason,
)
from game.combat.enemy_decision import (
    EnemyDecisionRuntimeContext,
    choose_enemy_skill_and_target,
    production_enemy_decision_policy,
)
from game.combat.formation import FormationSlot
from game.combat.targeting import can_use_skill_from_position, legal_targets
from game.combat.turn_order import roll_initiative
from game.content.definitions import GameDefinitions
from game.core.events import (
    BreachDiscoveredEvent,
    ContractCompletedEvent,
    DungeonActionEvent,
    EncounterEndedEvent,
    EncounterStartedEvent,
    ExpeditionEvent,
    GameEvent,
    LootGainedEvent,
    LoreDiscoveredEvent,
    RoundEndedEvent,
    RoundStartedEvent,
    TurnDelayedEvent,
)
from game.core.rng import GameRng
from game.data.schemas import ExpeditionNodeDefinition


def spend_ration(company_supplies: dict[str, int]) -> None:
    if company_supplies.get("rations", 0) > 0:
        company_supplies["rations"] -= 1


def node_event(
    node_id: str,
    message: str,
    *,
    first_visit: bool = True,
    major_beat: bool = False,
) -> ExpeditionEvent:
    return ExpeditionEvent(
        message=message,
        node_id=node_id,
        first_visit=first_visit,
        major_beat=major_beat,
    )


def opening_nodes(definitions: GameDefinitions) -> dict[str, ExpeditionNodeDefinition]:
    return {node.id: node for node in definitions.expeditions["opening"].nodes}


def event_for_node(
    node: ExpeditionNodeDefinition,
    *,
    first_visit: bool = True,
) -> ExpeditionEvent:
    return node_event(
        node.id,
        node.text,
        first_visit=first_visit,
        major_beat=node.major_beat,
    )


CAVE_REGIONAL_WORLD_IDS = frozenset(
    {"shallow_cave", "shallow_cave_breach", "pandoras_maze_depth_1"}
)
HAVEN_REGIONAL_ID = "haven"
CAVE_REGIONAL_ID = "shallow_cave"
REGIONAL_OVERWORLD_MAP_ID = "old_road_wilderness"
REGIONAL_EAST_GATE_NODE_ID = "town_gate"
REGIONAL_CAVE_ANCHOR_NODE_ID = "shallow_cave_entrance"
REGIONAL_OVERWORLD_NODE_IDS = frozenset(
    {
        "town_gate",
        "old_road",
        "abandoned_toll_post",
        "bandit_camp",
        "hunters_trail",
        "wolf_hollow",
        "dry_creek_bed",
        "wagon_cut",
        "carter_wreck",
        "bramble_shrine",
        "hidden_deer_path",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    }
)
REGIONAL_CHARTED_HOP_IDS = frozenset({HAVEN_REGIONAL_ID, CAVE_REGIONAL_ID})
REGIONAL_CHARTED_HOP_NODE_BY_WORLD_ID = {
    HAVEN_REGIONAL_ID: REGIONAL_EAST_GATE_NODE_ID,
    CAVE_REGIONAL_ID: REGIONAL_CAVE_ANCHOR_NODE_ID,
}
SHALLOW_CAVE_MEMORY_ID = "shallow_cave"


def regional_overworld_nodes(
    definitions: GameDefinitions,
) -> dict[str, ExpeditionNodeDefinition]:
    return {
        node.id: node
        for node in definitions.expeditions["opening"].nodes
        if node.map_id == REGIONAL_OVERWORLD_MAP_ID
    }


def get_regional_node_id(company: CompanyState) -> str:
    node_id = company.town_state.get("regional_node_id")
    if isinstance(node_id, str) and node_id in REGIONAL_OVERWORLD_NODE_IDS:
        return node_id
    return REGIONAL_EAST_GATE_NODE_ID


def set_regional_node_id(company: CompanyState, node_id: str) -> None:
    company.town_state["regional_node_id"] = node_id


def regional_node_id_for_world_location(world_location_id: str) -> str:
    if world_location_id in CAVE_REGIONAL_WORLD_IDS:
        return CAVE_REGIONAL_ID
    return HAVEN_REGIONAL_ID


def regional_node_id_for_safe_return(
    definitions: GameDefinitions,
    *,
    origin_node_id: str | None,
    world_location_id: str,
) -> str:
    if origin_node_id in REGIONAL_OVERWORLD_NODE_IDS:
        return origin_node_id
    return REGIONAL_CHARTED_HOP_NODE_BY_WORLD_ID.get(
        regional_node_id_for_world_location(world_location_id),
        REGIONAL_EAST_GATE_NODE_ID,
    )


def _regional_dungeon_memory(company: CompanyState) -> DungeonMemoryState | None:
    return company.dungeon_memory.get(SHALLOW_CAVE_MEMORY_ID)


def _memory_revealed_exit_node_ids(
    memory: DungeonMemoryState | None,
    node_id: str,
) -> list[str]:
    if memory is None:
        return []
    prefix = f"{node_id}->"
    return [
        revealed_exit_id.removeprefix(prefix)
        for revealed_exit_id in memory.revealed_exit_ids
        if revealed_exit_id.startswith(prefix)
    ]


def regional_available_exit_ids(
    company: CompanyState,
    definitions: GameDefinitions,
    node_id: str,
) -> tuple[str, ...]:
    memory = _regional_dungeon_memory(company)
    route_charted = "shallow_cave" in company.known_route_ids
    if (
        not route_charted
        and (memory is None or REGIONAL_CAVE_ANCHOR_NODE_ID not in memory.visited_node_ids)
    ):
        return ()
    nodes = regional_overworld_nodes(definitions)
    if node_id not in nodes:
        return ()
    exit_ids = list(
        dict.fromkeys(
            [
                *nodes[node_id].exits,
                *_memory_revealed_exit_node_ids(memory, node_id),
            ]
        )
    )
    return tuple(
        exit_id for exit_id in exit_ids if exit_id in REGIONAL_OVERWORLD_NODE_IDS
    )


def regional_known_exit_ids_by_node(
    definitions: GameDefinitions,
    known_ids: set[str],
    memory: DungeonMemoryState | None,
) -> dict[str, tuple[str, ...]]:
    """Map-display exits between known regional nodes (ignores walk unlock rules)."""
    nodes = regional_overworld_nodes(definitions)
    result: dict[str, tuple[str, ...]] = {}
    for node_id in known_ids:
        if node_id not in nodes:
            continue
        exit_ids = list(
            dict.fromkeys(
                [
                    *nodes[node_id].exits,
                    *_memory_revealed_exit_node_ids(memory, node_id),
                ]
            )
        )
        result[node_id] = tuple(
            exit_id
            for exit_id in exit_ids
            if exit_id in known_ids and exit_id in nodes
        )
    return result


def _regional_node_cleared(
    company: CompanyState,
    node_id: str,
    nodes: dict[str, ExpeditionNodeDefinition],
) -> bool:
    node = nodes[node_id]
    if node.encounter is None:
        return True
    memory = _regional_dungeon_memory(company)
    return memory is not None and node_id in memory.cleared_node_ids


def _remember_regional_visit(company: CompanyState, node_id: str) -> None:
    memory = company.dungeon_memory.setdefault(
        SHALLOW_CAVE_MEMORY_ID,
        DungeonMemoryState(dungeon_id=SHALLOW_CAVE_MEMORY_ID),
    )
    if node_id not in memory.visited_node_ids:
        memory.visited_node_ids.append(node_id)


def _remember_regional_clear(company: CompanyState, node_id: str) -> None:
    memory = company.dungeon_memory.setdefault(
        SHALLOW_CAVE_MEMORY_ID,
        DungeonMemoryState(dungeon_id=SHALLOW_CAVE_MEMORY_ID),
    )
    if node_id not in memory.cleared_node_ids:
        memory.cleared_node_ids.append(node_id)


def _remember_regional_world_clear(
    company: CompanyState,
    node: ExpeditionNodeDefinition,
) -> None:
    if node.encounter is None:
        return
    record_world_node_cleared(company, world_location_id_for_node(node), node.id)


def move_regional_node(
    company: CompanyState,
    definitions: GameDefinitions,
    node_id: str,
) -> list[GameEvent]:
    if company.active_expedition is not None:
        raise ValueError("Finish the active expedition first.")
    if node_id not in REGIONAL_OVERWORLD_NODE_IDS:
        raise ValueError("Choose a listed regional exit.")
    nodes = regional_overworld_nodes(definitions)
    current_id = get_regional_node_id(company)
    if node_id == current_id:
        raise ValueError("The company is already there.")
    available = regional_available_exit_ids(company, definitions, current_id)
    if node_id not in available:
        raise ValueError("Choose a listed regional exit.")
    if not _regional_node_cleared(company, current_id, nodes):
        raise ValueError("Clear or resolve this area before moving on.")

    destination = nodes[node_id]
    memory = _regional_dungeon_memory(company)
    first_visit = memory is None or node_id not in memory.visited_node_ids
    set_regional_node_id(company, node_id)
    set_company_node_location(company, destination)
    _remember_regional_visit(company, node_id)
    events: list[GameEvent] = [
        event_for_node(destination, first_visit=first_visit),
        *apply_node_rewards(company, destination, definitions),
        *unlock_known_route_for_node(company, destination),
    ]
    if destination.encounter is not None and not _regional_node_cleared(
        company,
        node_id,
        nodes,
    ):
        company.town_state["pending_regional_combat_node_id"] = node_id
    else:
        _remember_regional_clear(company, node_id)
        _remember_regional_world_clear(company, destination)
    return events


def regional_move_lands_on_place(
    node: ExpeditionNodeDefinition,
    *,
    first_visit: bool,
) -> bool:
    if not first_visit:
        return False
    if node.id in {REGIONAL_EAST_GATE_NODE_ID, REGIONAL_CAVE_ANCHOR_NODE_ID}:
        return True
    if node.major_beat or node.known_route_unlock is not None:
        return True
    if node.node_type.value == "hazard":
        return True
    return False


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _remember_regional_action(company: CompanyState, action_key: str) -> None:
    memory = company.dungeon_memory.setdefault(
        SHALLOW_CAVE_MEMORY_ID,
        DungeonMemoryState(dungeon_id=SHALLOW_CAVE_MEMORY_ID),
    )
    _append_once(memory.completed_action_ids, action_key)


def _reveal_regional_exit(
    company: CompanyState,
    definitions: GameDefinitions,
    node_id: str,
    exit_node_id: str,
) -> None:
    from game.expedition.dungeon import revealed_exit_key

    nodes = regional_overworld_nodes(definitions)
    memory = company.dungeon_memory.setdefault(
        SHALLOW_CAVE_MEMORY_ID,
        DungeonMemoryState(dungeon_id=SHALLOW_CAVE_MEMORY_ID),
    )
    for source_id, destination_id in ((node_id, exit_node_id), (exit_node_id, node_id)):
        if destination_id not in nodes:
            continue
        exit_key = revealed_exit_key(source_id, destination_id)
        _append_once(memory.revealed_exit_ids, exit_key)
        record_world_shortcut(
            company,
            world_location_id_for_node(nodes[source_id]),
            exit_key,
        )


def use_regional_action(
    company: CompanyState,
    definitions: GameDefinitions,
    action_id: str,
) -> list[GameEvent]:
    from game.expedition.dungeon import room_action_key

    if company.active_expedition is not None:
        raise ValueError("Finish the active expedition first.")
    if company.town_state.get("pending_regional_combat_node_id"):
        raise ValueError("Resolve the pending room combat first.")
    nodes = regional_overworld_nodes(definitions)
    node_id = get_regional_node_id(company)
    node = nodes[node_id]
    action = next((candidate for candidate in node.actions if candidate.id == action_id), None)
    if action is None:
        raise ValueError("Choose a listed room action.")
    memory = _regional_dungeon_memory(company)
    cleared_ids = set(memory.cleared_node_ids if memory else [])
    completed_action_ids = set(memory.completed_action_ids if memory else [])
    if action.requires_cleared and node_id not in cleared_ids:
        raise ValueError("Clear this room before using that action.")
    action_key = room_action_key(node.id, action.id)
    if action.once and action_key in completed_action_ids:
        raise ValueError("That room action has already been resolved.")
    for supply_id, quantity in action.supply_costs.items():
        if company.supplies.get(supply_id, 0) < quantity:
            raise ValueError(f"Not enough {supply_id} for that room action.")
    for item_id, quantity in action.inventory_requirements.items():
        if company.inventory.get(item_id, 0) < quantity:
            raise ValueError(f"Missing required item: {item_id}.")
    for contract_id in action.requires_active_contracts:
        if contract_id not in company.active_contract_ids:
            raise ValueError("That room action requires an active contract.")

    for supply_id, quantity in action.supply_costs.items():
        company.supplies[supply_id] = company.supplies.get(supply_id, 0) - quantity
    for supply_id, quantity in action.supply_rewards.items():
        company.supplies[supply_id] = company.supplies.get(supply_id, 0) + quantity
    for item_id, quantity in action.loot.items():
        company.inventory[item_id] = company.inventory.get(item_id, 0) + quantity
    if action.reputation_reward:
        add_reputation(company, action.reputation_reward)
    if action.coin_reward:
        add_coin(company, action.coin_reward)
    for flag_id, value in action.flags_set.items():
        company.flags[flag_id] = value
    if action.once:
        _remember_regional_action(company, action_key)
    for exit_node_id in action.reveal_exits:
        _reveal_regional_exit(company, definitions, node.id, exit_node_id)

    events: list[GameEvent] = [
        DungeonActionEvent(
            message=action.result_text,
            node_id=node.id,
            action_id=action.id,
            label=action.label,
            supply_costs=dict(action.supply_costs),
            supply_rewards=dict(action.supply_rewards),
            loot=dict(action.loot),
            reputation=action.reputation_reward,
            coin=action.coin_reward,
        )
    ]
    if action.complete_contract is not None:
        events.extend(
            complete_contract_at_node(
                company,
                definitions,
                contract_id=action.complete_contract,
                node_id=node.id,
            )
        )
    if action.history is not None and action.history not in company.expedition_history:
        company.expedition_history.append(action.history)
    return events


def mark_regional_combat_cleared(
    company: CompanyState,
    definitions: GameDefinitions,
) -> None:
    node_id = company.town_state.pop("pending_regional_combat_node_id", None)
    if node_id is None:
        return
    nodes = regional_overworld_nodes(definitions)
    node = nodes.get(node_id)
    if node is None:
        return
    _remember_regional_clear(company, node_id)
    _remember_regional_world_clear(company, node)


def regional_travel_flavor(*, origin_id: str, destination_id: str) -> str:
    origin_anchor = REGIONAL_CHARTED_HOP_NODE_BY_WORLD_ID.get(origin_id, origin_id)
    destination_anchor = REGIONAL_CHARTED_HOP_NODE_BY_WORLD_ID.get(
        destination_id,
        destination_id,
    )
    if (
        origin_anchor == REGIONAL_EAST_GATE_NODE_ID
        and destination_anchor == REGIONAL_CAVE_ANCHOR_NODE_ID
    ):
        return "The company follows the charted road, bypassing cleared stretches of Old Road."
    if (
        origin_anchor == REGIONAL_CAVE_ANCHOR_NODE_ID
        and destination_anchor == REGIONAL_EAST_GATE_NODE_ID
    ):
        return "The company follows the charted road, bypassing cleared stretches of Old Road."
    destination_name = destination_id.replace("_", " ").title()
    return f"The company travels toward {destination_name}."


def regional_return_flavor(origin_node_id: str) -> str:
    return {
        "town_gate": "The company turns back through the town gate.",
        "shallow_cave_entrance": "The company leaves the cave entrance and regroups in open air.",
        "shallow_cave_room_1": "The company climbs out through the cave mouth.",
        "stone_gate": "The company withdraws from the black gate to the cave mouth.",
        "maze_breach": "The company steps away from the breach to safer ground.",
    }.get(origin_node_id, "The company withdraws to the regional staging ground.")


def set_company_location(company: CompanyState, location_id: str, location_name: str) -> None:
    previous_location_id = str(company.town_state.get("location_id") or "")
    company.town_state["location_id"] = location_id
    company.town_state["location"] = location_name
    if previous_location_id != location_id:
        record_world_visit(company, location_id)


def set_company_node_location(
    company: CompanyState,
    node: ExpeditionNodeDefinition,
) -> None:
    location_id = world_location_id_for_node(node)
    previous_location_id = str(company.town_state.get("location_id") or "")
    company.town_state["location_id"] = location_id
    company.town_state["location"] = node.name
    if _is_generated_node_id(node.id):
        return
    if previous_location_id != location_id:
        record_world_visit(company, location_id)
    record_world_node_discovered(company, location_id, node.id)


def unlock_known_route_for_node(
    company: CompanyState,
    node: ExpeditionNodeDefinition,
) -> list[GameEvent]:
    if node.known_route_unlock is None:
        return []
    if node.known_route_unlock in company.known_route_ids:
        return []
    company.known_route_ids.add(node.known_route_unlock)
    record_world_shortcut(
        company,
        world_location_id_for_node(node),
        node.known_route_unlock,
    )
    return [
        node_event(
            node.id,
            f"Charted approach mapped: {node.known_route_unlock.replace('_', ' ')}.",
        )
    ]


def mark_regional_charted_route(
    company: CompanyState,
    definitions: GameDefinitions,
) -> list[GameEvent]:
    if company.active_expedition is not None:
        raise ValueError("Finish the active expedition first.")
    if get_regional_node_id(company) != REGIONAL_CAVE_ANCHOR_NODE_ID:
        raise ValueError("Mark the route from the Shallow Cave entrance.")
    nodes = regional_overworld_nodes(definitions)
    node = nodes[REGIONAL_CAVE_ANCHOR_NODE_ID]
    if node.known_route_unlock in company.known_route_ids:
        raise ValueError("This route is already charted.")
    return unlock_known_route_for_node(company, node)


def apply_node_rewards(
    company: CompanyState,
    node: ExpeditionNodeDefinition,
    definitions: GameDefinitions | None = None,
) -> list[GameEvent]:
    if node.history is not None and node.history in company.expedition_history:
        return []
    events: list[GameEvent] = []
    contract = (
        definitions.contracts.get(node.complete_contract)
        if definitions is not None and node.complete_contract is not None
        else None
    )
    contract_rewards_reputation = bool(
        contract is not None and contract.reward_reputation > 0
    )
    if node.reputation_reward and not contract_rewards_reputation:
        add_reputation(company, node.reputation_reward)
    if node.coin_reward:
        add_coin(company, node.coin_reward)
    if node.breach_id is not None:
        company.known_breaches.add(node.breach_id)
    for item_id, quantity in node.loot.items():
        company.inventory[item_id] = company.inventory.get(item_id, 0) + quantity
    for supply_id, quantity in node.supply_rewards.items():
        company.supplies[supply_id] = company.supplies.get(supply_id, 0) + quantity
    for flag_id, value in node.flags_set.items():
        company.flags[flag_id] = value
    for lore_id in node.lore_entries:
        if lore_id not in company.known_lore_entries:
            company.known_lore_entries.add(lore_id)
            record_world_rumor_consumed(company, world_location_id_for_node(node), lore_id)
            events.append(_lore_event(node.id, lore_id, definitions))
    if node.complete_contract is not None:
        events.extend(
            complete_contract_at_node(
                company,
                definitions,
                contract_id=node.complete_contract,
                node_id=node.id,
            )
        )
    if node.history is not None:
        company.expedition_history.append(node.history)
    node_reputation_reward = (
        0 if contract_rewards_reputation else node.reputation_reward
    )
    if node.reputation_reward or node.coin_reward or node.loot or node.supply_rewards:
        events.append(
            LootGainedEvent(
                message=_reward_message(node, reputation=node_reputation_reward),
                node_id=node.id,
                inventory=dict(node.loot),
                supplies=dict(node.supply_rewards),
                reputation=node_reputation_reward,
                coin=node.coin_reward,
            )
        )
    if node.breach_id is not None:
        events.append(
            BreachDiscoveredEvent(
                message=f"Breach discovered: {node.breach_id}.",
                node_id=node.id,
                breach_id=node.breach_id,
            )
        )
    return events


def world_location_id_for_node(node: ExpeditionNodeDefinition) -> str:
    if node.map_id == "haven_town" or node.node_type.value == "town":
        return "haven"
    if node.map_id == "old_road_wilderness":
        if node.id in {"town_gate", "old_road", "abandoned_toll_post", "bandit_camp"}:
            return "old_road"
        if node.id in {"hunters_trail", "wolf_hollow", "dry_creek_bed", "bramble_shrine"}:
            return "blackwood_forest"
        return "shallow_cave"
    if node.map_id == "shallow_cave":
        return "shallow_cave"
    if node.map_id == "shallow_cave_breach":
        return "shallow_cave_breach"
    if node.map_id == "maze_depth_1":
        return "pandoras_maze_depth_1"
    return node.map_id


def _is_generated_node_id(node_id: str) -> bool:
    return node_id.startswith("maze_run_")


def run_combat_to_end(
    state: CombatState,
    definitions: GameDefinitions,
    rng: GameRng,
    max_rounds: int = 20,
    *,
    encounter_id: str = "combat",
    encounter_name: str = "Combat Encounter",
    enemy_ai_mode: str = "learned_static",
    enemy_wait_mode: str = "package_only",
    enemy_movement_mode: str = "package_only",
) -> list[GameEvent]:
    events: list[GameEvent] = [
        EncounterStartedEvent(
            message=f"{encounter_name} begins.",
            encounter_id=encounter_id,
            encounter_name=encounter_name,
            actor_ids=sorted(state.all_combatants()),
        )
    ]
    while not state.is_victory() and not state.is_defeat() and state.round_number <= max_rounds:
        initiative = roll_initiative(state, rng)
        events.append(
            RoundStartedEvent(
                message=f"Round {state.round_number} begins.",
                encounter_id=encounter_id,
                round_number=state.round_number,
                actor_ids=[entry.actor_id for entry in initiative],
            )
        )
        waited_actor_ids: set[str] = set()
        turn_index = 0
        while turn_index < len(initiative):
            entry = initiative[turn_index]
            actor = state.actor(entry.actor_id)
            if not actor.can_act() or state.is_victory() or state.is_defeat():
                turn_index += 1
                continue
            if actor.team == Team.ENEMY:
                initiative_actor_ids = tuple(entry.actor_id for entry in initiative)
                runtime_context = EnemyDecisionRuntimeContext(
                    initiative_actor_ids=initiative_actor_ids,
                    current_turn_index=turn_index,
                )
                wait_reason = enemy_wait_reason(
                    state,
                    definitions,
                    actor,
                    runtime_context,
                    enemy_wait_mode,
                    waited_actor_ids,
                )
                if wait_reason is not None and turn_index + 1 < len(initiative):
                    delayed_entry = initiative.pop(turn_index)
                    initiative.append(delayed_entry)
                    waited_actor_ids.add(actor.actor_id)
                    events.append(
                        TurnDelayedEvent(
                            message=f"{actor.name} waits for {wait_reason}.",
                            actor_id=actor.actor_id,
                            encounter_id=encounter_id,
                        )
                    )
                    continue
                movement_events = enemy_proactive_move(
                    state,
                    definitions,
                    actor,
                    enemy_movement_mode,
                    runtime_context,
                )
                if movement_events:
                    events.extend(movement_events)
                    turn_index += 1
                    continue
                skill_and_target = choose_enemy_skill_and_target(
                    state,
                    definitions,
                    actor.actor_id,
                    runtime_context,
                    policy=production_enemy_decision_policy(enemy_ai_mode),
                )
            else:
                skill_and_target = _first_usable_skill_and_target(
                    state,
                    definitions,
                    actor.actor_id,
                )
            if skill_and_target is None:
                if actor.team == Team.ENEMY:
                    events.extend(
                        enemy_recovery_events(
                            state,
                            definitions,
                            actor,
                            enemy_movement_mode,
                        )
                    )
                turn_index += 1
                continue
            skill_id, target_id = skill_and_target
            result = use_skill(state, actor.actor_id, definitions.skills[skill_id], target_id, rng)
            events.extend(result.events)
            turn_index += 1
        events.append(
            RoundEndedEvent(
                message=f"Round {state.round_number} ends.",
                encounter_id=encounter_id,
                round_number=state.round_number,
            )
        )
        state.round_number += 1
    if state.is_victory():
        events.append(
            EncounterEndedEvent(
                message=f"{encounter_name} ends in victory.",
                encounter_id=encounter_id,
                victor="heroes",
            )
        )
    elif state.is_defeat():
        events.append(
            EncounterEndedEvent(
                message=f"{encounter_name} ends in defeat.",
                encounter_id=encounter_id,
                victor="enemies",
            )
        )
    else:
        events.append(node_event("combat", "The fight stalls and the company withdraws."))
        events.append(
            EncounterEndedEvent(
                message=f"{encounter_name} ends in withdrawal.",
                encounter_id=encounter_id,
                victor="withdrawal",
            )
        )
    return events


def _first_usable_skill_and_target(
    state: CombatState,
    definitions: GameDefinitions,
    actor_id: str,
) -> tuple[str, str] | None:
    actor = state.actor(actor_id)
    for skill_id in actor.skills:
        skill = definitions.skills[skill_id]
        if actor.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, actor_id, skill):
            continue
        targets = legal_targets(state, actor_id, skill.attack_type)
        if targets:
            return skill_id, sorted(targets)[0]
    return None


def _enemy_position_recovery_move(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
) -> list[GameEvent]:
    return list(enemy_recovery_move(state, definitions, actor))


def _enemy_recovery_slots(from_slot: FormationSlot) -> list[FormationSlot]:
    return enemy_recovery_slots(from_slot)


def _enemy_has_choice_after_swap(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
    from_slot: FormationSlot,
    to_slot: FormationSlot,
) -> bool:
    return enemy_has_choice_after_swap(state, definitions, actor, from_slot, to_slot)


def first_usable_skill_and_target(
    state: CombatState,
    definitions: GameDefinitions,
    actor_id: str,
) -> tuple[str, str] | None:
    return _first_usable_skill_and_target(state, definitions, actor_id)


def _reward_message(node: ExpeditionNodeDefinition, *, reputation: int | None = None) -> str:
    pieces: list[str] = []
    reputation_reward = node.reputation_reward if reputation is None else reputation
    if reputation_reward:
        pieces.append(f"{reputation_reward} reputation")
    if node.coin_reward:
        pieces.append(f"{node.coin_reward} Coin")
    pieces.extend(f"{quantity} {item_id}" for item_id, quantity in sorted(node.loot.items()))
    pieces.extend(
        f"{quantity} {supply_id}" for supply_id, quantity in sorted(node.supply_rewards.items())
    )
    return "Gained " + ", ".join(pieces) + "."


def _lore_event(
    node_id: str,
    lore_id: str,
    definitions: GameDefinitions | None,
) -> LoreDiscoveredEvent:
    rumor = definitions.rumors.get(lore_id) if definitions is not None else None
    if rumor is None:
        return LoreDiscoveredEvent(
            message=f"Lore recorded: {lore_id}.",
            node_id=node_id,
            lore_id=lore_id,
            title=lore_id,
        )
    return LoreDiscoveredEvent(
        message=f"Rumor recorded - {rumor.title}: {rumor.text}",
        node_id=node_id,
        lore_id=lore_id,
        title=rumor.title,
    )


def complete_contract_at_node(
    company: CompanyState,
    definitions: GameDefinitions | None,
    *,
    contract_id: str,
    node_id: str,
) -> list[GameEvent]:
    if contract_id in company.completed_contract_ids:
        return []
    company.completed_contract_ids.add(contract_id)
    company.active_contract_ids.discard(contract_id)
    record = contract_record(company, contract_id)
    record.state = "completed"
    record.completed_count += 1
    events: list[GameEvent] = [
        _contract_event(node_id, contract_id, definitions),
    ]
    if definitions is not None:
        contract = definitions.contracts.get(contract_id)
        if contract is not None:
            events.extend(grant_contract_rewards(company, contract, node_id))
    return events


def _contract_event(
    node_id: str,
    contract_id: str,
    definitions: GameDefinitions | None,
) -> ContractCompletedEvent:
    contract = definitions.contracts.get(contract_id) if definitions is not None else None
    if contract is None:
        return ContractCompletedEvent(
            message=f"Contract completed: {contract_id}.",
            node_id=node_id,
            contract_id=contract_id,
            name=contract_id,
        )
    return ContractCompletedEvent(
        message=f"Contract completed - {contract.name}.",
        node_id=node_id,
        contract_id=contract_id,
        name=contract.name,
    )
