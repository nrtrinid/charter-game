"""Interactive dungeon session helpers."""

from __future__ import annotations

from game.campaign.company import (
    CompanyState,
    DungeonMemoryState,
    ExpeditionReportState,
    ExpeditionSessionState,
    breach_memory,
    contract_record,
    record_world_node_cleared,
    record_world_node_discovered,
    record_world_shortcut,
)
from game.campaign.economy import add_coin
from game.campaign.memory import (
    capture_report_start,
    finalize_report_memory,
    record_report_event_signals,
)
from game.campaign.reputation import add_reputation
from game.campaign.rewards import grant_contract_rewards
from game.campaign.roster import living_roster
from game.content.definitions import GameDefinitions
from game.core.events import (
    BreachDiscoveredEvent,
    ContractCompletedEvent,
    DungeonActionEvent,
    EncounterEndedEvent,
    ExpeditionEvent,
    ExpeditionReturnedEvent,
    GameEvent,
    LootGainedEvent,
    MazeFrontierOpenedEvent,
    MazeRouteCollapsedEvent,
)
from game.core.rng import GameRng
from game.data.schemas import ExpeditionNodeDefinition
from game.expedition.expedition import OPENING_BREACH_PENDING_FLAG, OPENING_EXPEDITION_ID
from game.expedition.generated_maze import (
    GENERATED_MAZE_CONTRACT_ID,
    GENERATED_MAZE_HUNT_CONTRACT_ID,
    GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
    GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
    GENERATED_MAZE_REQUIRED_ROOMS,
    GeneratedDungeonState,
    frontier_exit_previews,
    generate_maze_breach_route,
    generated_nodes_by_id,
    resolve_generated_maze_travel,
)
from game.expedition.maze import run_maze_depth1
from game.expedition.maze_director import MazeDirectorPolicy, choose_maze_recipe
from game.expedition.travel import (
    HAVEN_REGIONAL_ID,
    apply_node_rewards,
    complete_contract_at_node,
    event_for_node,
    get_regional_node_id,
    node_event,
    opening_nodes,
    regional_node_id_for_safe_return,
    regional_node_id_for_world_location,
    regional_overworld_nodes,
    regional_return_flavor,
    set_company_location,
    set_company_node_location,
    set_regional_node_id,
    spend_ration,
    unlock_known_route_for_node,
    world_location_id_for_node,
)

SHALLOW_CAVE_DUNGEON_ID = "shallow_cave"
SHALLOW_CAVE_KNOWN_ROUTE_ID = "shallow_cave"
OPENING_START_NODE_ID = "town_gate"
OPENING_TRAVEL_NODE_IDS = (
    "known_route_road",
    "known_route_milestone",
    "known_route_creek",
    "known_route_black_stones",
    "known_route_deer_path",
    "known_route_sinkhole",
    "known_route_cave",
)
SHALLOW_CAVE_START_NODE_ID = "shallow_cave_room_1"
SHALLOW_CAVE_BOSS_NODE_ID = "maze_touched_lair"
SHALLOW_CAVE_BOSS_REWARD_NODE_ID = "cave_mini_boss"
SHALLOW_CAVE_BREACH_NODE_ID = "maze_breach"


def expedition_nodes(
    definitions: GameDefinitions,
    expedition_id: str,
) -> dict[str, ExpeditionNodeDefinition]:
    return {node.id: node for node in definitions.expeditions[expedition_id].nodes}


def active_dungeon_nodes(
    definitions: GameDefinitions,
    session: ExpeditionSessionState,
) -> dict[str, ExpeditionNodeDefinition]:
    nodes = expedition_nodes(definitions, session.expedition_id)
    nodes.update(generated_nodes_by_id(session.generated_dungeon))
    return nodes


def start_interactive_opening_dungeon(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    use_known_route: bool = True,
    skip_known_route_playback: bool = False,
) -> list[GameEvent]:
    nodes = opening_nodes(definitions)
    memory = _dungeon_memory(company, SHALLOW_CAVE_DUNGEON_ID)
    remembered_visited = _authored_node_ids(memory.visited_node_ids, nodes)
    remembered_cleared = _authored_node_ids(memory.cleared_node_ids, nodes)
    remembered_actions = list(memory.completed_action_ids)
    remembered_revealed = list(memory.revealed_exit_ids)
    events: list[GameEvent] = []
    report = ExpeditionReportState(
        expedition_id=OPENING_EXPEDITION_ID,
        dungeon_id=SHALLOW_CAVE_DUNGEON_ID,
    )
    capture_report_start(company, report)

    spend_ration(company.supplies)
    route_charted = (
        use_known_route and SHALLOW_CAVE_KNOWN_ROUTE_ID in company.known_route_ids
    )
    resume_at_cave_mouth = (
        use_known_route and SHALLOW_CAVE_START_NODE_ID in remembered_visited
    )
    if route_charted:
        for index, route_event in enumerate(_known_route_events(), start=1):
            if not skip_known_route_playback:
                events.append(route_event)
            record_room(report, f"known_route_{index}", route_event.node_id)
        if events:
            record_events(report, events)
        start_node_id = SHALLOW_CAVE_START_NODE_ID
        cleared_node_ids = list(OPENING_TRAVEL_NODE_IDS)
        visited_node_ids = list(OPENING_TRAVEL_NODE_IDS)
    elif resume_at_cave_mouth:
        start_node_id = SHALLOW_CAVE_START_NODE_ID
        cleared_node_ids = list(remembered_cleared)
        visited_node_ids = list(remembered_visited)
    else:
        start_node_id = OPENING_START_NODE_ID
        cleared_node_ids = []
        visited_node_ids = []

    start_node = nodes[start_node_id]
    first_visit = start_node.id not in remembered_visited
    set_company_node_location(company, start_node)
    start_events = [
        event_for_node(start_node, first_visit=first_visit),
        *apply_node_rewards(company, start_node, definitions),
    ]
    events.extend(start_events)
    record_room(report, start_node.id, start_node.name)
    record_events(report, start_events)

    session = ExpeditionSessionState(
        expedition_id=OPENING_EXPEDITION_ID,
        dungeon_id=SHALLOW_CAVE_DUNGEON_ID,
        current_node_id=start_node.id,
        visited_node_ids=_merge_ids(
            visited_node_ids,
            remembered_visited,
            [start_node.id],
        ),
        cleared_node_ids=_merge_ids(
            cleared_node_ids,
            remembered_cleared,
            [start_node.id],
        ),
        completed_action_ids=remembered_actions,
        revealed_exit_ids=remembered_revealed,
        report=report,
    )
    company.active_expedition = session
    _remember_visit(company, session, start_node.id)
    _remember_clear(company, session, start_node.id)
    company.last_expedition_report = None
    return events


def enter_dungeon_node(
    company: CompanyState,
    definitions: GameDefinitions,
    node_id: str,
) -> list[GameEvent]:
    session = require_active_session(company)
    nodes = active_dungeon_nodes(definitions, session)
    node = nodes[node_id]
    first_visit = node.id not in session.visited_node_ids
    previous_node_id = session.current_node_id
    if previous_node_id != node.id:
        session.previous_node_id = previous_node_id
    session.current_node_id = node.id
    if first_visit:
        session.visited_node_ids.append(node.id)
        _append_generated_visit(session, node.id)
    _remember_visit(company, session, node.id)
    set_company_node_location(company, node)
    events = [
        event_for_node(node, first_visit=first_visit),
        *apply_node_rewards(company, node, definitions),
    ]
    if session.report is not None:
        record_room(session.report, node.id, node.name)
        record_events(session.report, events)
    if node.encounter is None and node.id not in session.cleared_node_ids:
        session.cleared_node_ids.append(node.id)
        _append_generated_clear(session, node.id)
    if node.id in session.cleared_node_ids:
        _remember_clear(company, session, node.id)
        _remember_world_clear(company, session, node)
    if node.encounter is not None and node.id not in session.cleared_node_ids:
        session.pending_combat_node_id = node.id
    elif session.pending_combat_node_id == node.id and node.id in session.cleared_node_ids:
        session.pending_combat_node_id = None
    return events


def mark_pending_combat_cleared(
    company: CompanyState,
    definitions: GameDefinitions,
    encounter_id: str,
    events: list[GameEvent],
) -> None:
    session = require_active_session(company)
    nodes = active_dungeon_nodes(definitions, session)
    node_id = session.pending_combat_node_id
    if node_id is not None and node_id not in session.cleared_node_ids:
        session.cleared_node_ids.append(node_id)
        _append_generated_clear(session, node_id)
    if node_id is not None and node_id in session.cleared_node_ids:
        _remember_clear(company, session, node_id)
        node = nodes.get(node_id)
        if node is not None:
            _remember_world_clear(company, session, node)
    session.pending_combat_node_id = None
    if session.report is not None:
        if encounter_id not in session.report.encounters_resolved:
            session.report.encounters_resolved.append(encounter_id)
        record_events(session.report, events)


def use_dungeon_action(
    company: CompanyState,
    definitions: GameDefinitions,
    action_id: str,
) -> list[GameEvent]:
    session = require_active_session(company)
    nodes = active_dungeon_nodes(definitions, session)
    node = nodes[session.current_node_id]
    action = next((candidate for candidate in node.actions if candidate.id == action_id), None)
    if action is None:
        raise ValueError("Choose a listed room action.")
    if session.pending_combat_node_id is not None:
        raise ValueError("Resolve the pending room combat first.")
    if action.requires_cleared and node.id not in session.cleared_node_ids:
        raise ValueError("Clear this room before using that action.")
    action_key = room_action_key(node.id, action.id)
    if action.once and action_key in session.completed_action_ids:
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
        session.completed_action_ids.append(action_key)
        _remember_action(company, session, node.id, action_key)
        _complete_linked_room_actions(company, session, node.id, action.id)
    for exit_node_id in action.reveal_exits:
        _reveal_dungeon_exit(company, session, node.id, exit_node_id, nodes[node.id])
        if exit_node_id in nodes:
            _reveal_dungeon_exit(company, session, exit_node_id, node.id, nodes[exit_node_id])

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
    if session.report is not None:
        record_events(session.report, events)
    return events


def finish_shallow_cave_boss(
    company: CompanyState,
    definitions: GameDefinitions,
) -> list[GameEvent]:
    session = require_active_session(company)
    nodes = opening_nodes(definitions)
    events: list[GameEvent] = []

    reward_node = nodes[SHALLOW_CAVE_BOSS_REWARD_NODE_ID]
    reward_events = [
        event_for_node(reward_node),
        *apply_node_rewards(company, reward_node, definitions),
    ]
    events.extend(reward_events)

    breach_node = nodes[SHALLOW_CAVE_BREACH_NODE_ID]
    breach_events = [
        event_for_node(breach_node),
        *apply_node_rewards(company, breach_node, definitions),
    ]
    events.extend(breach_events)

    session.current_node_id = breach_node.id
    session.previous_node_id = reward_node.id
    for node_id in (reward_node.id, breach_node.id):
        if node_id not in session.visited_node_ids:
            session.visited_node_ids.append(node_id)
        if node_id not in session.cleared_node_ids:
            session.cleared_node_ids.append(node_id)
        _remember_visit(company, session, node_id)
        _remember_world_discovery(company, session, nodes[node_id])
        if node_id in session.cleared_node_ids:
            _remember_clear(company, session, node_id)
            _remember_world_clear(company, session, nodes[node_id])
        if session.report is not None:
            record_room(session.report, node_id, nodes[node_id].name)
    _reveal_dungeon_exit(
        company,
        session,
        SHALLOW_CAVE_BOSS_NODE_ID,
        breach_node.id,
        nodes[SHALLOW_CAVE_BOSS_NODE_ID],
    )
    if session.report is not None:
        record_events(session.report, events)
    company.flags[OPENING_BREACH_PENDING_FLAG] = True
    set_company_node_location(company, breach_node)
    return events


def open_opening_breach_room(
    company: CompanyState,
    definitions: GameDefinitions,
) -> list[GameEvent]:
    session = company.active_expedition
    if session is not None:
        if session.current_node_id == SHALLOW_CAVE_BREACH_NODE_ID:
            company.flags[OPENING_BREACH_PENDING_FLAG] = True
            return []
        return finish_shallow_cave_boss(company, definitions)

    nodes = opening_nodes(definitions)
    breach_node = nodes[SHALLOW_CAVE_BREACH_NODE_ID]
    events = [
        event_for_node(breach_node),
        *apply_node_rewards(company, breach_node, definitions),
    ]
    report = ExpeditionReportState(
        expedition_id=OPENING_EXPEDITION_ID,
        dungeon_id=SHALLOW_CAVE_DUNGEON_ID,
    )
    capture_report_start(company, report)
    record_room(report, breach_node.id, breach_node.name)
    record_events(report, events)
    company.active_expedition = ExpeditionSessionState(
        expedition_id=OPENING_EXPEDITION_ID,
        dungeon_id=SHALLOW_CAVE_DUNGEON_ID,
        current_node_id=breach_node.id,
        visited_node_ids=[breach_node.id],
        cleared_node_ids=[breach_node.id],
        report=report,
    )
    company.flags[OPENING_BREACH_PENDING_FLAG] = True
    set_company_node_location(company, breach_node)
    _remember_visit(company, company.active_expedition, breach_node.id)
    _remember_world_discovery(company, company.active_expedition, breach_node)
    _remember_clear(company, company.active_expedition, breach_node.id)
    _remember_world_clear(company, company.active_expedition, breach_node)
    return events


def enter_generated_maze(
    company: CompanyState,
    definitions: GameDefinitions,
    rng: GameRng,
    *,
    seed: int | None = None,
    director_policy: MazeDirectorPolicy | None = None,
) -> list[GameEvent]:
    session = require_active_session(company)
    if session.pending_combat_node_id is not None:
        raise ValueError("Resolve the pending room combat first.")
    if session.generated_dungeon is not None and not session.generated_dungeon.collapsed:
        raise ValueError("A Maze route is already active.")
    nodes = active_dungeon_nodes(definitions, session)
    source = nodes[session.current_node_id]
    if source.node_type.value != "breach":
        raise ValueError("Enter the Maze from a breach.")

    run_number = _next_generated_run_number(company, source.id)
    route_seed = seed if seed is not None else rng.randint(1, 999_999)
    recipe = choose_maze_recipe(
        company,
        source_node_id=source.id,
        run_number=run_number,
        rng=GameRng(route_seed),
        definitions=definitions,
        policy=director_policy,
    )
    generated = generate_maze_breach_route(
        run_number=run_number,
        source_node_id=source.id,
        return_node_id=source.id,
        rng=rng,
        seed=route_seed,
        recipe=recipe,
    )
    source_memory = breach_memory(company, source.id)
    source_memory.run_count = max(source_memory.run_count, run_number)
    source_memory.last_seed = route_seed
    source_memory.last_pressure_id = recipe.pressure_id
    source_memory.pressure_counts[recipe.pressure_id] = (
        source_memory.pressure_counts.get(recipe.pressure_id, 0) + 1
    )
    session.generated_dungeon = generated
    session.previous_node_id = source.id
    session.current_node_id = generated.entry_node_id
    _append_once(session.visited_node_ids, generated.entry_node_id)
    _append_once(session.cleared_node_ids, generated.entry_node_id)
    _append_generated_visit(session, generated.entry_node_id)
    _append_generated_clear(session, generated.entry_node_id)

    node = generated_nodes_by_id(generated)[generated.entry_node_id]
    set_company_node_location(company, node)
    events: list[GameEvent] = [event_for_node(node)]
    if session.report is not None:
        record_room(session.report, node.id, node.name)
        record_events(session.report, events)
    return events


def retrace_generated_maze(
    company: CompanyState,
    definitions: GameDefinitions,
) -> list[GameEvent]:
    session = require_active_session(company)
    generated = session.generated_dungeon
    if generated is None or generated.collapsed:
        raise ValueError("No active Maze route is available.")
    if session.pending_combat_node_id is not None:
        raise ValueError("Resolve the pending room combat first.")
    generated_nodes = generated_nodes_by_id(generated)
    if session.current_node_id not in generated_nodes:
        raise ValueError("Retrace is only available inside the generated Maze.")
    if session.current_node_id == generated.entry_node_id:
        raise ValueError("Already at the generated Maze threshold.")
    if session.current_node_id not in session.cleared_node_ids:
        raise ValueError("Clear this room before retracing.")

    events: list[GameEvent] = [
        node_event(
            generated.entry_node_id,
            "The company follows its marks back to the Maze threshold.",
            first_visit=False,
        )
    ]
    session.previous_node_id = session.current_node_id
    session.current_node_id = generated.entry_node_id
    set_company_node_location(company, generated_nodes[generated.entry_node_id])
    if session.report is not None:
        record_events(session.report, events)
    return events


def generated_maze_frontier_exit_ids(
    session: ExpeditionSessionState,
) -> tuple[str, ...]:
    generated = session.generated_dungeon
    if generated is None or generated.collapsed:
        return ()
    return tuple(
        preview.exit_id
        for preview in frontier_exit_previews(generated, session.current_node_id)
    )


def move_generated_maze_if_needed(
    company: CompanyState,
    definitions: GameDefinitions,
    target_node_id: str,
) -> list[GameEvent] | None:
    session = require_active_session(company)
    generated = session.generated_dungeon
    if generated is None or generated.collapsed:
        return None
    if target_node_id in generated_nodes_by_id(generated):
        return None
    if target_node_id not in generated_maze_frontier_exit_ids(session):
        return None
    if session.pending_combat_node_id is not None:
        raise ValueError("Resolve the pending room combat first.")
    if session.current_node_id not in session.cleared_node_ids:
        raise ValueError("Clear or inspect this room before moving on.")

    extension = resolve_generated_maze_travel(
        generated,
        current_node_id=session.current_node_id,
        target_node_id=target_node_id,
    )
    if extension is None:
        return None
    reported_depth = (
        extension.new_depth
        if extension.node.id.startswith(f"{generated.run_id}_room_")
        else generated.main_spine_length
    )
    frontier_event = MazeFrontierOpenedEvent(
        message=(
            f"The company opens {extension.node.name} at depth {reported_depth}. "
            "The breach holds; the route does not end here."
        ),
        run_id=generated.run_id,
        node_id=extension.node.id,
        depth=reported_depth,
    )
    events = [frontier_event, *enter_dungeon_node(company, definitions, extension.node.id)]
    if session.report is not None:
        record_events(session.report, [frontier_event])
    return events


def withdraw_generated_maze(
    company: CompanyState,
    definitions: GameDefinitions,
) -> list[GameEvent]:
    session = require_active_session(company)
    generated = session.generated_dungeon
    if generated is None or generated.collapsed:
        raise ValueError("No active Maze route is available.")
    if session.pending_combat_node_id is not None:
        raise ValueError("Resolve the pending room combat first.")
    if session.current_node_id != generated.entry_node_id:
        raise ValueError("Withdraw to Shallow Cave from the generated Maze threshold.")
    if session.current_node_id not in session.cleared_node_ids:
        raise ValueError("Clear the threshold before withdrawing.")

    rooms_visited = len(generated.visited_node_ids)
    events: list[GameEvent] = []
    events.extend(_complete_generated_maze_contracts(company, definitions, session))
    generated.collapsed = True
    history_key = f"generated_maze_route_{generated.run_id}_collapsed"
    if history_key not in company.expedition_history:
        company.expedition_history.append(history_key)
    _append_once(
        breach_memory(company, generated.source_node_id).collapsed_run_ids,
        generated.run_id,
    )
    main_depth = generated.main_spine_length
    collapse_event = MazeRouteCollapsedEvent(
        message=(
            f"{generated.run_id.replace('_', ' ').title()} held open to depth {main_depth}; "
            f"the company withdrew after {rooms_visited} rooms. The breach remains exploitable."
        ),
        run_id=generated.run_id,
        source_node_id=generated.source_node_id,
        rooms_visited=rooms_visited,
        main_depth_reached=main_depth,
    )
    events.append(collapse_event)

    session.previous_node_id = session.current_node_id
    session.current_node_id = generated.return_node_id
    set_company_node_location(
        company,
        active_dungeon_nodes(definitions, session)[session.current_node_id],
    )
    if session.report is not None:
        record_events(session.report, events)
    return events


def retreat_generated_maze(
    company: CompanyState,
    definitions: GameDefinitions,
) -> list[GameEvent]:
    return retrace_generated_maze(company, definitions)


def return_from_dungeon(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    outcome: str = "returned_to_haven",
    origin_node_id: str | None = None,
    message: str | None = None,
) -> list[GameEvent]:
    session = company.active_expedition
    if outcome == "defeat":
        regional_id = HAVEN_REGIONAL_ID
        final_message = message or "The company is carried back to Haven after defeat."
    else:
        if session is not None:
            nodes = active_dungeon_nodes(definitions, session)
            node_id = origin_node_id or session.current_node_id
            node = nodes[node_id]
            world_location_id = world_location_id_for_node(node)
            if origin_node_id is None:
                origin_node_id = node_id
        else:
            world_location_id = HAVEN_REGIONAL_ID
        regional_id = regional_node_id_for_world_location(world_location_id)
        if message is not None:
            final_message = message
        elif regional_id == HAVEN_REGIONAL_ID:
            final_message = "The company returns to Haven."
        else:
            final_message = regional_return_flavor(origin_node_id or "")
        set_regional_node_id(
            company,
            regional_node_id_for_safe_return(
                definitions,
                origin_node_id=origin_node_id,
                world_location_id=world_location_id,
            ),
        )
    location = definitions.locations[regional_id]
    set_company_location(company, regional_id, location.name)
    company.flags[OPENING_BREACH_PENDING_FLAG] = False
    if outcome == "returned_to_haven" and "returned_to_haven" not in company.expedition_history:
        company.expedition_history.append("returned_to_haven")
    event = ExpeditionReturnedEvent(
        message=final_message,
        expedition_id=OPENING_EXPEDITION_ID,
        location=location.name,
    )
    report = active_report(company)
    if report is not None:
        record_events(report, [event])
    finish_report(company, outcome)
    events: list[GameEvent] = [event]
    if outcome != "defeat":
        regional_nodes = regional_overworld_nodes(definitions)
        regional_node_id = get_regional_node_id(company)
        if regional_node_id in regional_nodes:
            events.extend(
                unlock_known_route_for_node(company, regional_nodes[regional_node_id])
            )
    return events


def descend_from_interactive_breach(
    company: CompanyState,
    definitions: GameDefinitions,
    rng: GameRng,
) -> list[GameEvent]:
    events = run_maze_depth1(company, definitions, rng)
    report = active_report(company)
    if report is not None:
        record_events(report, events)
    events.extend(
        return_from_dungeon(
            company,
            definitions,
            outcome="descended_maze_depth_1",
            message="The company returns to Haven after the first impossible descent.",
        )
    )
    return events


def finish_report(company: CompanyState, outcome: str) -> ExpeditionReportState | None:
    session = company.active_expedition
    if session is None or session.report is None:
        return None
    _remember_session_state(company, session)
    finalize_report_memory(company, session.report, outcome)
    company.last_expedition_report = session.report
    company.expedition_reports.append(session.report)
    company.active_expedition = None
    return company.last_expedition_report


def active_report(company: CompanyState) -> ExpeditionReportState | None:
    session = company.active_expedition
    return session.report if session is not None else None


def require_active_session(company: CompanyState) -> ExpeditionSessionState:
    if company.active_expedition is None:
        raise ValueError("No active dungeon expedition.")
    return company.active_expedition


def record_room(
    report: ExpeditionReportState,
    node_id: str,
    node_name: str | None = None,
) -> None:
    _append_once(report.rooms_entered, node_id)
    if node_name is not None:
        report.room_names[node_id] = node_name


def record_events(report: ExpeditionReportState, events: list[GameEvent]) -> None:
    record_report_event_signals(report, events)
    for event in events:
        if isinstance(event, LootGainedEvent):
            report.reputation_gained += event.reputation
            report.coin_gained += event.coin
            _add_counts(report.loot, event.inventory)
            _add_counts(report.supplies, event.supplies)
            _add_counts(report.gear, event.gear)
        elif isinstance(event, DungeonActionEvent):
            report.reputation_gained += event.reputation
            report.coin_gained += event.coin
            _add_counts(report.loot, event.loot)
            _add_counts(report.supplies, event.supply_rewards)
            _add_counts(report.supplies_spent, event.supply_costs)
            _append_once(report.room_actions, room_action_key(event.node_id, event.action_id))
        elif isinstance(event, BreachDiscoveredEvent):
            _append_once(report.breaches_discovered, event.breach_id)
        elif isinstance(event, EncounterEndedEvent):
            _append_once(report.encounters_resolved, event.encounter_id)


def room_action_key(node_id: str, action_id: str) -> str:
    return f"{node_id}:{action_id}"


def _complete_linked_room_actions(
    company: CompanyState,
    session: ExpeditionSessionState,
    node_id: str,
    action_id: str,
) -> None:
    linked_action_ids = {
        ("stone_gate", "unlock_black_gate"): ("force_black_gate",),
        ("stone_gate", "force_black_gate"): ("unlock_black_gate",),
    }.get((node_id, action_id), ())
    for linked_action_id in linked_action_ids:
        action_key = room_action_key(node_id, linked_action_id)
        _append_once(session.completed_action_ids, action_key)
        _remember_action(company, session, node_id, action_key)


def revealed_exit_key(node_id: str, exit_node_id: str) -> str:
    return f"{node_id}->{exit_node_id}"


def _reveal_dungeon_exit(
    company: CompanyState,
    session: ExpeditionSessionState,
    node_id: str,
    exit_node_id: str,
    node: ExpeditionNodeDefinition,
) -> None:
    exit_key = revealed_exit_key(node_id, exit_node_id)
    _append_once(session.revealed_exit_ids, exit_key)
    _remember_revealed_exit(company, session, node_id, exit_key)
    _remember_world_shortcut(company, session, node, exit_key)


def revealed_exit_node_ids(session: ExpeditionSessionState, node_id: str) -> list[str]:
    prefix = f"{node_id}->"
    return [
        revealed_exit_id.removeprefix(prefix)
        for revealed_exit_id in session.revealed_exit_ids
        if revealed_exit_id.startswith(prefix)
    ]


def _known_route_events() -> list[ExpeditionEvent]:
    return [
        node_event(
            "known_route_road",
            "The company takes the charted road out of Haven before dawn.",
        ),
        node_event(
            "known_route_milestone",
            "The repeated milestone is passed without argument this time.",
        ),
        node_event(
            "known_route_creek",
            "Boots find the dry creek bed by old chalk marks and memory.",
        ),
        node_event(
            "known_route_black_stones",
            "Black stones mark the turn south along the known approach.",
        ),
        node_event(
            "known_route_deer_path",
            "The hidden deer path cuts away the worst brambles again.",
        ),
        node_event(
            "known_route_sinkhole",
            "The sinkhole rim comes into view through low pines.",
        ),
        node_event(
            "known_route_cave",
            "The charted approach ends where the cave mouth waits ahead.",
        ),
    ]


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _merge_ids(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for value in group:
            _append_once(merged, value)
    return merged


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _dungeon_memory(company: CompanyState, dungeon_id: str) -> DungeonMemoryState:
    memory = company.dungeon_memory.get(dungeon_id)
    if memory is None:
        memory = DungeonMemoryState(dungeon_id=dungeon_id)
        company.dungeon_memory[dungeon_id] = memory
    return memory


def _authored_node_ids(
    node_ids: list[str],
    nodes: dict[str, ExpeditionNodeDefinition],
) -> list[str]:
    return [node_id for node_id in node_ids if node_id in nodes]


def _remember_session_state(
    company: CompanyState,
    session: ExpeditionSessionState,
) -> None:
    for node_id in session.visited_node_ids:
        _remember_visit(company, session, node_id)
    for node_id in session.cleared_node_ids:
        _remember_clear(company, session, node_id)
    for action_key in session.completed_action_ids:
        node_id = action_key.split(":", 1)[0]
        _remember_action(company, session, node_id, action_key)
    for exit_key in session.revealed_exit_ids:
        node_id = exit_key.split("->", 1)[0]
        _remember_revealed_exit(company, session, node_id, exit_key)


def _remember_visit(
    company: CompanyState,
    session: ExpeditionSessionState,
    node_id: str,
) -> None:
    if _is_generated_route_node(session, node_id):
        return
    _append_once(_dungeon_memory(company, session.dungeon_id).visited_node_ids, node_id)


def _remember_clear(
    company: CompanyState,
    session: ExpeditionSessionState,
    node_id: str,
) -> None:
    if _is_generated_route_node(session, node_id):
        return
    _append_once(_dungeon_memory(company, session.dungeon_id).cleared_node_ids, node_id)


def _remember_action(
    company: CompanyState,
    session: ExpeditionSessionState,
    node_id: str,
    action_key: str,
) -> None:
    if _is_generated_route_node(session, node_id):
        return
    _append_once(_dungeon_memory(company, session.dungeon_id).completed_action_ids, action_key)


def _remember_revealed_exit(
    company: CompanyState,
    session: ExpeditionSessionState,
    node_id: str,
    exit_key: str,
) -> None:
    if _is_generated_route_node(session, node_id):
        return
    _append_once(_dungeon_memory(company, session.dungeon_id).revealed_exit_ids, exit_key)


def _remember_world_discovery(
    company: CompanyState,
    session: ExpeditionSessionState,
    node: ExpeditionNodeDefinition,
) -> None:
    if _is_generated_route_node(session, node.id):
        return
    record_world_node_discovered(company, world_location_id_for_node(node), node.id)


def _remember_world_clear(
    company: CompanyState,
    session: ExpeditionSessionState,
    node: ExpeditionNodeDefinition,
) -> None:
    if _is_generated_route_node(session, node.id) or not _is_threat_node(node):
        return
    record_world_node_cleared(company, world_location_id_for_node(node), node.id)


def _remember_world_shortcut(
    company: CompanyState,
    session: ExpeditionSessionState,
    node: ExpeditionNodeDefinition,
    shortcut_id: str,
) -> None:
    if _is_generated_route_node(session, node.id):
        return
    record_world_shortcut(company, world_location_id_for_node(node), shortcut_id)


def _is_threat_node(node: ExpeditionNodeDefinition) -> bool:
    return node.encounter is not None or node.node_type.value in {"combat", "boss", "hazard"}


def _is_generated_route_node(session: ExpeditionSessionState, node_id: str) -> bool:
    if node_id.startswith("maze_run_"):
        return True
    generated = session.generated_dungeon
    return generated is not None and any(node.id == node_id for node in generated.nodes)


def _append_generated_visit(session: ExpeditionSessionState, node_id: str) -> None:
    generated = session.generated_dungeon
    if generated is not None and node_id in generated_nodes_by_id(generated):
        _append_once(generated.visited_node_ids, node_id)


def _append_generated_clear(session: ExpeditionSessionState, node_id: str) -> None:
    generated = session.generated_dungeon
    if generated is not None and node_id in generated_nodes_by_id(generated):
        _append_once(generated.cleared_node_ids, node_id)


def _next_generated_run_number(company: CompanyState, source_node_id: str) -> int:
    memory = breach_memory(company, source_node_id)
    if memory.run_count:
        return memory.run_count + 1
    legacy_count = sum(
        1
        for entry in company.expedition_history
        if entry.startswith("generated_maze_route_") and entry.endswith("_collapsed")
    )
    return legacy_count + 1


def _complete_generated_maze_contracts(
    company: CompanyState,
    definitions: GameDefinitions,
    session: ExpeditionSessionState,
) -> list[GameEvent]:
    generated = session.generated_dungeon
    if generated is None:
        return []
    events: list[GameEvent] = []
    events.extend(_complete_generated_maze_scout_contract(company, definitions, session))
    events.extend(_complete_generated_maze_hunt_contract(company, definitions, session))
    return events


def _complete_generated_maze_scout_contract(
    company: CompanyState,
    definitions: GameDefinitions,
    session: ExpeditionSessionState,
) -> list[GameEvent]:
    generated = session.generated_dungeon
    if generated is None:
        return []
    contract_ids = _active_generated_scout_contract_ids(company)
    if not contract_ids:
        return []
    if not living_roster(company):
        return []

    events: list[GameEvent] = []
    for contract_id in contract_ids:
        if not _generated_maze_contract_requirements_met(
            company,
            definitions,
            session,
            contract_id,
            default_required_rooms=GENERATED_MAZE_REQUIRED_ROOMS,
        ):
            continue
        events.extend(
            _complete_generated_maze_contract(
                company,
                definitions,
                contract_id,
                session.current_node_id,
                run_id=generated.run_id,
                rooms_scouted=_generated_actual_room_count(generated),
                hunt_cleared=False,
            )
        )
    if events:
        company.expedition_history.append(f"generated_maze_route_{generated.run_id}_scouted")
        _append_once(
            breach_memory(company, generated.source_node_id).scouted_run_ids,
            generated.run_id,
        )
    return events


def _complete_generated_maze_hunt_contract(
    company: CompanyState,
    definitions: GameDefinitions,
    session: ExpeditionSessionState,
) -> list[GameEvent]:
    generated = session.generated_dungeon
    if generated is None:
        return []
    contract_ids = _active_generated_hunt_contract_ids(company)
    if not contract_ids:
        return []
    if not living_roster(company):
        return []

    events: list[GameEvent] = []
    for contract_id in contract_ids:
        if not _generated_maze_contract_requirements_met(
            company,
            definitions,
            session,
            contract_id,
        ):
            continue
        hunt_node_id = _generated_hunt_node_id(generated)
        events.extend(
            _complete_generated_maze_contract(
                company,
                definitions,
                contract_id,
                hunt_node_id,
                run_id=generated.run_id,
                rooms_scouted=_generated_actual_room_count(generated),
                hunt_cleared=True,
            )
        )
    if events:
        company.expedition_history.append(f"generated_maze_route_{generated.run_id}_hunt")
        _append_once(
            breach_memory(company, generated.source_node_id).hunt_run_ids,
            generated.run_id,
        )
    return events


def _generated_maze_contract_requirements_met(
    company: CompanyState,
    definitions: GameDefinitions,
    session: ExpeditionSessionState,
    contract_id: str,
    *,
    default_required_rooms: int = 0,
) -> bool:
    generated = session.generated_dungeon
    if generated is None:
        return False
    contract = definitions.contracts[contract_id]
    required_rooms = (
        contract.generated_maze_required_rooms
        if contract.generated_maze_required_rooms
        else default_required_rooms
    )
    if _generated_actual_room_count(generated) < required_rooms:
        return False
    if (
        _generated_action_count(session, generated)
        < contract.generated_maze_required_action_count
    ):
        return False
    if not _generated_loot_requirements_met(session, contract.generated_maze_required_loot):
        return False
    if (
        _generated_combat_clear_count(generated)
        < contract.generated_maze_required_combat_clears
    ):
        return False
    if contract.generated_maze_requires_hunt:
        hunt_node_id = _generated_hunt_node_id(generated)
        if not hunt_node_id or hunt_node_id not in generated.cleared_node_ids:
            return False
    return bool(living_roster(company))


def _generated_actual_room_count(generated: GeneratedDungeonState) -> int:
    return sum(
        1
        for node_id in generated.visited_node_ids
        if node_id.startswith("maze_run_") and node_id != generated.entry_node_id
    )


def _generated_action_count(
    session: ExpeditionSessionState,
    generated: GeneratedDungeonState,
) -> int:
    generated_node_ids = {node.id for node in generated.nodes}
    return sum(
        1
        for action_key in session.completed_action_ids
        if action_key.split(":", 1)[0] in generated_node_ids
    )


def _generated_loot_requirements_met(
    session: ExpeditionSessionState,
    required_loot: dict[str, int],
) -> bool:
    if not required_loot:
        return True
    route_loot = session.report.loot if session.report is not None else {}
    return all(
        route_loot.get(item_id, 0) >= quantity
        for item_id, quantity in required_loot.items()
    )


def _generated_combat_clear_count(generated: GeneratedDungeonState) -> int:
    return sum(
        1
        for node in generated.nodes
        if node.encounter is not None and node.id in generated.cleared_node_ids
    )


def _generated_hunt_node_id(generated: GeneratedDungeonState) -> str:
    return next(
        (node.id for node in generated.nodes if node.id.endswith("_hunt_lair")),
        "",
    )


def _complete_generated_maze_contract(
    company: CompanyState,
    definitions: GameDefinitions,
    contract_id: str,
    node_id: str,
    *,
    run_id: str,
    rooms_scouted: int,
    hunt_cleared: bool,
) -> list[GameEvent]:
    contract = definitions.contracts[contract_id]
    company.active_contract_ids.discard(contract.id)
    record = contract_record(company, contract.id)
    record.completed_count += 1
    record.last_run_id = run_id
    record.rooms_scouted = max(record.rooms_scouted, rooms_scouted)
    record.hunt_cleared = record.hunt_cleared or hunt_cleared
    if _generated_maze_contract_is_repeatable(definitions, contract.id):
        history_id = f"{contract.id}_completed"
        company.expedition_history.append(history_id)
        record.state = "repeatable_completed"
    else:
        company.completed_contract_ids.add(contract.id)
        record.state = "completed"
    return [
        ContractCompletedEvent(
            message=f"Contract completed - {contract.name}.",
            node_id=node_id,
            contract_id=contract.id,
            name=contract.name,
        )
    ] + grant_contract_rewards(company, contract, node_id)


def _generated_maze_contract_is_repeatable(
    definitions: GameDefinitions,
    contract_id: str,
) -> bool:
    contract = definitions.contracts.get(contract_id)
    return bool(contract is not None and "repeatable" in contract.tags)


def _active_generated_scout_contract_ids(company: CompanyState) -> tuple[str, ...]:
    return tuple(
        contract_id
        for contract_id in (
            GENERATED_MAZE_CONTRACT_ID,
            GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
        )
        if contract_id in company.active_contract_ids
    )


def _active_generated_hunt_contract_ids(company: CompanyState) -> tuple[str, ...]:
    return tuple(
        contract_id
        for contract_id in (
            GENERATED_MAZE_HUNT_CONTRACT_ID,
            GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
        )
        if contract_id in company.active_contract_ids
    )
