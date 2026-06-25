"""App-facing view models for terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game.app.actions import (
    ActionProvider,
)
from game.app.views.dungeon import (
    DungeonActionView,
    DungeonMapNodeView,
    DungeonRoomView,
    DungeonView,
    _dungeon_action_view,
    _dungeon_node_view,
    _quest_target_node_ids,
    _room_action_visible,
)
from game.app.views.formatting import _dedupe, _join_detail
from game.campaign.company import (
    CompanyState,
    ExpeditionReportState,
)
from game.campaign.gear import (
    effective_hero_stats,
)
from game.campaign.objectives import (
    CampaignObjectiveView,
    build_campaign_objective,
)
from game.campaign.roster import active_roster, living_roster, reserve_roster
from game.combat.combat_state import LifeState
from game.content.definitions import GameDefinitions
from game.expedition.travel import (
    CAVE_REGIONAL_ID,
    HAVEN_REGIONAL_ID,
    REGIONAL_CAVE_ANCHOR_NODE_ID,
    REGIONAL_EAST_GATE_NODE_ID,
    REGIONAL_OVERWORLD_MAP_ID,
    REGIONAL_OVERWORLD_NODE_IDS,
    get_regional_node_id,
    regional_available_exit_ids,
    regional_known_exit_ids_by_node,
    regional_overworld_nodes,
    world_location_id_for_node,
)


@dataclass(frozen=True)
class WorldLocationView:
    location_id: str
    name: str
    kind: str
    description: str
    known: bool
    visit_count: int = 0
    discovered_count: int = 0
    cleared_count: int = 0
    shortcut_count: int = 0
    consumed_rumor_count: int = 0
    related_contracts: tuple[str, ...] = ()
    memory_summary: str = ""

@dataclass(frozen=True)
class WorldDestinationView:
    destination_id: str
    location_id: str
    label: str
    description: str
    available: bool = True
    route_kind: str = "travel"

@dataclass(frozen=True)
class WorldView:
    company_name: str
    current_location_id: str
    current_location_name: str
    current_location_kind: str
    current_location_description: str
    breadcrumb: str
    objective: CampaignObjectiveView
    known_locations: tuple[WorldLocationView, ...]
    travel_destinations: tuple[WorldDestinationView, ...]
    active_contracts: tuple[tuple[str, str, str], ...]
    known_routes: tuple[str, ...]

@dataclass(frozen=True)
class ArrivalBriefView:
    title: str
    location_id: str
    location_name: str
    origin_name: str
    path: tuple[str, ...]
    flavor_lines: tuple[str, ...]
    what_changed: tuple[str, ...]
    next_objective: str
    record_label: str = "Latest filed company record"

@dataclass(frozen=True)
class RegionalMapView:
    current_node_id: str
    current_node_name: str
    other_node_id: str
    other_node_name: str
    route_charted: bool
    travel_available: bool
    travel_cost: str
    travel_flavor: str | None
    objective: CampaignObjectiveView
    arrival_context: ArrivalBriefView | None
    active_contracts: tuple[tuple[str, str, str], ...]
    breadcrumb: str = ""
    location_description: str = ""
    place_text: str = ""
    anchor_kind: str = "overworld"
    current_map_id: str = REGIONAL_OVERWORLD_MAP_ID
    map_nodes: tuple[DungeonMapNodeView, ...] = ()
    exits: tuple[DungeonMapNodeView, ...] = ()
    inventory: tuple[tuple[str, int], ...] = ()
    supplies: tuple[tuple[str, int], ...] = ()
    place_title: str = ""
    opening_route_available: bool = False
    opening_route_label: str = "Take the Old Road"
    room_actions: tuple[DungeonActionView, ...] = ()

@dataclass(frozen=True)
class ShellStatusView:
    company_name: str | None
    location: str = "No company loaded"
    reputation: int = 0
    coin: int = 0
    active_count: int = 0
    reserve_count: int = 0
    wounded_count: int = 0
    downed_count: int = 0
    breaches: str = "none"
    save_status: str = "empty"
    save_path: str = ""

def build_shell_status(
    company: CompanyState | None,
    save_path: str,
    *,
    save_exists: bool,
    definitions: GameDefinitions | None = None,
) -> ShellStatusView:
    if company is None:
        return ShellStatusView(
            company_name=None,
            save_status="present" if save_exists else "empty",
            save_path=save_path,
        )
    active = active_roster(company)
    reserves = reserve_roster(company)
    living = living_roster(company)
    wounded_count = sum(
        1
        for hero in living
        if hero.hp
        < (
            effective_hero_stats(hero, definitions).max_hp
            if definitions is not None
            else hero.max_hp
        )
        or hero.mortal_wounds > 0
    )
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    return ShellStatusView(
        company_name=company.name,
        location=str(company.town_state.get("location", "Haven Town")),
        reputation=company.reputation,
        coin=company.coin,
        active_count=len(active),
        reserve_count=len(reserves),
        wounded_count=wounded_count,
        downed_count=downed_count,
        breaches=", ".join(sorted(company.known_breaches)) or "none",
        save_status="present" if save_exists else "empty",
        save_path=save_path,
    )

def build_world_view(company: CompanyState, definitions: GameDefinitions) -> WorldView:
    current_location_id = _current_world_location_id(company, definitions)
    current_location = definitions.locations[current_location_id]
    known_location_ids = _known_world_location_ids(company, definitions, current_location_id)
    objective = build_campaign_objective(company)
    active_contracts = tuple(
        (
            contract.id,
            contract.name,
            contract.summary,
        )
        for contract in sorted(
            (
                contract
                for contract in definitions.contracts.values()
                if contract.id in company.active_contract_ids
            ),
            key=lambda contract: contract.name,
        )
    )
    return WorldView(
        company_name=company.name,
        current_location_id=current_location.id,
        current_location_name=current_location.name,
        current_location_kind=current_location.kind,
        current_location_description=current_location.description,
        breadcrumb=_world_breadcrumb(current_location),
        objective=objective,
        known_locations=tuple(
            _world_location_view(
                company,
                definitions,
                location.id,
                known=location.id in known_location_ids,
            )
            for location in definitions.locations.values()
            if location.id in known_location_ids
        ),
        travel_destinations=_travel_destinations(
            company,
            definitions,
            current_location.id,
        ),
        active_contracts=active_contracts,
        known_routes=tuple(sorted(company.known_route_ids)),
    )

def build_regional_map_view(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    travel_flavor: str | None = None,
    arrival_context: ArrivalBriefView | None = None,
) -> RegionalMapView:
    nodes = regional_overworld_nodes(definitions)
    current_node_id = _current_regional_node_id(company, definitions)
    current_node = nodes[current_node_id]
    route_charted = "shallow_cave" in company.known_route_ids
    charted_hop_id = (
        CAVE_REGIONAL_ID
        if current_node_id == REGIONAL_EAST_GATE_NODE_ID
        else HAVEN_REGIONAL_ID
    )
    other_node_id = charted_hop_id
    other_location = definitions.locations[other_node_id]
    active_contracts = tuple(
        (
            contract.id,
            contract.name,
            contract.summary,
        )
        for contract in sorted(
            (
                contract
                for contract in definitions.contracts.values()
                if contract.id in company.active_contract_ids
            ),
            key=lambda contract: contract.name,
        )
    )
    memory = company.dungeon_memory.get("shallow_cave")
    visited_ids = set(memory.visited_node_ids if memory else [])
    cleared_ids = set(memory.cleared_node_ids if memory else [])
    visited_ids.add(current_node_id)
    known_ids = set(visited_ids)
    for world_memory in company.world_memory.values():
        known_ids.update(
            node_id for node_id in world_memory.discovered_node_ids if node_id in nodes
        )
    for node_id in list(known_ids):
        if node_id not in nodes:
            continue
        known_ids.update(nodes[node_id].exits)
        if memory is not None:
            prefix = f"{node_id}->"
            known_ids.update(
                revealed_exit_id.removeprefix(prefix)
                for revealed_exit_id in memory.revealed_exit_ids
                if revealed_exit_id.startswith(prefix)
            )
    known_ids = {node_id for node_id in known_ids if node_id in REGIONAL_OVERWORLD_NODE_IDS}
    known_ids.add(current_node_id)
    known_exit_ids_by_node = regional_known_exit_ids_by_node(
        definitions,
        known_ids,
        memory,
    )
    quest_target_node_ids = _quest_target_node_ids(company, None)
    map_nodes = tuple(
        _dungeon_node_view(
            node_id,
            nodes,
            company=company,
            session=None,
            current_id=current_node_id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=set(memory.completed_action_ids if memory else []),
            exit_node_ids=known_exit_ids_by_node.get(node_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for node_id in nodes
        if node_id in known_ids
    )
    walk_exit_ids = regional_available_exit_ids(company, definitions, current_node_id)
    exits = tuple(
        _dungeon_node_view(
            exit_id,
            nodes,
            company=company,
            session=None,
            current_id=current_node_id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=set(memory.completed_action_ids if memory else []),
            exit_node_ids=known_exit_ids_by_node.get(exit_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for exit_id in walk_exit_ids
    )
    if current_node_id == REGIONAL_EAST_GATE_NODE_ID:
        breadcrumb = "World > Haven > East Gate"
        place_title = "East Gate"
        anchor_kind = "east_gate"
    elif current_node_id == REGIONAL_CAVE_ANCHOR_NODE_ID:
        breadcrumb = "World > Shallow Cave"
        place_title = current_node.name
        anchor_kind = "shallow_cave"
    else:
        breadcrumb = "World > Old Road Wilds"
        place_title = current_node.name
        anchor_kind = "overworld"
    first_visit = memory is None or current_node_id not in memory.visited_node_ids
    place_text = (
        current_node.text
        if first_visit or not current_node.revisit_text
        else current_node.revisit_text
    )
    pending_regional_combat = bool(company.town_state.get("pending_regional_combat_node_id"))
    current_cleared = (
        current_node_id in cleared_ids or current_node.encounter is None
    )
    room_actions: tuple[DungeonActionView, ...] = ()
    if not pending_regional_combat:
        room_actions = tuple(
            _dungeon_action_view(
                action,
                company,
                current_node_id=current_node_id,
                current_cleared=current_cleared,
                completed_action_ids=set(memory.completed_action_ids if memory else []),
            )
            for action in current_node.actions
            if _room_action_visible(company, action)
        )
    return RegionalMapView(
        current_node_id=current_node_id,
        current_node_name=current_node.name,
        other_node_id=other_node_id,
        other_node_name=other_location.name,
        route_charted=route_charted,
        travel_available=route_charted,
        travel_cost="1 ration when available",
        travel_flavor=travel_flavor,
        objective=build_campaign_objective(company),
        arrival_context=arrival_context,
        active_contracts=active_contracts,
        breadcrumb=breadcrumb,
        location_description=current_node.text,
        place_text=place_text,
        anchor_kind=anchor_kind,
        current_map_id=REGIONAL_OVERWORLD_MAP_ID,
        map_nodes=map_nodes,
        exits=exits,
        inventory=tuple(sorted(company.inventory.items())),
        supplies=tuple(sorted(company.supplies.items())),
        place_title=place_title,
        opening_route_available=(
            current_node_id == REGIONAL_EAST_GATE_NODE_ID and not route_charted
        ),
        opening_route_label="Take the Old Road",
        room_actions=room_actions,
    )

def build_regional_render_view(regional: RegionalMapView) -> DungeonView:
    """Adapt a regional map view for dungeon-style map/minimap rendering."""
    current_node = next(
        node for node in regional.map_nodes if node.node_id == regional.current_node_id
    )
    return DungeonView(
        expedition_id="",
        dungeon_id="",
        current_map_id=regional.current_map_id,
        current_room=DungeonRoomView(
            node_id=current_node.node_id,
            name=current_node.name,
            node_type=current_node.node_type,
            text=regional.place_text or regional.location_description,
            safe_return=True,
            cleared=True,
        ),
        map_nodes=regional.map_nodes,
        exits=regional.exits,
        room_actions=(),
        actions=ActionProvider.regional_map_travel_actions(regional),
        inventory=regional.inventory,
        supplies=regional.supplies,
    )

def build_regional_arrival_context(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    origin_name: str = "",
    flavor_line: str = "",
) -> ArrivalBriefView | None:
    report = company.last_expedition_report
    if report is None:
        return None
    regional_node_id = _current_regional_node_id(company, definitions)
    nodes = regional_overworld_nodes(definitions)
    if regional_node_id in nodes:
        location_name = nodes[regional_node_id].name
        location_id = world_location_id_for_node(nodes[regional_node_id])
    else:
        location_id = _current_world_location_id(company, definitions)
        location_name = definitions.locations[location_id].name
    objective = build_campaign_objective(company)
    flavor_lines = (flavor_line,) if flavor_line else ()
    path = (origin_name, location_name) if origin_name else (location_name,)
    return ArrivalBriefView(
        title=f"Returned to {location_name}",
        location_id=location_id,
        location_name=location_name,
        origin_name=origin_name,
        path=path,
        flavor_lines=flavor_lines,
        what_changed=_arrival_change_lines(company, report),
        next_objective=objective.next_step,
    )

def _world_location_view(
    company: CompanyState,
    definitions: GameDefinitions,
    location_id: str,
    *,
    known: bool,
) -> WorldLocationView:
    location = definitions.locations[location_id]
    memory = company.world_memory.get(location_id)
    visit_count = memory.visit_count if memory is not None else 0
    discovered_count = len(memory.discovered_node_ids) if memory is not None else 0
    cleared_count = len(memory.cleared_threat_node_ids) if memory is not None else 0
    shortcut_count = len(memory.unlocked_shortcut_ids) if memory is not None else 0
    consumed_rumor_count = len(memory.consumed_rumor_ids) if memory is not None else 0
    memory_summary = _join_detail(
        f"visited {visit_count}x" if visit_count else "",
        f"{discovered_count} rooms discovered" if discovered_count else "",
        f"{cleared_count} threats cleared" if cleared_count else "",
        f"{shortcut_count} shortcuts" if shortcut_count else "",
        f"{consumed_rumor_count} rumors filed" if consumed_rumor_count else "",
    )
    return WorldLocationView(
        location_id=location.id,
        name=location.name,
        kind=location.kind,
        description=location.description,
        known=known,
        visit_count=visit_count,
        discovered_count=discovered_count,
        cleared_count=cleared_count,
        shortcut_count=shortcut_count,
        consumed_rumor_count=consumed_rumor_count,
        related_contracts=_location_contract_lines(company, definitions, location_id),
        memory_summary=memory_summary,
    )

_WORLD_LOCATION_FALLBACKS: dict[str, str] = {
    "Haven Town": "haven",
    "Haven East Gate": "old_road",
    "Old Road": "old_road",
    "Shallow Cave Entrance": "shallow_cave",
    "Cave Mouth": "shallow_cave",
    "Shallow Cave": "shallow_cave",
    "Shallow Cave Breach": "shallow_cave_breach",
    "Pandora's Maze Depth 1": "pandoras_maze_depth_1",
}


def _current_world_location_id(
    company: CompanyState,
    definitions: GameDefinitions,
) -> str:
    raw_location_id = str(company.town_state.get("location_id") or "")
    if raw_location_id in definitions.locations:
        return raw_location_id
    location_name = str(company.town_state.get("location") or "")
    for location in definitions.locations.values():
        if location.name == location_name:
            return location.id
    fallback = _WORLD_LOCATION_FALLBACKS.get(location_name)
    if fallback in definitions.locations:
        return fallback
    return definitions.world.starting_settlement

def _current_regional_node_id(
    company: CompanyState,
    definitions: GameDefinitions,
) -> str:
    node_id = get_regional_node_id(company)
    nodes = regional_overworld_nodes(definitions)
    if node_id in nodes:
        return node_id
    location_id = _current_world_location_id(company, definitions)
    if location_id == CAVE_REGIONAL_ID:
        return REGIONAL_CAVE_ANCHOR_NODE_ID
    return REGIONAL_EAST_GATE_NODE_ID

def _known_world_location_ids(
    company: CompanyState,
    definitions: GameDefinitions,
    current_location_id: str,
) -> set[str]:
    known = {definitions.world.starting_settlement, current_location_id}
    known.update(
        location_id for location_id, memory in company.world_memory.items() if memory.visited
    )
    known.update(
        contract.location_id
        for contract in definitions.contracts.values()
        if contract.id in company.active_contract_ids
        or contract.id in company.completed_contract_ids
        or (
            contract.id in company.contract_records
            and company.contract_records[contract.id].state
            in {"active", "completed", "repeatable_completed"}
        )
    )
    if "shallow_cave" in company.known_route_ids or company.flags.get(
        "shallow_cave_discovered",
        False,
    ):
        known.add("shallow_cave")
    known.update(
        breach_id for breach_id in company.known_breaches if breach_id in definitions.locations
    )
    if company.flags.get("maze_leak_confirmed", False):
        known.add("shallow_cave_breach")
    if "maze_depth_1_scouted" in company.expedition_history:
        known.add("pandoras_maze_depth_1")
    return {location_id for location_id in known if location_id in definitions.locations}

def _travel_destinations(
    company: CompanyState,
    definitions: GameDefinitions,
    current_location_id: str,
) -> tuple[WorldDestinationView, ...]:
    if current_location_id == definitions.world.starting_settlement:
        destinations = [
            _destination(
                definitions,
                "old_road",
                "Old Road",
                _join_detail(
                    "Leave through Haven East Gate and travel the wilderness route.",
                    "Cost: 1 ration when available.",
                    _contract_context(company, definitions, "old_road"),
                ),
                route_kind="road",
            )
        ]
        if "shallow_cave" in company.known_route_ids:
            destinations.append(
                _destination(
                    definitions,
                    "shallow_cave",
                    "Charted Approach: Shallow Cave",
                    (
                        "Follow the charted road to the cave mouth. This summarizes "
                        "the solved approach and spends normal route supplies."
                    ),
                    route_kind="charted_approach",
                )
            )
        return tuple(destinations)
    if current_location_id == "shallow_cave":
        return (
            _destination(
                definitions,
                definitions.world.starting_settlement,
                "Haven Town",
                "Return along the charted road to Haven.",
                route_kind="return_road",
            ),
        )
    return (
        _destination(
            definitions,
            definitions.world.starting_settlement,
            "Haven Town",
            "Return to the charter town.",
        ),
    )

def _location_contract_lines(
    company: CompanyState,
    definitions: GameDefinitions,
    location_id: str,
) -> tuple[str, ...]:
    lines: list[str] = []
    for contract in sorted(
        (
            contract
            for contract in definitions.contracts.values()
            if contract.location_id == location_id
            and (
                contract.id in company.active_contract_ids
                or contract.id in company.completed_contract_ids
                or contract.id in company.contract_records
            )
        ),
        key=lambda contract: contract.name,
    ):
        lines.append(f"{_contract_state_label(company, contract.id)}: {contract.name}")
    return tuple(lines)

def _contract_state_label(company: CompanyState, contract_id: str) -> str:
    record = company.contract_records.get(contract_id)
    if contract_id in company.active_contract_ids or (
        record is not None and record.state == "active"
    ):
        return "Active"
    if contract_id in company.completed_contract_ids or (
        record is not None and record.state in {"completed", "repeatable_completed"}
    ):
        return "Completed"
    return "Known"

def _destination(
    definitions: GameDefinitions,
    location_id: str,
    label: str,
    description: str,
    *,
    route_kind: str = "travel",
) -> WorldDestinationView:
    location = definitions.locations[location_id]
    return WorldDestinationView(
        destination_id=location.id,
        location_id=location.id,
        label=label,
        description=description,
        route_kind=route_kind,
    )

def _contract_context(
    company: CompanyState,
    definitions: GameDefinitions,
    location_id: str,
) -> str:
    contracts = [
        contract
        for contract in definitions.contracts.values()
        if contract.id in company.active_contract_ids and contract.location_id == location_id
    ]
    if not contracts:
        return ""
    return "Contract: " + "; ".join(contract.name for contract in contracts)

def _world_breadcrumb(location: Any) -> str:
    if location.id == "haven":
        return "World > Haven"
    return f"World > {location.name}"

def _arrival_change_lines(
    company: CompanyState,
    report: ExpeditionReportState,
) -> tuple[str, ...]:
    lines: list[str] = []
    for signal in report.event_signals:
        if signal.kind in {
            "breach_discovered",
            "contract_completed",
            "known_route_unlocked",
            "maze_route_collapsed",
            "maze_frontier_opened",
        }:
            lines.append(signal.message)
    if report.breaches_discovered:
        lines.append("Breach recorded: " + ", ".join(report.breaches_discovered) + ".")
    if report.reputation_gained:
        lines.append(f"Reputation +{report.reputation_gained}.")
    if report.coin_gained:
        lines.append(f"Coin +{report.coin_gained}.")
    if report.gear:
        lines.append(
            "Gear added: "
            + ", ".join(f"{gear_id} x{quantity}" for gear_id, quantity in report.gear.items())
            + "."
        )
    living = living_roster(company)
    wounded_count = sum(1 for hero in living if hero.hp < hero.max_hp or hero.mortal_wounds > 0)
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    if wounded_count:
        lines.append(f"Wounded: {wounded_count}.")
    if downed_count:
        lines.append(f"Downed: {downed_count}.")
    if company.deceased_heroes:
        lines.append(f"Memorial: {len(company.deceased_heroes)}.")
    if not lines:
        lines.append("Company record filed.")
    return tuple(_dedupe(lines))
