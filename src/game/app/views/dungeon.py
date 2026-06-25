"""App-facing view models for terminal rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from game.app.actions import (
    ActionProvider,
    ScreenAction,
)
from game.app.views.art import _art_lines, _dungeon_node_art_asset
from game.app.views.constants import (
    BLACKWOOD_BOSS_TARGET_NODE_ID,
    BLACKWOOD_CAVE_TARGET_NODE_ID,
    BREACH_TARGET_NODE_ID,
    QUEST_PICKUP_NODE_ID,
)
from game.app.views.formatting import _dedupe, _dedupe_lines, _join_detail, _signed
from game.campaign.company import (
    CompanyState,
    ExpeditionReportState,
    HeroReportOutcome,
)
from game.campaign.objectives import (
    BLACKWOOD_ROAD_CHARTER_ID,
    BREACH_STALKER_HUNT_ID,
    SHALLOW_CAVE_BREACH_SCOUT_ID,
    CampaignObjectiveView,
    build_campaign_objective,
)
from game.campaign.roster import living_roster
from game.combat.combat_state import LifeState
from game.content.definitions import GameDefinitions
from game.core.events import GameEvent
from game.data.schemas import ExpeditionNodeDefinition, ExpeditionRoomActionDefinition
from game.expedition.dungeon import (
    SHALLOW_CAVE_BREACH_NODE_ID,
    active_dungeon_nodes,
    revealed_exit_node_ids,
    room_action_key,
)
from game.expedition.expedition import OPENING_BREACH_PENDING_FLAG
from game.expedition.generated_maze import (
    GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
    GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
    FrontierExitPreview,
    frontier_exit_previews,
    frontier_node_id,
    frontier_preview_map_positions,
    generated_nodes_by_id,
    is_generated_spur_room,
    is_main_spine_room,
)
from game.expedition.travel import (
    world_location_id_for_node,
)
from game.ui.wounds import mortal_wound_badge


@dataclass(frozen=True)
class DungeonRoomView:
    node_id: str
    name: str
    node_type: str
    text: str
    safe_return: bool
    cleared: bool
    scene_state: str = ""
    route_hint: str = ""
    party_hint: str = ""
    first_visit: bool = False
    major_beat: bool = False
    art_lines: tuple[str, ...] = ()

@dataclass(frozen=True)
class DungeonMapNodeView:
    node_id: str
    name: str
    node_type: str
    status: str
    current: bool
    known: bool
    visited: bool
    cleared: bool
    safe_return: bool
    map_id: str = ""
    exit_node_ids: tuple[str, ...] = ()
    map_x: int | None = None
    map_y: int | None = None
    direction: str = ""
    location_id: str = ""
    memory_summary: str = ""
    memory_notes: tuple[str, ...] = ()
    inventory_rewards: tuple[tuple[str, int], ...] = ()
    supply_rewards: tuple[tuple[str, int], ...] = ()
    reputation_reward: int = 0
    coin_reward: int = 0
    inventory_requirements: tuple[tuple[str, int], ...] = ()
    supply_costs: tuple[tuple[str, int], ...] = ()
    action_summaries: tuple[str, ...] = ()
    quest_marker: bool = False

@dataclass(frozen=True)
class DungeonActionView:
    action_id: str
    label: str
    description: str
    state: str
    requirements: tuple[tuple[str, int], ...]
    cost: tuple[tuple[str, int], ...]
    reward: tuple[str, ...]

@dataclass(frozen=True)
class DungeonView:
    expedition_id: str
    dungeon_id: str
    current_room: DungeonRoomView
    map_nodes: tuple[DungeonMapNodeView, ...]
    exits: tuple[DungeonMapNodeView, ...]
    room_actions: tuple[DungeonActionView, ...]
    actions: tuple[ScreenAction, ...]
    recent_events: tuple[GameEvent, ...] = ()
    current_map_id: str = ""
    previous_node_id: str = ""
    inventory: tuple[tuple[str, int], ...] = ()
    supplies: tuple[tuple[str, int], ...] = ()
    maze_main_depth: int = 0
    maze_frontier_depth: int = 0
    maze_off_spine_hint: str = ""

@dataclass(frozen=True)
class ExpeditionReportView:
    expedition_id: str
    dungeon_id: str | None
    outcome: str
    rooms_entered: tuple[str, ...]
    encounters_resolved: tuple[str, ...]
    loot: tuple[tuple[str, int], ...]
    supplies: tuple[tuple[str, int], ...]
    gear: tuple[tuple[str, int], ...]
    reputation_gained: int
    coin_gained: int
    breaches_discovered: tuple[str, ...]
    room_actions: tuple[str, ...]
    supplies_spent: tuple[tuple[str, int], ...]
    reputation_start: int
    reputation_end: int
    coin_start: int
    coin_end: int
    hero_outcomes: tuple[str, ...]
    notable_moments: tuple[str, ...]
    what_changed: tuple[str, ...]
    objective: CampaignObjectiveView
    next_objective: str
    wounded_count: int
    downed_count: int
    deceased_count: int
    actions: tuple[ScreenAction, ...]
    reputation_delta: int = 0
    coin_delta: int = 0
    supply_deltas: tuple[tuple[str, int, int, int], ...] = ()
    inventory_deltas: tuple[tuple[str, int, int, int], ...] = ()
    gear_deltas: tuple[tuple[str, int, int, int], ...] = ()

def build_dungeon_view(
    company: CompanyState,
    definitions: GameDefinitions,
    recent_events: Sequence[GameEvent] = (),
    first_visit_node_id: str | None = None,
) -> DungeonView:
    session = company.active_expedition
    if session is None:
        raise ValueError("No active dungeon expedition.")
    nodes = active_dungeon_nodes(definitions, session)
    current = nodes[session.current_node_id]
    current_map_id = current.map_id
    current_cleared = current.id in session.cleared_node_ids
    current_first_visit = current.id == first_visit_node_id
    current_text = (
        current.text if current_first_visit or not current.revisit_text else current.revisit_text
    )
    current_exit_ids = list(
        dict.fromkeys([*current.exits, *revealed_exit_node_ids(session, current.id)])
    )
    current_exit_ids = _ordered_route_ids(
        current_exit_ids,
        nodes=nodes,
        current_node_id=current.id,
    )
    anchor_ids = set(session.visited_node_ids)
    anchor_ids.add(current.id)
    known_ids = set(anchor_ids)
    for node_id in anchor_ids:
        if node_id in nodes:
            known_ids.update(nodes[node_id].exits)
            known_ids.update(revealed_exit_node_ids(session, node_id))
    movement_enabled = current_cleared and session.pending_combat_node_id is None
    generated_nodes = generated_nodes_by_id(session.generated_dungeon)
    inside_generated_maze = current.id in generated_nodes
    at_generated_threshold = (
        session.generated_dungeon is not None
        and not session.generated_dungeon.collapsed
        and current.id == session.generated_dungeon.entry_node_id
    )
    opening_breach_at_breach = bool(
        company.flags.get(OPENING_BREACH_PENDING_FLAG, False)
        and session.current_node_id == SHALLOW_CAVE_BREACH_NODE_ID
    )
    can_enter_generated_maze = (
        current.node_type.value == "breach"
        and (session.generated_dungeon is None or session.generated_dungeon.collapsed)
        and not opening_breach_at_breach
    )
    generated_active = (
        session.generated_dungeon is not None and not session.generated_dungeon.collapsed
    )
    generated_dungeon = (
        session.generated_dungeon
        if session.generated_dungeon is not None and not session.generated_dungeon.collapsed
        else None
    )
    maze_frontier_depth = generated_dungeon.main_spine_length if generated_dungeon else 0
    at_maze_frontier = (
        generated_dungeon is not None
        and inside_generated_maze
        and session.current_node_id == frontier_node_id(generated_dungeon)
    )
    maze_main_depth = 0
    if generated_dungeon is not None and is_main_spine_room(
        session.current_node_id,
        generated_dungeon.run_id,
    ):
        room_suffix = session.current_node_id.rsplit("_room_", 1)[-1]
        if room_suffix.isdigit():
            maze_main_depth = int(room_suffix)
    frontier_previews = (
        frontier_exit_previews(generated_dungeon, current.id)
        if generated_dungeon is not None
        and current_cleared
        and session.pending_combat_node_id is None
        else ()
    )
    preview_by_id = {preview.exit_id: preview for preview in frontier_previews}
    if frontier_previews:
        current_exit_ids = list(
            dict.fromkeys([*current_exit_ids, *(preview.exit_id for preview in frontier_previews)])
        )

    maze_off_spine_hint = ""
    if (
        generated_active
        and session.generated_dungeon is not None
        and not at_maze_frontier
        and is_generated_spur_room(
            session.current_node_id,
            session.generated_dungeon.run_id,
        )
    ):
        frontier_id = frontier_node_id(session.generated_dungeon)
        if frontier_id in current.exits:
            maze_off_spine_hint = (
                "Return to the main route to push deeper or take unstable exits."
            )

    preview_exit_ids = {
        preview.exit_id for preview in frontier_previews if preview.exit_id not in nodes
    }
    if preview_exit_ids:
        known_ids.update(preview_exit_ids)

    known_exit_ids_by_node: dict[str, tuple[str, ...]] = {}
    for node_id in known_ids:
        if node_id not in nodes:
            continue
        exit_ids = tuple(
            dict.fromkeys(
                [*nodes[node_id].exits, *revealed_exit_node_ids(session, node_id)]
            )
        )
        known_exit_ids_by_node[node_id] = tuple(
            exit_id
            for exit_id in exit_ids
            if exit_id in preview_exit_ids
            or (
                exit_id in known_ids
                and exit_id in nodes
                and nodes[exit_id].map_id == nodes[node_id].map_id
            )
        )
    if preview_exit_ids and generated_active and session.generated_dungeon is not None:
        frontier_id = frontier_node_id(session.generated_dungeon)
        if frontier_id in known_exit_ids_by_node:
            preview_ordered = tuple(
                preview.exit_id
                for preview in frontier_previews
                if preview.exit_id in preview_exit_ids
            )
            existing = known_exit_ids_by_node[frontier_id]
            known_exit_ids_by_node[frontier_id] = tuple(
                dict.fromkeys([*existing, *preview_ordered])
            )
    visited_ids = set(session.visited_node_ids)
    cleared_ids = set(session.cleared_node_ids)
    completed_action_ids = set(session.completed_action_ids)
    quest_target_node_ids = _quest_target_node_ids(company, session)
    map_nodes = tuple(
        _dungeon_node_view(
            node_id,
            nodes,
            company=company,
            session=session,
            current_id=current.id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=completed_action_ids,
            exit_node_ids=known_exit_ids_by_node.get(node_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for node_id in nodes
        if node_id in known_ids and nodes[node_id].map_id == current_map_id
    )
    if frontier_previews and session.generated_dungeon is not None:
        frontier_position = current.position
        preview_positions = frontier_preview_map_positions(
            session.generated_dungeon,
            frontier_previews,
        )
        preview_map_nodes: list[DungeonMapNodeView] = []
        for preview in frontier_previews:
            if preview.exit_id in nodes:
                continue
            position = preview_positions[preview.exit_id]
            preview_map_nodes.append(
                _frontier_preview_exit_view(
                    preview,
                    current_map_id=current_map_id,
                    map_x=position[0],
                    map_y=position[1],
                    direction=(
                        _direction_between_positions(frontier_position, position)
                        if frontier_position is not None
                        else ""
                    ),
                )
            )
        if preview_map_nodes:
            map_nodes = (*map_nodes, *preview_map_nodes)
    exits = tuple(
        _frontier_preview_exit_view(preview_by_id[node_id], current_map_id=current_map_id)
        if node_id in preview_by_id and node_id not in nodes
        else _dungeon_node_view(
            node_id,
            nodes,
            company=company,
            session=session,
            current_id=current.id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=completed_action_ids,
            exit_node_ids=known_exit_ids_by_node.get(node_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for node_id in current_exit_ids
    )
    room_actions = tuple(
        _dungeon_action_view(
            action,
            company,
            current_node_id=current.id,
            current_cleared=current_cleared,
            completed_action_ids=set(session.completed_action_ids),
        )
        for action in current.actions
        if _room_action_visible(company, action)
    )

    actions = ActionProvider.dungeon_actions(
        room_actions,
        exits,
        movement_enabled=movement_enabled,
        pending_combat=session.pending_combat_node_id is not None,
        current_map_id=current_map_id,
        current_safe_return=current.safe_return,
        can_enter_generated_maze=can_enter_generated_maze,
        can_retrace_generated_maze=inside_generated_maze and not at_generated_threshold,
        can_withdraw_generated_maze=at_generated_threshold,
        unstable_frontier_exit_ids=tuple(preview.exit_id for preview in frontier_previews),
        previous_node_id=session.previous_node_id,
        current_node_id=current.id,
        opening_breach_at_breach=opening_breach_at_breach,
    )
    return DungeonView(
        expedition_id=session.expedition_id,
        dungeon_id=session.dungeon_id,
        current_map_id=current_map_id,
        previous_node_id=session.previous_node_id,
        maze_main_depth=maze_main_depth,
        maze_frontier_depth=maze_frontier_depth,
        maze_off_spine_hint=maze_off_spine_hint,
        current_room=DungeonRoomView(
            node_id=current.id,
            name=current.name,
            node_type=current.node_type.value,
            text=current_text,
            safe_return=current.safe_return,
            cleared=current_cleared,
            scene_state=current.scene_state,
            route_hint=current.route_hint,
            party_hint=current.party_hint,
            first_visit=current_first_visit,
            major_beat=current.major_beat,
            art_lines=_art_lines(_dungeon_node_art_asset(definitions, session, current)),
        ),
        map_nodes=map_nodes,
        exits=exits,
        room_actions=room_actions,
        actions=tuple(actions),
        recent_events=tuple(recent_events[-8:]),
        inventory=tuple(sorted(company.inventory.items())),
        supplies=tuple(sorted(company.supplies.items())),
    )

def build_expedition_report_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> ExpeditionReportView:
    report = company.last_expedition_report
    if report is None:
        raise ValueError("No expedition report is available.")
    nodes = {
        node.id: node
        for expedition in definitions.expeditions.values()
        for node in expedition.nodes
    }
    living = living_roster(company)
    wounded_count = sum(1 for hero in living if hero.hp < hero.max_hp or hero.mortal_wounds > 0)
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    actions = ActionProvider.report_actions(company, definitions)
    objective = build_campaign_objective(company)
    return ExpeditionReportView(
        expedition_id=report.expedition_id,
        dungeon_id=report.dungeon_id,
        outcome=report.outcome,
        rooms_entered=tuple(
            _node_name(nodes, node_id, report.room_names) for node_id in report.rooms_entered
        ),
        encounters_resolved=tuple(
            encounter_id.replace("_", " ").title() for encounter_id in report.encounters_resolved
        ),
        loot=tuple(sorted(report.loot.items())),
        supplies=tuple(sorted(report.supplies.items())),
        gear=tuple(sorted(report.gear.items())),
        reputation_gained=report.reputation_gained,
        coin_gained=report.coin_gained,
        breaches_discovered=tuple(report.breaches_discovered),
        room_actions=tuple(
            _room_action_name(nodes, action_key) for action_key in report.room_actions
        ),
        supplies_spent=tuple(sorted(report.supplies_spent.items())),
        reputation_start=report.start_reputation,
        reputation_end=report.end_reputation,
        reputation_delta=report.end_reputation - report.start_reputation,
        coin_start=report.start_coin,
        coin_end=report.end_coin,
        coin_delta=report.end_coin - report.start_coin,
        supply_deltas=_item_deltas(report.start_supplies, report.end_supplies),
        inventory_deltas=_item_deltas(report.start_inventory, report.end_inventory),
        gear_deltas=_item_deltas(
            report.start_gear_inventory,
            report.end_gear_inventory,
        ),
        hero_outcomes=tuple(_hero_outcome_line(outcome) for outcome in report.hero_outcomes),
        notable_moments=tuple(report.notable_moments),
        what_changed=_report_change_lines(report),
        objective=objective,
        next_objective=objective.next_step,
        wounded_count=wounded_count,
        downed_count=downed_count,
        deceased_count=len(company.deceased_heroes),
        actions=actions,
    )

def _ordered_route_ids(
    node_ids: Sequence[str],
    *,
    nodes: Mapping[str, ExpeditionNodeDefinition],
    current_node_id: str,
) -> list[str]:
    current = nodes[current_node_id]
    if current.position is None:
        return list(node_ids)
    indexed_ids = list(enumerate(node_ids))
    indexed_ids.sort(
        key=lambda item: _route_sort_key(
            nodes[item[1]],
            current=current,
            authored_index=item[0],
        )
    )
    return [node_id for _index, node_id in indexed_ids]

def _route_sort_key(
    node: ExpeditionNodeDefinition,
    *,
    current: ExpeditionNodeDefinition,
    authored_index: int,
) -> tuple[int, int, int, int]:
    if node.position is None or current.position is None:
        return (1, 99, 99, authored_index)
    dx = node.position[0] - current.position[0]
    dy = node.position[1] - current.position[1]
    return (0, _compass_rank(dx, dy), abs(dx) + abs(dy), authored_index)

def _compass_rank(dx: int, dy: int) -> int:
    if dx == 0 and dy < 0:
        return 0
    if dx > 0 and dy == 0:
        return 1
    if dx == 0 and dy > 0:
        return 2
    if dx < 0 and dy == 0:
        return 3
    if dy < 0:
        return 0
    if dx > 0:
        return 1
    if dy > 0:
        return 2
    if dx < 0:
        return 3
    return 4

def _item_deltas(
    start: Mapping[str, int],
    end: Mapping[str, int],
) -> tuple[tuple[str, int, int, int], ...]:
    item_ids = sorted(set(start) | set(end))
    return tuple(
        (
            item_id,
            start.get(item_id, 0),
            end.get(item_id, 0),
            end.get(item_id, 0) - start.get(item_id, 0),
        )
        for item_id in item_ids
        if start.get(item_id, 0) != end.get(item_id, 0)
    )

def _report_change_lines(report: ExpeditionReportState) -> tuple[str, ...]:
    lines: list[str] = []
    if report.end_reputation != report.start_reputation:
        lines.append(
            "Reputation "
            f"{report.start_reputation}->{report.end_reputation} "
            f"({_signed(report.end_reputation - report.start_reputation)})."
        )
    if report.end_coin != report.start_coin:
        lines.append(
            "Coin "
            f"{report.start_coin}->{report.end_coin} "
            f"({_signed(report.end_coin - report.start_coin)})."
        )
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
        lines.append("Breach memory updated: " + ", ".join(report.breaches_discovered) + ".")
    if report.room_actions:
        lines.append(f"Ledger recorded {len(report.room_actions)} room action(s).")
    if report.gear:
        lines.append(
            "Armory updated: "
            + ", ".join(f"{gear_id} x{quantity}" for gear_id, quantity in report.gear.items())
            + "."
        )
    if report.hero_outcomes:
        lines.append(f"Roster report updated for {len(report.hero_outcomes)} participant(s).")
    return tuple(_dedupe(lines))

def _hero_outcome_line(outcome: HeroReportOutcome) -> str:
    pieces = [
        f"{outcome.hero_name} ({outcome.class_id})",
        outcome.status,
        f"HP {outcome.start_hp}->{outcome.end_hp}/{outcome.max_hp}",
    ]
    if outcome.mortal_wounds_delta:
        sign = "+" if outcome.mortal_wounds_delta > 0 else ""
        pieces.append(
            f"Mortal Wounds {sign}{outcome.mortal_wounds_delta} "
            f"({mortal_wound_badge(outcome.end_mortal_wounds)})"
        )
    else:
        pieces.append(mortal_wound_badge(outcome.end_mortal_wounds))
    return _join_detail(*pieces)

def _frontier_preview_exit_view(
    preview: FrontierExitPreview,
    *,
    current_map_id: str,
    map_x: int | None = None,
    map_y: int | None = None,
    direction: str = "",
) -> DungeonMapNodeView:
    return DungeonMapNodeView(
        node_id=preview.exit_id,
        name=preview.label,
        map_id=current_map_id,
        node_type=preview.node_type,
        status="unstable",
        exit_node_ids=(),
        map_x=map_x,
        map_y=map_y,
        direction=direction,
        location_id="",
        memory_summary="Unstable passage — room not yet mapped.",
        memory_notes=(),
        inventory_rewards=(),
        supply_rewards=(),
        reputation_reward=0,
        coin_reward=0,
        inventory_requirements=(),
        supply_costs=(),
        action_summaries=(),
        current=False,
        known=True,
        visited=False,
        cleared=False,
        safe_return=False,
    )

def _dungeon_quest_marker(
    node: ExpeditionNodeDefinition,
    *,
    quest_target_node_ids: set[str],
) -> bool:
    return node.id in quest_target_node_ids

def _quest_target_node_ids(
    company: CompanyState,
    session: Any,
) -> set[str]:
    active_contract_ids = set(company.active_contract_ids)
    if not active_contract_ids:
        return {QUEST_PICKUP_NODE_ID}
    if BLACKWOOD_ROAD_CHARTER_ID in active_contract_ids:
        cave_memory = company.dungeon_memory.get("shallow_cave")
        visited_ids = set(cave_memory.visited_node_ids if cave_memory else ())
        cleared_ids = set(cave_memory.cleared_node_ids if cave_memory else ())
        if BLACKWOOD_CAVE_TARGET_NODE_ID not in visited_ids:
            return {BLACKWOOD_CAVE_TARGET_NODE_ID}
        if BLACKWOOD_BOSS_TARGET_NODE_ID not in cleared_ids:
            return {BLACKWOOD_BOSS_TARGET_NODE_ID}
        return set()
    if _active_hunt_contract_ids(active_contract_ids):
        generated = getattr(session, "generated_dungeon", None)
        if generated is not None and not getattr(generated, "collapsed", False):
            hunt_node_id = next(
                (node.id for node in generated.nodes if node.id.endswith("_hunt_lair")),
                "",
            )
            if hunt_node_id and hunt_node_id not in generated.cleared_node_ids:
                return {hunt_node_id}
        return {BREACH_TARGET_NODE_ID}
    if _active_scout_contract_ids(active_contract_ids):
        return {BREACH_TARGET_NODE_ID}
    return set()

def _active_hunt_contract_ids(active_contract_ids: set[str]) -> set[str]:
    return active_contract_ids & {
        BREACH_STALKER_HUNT_ID,
        GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
    }

def _active_scout_contract_ids(active_contract_ids: set[str]) -> set[str]:
    return active_contract_ids & {
        SHALLOW_CAVE_BREACH_SCOUT_ID,
        GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
    }

def _dungeon_node_view(
    node_id: str,
    nodes: Mapping[str, ExpeditionNodeDefinition],
    *,
    company: CompanyState,
    session: Any,
    current_id: str,
    visited_ids: set[str],
    cleared_ids: set[str],
    known_ids: set[str],
    completed_action_ids: set[str],
    exit_node_ids: Sequence[str] = (),
    quest_target_node_ids: set[str] | None = None,
) -> DungeonMapNodeView:
    node = nodes[node_id]
    visited = node_id in visited_ids
    cleared = node_id in cleared_ids
    current = node_id == current_id
    known = node_id in known_ids
    if current:
        status = "current"
    elif cleared:
        status = "cleared"
    elif visited:
        status = "visited"
    else:
        status = "known"
    return DungeonMapNodeView(
        node_id=node_id,
        name=node.name,
        map_id=node.map_id,
        node_type=node.node_type.value,
        status=status,
        exit_node_ids=tuple(exit_node_ids),
        map_x=node.position[0] if node.position is not None else None,
        map_y=node.position[1] if node.position is not None else None,
        direction=_route_direction(nodes[current_id], node),
        location_id=world_location_id_for_node(node),
        memory_summary=_node_memory_summary(company, node),
        memory_notes=_node_memory_notes(company, session, node),
        inventory_rewards=_node_inventory_rewards(node),
        supply_rewards=_node_supply_rewards(node),
        reputation_reward=_node_reputation_reward(node),
        coin_reward=_node_coin_reward(node),
        inventory_requirements=_node_inventory_requirements(node),
        supply_costs=_node_supply_costs(node),
        action_summaries=_node_action_summaries(
            node,
            nodes,
            completed_action_ids=completed_action_ids,
        ),
        current=current,
        known=known,
        visited=visited,
        cleared=cleared,
        safe_return=node.safe_return,
        quest_marker=_dungeon_quest_marker(
            node,
            quest_target_node_ids=quest_target_node_ids or set(),
        ),
    )

def _node_memory_summary(
    company: CompanyState,
    node: ExpeditionNodeDefinition,
) -> str:
    location_id = world_location_id_for_node(node)
    memory = company.world_memory.get(location_id)
    if memory is None:
        return ""
    pieces = [
        "discovered" if node.id in memory.discovered_node_ids else "",
        "threat cleared" if node.id in memory.cleared_threat_node_ids else "",
        f"{memory.visit_count} location visits" if memory.visit_count else "",
    ]
    if node.known_route_unlock and node.known_route_unlock in memory.unlocked_shortcut_ids:
        pieces.append("shortcut charted")
    return _join_detail(*pieces)

def _node_memory_notes(
    company: CompanyState,
    session: Any,
    node: ExpeditionNodeDefinition,
) -> tuple[str, ...]:
    notes: list[str] = []
    report = getattr(session, "report", None)
    if report is not None:
        notes.extend(
            signal.message
            for signal in getattr(report, "event_signals", ())
            if signal.node_id == node.id
        )
    notes.extend(entry.summary for entry in company.company_timeline if entry.node_id == node.id)
    notes.extend(memory.summary for memory in company.hero_memories if memory.node_id == node.id)
    return tuple(_dedupe_lines(notes)[-3:])

def _node_inventory_rewards(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    rewards = dict(node.loot)
    for action in node.actions:
        for item_id, quantity in action.loot.items():
            rewards[item_id] = rewards.get(item_id, 0) + quantity
    return tuple(sorted((item_id, quantity) for item_id, quantity in rewards.items() if quantity))

def _node_supply_rewards(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    rewards = dict(node.supply_rewards)
    for action in node.actions:
        for supply_id, quantity in action.supply_rewards.items():
            rewards[supply_id] = rewards.get(supply_id, 0) + quantity
    return tuple(
        sorted((supply_id, quantity) for supply_id, quantity in rewards.items() if quantity)
    )

def _node_reputation_reward(node: ExpeditionNodeDefinition) -> int:
    return node.reputation_reward + sum(action.reputation_reward for action in node.actions)

def _node_coin_reward(node: ExpeditionNodeDefinition) -> int:
    return node.coin_reward + sum(action.coin_reward for action in node.actions)

def _node_inventory_requirements(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    requirements: dict[str, int] = {}
    for action in node.actions:
        for item_id, quantity in action.inventory_requirements.items():
            requirements[item_id] = max(requirements.get(item_id, 0), quantity)
    return tuple(sorted(requirements.items()))

def _node_supply_costs(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    costs: dict[str, int] = {}
    for action in node.actions:
        for supply_id, quantity in action.supply_costs.items():
            costs[supply_id] = max(costs.get(supply_id, 0), quantity)
    return tuple(sorted(costs.items()))

def _node_action_summaries(
    node: ExpeditionNodeDefinition,
    nodes: Mapping[str, ExpeditionNodeDefinition],
    *,
    completed_action_ids: set[str],
) -> tuple[str, ...]:
    lines: list[str] = []
    for action in node.actions:
        state = "completed" if room_action_key(node.id, action.id) in completed_action_ids else ""
        reward_detail = _action_reward_detail(action)
        detail = _join_detail(
            state,
            f"needs {_quantity_detail(action.inventory_requirements)}"
            if action.inventory_requirements
            else "",
            f"costs {_quantity_detail(action.supply_costs)}" if action.supply_costs else "",
            f"yields {reward_detail}" if reward_detail else "",
            _revealed_exit_detail(action.reveal_exits, nodes),
        )
        lines.append(f"{action.label}: {detail}" if detail else action.label)
    return tuple(lines)

def _quantity_detail(values: Mapping[str, int]) -> str:
    return ", ".join(
        f"{quantity} {_display_id(item_id)}"
        for item_id, quantity in sorted(values.items())
        if quantity
    )

def _action_reward_detail(action: ExpeditionRoomActionDefinition) -> str:
    lines: list[str] = []
    if action.reputation_reward:
        lines.append(f"{action.reputation_reward} reputation")
    if action.coin_reward:
        lines.append(f"{action.coin_reward} Coin")
    lines.extend(
        f"{quantity} {_display_id(item_id)}" for item_id, quantity in sorted(action.loot.items())
    )
    lines.extend(
        f"{quantity} {_display_id(supply_id)}"
        for supply_id, quantity in sorted(action.supply_rewards.items())
    )
    return ", ".join(lines)

def _revealed_exit_detail(
    exit_node_ids: Sequence[str],
    nodes: Mapping[str, ExpeditionNodeDefinition],
) -> str:
    if not exit_node_ids:
        return ""
    labels = [nodes[exit_id].name if exit_id in nodes else exit_id for exit_id in exit_node_ids]
    return f"reveals {', '.join(labels)}"

def _display_id(value: str) -> str:
    return value.replace("_", " ")

def _direction_between_positions(
    from_position: tuple[int, int],
    to_position: tuple[int, int],
) -> str:
    dx = to_position[0] - from_position[0]
    dy = to_position[1] - from_position[1]
    if dx == 0 and dy == 0:
        return ""
    if dx == 0 and dy < 0:
        return "North"
    if dx > 0 and dy == 0:
        return "East"
    if dx == 0 and dy > 0:
        return "South"
    if dx < 0 and dy == 0:
        return "West"
    vertical = "North" if dy < 0 else "South" if dy > 0 else ""
    horizontal = "East" if dx > 0 else "West" if dx < 0 else ""
    return " ".join(part for part in (vertical, horizontal) if part)

def _route_direction(
    current: ExpeditionNodeDefinition,
    node: ExpeditionNodeDefinition,
) -> str:
    if current.position is None or node.position is None or current.id == node.id:
        return ""
    return _direction_between_positions(current.position, node.position)

def _room_action_visible(
    company: CompanyState,
    action: ExpeditionRoomActionDefinition,
) -> bool:
    if not action.requires_active_contracts:
        return True
    return all(
        contract_id in company.active_contract_ids
        for contract_id in action.requires_active_contracts
    )

def _dungeon_action_view(
    action: ExpeditionRoomActionDefinition,
    company: CompanyState,
    *,
    current_node_id: str,
    current_cleared: bool,
    completed_action_ids: set[str],
) -> DungeonActionView:
    action_key = room_action_key(current_node_id, action.id)
    if action.once and action_key in completed_action_ids:
        state = "completed"
    elif action.requires_cleared and not current_cleared:
        state = "requires cleared room"
    elif any(
        company.inventory.get(item_id, 0) < quantity
        for item_id, quantity in action.inventory_requirements.items()
    ):
        state = "missing item"
    elif any(
        company.supplies.get(supply_id, 0) < quantity
        for supply_id, quantity in action.supply_costs.items()
    ):
        state = "missing supplies"
    else:
        state = "available"
    return DungeonActionView(
        action_id=action.id,
        label=action.label,
        description=action.description,
        state=state,
        requirements=tuple(sorted(action.inventory_requirements.items())),
        cost=tuple(sorted(action.supply_costs.items())),
        reward=tuple(_action_reward_lines(action)),
    )

def _action_reward_lines(action: ExpeditionRoomActionDefinition) -> list[str]:
    lines: list[str] = []
    if action.reputation_reward:
        lines.append(f"{action.reputation_reward} reputation")
    if action.coin_reward:
        lines.append(f"{action.coin_reward} Coin")
    lines.extend(f"{quantity} {item_id}" for item_id, quantity in sorted(action.loot.items()))
    lines.extend(
        f"{quantity} {supply_id}" for supply_id, quantity in sorted(action.supply_rewards.items())
    )
    return lines

def _node_name(
    nodes: Mapping[str, ExpeditionNodeDefinition],
    node_id: str,
    room_names: Mapping[str, str] | None = None,
) -> str:
    node = nodes.get(node_id)
    if node is None:
        if room_names is not None and node_id in room_names:
            return room_names[node_id]
        return node_id.replace("_", " ").title()
    return node.name

def _room_action_name(nodes: Mapping[str, ExpeditionNodeDefinition], action_key: str) -> str:
    node_id, _separator, action_id = action_key.partition(":")
    node = nodes.get(node_id)
    if node is None:
        return action_key
    action = next((candidate for candidate in node.actions if candidate.id == action_id), None)
    if action is None:
        return action_key
    return f"{node.name}: {action.label}"
