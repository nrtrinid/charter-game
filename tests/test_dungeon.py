from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from game.app.commands import (
    AcceptContract,
    EnterGeneratedMaze,
    MarkRegionalRoute,
    MoveDungeon,
    MoveRegional,
    PassCombatTurn,
    ResolveCombatAction,
    ResolveCombatReaction,
    RetraceGeneratedMaze,
    RetreatCombat,
    RetreatGeneratedMaze,
    ReturnFromDungeon,
    ReturnToHavenTown,
    StartExpedition,
    StartNewCompany,
    TakeExpeditionChoice,
    TravelRegional,
    UseDungeonAction,
    UseRegionalAction,
    ViewDungeon,
    ViewExpeditionReport,
    ViewRegionalMap,
    VisitEastGate,
    WithdrawGeneratedMaze,
)
from game.app.controller import AppController
from game.app.manual_combat import legal_skill_ids, legal_target_ids
from game.app.views import (
    CombatView,
    DungeonView,
    ExpeditionReportView,
    RegionalMapView,
    TownDashboardView,
)
from game.campaign.company import DungeonMemoryState
from game.campaign.save_load import load_company, save_company
from game.combat.combat_state import LifeState
from game.combat.turn_order import InitiativeEntry
from game.core.events import (
    CombatRetreatDeclaredEvent,
    CombatRetreatedEvent,
    ContractCompletedEvent,
    DungeonActionEvent,
    EncounterStartedEvent,
    ExpeditionEvent,
    LootGainedEvent,
    MazeFrontierOpenedEvent,
    MazeRouteCollapsedEvent,
    RoundEndedEvent,
    SkillUsedEvent,
)
from game.data.loaders import load_game_definitions
from tests.conftest import get_definitions


def test_interactive_expedition_starts_travel_then_dungeon_session() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())

    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))

    assert result.success
    assert controller.company is not None
    assert controller.manual_combat is None
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.current_node_id == "town_gate"
    assert "shallow_cave" not in controller.company.known_route_ids
    assert result.events

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    view = view_result.value
    assert view.current_room.node_id == "town_gate"
    assert view.current_room.art_lines
    assert view.current_map_id == "old_road_wilderness"
    assert view.exits[0].node_id == "old_road"
    map_nodes = {node.node_id: node for node in view.map_nodes}
    assert map_nodes["town_gate"].map_id == "old_road_wilderness"
    assert (map_nodes["town_gate"].map_x, map_nodes["town_gate"].map_y) == (0, 1)
    assert (map_nodes["old_road"].map_x, map_nodes["old_road"].map_y) == (0, 0)
    assert map_nodes["town_gate"].exit_node_ids == ("old_road",)
    assert not view.room_actions
    assert all(action.value != "inspect" for action in view.actions)
    assert any(action.value == "return" and action.enabled for action in view.actions)


def test_dungeon_room_view_uses_authored_scene_and_revisit_text() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())

    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))

    assert result.success
    assert isinstance(result.value, DungeonView)
    start_view = result.value
    assert start_view.current_room.first_visit
    assert "townsman at the gate" in start_view.current_room.text
    assert (
        start_view.current_room.scene_state == "Haven's lamps are still close enough to retreat by."
    )
    assert start_view.current_room.route_hint
    assert start_view.current_room.party_hint

    first_road = controller.handle(MoveDungeon("old_road"))

    assert first_road.success
    assert isinstance(first_road.value, DungeonView)
    assert first_road.value.current_room.first_visit
    assert "old road splits" in first_road.value.current_room.text

    revisit_gate = controller.handle(MoveDungeon("town_gate"))

    assert revisit_gate.success
    assert isinstance(revisit_gate.value, DungeonView)
    assert not revisit_gate.value.current_room.first_visit
    assert "east gate stands open" in revisit_gate.value.current_room.text
    assert revisit_gate.value.current_room.major_beat is False


def test_dungeon_routes_use_compass_order() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))

    result = controller.handle(MoveDungeon("old_road"))

    assert result.success
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.previous_node_id == "town_gate"
    assert isinstance(result.value, DungeonView)
    view = result.value
    route_actions = [action for action in view.actions if str(action.kind) == "travel"]
    assert [action.value for action in route_actions] == [
        "bramble_shrine",
        "hunters_trail",
        "town_gate",
        "abandoned_toll_post",
    ]
    assert "direction north" in route_actions[0].preview
    assert "direction south" in route_actions[2].preview


def test_node_presentation_schema_loads_major_beat_thresholds() -> None:
    definitions = get_definitions()
    nodes = {node.id: node for node in definitions.expeditions["opening"].nodes}

    assert nodes["maze_breach"].major_beat
    assert nodes["maze_breach"].scene_state
    assert nodes["maze_breach"].revisit_text
    assert nodes["maze_breach"].route_hint
    assert nodes["maze_breach"].party_hint


def test_map_region_scope_keeps_cave_and_wilderness_maps_separate() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    entrance_view = view_result.value
    assert entrance_view.current_map_id == "old_road_wilderness"
    assert "shallow_cave_room_1" not in {node.node_id for node in entrance_view.map_nodes}
    cave_route = next(
        action for action in entrance_view.actions if action.value == "shallow_cave_room_1"
    )
    assert "enter shallow cave" in cave_route.description

    _move_along(controller, "shallow_cave_room_1", "cave_fork")
    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    cave_view = view_result.value
    assert cave_view.current_map_id == "shallow_cave"
    map_node_ids = {node.node_id for node in cave_view.map_nodes}
    assert "old_road" not in map_node_ids
    assert "shallow_cave_room_1" in map_node_ids
    assert "cave_fork" in map_node_ids


def test_regional_east_gate_minimap_draws_old_road_connection() -> None:
    from game.app.commands import StartNewCompany
    from game.app.controller import AppController
    from game.app.views import build_regional_map_view, build_regional_render_view
    from game.ui.tui_widgets import DungeonMapPanel

    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    assert controller.company is not None
    assert "shallow_cave" not in controller.company.known_route_ids

    view = build_regional_map_view(controller.company, controller.definitions)
    map_nodes = {node.node_id: node for node in view.map_nodes}
    assert view.current_node_id == "town_gate"
    assert "old_road" in map_nodes
    assert map_nodes["town_gate"].exit_node_ids == ("old_road",)

    render_view = build_regional_render_view(view)
    map_text = DungeonMapPanel.render_minimap_text(render_view)
    map_body = "\n".join(
        line
        for line in map_text.splitlines()[3:]
        if line and not line.startswith("Legend:")
    )
    assert "|" in map_body or "-" in map_body


def test_dungeon_minimap_draws_edges_between_known_unvisited_submap_nodes() -> None:
    from game.ui.tui_widgets import DungeonMapPanel

    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
    )

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    cave_view = view_result.value
    map_nodes = {node.node_id: node for node in cave_view.map_nodes}
    assert cave_view.current_map_id == "shallow_cave"
    assert "cave_fork" in map_nodes
    assert "cave_fork" in map_nodes["shallow_cave_room_1"].exit_node_ids
    assert "shallow_cave_room_1" in map_nodes["cave_fork"].exit_node_ids

    map_text = DungeonMapPanel.render_minimap_text(cave_view)
    map_body = "\n".join(
        line
        for line in map_text.splitlines()[3:]
        if line and not line.startswith("Legend:")
    )
    assert "|" in map_body or "-" in map_body


def test_dungeon_map_marks_town_gate_when_contract_available() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    assert controller.company is not None
    controller.company.active_contract_ids.clear()
    controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))
    _move_along(controller, "old_road")

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    town_gate = {node.node_id: node for node in view_result.value.map_nodes}["town_gate"]
    assert town_gate.quest_marker


def test_dungeon_map_clears_town_gate_quest_marker_when_contract_active() -> None:
    controller = _started_interactive_controller()
    _move_along(controller, "old_road")

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    town_gate = {node.node_id: node for node in view_result.value.map_nodes}["town_gate"]
    assert not town_gate.quest_marker


def test_dungeon_map_marks_shallow_cave_entrance_for_active_charter() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
    )

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    map_nodes = {node.node_id: node for node in view_result.value.map_nodes}
    assert map_nodes["shallow_cave_entrance"].quest_marker
    assert not map_nodes["town_gate"].quest_marker


def test_dungeon_map_marks_uncleared_boss_room_as_quest() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)
    _win_active_manual_combat(controller)
    controller.handle(UseDungeonAction("recover_gate_key"))
    _move_along(controller, "fungus_chamber", "stone_gate")
    controller.handle(UseDungeonAction("unlock_black_gate"))

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    map_nodes = {node.node_id: node for node in view_result.value.map_nodes}
    assert map_nodes["maze_touched_lair"].quest_marker


def test_dungeon_map_marks_generated_hunt_lair_as_quest() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    controller.company.completed_contract_ids.add("shallow_cave_breach_scout")
    accepted = controller.handle(AcceptContract("shallow_cave_breach_hunt"))
    assert accepted.success, accepted.error
    entered = controller.handle(EnterGeneratedMaze(seed=9))
    assert entered.success, entered.error
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    hunt_node_id = _generated_node_id(controller, "_hunt_lair")
    hunt_node = next(node for node in session.generated_dungeon.nodes if node.id == hunt_node_id)
    final_main_index = int(hunt_node.exits[0].rsplit("_room_", 1)[1])
    for index in range(1, final_main_index + 1):
        _move_generated_main_room(controller, index)
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    map_nodes = {node.node_id: node for node in view_result.value.map_nodes}
    assert map_nodes[hunt_node_id].quest_marker


def test_dungeon_map_clears_boss_quest_marker_after_victory() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    _move_along(controller, "maze_touched_lair")

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    map_nodes = {node.node_id: node for node in view_result.value.map_nodes}
    assert map_nodes["maze_touched_lair"].quest_marker is False


def test_cleared_boss_route_drops_route_warning() -> None:
    from game.app.actions import route_action_warns_player

    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    _move_along(controller, "maze_touched_lair", "stone_gate")

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    lair_exit = next(
        exit_view
        for exit_view in view_result.value.exits
        if exit_view.node_id == "maze_touched_lair"
    )
    assert lair_exit.cleared
    assert not route_action_warns_player(lair_exit)
    lair_action = next(
        action for action in view_result.value.actions if action.value == "maze_touched_lair"
    )
    assert not lair_action.route_warning


def test_dungeon_map_nodes_include_memory_and_inventory_metadata() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)

    result = controller.handle(ViewDungeon())

    assert result.success
    assert isinstance(result.value, DungeonView)
    view = result.value
    nodes = {node.node_id: node for node in view.map_nodes}
    cache = nodes["old_works_cache"]
    assert cache.location_id == "shallow_cave"
    assert "discovered" in cache.memory_summary
    assert cache.inventory_rewards == (("cave_key", 1),)
    assert "Recover Gate Key: yields 3 Coin, 1 cave key" in cache.action_summaries

    _win_active_manual_combat(controller)
    result = controller.handle(UseDungeonAction("recover_gate_key"))
    assert result.success, result.error
    _move_along(controller, "fungus_chamber", "stone_gate")

    result = controller.handle(ViewDungeon())

    assert result.success
    assert isinstance(result.value, DungeonView)
    view = result.value
    gate = {node.node_id: node for node in view.map_nodes}["stone_gate"]
    assert ("cave_key", 1) in gate.inventory_requirements
    assert ("rope", 1) in gate.supply_costs
    assert ("cave_key", 1) in view.inventory
    assert any("Unlock Black Gate: needs 1 cave key" in line for line in gate.action_summaries)
    assert any("Force Black Gate: costs 1 rope" in line for line in gate.action_summaries)


def test_locked_boss_room_requires_key_action_after_guard_fight() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)
    _win_active_manual_combat(controller)
    controller.handle(UseDungeonAction("recover_gate_key"))
    _move_along(controller, "fungus_chamber", "stone_gate")

    blocked = controller.handle(MoveDungeon("maze_touched_lair"))

    assert not blocked.success
    assert blocked.error == "Choose a listed dungeon exit."
    assert controller.company.active_expedition is not None

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    key_action = next(
        action for action in view_result.value.actions if action.value == "action:unlock_black_gate"
    )
    assert key_action.enabled
    assert not any(action.value == "maze_touched_lair" for action in view_result.value.actions)


def test_key_and_gate_actions_reveal_boss_room_and_update_report() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None

    result = controller.handle(UseDungeonAction("recover_gate_key"))

    assert result.success
    assert controller.company.inventory["cave_key"] == 1
    assert controller.company.active_expedition is not None
    session = controller.company.active_expedition
    assert "old_works_cache:recover_gate_key" in session.completed_action_ids
    report = controller.company.active_expedition.report
    assert report is not None
    assert report.loot["cave_key"] == 1
    assert "old_works_cache:recover_gate_key" in report.room_actions
    assert any(isinstance(event, DungeonActionEvent) for event in result.events)

    _move_along(controller, "fungus_chamber", "stone_gate")
    result = controller.handle(UseDungeonAction("unlock_black_gate"))

    assert result.success
    assert "stone_gate->maze_touched_lair" in session.revealed_exit_ids
    assert "stone_gate:unlock_black_gate" in report.room_actions
    assert "stone_gate:unlock_black_gate" in session.completed_action_ids
    assert "stone_gate:force_black_gate" in session.completed_action_ids

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "maze_touched_lair" and action.enabled
        for action in view_result.value.actions
    )
    gate_actions = {action.action_id: action.state for action in view_result.value.room_actions}
    assert gate_actions["unlock_black_gate"] == "completed"
    assert gate_actions["force_black_gate"] == "completed"


def test_forcing_black_gate_clears_key_unlock_interaction_too() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
        "narrow_crawl",
        "stone_gate",
    )
    assert controller.company is not None
    controller.company.supplies["rope"] = 1

    result = controller.handle(UseDungeonAction("force_black_gate"))

    assert result.success, result.error
    session = controller.company.active_expedition
    assert session is not None
    assert "stone_gate->maze_touched_lair" in session.revealed_exit_ids
    assert "stone_gate:force_black_gate" in session.completed_action_ids
    assert "stone_gate:unlock_black_gate" in session.completed_action_ids

    view_result = controller.handle(ViewDungeon())
    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    gate_actions = {action.action_id: action.state for action in view_result.value.room_actions}
    assert gate_actions["unlock_black_gate"] == "completed"
    assert gate_actions["force_black_gate"] == "completed"


def test_move_to_combat_room_starts_manual_combat() -> None:
    controller = _started_interactive_controller()

    result = _move_to_cache(controller)

    assert result.success
    assert controller.company is not None
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.pending_combat_node_id == "old_works_cache"
    assert controller.manual_combat is not None
    assert any(isinstance(event, EncounterStartedEvent) for event in result.events)


def test_retreat_from_combat_returns_to_latest_safe_room_without_clearing_encounter() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)
    assert controller.company is not None
    assert controller.company.active_expedition is not None
    assert controller.manual_combat is not None
    controller.manual_combat.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("bone_soldier_1", 98),
        InitiativeEntry("hero_cutpurse", 97),
    ]
    controller.manual_combat.turn_index = 0
    controller.manual_combat.state.heroes["hero_watchman"].hp = 7

    result = controller.handle(RetreatCombat())

    assert result.success
    assert controller.manual_combat is None
    session = controller.company.active_expedition
    assert session is not None
    assert session.current_node_id == "shallow_cave_room_1"
    assert session.pending_combat_node_id is None
    assert "old_works_cache" not in session.cleared_node_ids
    assert controller.company.last_expedition_report is None
    assert session.report is not None
    assert session.report.outcome == "in_progress"
    assert any(signal.kind == "combat_retreat" for signal in session.report.event_signals)
    watchman = next(hero for hero in controller.company.roster if hero.hero_id == "hero_watchman")
    assert watchman.hp == 7
    assert any(isinstance(event, CombatRetreatDeclaredEvent) for event in result.events)
    assert any(
        isinstance(event, SkillUsedEvent) and event.actor_id == "bone_soldier_1"
        for event in result.events
    )
    assert any(isinstance(event, RoundEndedEvent) for event in result.events)
    assert any(isinstance(event, CombatRetreatedEvent) for event in result.events)
    assert _event_index(result.events, CombatRetreatDeclaredEvent) < _event_index(
        result.events,
        CombatRetreatedEvent,
    )

    _move_along(controller, "cave_fork", "fungus_chamber")
    restart = controller.handle(MoveDungeon("old_works_cache"))

    assert restart.success
    assert controller.manual_combat is not None
    assert session.pending_combat_node_id == "old_works_cache"
    assert controller.manual_combat.state.enemies["bone_soldier_1"].hp == (
        controller.manual_combat.state.enemies["bone_soldier_1"].max_hp
    )


def test_combat_victory_clears_room_and_returns_to_dungeon_navigation() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)

    _win_active_manual_combat(controller)

    assert controller.company is not None
    assert controller.manual_combat is None
    assert controller.company.active_expedition is not None
    assert "old_works_cache" in controller.company.active_expedition.cleared_node_ids

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "action:recover_gate_key" and action.enabled
        for action in view_result.value.actions
    )
    assert not any(action.value == "maze_touched_lair" for action in view_result.value.actions)

    controller.handle(UseDungeonAction("recover_gate_key"))
    _move_along(controller, "fungus_chamber", "stone_gate")
    controller.handle(UseDungeonAction("unlock_black_gate"))
    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "maze_touched_lair" and action.enabled
        for action in view_result.value.actions
    )


def test_cleared_combat_room_can_be_reentered_for_room_action() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    assert controller.company.active_expedition is not None

    _move_along(controller, "fungus_chamber", "old_works_cache")

    session = controller.company.active_expedition
    assert session.pending_combat_node_id is None
    view_result = controller.handle(ViewDungeon())
    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "action:recover_gate_key" and action.enabled
        for action in view_result.value.actions
    )

    result = controller.handle(UseDungeonAction("recover_gate_key"))

    assert result.success
    assert controller.company.inventory["cave_key"] == 1


def test_boss_victory_moves_to_breach_and_return_creates_report() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)

    assert controller.company is not None
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.current_node_id == "maze_breach"
    assert controller.company.flags["opening_breach_pending"]

    view_result = controller.handle(ViewDungeon())
    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    action_values = [action.value for action in view_result.value.actions]
    assert "return_to_haven" in action_values
    assert "descend_maze_depth_1" in action_values
    assert "enter_generated_maze" not in action_values

    result = controller.handle(TakeExpeditionChoice("return_to_haven"))

    assert result.success
    assert controller.company.active_expedition is None
    assert controller.company.last_expedition_report is not None
    report = controller.company.last_expedition_report
    assert report.outcome == "returned_to_haven"
    assert "cave_mini_boss" in report.encounters_resolved
    assert "shallow_cave_breach" in report.breaches_discovered
    assert "breach_discovered" in {entry.kind for entry in controller.company.company_timeline}
    assert "contract_completed" in {entry.kind for entry in controller.company.company_timeline}
    assert any(memory.kind == "breach_discovered" for memory in controller.company.hero_memories)
    assert any("Breach discovered" in moment for moment in report.notable_moments)


def test_breach_descent_creates_report() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)

    result = controller.handle(TakeExpeditionChoice("descend_maze_depth_1"))

    assert result.success
    assert controller.company is not None
    assert controller.company.active_expedition is None
    assert controller.company.last_expedition_report is not None
    assert controller.company.last_expedition_report.outcome == "descended_maze_depth_1"
    assert "maze_depth_1_scouted" in controller.company.expedition_history


def test_opening_route_memory_pacing_does_not_saturate_earned_quirks() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)

    result = controller.handle(TakeExpeditionChoice("descend_maze_depth_1"))

    assert result.success
    assert controller.company is not None
    active_hero_ids = {
        hero_id for hero_id in controller.company.active_party_slots.values() if hero_id
    }
    active_heroes = [
        hero for hero in controller.company.roster if hero.hero_id in active_hero_ids
    ]
    earned_counts = [len(hero.earned_quirk_slots) for hero in active_heroes]
    thread_sense_count = sum(
        1
        for hero in active_heroes
        if "thread_sense" in {slot.quirk_id for slot in hero.earned_quirk_slots}
    )
    gold_fever_count = sum(
        1
        for hero in active_heroes
        if "gold_fever" in {slot.quirk_id for slot in hero.earned_quirk_slots}
    )
    future_hook_quirks = {"thread_sense", "bad_geometry", "loaded_pockets"}
    natural_quirks = {
        slot.quirk_id for hero in active_heroes for slot in hero.earned_quirk_slots
    }
    total_earned = sum(earned_counts)

    assert thread_sense_count < len(active_heroes)
    assert gold_fever_count < len(active_heroes)
    assert not all(count >= 2 for count in earned_counts)
    assert total_earned <= 2
    assert natural_quirks.isdisjoint(future_hook_quirks)
    assert any(hero.fresh_memories for hero in active_heroes)
    assert all(hero.career_signals.get("relic_greed", 0) >= 3 for hero in active_heroes)
    assert all(hero.career_signals.get("maze_thread", 0) >= 1 for hero in active_heroes)


def test_generated_maze_enters_from_breach_and_renders_nodes() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    breach_view_result = controller.handle(ViewDungeon())
    assert breach_view_result.success
    assert isinstance(breach_view_result.value, DungeonView)
    assert any(
        action.value == "return_to_haven" for action in breach_view_result.value.actions
    )
    assert controller.company is not None
    controller.company.flags["opening_breach_pending"] = False
    breach_view_result = controller.handle(ViewDungeon())
    assert breach_view_result.success
    assert isinstance(breach_view_result.value, DungeonView)
    assert any(
        action.value == "enter_generated_maze" and action.label == "Enter The Breach"
        for action in breach_view_result.value.actions
    )

    result = controller.handle(EnterGeneratedMaze(seed=11))

    assert result.success, result.error
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    assert session.current_node_id == session.generated_dungeon.entry_node_id
    assert session.generated_dungeon.run_id == "maze_run_0001"
    assert isinstance(result.value, DungeonView)
    view = result.value
    assert view.current_map_id == "generated_maze_breach"
    assert view.current_room.node_id == session.generated_dungeon.entry_node_id
    assert view.current_room.art_lines == tuple(
        controller.definitions.art.dungeon_nodes["generated_maze_entry"].lines
    )
    assert any(node.node_id.startswith("maze_run_0001") for node in view.map_nodes)
    assert any(
        action.value == "withdraw_generated_maze" and action.label == "Withdraw to Shallow Cave"
        for action in view.actions
    )
    assert not any(action.value == "retrace_generated_maze" for action in view.actions)
    assert not any(action.value == "return" for action in view.actions)

    first_room = next(
        node for node in session.generated_dungeon.nodes if node.id.endswith("_room_1")
    )
    first_room_result = controller.handle(MoveDungeon(first_room.id))

    assert first_room_result.success, first_room_result.error
    assert isinstance(first_room_result.value, DungeonView)
    assert first_room_result.value.current_room.art_lines == tuple(
        controller.definitions.art.dungeon_nodes["generated_maze_glass_room"].lines
    )
    assert any(
        action.value == "retrace_generated_maze" and action.label == "Retrace Steps"
        for action in first_room_result.value.actions
    )
    assert not any(
        action.value == "withdraw_generated_maze" for action in first_room_result.value.actions
    )


def test_generated_maze_entry_is_blocked_away_from_breach() -> None:
    controller = _started_interactive_controller()

    result = controller.handle(EnterGeneratedMaze(seed=3))

    assert not result.success
    assert result.error == "Enter the Maze from a breach."


def test_generated_maze_retreat_completes_scout_contract_and_returns_to_breach() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    assert "shallow_cave_breach_scout" not in controller.company.active_contract_ids
    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout"))
    assert accepted.success, accepted.error

    result = controller.handle(EnterGeneratedMaze(seed=5))
    assert result.success, result.error
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    _complete_initial_scout_requirements(controller)
    coin_before = controller.company.coin
    expected_coin = controller.definitions.world.contracts["shallow_cave_breach_scout"].coin_reward
    assert controller.company.active_expedition.report is not None
    report_coin_before = controller.company.active_expedition.report.coin_gained

    result = controller.handle(RetraceGeneratedMaze())

    assert result.success, result.error
    assert session.generated_dungeon is not None
    assert session.current_node_id == session.generated_dungeon.entry_node_id
    assert not session.generated_dungeon.collapsed
    assert "shallow_cave_breach_scout" in controller.company.active_contract_ids
    assert "shallow_cave_breach_scout" not in controller.company.completed_contract_ids
    assert not any(isinstance(event, ContractCompletedEvent) for event in result.events)
    assert not any(isinstance(event, MazeRouteCollapsedEvent) for event in result.events)
    assert controller.company.active_expedition is not None
    assert controller.company.last_expedition_report is None

    result = controller.handle(WithdrawGeneratedMaze())

    assert result.success, result.error
    assert session.current_node_id == "maze_breach"
    assert session.generated_dungeon.collapsed
    assert "shallow_cave_breach_scout" in controller.company.completed_contract_ids
    assert "shallow_cave_breach_scout" not in controller.company.active_contract_ids
    assert "shallow_cave_breach_hunt" not in controller.company.active_contract_ids
    assert any(isinstance(event, ContractCompletedEvent) for event in result.events)
    assert any(
        isinstance(event, LootGainedEvent) and event.coin == expected_coin
        for event in result.events
    )
    assert any(isinstance(event, MazeRouteCollapsedEvent) for event in result.events)
    assert controller.company.reputation >= 2
    assert controller.company.coin == coin_before + expected_coin
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.report is not None
    assert (
        controller.company.active_expedition.report.coin_gained
        == report_coin_before + expected_coin
    )

    accepted = controller.handle(AcceptContract("shallow_cave_breach_hunt"))
    assert accepted.success, accepted.error
    result = controller.handle(EnterGeneratedMaze(seed=6))

    assert result.success, result.error
    assert session.generated_dungeon is not None
    assert session.generated_dungeon.run_id == "maze_run_0002"
    assert any(node.id.endswith("_hunt_lair") for node in session.generated_dungeon.nodes)


def test_seamless_forward_move_blocked_away_from_frontier() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    result = controller.handle(EnterGeneratedMaze(seed=21))
    assert result.success, result.error
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    forward_id = _forward_preview_id(session)

    result = controller.handle(MoveDungeon(forward_id))
    assert not result.success
    assert result.error == "Choose a listed dungeon exit."

    first_room = session.generated_dungeon.nodes[0].exits[0]
    _move_along(controller, first_room)
    result = controller.handle(MoveDungeon(forward_id))
    assert not result.success
    assert result.error == "Choose a listed dungeon exit."


def test_seamless_forward_move_blocked_when_room_not_cleared() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=22))
    _reach_generated_frontier(controller, clear_frontier=False)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    result = controller.handle(MoveDungeon(_forward_preview_id(session)))
    assert not result.success
    assert result.error in {
        "Resolve the pending room combat first.",
        "Clear or inspect this room before moving on.",
    }


def test_seamless_forward_move_extends_route() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=23))
    _reach_generated_frontier(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    spine_before = session.generated_dungeon.main_spine_length

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    assert not any(
        action.value == "push_generated_maze_deeper"
        for action in view_result.value.actions
    )
    assert any(
        exit_view.status == "unstable"
        for exit_view in view_result.value.exits
    )
    forward_id = _forward_preview_id(session)
    map_nodes = {node.node_id: node for node in view_result.value.map_nodes}
    assert forward_id in map_nodes
    assert map_nodes[forward_id].status == "unstable"
    assert map_nodes[forward_id].map_x is not None
    assert map_nodes[forward_id].map_y is not None
    assert forward_id in map_nodes[session.current_node_id].exit_node_ids

    result = _move_forward_from_frontier(controller)

    assert result.success, result.error
    assert session.generated_dungeon.main_spine_length == spine_before + 1
    assert any(isinstance(event, MazeFrontierOpenedEvent) for event in result.events)
    assert isinstance(result.value, DungeonView)
    assert any(
        action.value == "retrace_generated_maze" for action in result.value.actions
    )
    assert result.value.current_room.node_id.endswith(f"_room_{spine_before + 1}")


def test_seamless_forward_move_past_initial_route_length() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=24))
    _reach_generated_frontier(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    initial_spine = session.generated_dungeon.main_spine_length

    for _ in range(2):
        result = _move_forward_from_frontier(controller)
        assert result.success, result.error
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)

    assert session.generated_dungeon.main_spine_length == initial_spine + 2
    assert any(
        node.id == f"{session.generated_dungeon.run_id}_room_{initial_spine + 2}"
        for node in session.generated_dungeon.nodes
    )


def test_seamless_forward_move_does_not_inflate_coin_without_combat() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=25))
    _reach_generated_frontier(controller)
    assert controller.company is not None
    coin_before = controller.company.coin

    for _ in range(4):
        result = _move_forward_from_frontier(controller)
        assert result.success, result.error
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)

    assert controller.company.coin == coin_before


def test_frontier_previews_all_appear_on_minimap() -> None:
    from game.ui.tui_widgets import DungeonMapPanel

    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=29))
    _reach_generated_frontier(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    forward_id = _forward_preview_id(session)
    spur_id = _spur_preview_id(session)
    if spur_id is None:
        pytest.skip("Seed 29 did not offer a spur preview at this frontier.")

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    view = view_result.value
    map_nodes = {node.node_id: node for node in view.map_nodes}
    assert forward_id in map_nodes
    assert spur_id in map_nodes
    assert map_nodes[forward_id].map_x is not None
    assert map_nodes[spur_id].map_x is not None
    assert (
        map_nodes[forward_id].map_x,
        map_nodes[forward_id].map_y,
    ) != (
        map_nodes[spur_id].map_x,
        map_nodes[spur_id].map_y,
    )

    map_text = DungeonMapPanel.render_minimap_text(view, actions=view.actions)
    forward_number = next(
        action.number for action in view.actions if action.value == forward_id
    )
    spur_number = next(action.number for action in view.actions if action.value == spur_id)
    assert f"{forward_number}?" in map_text
    assert f"{spur_number}?" in map_text


def test_view_dungeon_after_spur_still_shows_forward_preview() -> None:
    from game.expedition.generated_maze import frontier_node_id

    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=29))
    _reach_generated_frontier(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    spur_id = _spur_preview_id(session)
    if spur_id is None:
        pytest.skip("Seed 29 did not offer a spur preview at this frontier.")
    forward_id = _forward_preview_id(session)
    frontier_id = frontier_node_id(session.generated_dungeon)

    spur_result = controller.handle(MoveDungeon(spur_id))
    assert spur_result.success, spur_result.error
    return_result = controller.handle(MoveDungeon(frontier_id))
    assert return_result.success, return_result.error

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    map_nodes = {node.node_id: node for node in view_result.value.map_nodes}
    assert forward_id in map_nodes
    assert map_nodes[forward_id].status == "unstable"


def test_off_spine_hint_shown_on_spur_and_cleared_on_spine_return() -> None:
    from game.expedition.generated_maze import frontier_node_id
    from game.ui.tui_widgets import DungeonRoomPanel

    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=29))
    _reach_generated_frontier(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    spur_id = _spur_preview_id(session)
    if spur_id is None:
        pytest.skip("Seed 29 did not offer a spur preview at this frontier.")
    forward_id = _forward_preview_id(session)
    frontier_id = frontier_node_id(session.generated_dungeon)

    spur_result = controller.handle(MoveDungeon(spur_id))
    assert spur_result.success, spur_result.error
    if controller.manual_combat is not None:
        _win_active_manual_combat(controller)

    on_spur = controller.handle(ViewDungeon())
    assert on_spur.success, on_spur.error
    assert isinstance(on_spur.value, DungeonView)
    assert on_spur.value.maze_off_spine_hint
    assert "main route" in on_spur.value.maze_off_spine_hint
    assert forward_id not in {exit_view.node_id for exit_view in on_spur.value.exits}
    assert "main route" in DungeonRoomPanel.render_text(on_spur.value)

    return_result = controller.handle(MoveDungeon(frontier_id))
    assert return_result.success, return_result.error

    on_spine = controller.handle(ViewDungeon())
    assert on_spine.success, on_spine.error
    assert isinstance(on_spine.value, DungeonView)
    assert not on_spine.value.maze_off_spine_hint
    assert forward_id in {exit_view.node_id for exit_view in on_spine.value.exits}


def test_seamless_spur_move_generates_branch() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=29))
    _reach_generated_frontier(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    spur_id = _spur_preview_id(session)
    if spur_id is None:
        pytest.skip("Seed 29 did not offer a spur preview at this frontier.")
    spine_before = session.generated_dungeon.main_spine_length
    result = controller.handle(MoveDungeon(spur_id))
    assert result.success, result.error
    assert session.generated_dungeon.main_spine_length == spine_before
    assert any(node.id == spur_id for node in session.generated_dungeon.nodes)


def test_scout_contract_completes_after_push_to_required_depth() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(AcceptContract("shallow_cave_breach_scout"))
    controller.handle(EnterGeneratedMaze(seed=26))
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    initial_spine = session.generated_dungeon.main_spine_length

    for index in range(1, initial_spine + 1):
        _move_generated_main_room(controller, index)
        if index == 2:
            _use_current_generated_room_action(controller)
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)

    while _generated_actual_main_room_count(session) < 4:
        result = _move_forward_from_frontier(controller)
        assert result.success, result.error
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)
        current = session.current_node_id
        if current.endswith("_room_2") or (
            session.generated_dungeon is not None
            and any(
                node.id == current and node.actions
                for node in session.generated_dungeon.nodes
            )
        ):
            node = next(
                node
                for node in session.generated_dungeon.nodes
                if node.id == current and node.actions
            )
            if node.actions:
                action_result = controller.handle(UseDungeonAction(node.actions[0].id))
                assert action_result.success, action_result.error

    result = _retrace_then_withdraw_generated_maze(controller)
    assert result.success, result.error
    assert "shallow_cave_breach_scout" in controller.company.completed_contract_ids


def test_retrace_and_withdraw_after_deep_push() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=27))
    _reach_generated_frontier(controller)
    _move_forward_from_frontier(controller)
    if controller.manual_combat is not None:
        _win_active_manual_combat(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    assert (
        session.generated_dungeon.main_spine_length
        > session.generated_dungeon.recipe.route_length
    )

    result = _retrace_then_withdraw_generated_maze(controller)
    assert result.success, result.error
    collapse = next(
        event for event in result.events if isinstance(event, MazeRouteCollapsedEvent)
    )
    assert collapse.main_depth_reached > 0
    assert "breach remains" in collapse.message.lower()


def test_generated_maze_save_load_preserves_extended_spine(tmp_path: Path) -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=28))
    _reach_generated_frontier(controller)
    _move_forward_from_frontier(controller)
    if controller.manual_combat is not None:
        _win_active_manual_combat(controller)
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    spine_length = session.generated_dungeon.main_spine_length
    current_room = session.current_node_id
    save_path = tmp_path / "extended_maze.json"
    save_company(controller.company, save_path)
    loaded, _event = load_company(save_path)
    loaded_session = loaded.active_expedition
    assert loaded_session is not None
    assert loaded_session.generated_dungeon is not None
    assert loaded_session.generated_dungeon.main_spine_length == spine_length
    assert loaded_session.current_node_id == current_room
    assert any(
        node.id == f"{loaded_session.generated_dungeon.run_id}_room_{spine_length}"
        for node in loaded_session.generated_dungeon.nodes
    )


def test_generated_maze_scout_requires_survey_action_before_completion() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout"))
    assert accepted.success, accepted.error

    result = controller.handle(EnterGeneratedMaze(seed=5))
    assert result.success, result.error
    for index in range(1, 5):
        _move_generated_main_room(controller, index)
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)

    result = _retrace_then_withdraw_generated_maze(controller)

    assert result.success, result.error
    assert "shallow_cave_breach_scout" in controller.company.active_contract_ids
    assert "shallow_cave_breach_scout" not in controller.company.completed_contract_ids
    assert not any(
        isinstance(event, ContractCompletedEvent)
        and event.contract_id == "shallow_cave_breach_scout"
        for event in result.events
    )


def test_legacy_generated_maze_retreat_alias_retraces_without_collapsing() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    result = controller.handle(EnterGeneratedMaze(seed=5))
    assert result.success, result.error
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    first_room = session.generated_dungeon.nodes[0].exits[0]
    _move_along(controller, first_room)

    result = controller.handle(RetreatGeneratedMaze())

    assert result.success, result.error
    assert session.current_node_id == session.generated_dungeon.entry_node_id
    assert not session.generated_dungeon.collapsed
    assert controller.company.active_expedition is not None
    assert controller.company.last_expedition_report is None


def test_generated_maze_hunt_contract_completes_after_marked_lair() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    result = controller.handle(AcceptContract("shallow_cave_breach_scout"))
    assert result.success, result.error
    controller.handle(EnterGeneratedMaze(seed=5))
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    _complete_initial_scout_requirements(controller)
    result = _retrace_then_withdraw_generated_maze(controller)
    assert result.success, result.error
    assert "shallow_cave_breach_hunt" not in controller.company.active_contract_ids
    result = controller.handle(AcceptContract("shallow_cave_breach_hunt"))
    assert result.success, result.error
    assert "shallow_cave_breach_hunt" in controller.company.active_contract_ids

    result = controller.handle(EnterGeneratedMaze(seed=9))
    assert result.success, result.error
    assert session.generated_dungeon is not None
    _complete_generated_hunt_requirements(controller)
    coin_before = controller.company.coin
    expected_coin = controller.definitions.world.contracts["shallow_cave_breach_hunt"].coin_reward
    assert controller.company.active_expedition.report is not None
    report_coin_before = controller.company.active_expedition.report.coin_gained

    result = _retrace_then_withdraw_generated_maze(controller)

    assert result.success, result.error
    assert "shallow_cave_breach_hunt" in controller.company.completed_contract_ids
    assert "shallow_cave_breach_hunt" not in controller.company.active_contract_ids
    assert controller.company.gear_inventory["maze_glass_talisman"] == 1
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.report is not None
    assert controller.company.active_expedition.report.gear == {"maze_glass_talisman": 1}
    assert controller.company.coin == coin_before + expected_coin
    assert (
        controller.company.active_expedition.report.coin_gained
        == report_coin_before + expected_coin
    )
    assert any(
        isinstance(event, ContractCompletedEvent)
        and event.contract_id == "shallow_cave_breach_hunt"
        for event in result.events
    )
    assert any(
        isinstance(event, LootGainedEvent) and event.coin == expected_coin
        for event in result.events
    )


def test_repeatable_generated_maze_scout_can_be_accepted_again_after_completion() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    controller.company.completed_contract_ids.add("shallow_cave_breach_scout")

    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout_posting"))
    assert accepted.success, accepted.error
    result = controller.handle(EnterGeneratedMaze(seed=21))
    assert result.success, result.error
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    _complete_repeatable_scout_requirements(controller)

    result = _retrace_then_withdraw_generated_maze(controller)

    assert result.success, result.error
    assert "shallow_cave_breach_scout_posting" not in controller.company.active_contract_ids
    assert "shallow_cave_breach_scout_posting" not in controller.company.completed_contract_ids
    assert any(
        isinstance(event, ContractCompletedEvent)
        and event.contract_id == "shallow_cave_breach_scout_posting"
        for event in result.events
    )
    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout_posting"))
    assert accepted.success, accepted.error


def test_repeatable_generated_maze_hunt_adds_marked_lair_and_repeats() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    controller.company.completed_contract_ids.update(
        {
            "shallow_cave_breach_scout",
            "shallow_cave_breach_hunt",
        }
    )

    accepted = controller.handle(AcceptContract("shallow_cave_breach_hunt_posting"))
    assert accepted.success, accepted.error
    result = controller.handle(EnterGeneratedMaze(seed=22))
    assert result.success, result.error
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    assert any(node.id.endswith("_hunt_lair") for node in session.generated_dungeon.nodes)

    _complete_generated_hunt_requirements(controller)

    result = _retrace_then_withdraw_generated_maze(controller)

    assert result.success, result.error
    assert "shallow_cave_breach_hunt_posting" not in controller.company.active_contract_ids
    assert "shallow_cave_breach_hunt_posting" not in controller.company.completed_contract_ids
    assert any(
        isinstance(event, ContractCompletedEvent)
        and event.contract_id == "shallow_cave_breach_hunt_posting"
        for event in result.events
    )
    accepted = controller.handle(AcceptContract("shallow_cave_breach_hunt_posting"))
    assert accepted.success, accepted.error


def test_generated_maze_combat_room_resolves_and_can_retrace_then_withdraw() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=7))
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    combat_node = next(node for node in session.generated_dungeon.nodes if node.encounter)
    room_1 = next(
        node.id for node in session.generated_dungeon.nodes if node.id.endswith("_room_1")
    )
    room_2 = next(
        node.id for node in session.generated_dungeon.nodes if node.id.endswith("_room_2")
    )
    _move_along(controller, room_1, room_2, combat_node.id)

    assert controller.manual_combat is not None
    assert session.pending_combat_node_id == combat_node.id
    _win_active_manual_combat(controller)

    assert controller.manual_combat is None
    assert combat_node.id in session.cleared_node_ids
    result = _retrace_then_withdraw_generated_maze(controller)
    assert result.success, result.error
    assert session.current_node_id == "maze_breach"


def test_generated_maze_combat_retreat_returns_to_threshold_without_clearing_room() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=7))
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    combat_node = next(node for node in session.generated_dungeon.nodes if node.encounter)
    room_1 = next(
        node.id for node in session.generated_dungeon.nodes if node.id.endswith("_room_1")
    )
    room_2 = next(
        node.id for node in session.generated_dungeon.nodes if node.id.endswith("_room_2")
    )
    _move_along(controller, room_1, room_2, combat_node.id)
    assert controller.manual_combat is not None
    actor = controller.manual_combat.pending_hero()
    assert actor is not None
    controller.manual_combat.initiative = [InitiativeEntry(actor.actor_id, 99)]
    controller.manual_combat.turn_index = 0

    result = controller.handle(RetreatCombat())

    assert result.success, result.error
    assert controller.manual_combat is None
    assert session.current_node_id == session.generated_dungeon.entry_node_id
    assert session.pending_combat_node_id is None
    assert combat_node.id not in session.cleared_node_ids
    assert not session.generated_dungeon.collapsed
    assert any(isinstance(event, CombatRetreatedEvent) for event in result.events)

    _move_along(controller, room_1, room_2)
    restart = controller.handle(MoveDungeon(combat_node.id))

    assert restart.success, restart.error
    assert controller.manual_combat is not None
    assert session.pending_combat_node_id == combat_node.id


def test_generated_maze_save_load_preserves_active_route(tmp_path: Path) -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    controller.handle(EnterGeneratedMaze(seed=17))
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    assert session.generated_dungeon.recipe is not None
    first_room = session.generated_dungeon.nodes[0].exits[0]
    _move_along(controller, first_room)
    save_path = tmp_path / "company.json"

    save_company(controller.company, save_path)
    loaded, _event = load_company(save_path)
    loaded_controller = AppController(
        definitions=controller.definitions,
        rng=controller.rng,
        company=loaded,
    )
    result = loaded_controller.handle(ViewDungeon())

    assert result.success, result.error
    assert isinstance(result.value, DungeonView)
    loaded_session = loaded.active_expedition
    assert loaded_session is not None
    assert loaded_session.generated_dungeon is not None
    assert loaded_session.generated_dungeon.seed == 17
    assert loaded_session.generated_dungeon.run_id == "maze_run_0001"
    assert loaded_session.generated_dungeon.recipe is not None
    assert loaded_session.generated_dungeon.recipe.pressure_id == (
        session.generated_dungeon.recipe.pressure_id
    )
    assert loaded_session.current_node_id == first_room
    assert first_room in loaded_session.generated_dungeon.visited_node_ids
    assert result.value.current_room.node_id == first_room


def test_safe_room_return_creates_regional_map_and_filed_record() -> None:
    controller = _started_interactive_controller()

    result = controller.handle(ReturnFromDungeon())

    assert result.success
    assert isinstance(result.value, RegionalMapView)
    assert result.value.current_node_id == "town_gate"
    assert result.value.arrival_context is not None
    assert result.value.arrival_context.origin_name == "Haven East Gate"
    assert result.value.arrival_context.location_id == "old_road"
    assert "Company record filed." in result.value.arrival_context.what_changed
    assert controller.company is not None
    assert controller.company.active_expedition is None
    assert controller.company.last_expedition_report is not None
    assert controller.company.last_expedition_report.outcome == "returned_to_haven"
    assert controller.company.expedition_reports == [controller.company.last_expedition_report]
    assert controller.company.company_timeline[0].kind == "expedition_started"
    assert len(controller.company.hero_memories) == len(
        controller.company.last_expedition_report.participant_ids
    )

    view_result = controller.handle(ViewExpeditionReport())

    assert view_result.success
    assert isinstance(view_result.value, ExpeditionReportView)
    view = view_result.value
    assert any(delta[0] == "rations" and delta[3] == -1 for delta in view.supply_deltas)
    assert view.hero_outcomes
    assert view.notable_moments


def test_regional_travel_charted_hop_compresses_opening_route() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    assert controller.company is not None
    assert "shallow_cave" not in controller.company.known_route_ids

    result = controller.handle(ReturnFromDungeon())

    assert result.success
    assert isinstance(result.value, RegionalMapView)
    assert result.value.current_node_id == "shallow_cave_entrance"
    assert controller.company.active_expedition is None
    assert len(controller.company.expedition_reports) == 1
    assert "shallow_cave" in controller.company.known_route_ids

    map_result = controller.handle(ViewRegionalMap())
    assert map_result.success
    assert map_result.value.route_charted is True
    assert map_result.value.travel_available is True

    rations_before = controller.company.supplies.get("rations", 0)
    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))

    assert result.success
    assert isinstance(result.value, DungeonView)
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.current_node_id == "shallow_cave_room_1"
    assert controller.company.supplies.get("rations", 0) == max(0, rations_before - 1)
    assert any(
        isinstance(event, ExpeditionEvent)
        and event.node_id == "known_route_road"
        and "charted road" in event.message
        for event in result.events
    )
    result = controller.handle(ReturnFromDungeon())
    assert result.success, result.error
    assert isinstance(result.value, RegionalMapView)
    assert result.value.current_node_id == "shallow_cave_entrance"

    direct_result = controller.handle(
        StartExpedition(
            manual_combat=True,
            interactive_dungeon=True,
            skip_known_route_playback=True,
        )
    )
    assert direct_result.success, direct_result.error
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.current_node_id == "shallow_cave_room_1"
    direct_event_node_ids = [
        event.node_id for event in direct_result.events if isinstance(event, ExpeditionEvent)
    ]
    assert "known_route_road" not in direct_event_node_ids
    assert direct_event_node_ids[0] == "shallow_cave_room_1"

    result = controller.handle(ReturnFromDungeon())
    assert result.success, result.error
    assert isinstance(result.value, RegionalMapView)
    assert result.value.current_node_id == "shallow_cave_entrance"
    assert len(controller.company.expedition_reports) == 3
    assert controller.company.last_expedition_report == controller.company.expedition_reports[-1]

    travel_result = controller.handle(TravelRegional("haven"))
    assert travel_result.success, travel_result.error
    assert isinstance(travel_result.value, RegionalMapView)
    assert travel_result.value.current_node_id == "town_gate"
    assert travel_result.value.travel_flavor is not None

    result = controller.handle(
        StartExpedition(
            manual_combat=True,
            interactive_dungeon=True,
            use_known_route=False,
        )
    )
    assert result.success
    assert isinstance(result.value, DungeonView)
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.current_node_id == "town_gate"
    event_node_ids = [
        event.node_id for event in result.events if isinstance(event, ExpeditionEvent)
    ]
    assert event_node_ids[0] == "town_gate"
    assert "known_route_road" not in event_node_ids


def test_travel_regional_blocked_before_route_is_charted() -> None:
    controller = _started_interactive_controller()
    result = controller.handle(TravelRegional("shallow_cave"))
    assert not result.success
    assert "Chart the route" in (result.error or "")


def test_return_fails_at_stone_gate() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)
    _win_active_manual_combat(controller)
    controller.handle(UseDungeonAction("recover_gate_key"))
    _move_along(controller, "fungus_chamber", "stone_gate")

    result = controller.handle(ReturnFromDungeon())
    assert not result.success
    assert result.error == "Return is only available from safe return rooms."


def test_return_fails_at_maze_breach() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.current_node_id == "maze_breach"

    result = controller.handle(ReturnFromDungeon())
    assert not result.success
    assert result.error == "Return is only available from safe return rooms."


def test_travel_regional_haven_lands_at_east_gate_view() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    assert controller.company.town_state["location_id"] == "shallow_cave"
    _mark_charted_route_at_cave_entrance(controller)

    result = controller.handle(TravelRegional("haven"))
    assert result.success, result.error
    view = result.value
    assert isinstance(view, RegionalMapView)
    assert view.current_node_id == "town_gate"
    assert view.place_title == "East Gate"
    assert controller.company.town_state["location_id"] == "haven"
    assert controller.company.town_state["regional_node_id"] == "town_gate"


def test_visit_east_gate_from_haven_sets_regional_view() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    assert controller.company is not None
    assert controller.company.town_state["location_id"] == "haven"
    assert controller.company.active_expedition is None

    result = controller.handle(VisitEastGate())
    assert result.success, result.error
    view = result.value
    assert isinstance(view, RegionalMapView)
    assert view.current_node_id == "town_gate"
    assert view.anchor_kind == "east_gate"
    assert controller.company.town_state["regional_node_id"] == "town_gate"


def test_return_to_haven_town_from_east_gate() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    assert controller.company is not None

    visit_result = controller.handle(VisitEastGate())
    assert visit_result.success, visit_result.error
    regional_node_id = controller.company.town_state["regional_node_id"]

    result = controller.handle(ReturnToHavenTown())
    assert result.success, result.error
    assert isinstance(result.value, TownDashboardView)
    assert controller.company.town_state["location_id"] == "haven"
    assert controller.company.town_state["regional_node_id"] == regional_node_id


def test_defeat_return_still_lands_in_haven() -> None:
    from game.expedition.dungeon import return_from_dungeon

    controller = _started_interactive_controller()
    assert controller.company is not None
    return_from_dungeon(controller.company, controller.definitions, outcome="defeat")
    assert controller.company.town_state["location_id"] == "haven"
    assert controller.company.active_expedition is None


def test_return_to_haven_is_only_available_from_safe_nodes() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
    )

    view_result = controller.handle(ViewDungeon())

    assert view_result.success
    assert isinstance(view_result.value, DungeonView)
    assert not any(action.value == "return" for action in view_result.value.actions)

    result = controller.handle(ReturnFromDungeon())

    assert not result.success
    assert result.error == "Return is only available from safe return rooms."


def test_stable_dungeon_memory_survives_return_to_town() -> None:
    controller = _started_interactive_controller()
    _reach_cache(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None

    result = controller.handle(UseDungeonAction("recover_gate_key"))
    assert result.success, result.error
    _move_along(controller, "fungus_chamber", "stone_gate")
    result = controller.handle(UseDungeonAction("unlock_black_gate"))
    assert result.success, result.error

    memory = controller.company.dungeon_memory["shallow_cave"]
    assert "old_works_cache" in memory.cleared_node_ids
    assert "old_works_cache:recover_gate_key" in memory.completed_action_ids
    assert "stone_gate->maze_touched_lair" in memory.revealed_exit_ids
    world_memory = controller.company.world_memory["shallow_cave"]
    assert "old_works_cache" in world_memory.discovered_node_ids
    assert "old_works_cache" in world_memory.cleared_threat_node_ids
    assert "stone_gate->maze_touched_lair" in world_memory.unlocked_shortcut_ids

    return_at_gate = controller.handle(ReturnFromDungeon())
    assert not return_at_gate.success
    assert return_at_gate.error == "Return is only available from safe return rooms."

    _move_along(controller, "narrow_crawl", "cave_fork", "shallow_cave_room_1")
    result = controller.handle(ReturnFromDungeon())
    assert result.success, result.error
    assert isinstance(result.value, RegionalMapView)
    assert result.value.current_node_id == "shallow_cave_entrance"

    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))
    assert result.success, result.error
    assert isinstance(result.value, DungeonView)
    assert not result.value.current_room.first_visit

    session = controller.company.active_expedition
    assert session is not None
    assert "old_works_cache" in session.cleared_node_ids
    assert "old_works_cache:recover_gate_key" in session.completed_action_ids
    assert "stone_gate->maze_touched_lair" in session.revealed_exit_ids

    _move_along(controller, "cave_fork", "fungus_chamber", "old_works_cache")

    assert controller.manual_combat is None
    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    recover_key = next(
        action
        for action in view_result.value.room_actions
        if action.action_id == "recover_gate_key"
    )
    assert recover_key.state == "completed"

    _move_along(controller, "fungus_chamber", "stone_gate")
    view_result = controller.handle(ViewDungeon())

    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "maze_touched_lair" and action.enabled
        for action in view_result.value.actions
    )


def test_cleared_boss_room_remembers_route_to_breach() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None
    assert controller.company.active_expedition is not None

    memory = controller.company.dungeon_memory["shallow_cave"]
    assert "maze_touched_lair" in memory.cleared_node_ids
    assert "maze_breach" in memory.cleared_node_ids
    assert "maze_touched_lair->maze_breach" in memory.revealed_exit_ids

    result = controller.handle(TakeExpeditionChoice("return_to_haven"))
    assert result.success, result.error
    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))
    assert result.success, result.error

    _move_along(
        controller,
        "cave_fork",
        "fungus_chamber",
        "stone_gate",
        "maze_touched_lair",
    )

    assert controller.manual_combat is None
    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "maze_breach" and action.enabled for action in view_result.value.actions
    )
    assert any(
        action.value == "stone_gate" and action.enabled for action in view_result.value.actions
    )

    result = controller.handle(MoveDungeon("maze_breach"))

    assert result.success, result.error
    assert controller.company.active_expedition is not None
    assert controller.company.active_expedition.current_node_id == "maze_breach"

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "maze_touched_lair" and action.enabled
        for action in view_result.value.actions
    )

    result = controller.handle(MoveDungeon("maze_touched_lair"))
    assert result.success, result.error

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert isinstance(view_result.value, DungeonView)
    assert any(
        action.value == "stone_gate" and action.enabled for action in view_result.value.actions
    )

    result = controller.handle(MoveDungeon("stone_gate"))
    assert result.success, result.error
    assert controller.company.active_expedition.current_node_id == "stone_gate"

    view_result = controller.handle(ViewDungeon())
    assert view_result.success, view_result.error
    assert any(
        action.value == "fungus_chamber" and action.enabled for action in view_result.value.actions
    )
    assert not any(action.value == "return" for action in view_result.value.actions)

    result = controller.handle(ReturnFromDungeon())
    assert not result.success
    assert result.error == "Return is only available from safe return rooms."


def test_generated_maze_contract_routes_stay_out_of_persistent_dungeon_memory() -> None:
    controller = _started_interactive_controller()
    _reach_boss(controller)
    _win_active_manual_combat(controller)
    assert controller.company is not None

    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout"))
    assert accepted.success, accepted.error
    result = controller.handle(EnterGeneratedMaze(seed=5))
    assert result.success, result.error
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    _complete_initial_scout_requirements(controller)
    result = _retrace_then_withdraw_generated_maze(controller)
    assert result.success, result.error

    memory = controller.company.dungeon_memory["shallow_cave"]
    remembered_values = [
        *memory.visited_node_ids,
        *memory.cleared_node_ids,
        *memory.completed_action_ids,
        *memory.revealed_exit_ids,
    ]
    assert not any(value.startswith("maze_run_") for value in remembered_values)
    assert "maze_breach" in controller.company.breach_memory
    breach_memory = controller.company.breach_memory["maze_breach"]
    assert breach_memory.run_count == 1
    assert breach_memory.last_seed == 5
    assert breach_memory.collapsed_run_ids == ["maze_run_0001"]
    assert breach_memory.scouted_run_ids == ["maze_run_0001"]
    world_values = [
        value
        for memory in controller.company.world_memory.values()
        for value in [
            *memory.discovered_node_ids,
            *memory.cleared_threat_node_ids,
            *memory.consumed_rumor_ids,
            *memory.unlocked_shortcut_ids,
        ]
    ]
    assert not any(value.startswith("maze_run_") for value in world_values)

    accepted = controller.handle(AcceptContract("shallow_cave_breach_hunt"))
    assert accepted.success, accepted.error
    result = controller.handle(EnterGeneratedMaze(seed=6))

    assert result.success, result.error
    assert session.generated_dungeon is not None
    assert session.generated_dungeon.run_id == "maze_run_0002"
    assert controller.company.breach_memory["maze_breach"].run_count == 2
    assert controller.company.breach_memory["maze_breach"].last_seed == 6
    assert any(node.id.endswith("_hunt_lair") for node in session.generated_dungeon.nodes)


def test_active_dungeon_progress_saves_and_loads(tmp_path: Path) -> None:
    controller = _started_interactive_controller()
    assert controller.company is not None
    save_path = tmp_path / "company.json"

    save_company(controller.company, save_path)
    loaded, _event = load_company(save_path)

    assert loaded.active_expedition is not None
    assert loaded.active_expedition.current_node_id == "town_gate"
    assert "town_gate" in loaded.active_expedition.visited_node_ids
    assert loaded.last_expedition_report is None


def test_old_save_without_expedition_fields_loads(tmp_path: Path) -> None:
    definitions = get_definitions()
    controller = AppController(definitions=definitions)
    controller.handle(StartNewCompany())
    assert controller.company is not None
    raw = controller.company.to_dict()
    raw.pop("active_expedition")
    raw.pop("last_expedition_report")
    raw.pop("hero_memories")
    raw.pop("company_timeline")
    raw.pop("dungeon_memory")
    raw.pop("world_memory")
    raw.pop("recruitment_state")
    raw.pop("contract_records")
    raw.pop("breach_memory")
    raw.pop("expedition_reports")
    raw["save_version"] = 2
    save_path = tmp_path / "old.json"
    save_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded, _event = load_company(save_path)

    assert loaded.active_expedition is None
    assert loaded.last_expedition_report is None
    assert loaded.hero_memories == []
    assert loaded.company_timeline == []
    assert loaded.dungeon_memory == {}
    assert loaded.world_memory == {}
    assert loaded.recruitment_state.current_offers == []
    assert loaded.recruitment_state.refresh_count == 0
    assert loaded.contract_records == {}
    assert loaded.breach_memory == {}
    assert loaded.expedition_reports == []


def test_last_expedition_report_saves_and_loads(tmp_path: Path) -> None:
    controller = _started_interactive_controller()
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    save_path = tmp_path / "report.json"

    save_company(controller.company, save_path)
    loaded, _event = load_company(save_path)

    assert loaded.active_expedition is None
    assert loaded.last_expedition_report is not None
    assert loaded.last_expedition_report.outcome == "returned_to_haven"
    assert len(loaded.expedition_reports) == 1
    assert loaded.expedition_reports[0].outcome == "returned_to_haven"
    assert loaded.hero_memories
    assert loaded.company_timeline
    assert loaded.last_expedition_report.notable_moments
    assert loaded.last_expedition_report.start_supplies
    assert loaded.last_expedition_report.end_supplies


def test_invalid_exit_reference_fails_loader_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(Path("data"), data_dir)
    expedition_path = data_dir / "expeditions.yaml"
    expedition_path.write_text(
        expedition_path.read_text(encoding="utf-8").replace(
            "exits: [town_gate, abandoned_toll_post, hunters_trail, bramble_shrine]",
            "exits: [missing_room]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown exit node: missing_room"):
        load_game_definitions(data_dir)


def test_duplicate_node_position_fails_loader_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(Path("data"), data_dir)
    expedition_path = data_dir / "expeditions.yaml"
    expedition_path.write_text(
        expedition_path.read_text(encoding="utf-8").replace(
            "position: [0, 1]",
            "position: [0, 0]",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate map position"):
        load_game_definitions(data_dir)


def test_diagonal_node_link_fails_loader_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(Path("data"), data_dir)
    expedition_path = data_dir / "expeditions.yaml"
    expedition_path.write_text(
        expedition_path.read_text(encoding="utf-8").replace(
            "position: [2, -1]",
            "position: [1, -1]",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Map link must be cardinal"):
        load_game_definitions(data_dir)


def _started_interactive_controller() -> AppController:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))
    assert result.success
    return controller


def _move_along(controller: AppController, *node_ids: str) -> None:
    for node_id in node_ids:
        result = controller.handle(MoveDungeon(node_id))
        assert result.success, result.error


def _mark_charted_route_at_cave_entrance(controller: AppController) -> None:
    assert controller.company is not None
    if "shallow_cave" in controller.company.known_route_ids:
        return
    result = controller.handle(MarkRegionalRoute())
    assert result.success, result.error


def _generated_actual_main_room_count(session) -> int:
    assert session.generated_dungeon is not None
    entry_id = session.generated_dungeon.entry_node_id
    return sum(
        1
        for node_id in session.generated_dungeon.visited_node_ids
        if node_id.startswith("maze_run_")
        and node_id != entry_id
        and "_room_" in node_id
    )


def _forward_preview_id(session) -> str:
    from game.expedition.generated_maze import frontier_exit_previews, frontier_node_id

    assert session.generated_dungeon is not None
    previews = frontier_exit_previews(
        session.generated_dungeon,
        frontier_node_id(session.generated_dungeon),
    )
    return next(preview.exit_id for preview in previews if preview.kind == "forward")


def _spur_preview_id(session) -> str | None:
    from game.expedition.generated_maze import frontier_exit_previews, frontier_node_id

    assert session.generated_dungeon is not None
    previews = frontier_exit_previews(
        session.generated_dungeon,
        frontier_node_id(session.generated_dungeon),
    )
    for preview in previews:
        if preview.kind == "spur":
            return preview.exit_id
    return None


def _move_forward_from_frontier(controller: AppController):
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    return controller.handle(MoveDungeon(_forward_preview_id(session)))


def _reach_generated_frontier(
    controller: AppController,
    *,
    clear_frontier: bool = True,
) -> None:
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    spine_length = session.generated_dungeon.main_spine_length
    for index in range(1, spine_length + 1):
        _move_generated_main_room(controller, index)
        if controller.manual_combat is not None and clear_frontier:
            _win_active_manual_combat(controller)
        elif controller.manual_combat is not None and index < spine_length:
            _win_active_manual_combat(controller)


def _complete_initial_scout_requirements(controller: AppController) -> None:
    for index in range(1, 5):
        _move_generated_main_room(controller, index)
        if index == 2:
            _use_current_generated_room_action(controller)
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)


def _complete_repeatable_scout_requirements(controller: AppController) -> None:
    _move_generated_main_room(controller, 1)
    _move_generated_main_room(controller, 2)
    reward_node_id = _generated_node_id(controller, "_reward")
    result = controller.handle(MoveDungeon(reward_node_id))
    assert result.success, result.error
    _use_current_generated_room_action(controller)
    _move_generated_main_room(controller, 2)
    _move_generated_main_room(controller, 3)
    if controller.manual_combat is not None:
        _win_active_manual_combat(controller)


def _complete_generated_hunt_requirements(controller: AppController) -> None:
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    hunt_node_id = _generated_node_id(controller, "_hunt_lair")
    hunt_node = next(
        node for node in session.generated_dungeon.nodes if node.id == hunt_node_id
    )
    final_main_index = int(hunt_node.exits[0].rsplit("_room_", 1)[1])
    for index in range(1, final_main_index + 1):
        _move_generated_main_room(controller, index)
        if controller.manual_combat is not None:
            _win_active_manual_combat(controller)
    result = controller.handle(MoveDungeon(hunt_node_id))
    assert result.success, result.error
    assert controller.manual_combat is not None
    _win_active_manual_combat(controller)


def _move_generated_main_room(controller: AppController, index: int) -> None:
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    result = controller.handle(MoveDungeon(f"{session.generated_dungeon.run_id}_room_{index}"))
    assert result.success, result.error


def _generated_node_id(controller: AppController, suffix: str) -> str:
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    return next(
        node.id
        for node in session.generated_dungeon.nodes
        if node.id.endswith(suffix)
    )


def _use_current_generated_room_action(controller: AppController) -> None:
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    node = next(
        node
        for node in session.generated_dungeon.nodes
        if node.id == session.current_node_id
    )
    assert node.actions
    result = controller.handle(UseDungeonAction(node.actions[0].id))
    assert result.success, result.error


def _retrace_then_withdraw_generated_maze(controller: AppController):
    assert controller.company is not None
    session = controller.company.active_expedition
    assert session is not None
    assert session.generated_dungeon is not None
    entry_node_id = session.generated_dungeon.entry_node_id

    result = controller.handle(RetraceGeneratedMaze())
    assert result.success, result.error
    assert session.current_node_id == entry_node_id
    assert not session.generated_dungeon.collapsed
    assert controller.company.active_expedition is not None
    assert controller.company.last_expedition_report is None

    result = controller.handle(WithdrawGeneratedMaze())
    assert result.success, result.error
    return result


def _move_to_cache(controller: AppController):
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
        "fungus_chamber",
    )
    return controller.handle(MoveDungeon("old_works_cache"))


def _reach_cache(controller: AppController) -> None:
    result = _move_to_cache(controller)
    assert result.success, result.error


def test_regional_node_id_migration_from_legacy_save() -> None:
    from game.campaign.company import _normalize_town_state

    haven_state = _normalize_town_state({"location_id": "haven"})
    cave_state = _normalize_town_state({"location_id": "shallow_cave"})

    assert haven_state["regional_node_id"] == "town_gate"
    assert cave_state["regional_node_id"] == "shallow_cave_entrance"


def test_build_regional_map_view_includes_discovered_wilderness_nodes() -> None:
    from game.app.views import build_regional_map_view

    controller = _started_interactive_controller()
    _move_along(controller, "old_road", "hunters_trail", "old_road", "town_gate")
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None

    view = build_regional_map_view(controller.company, controller.definitions)
    map_node_ids = {node.node_id for node in view.map_nodes}

    assert view.current_node_id == "town_gate"
    assert "old_road" in map_node_ids
    assert "hunters_trail" in map_node_ids
    assert "dry_creek_bed" in map_node_ids
    assert len(map_node_ids) > 2


def test_regional_map_marks_shallow_cave_entrance_for_active_charter() -> None:
    from game.app.views import build_regional_map_view

    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
    )
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None

    view = build_regional_map_view(controller.company, controller.definitions)
    map_nodes = {node.node_id: node for node in view.map_nodes}

    assert map_nodes["shallow_cave_entrance"].quest_marker
    assert not map_nodes["town_gate"].quest_marker


def test_move_regional_walks_adjacent_node_after_route_charted() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    _mark_charted_route_at_cave_entrance(controller)
    controller.handle(TravelRegional("haven"))
    assert controller.company is not None
    rations_before = controller.company.supplies.get("rations", 0)

    result = controller.handle(MoveRegional("old_road"))

    assert result.success, result.error
    assert isinstance(result.value, RegionalMapView)
    assert result.value.current_node_id == "old_road"
    assert controller.company.town_state["regional_node_id"] == "old_road"
    assert controller.company.supplies.get("rations", 0) == rations_before


def test_move_regional_blocked_before_route_is_charted() -> None:
    controller = _started_interactive_controller()
    assert controller.company is not None
    controller.company.town_state["regional_node_id"] = "town_gate"
    assert "shallow_cave" not in controller.company.known_route_ids

    result = controller.handle(MoveRegional("old_road"))

    assert not result.success
    assert "active expedition" in (result.error or "")


def test_move_regional_walks_cave_exit_nodes_before_route_charted() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    assert "shallow_cave" in controller.company.known_route_ids

    result = controller.handle(MoveRegional("black_stone_sinkhole"))

    assert result.success, result.error
    assert isinstance(result.value, RegionalMapView)
    assert result.value.current_node_id == "black_stone_sinkhole"

    next_result = controller.handle(MoveRegional("dry_creek_bed"))

    assert next_result.success, next_result.error
    assert isinstance(next_result.value, RegionalMapView)
    assert next_result.value.current_node_id == "dry_creek_bed"


def test_move_regional_blocked_for_non_adjacent_node() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    _mark_charted_route_at_cave_entrance(controller)
    controller.handle(TravelRegional("haven"))

    result = controller.handle(MoveRegional("bandit_camp"))

    assert not result.success
    assert "listed regional exit" in (result.error or "")


def test_travel_regional_sets_anchor_regional_node_id() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    assert controller.company.town_state["regional_node_id"] == "shallow_cave_entrance"
    assert "shallow_cave" in controller.company.known_route_ids

    controller.handle(TravelRegional("haven"))
    result = controller.handle(TravelRegional("shallow_cave"))

    assert result.success, result.error
    assert controller.company.town_state["regional_node_id"] == "shallow_cave_entrance"
    assert isinstance(result.value, RegionalMapView)
    assert result.value.anchor_kind == "shallow_cave"


def test_auto_chart_on_cave_entrance_arrival() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    assert controller.company is not None
    assert "shallow_cave" not in controller.company.known_route_ids

    result = controller.handle(ReturnFromDungeon())

    assert result.success
    assert "shallow_cave" in controller.company.known_route_ids
    assert any(
        "Charted approach mapped" in event.message
        for event in result.events
        if hasattr(event, "message")
    )


def test_regional_anchor_actions_follow_current_node() -> None:
    from game.app.actions import ActionProvider
    from game.app.views import build_regional_map_view

    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    assert "shallow_cave" in controller.company.known_route_ids

    cave_view = build_regional_map_view(controller.company, controller.definitions)
    cave_actions = ActionProvider.regional_place_actions(cave_view)
    cave_values = [action.value for action in cave_actions]
    assert cave_values.index("black_stone_sinkhole") < cave_values.index("enter_cave")
    assert cave_values.index("enter_cave") < cave_values.index("survey_route")
    assert "mark_route" not in cave_values
    assert "survey_route" in cave_values
    assert not any(action.value == "enter_haven" for action in cave_actions)
    assert not any(action.value == "system" for action in cave_actions)

    controller.handle(TravelRegional("haven"))
    controller.company.town_state["regional_node_id"] = "old_road"
    road_view = build_regional_map_view(controller.company, controller.definitions)
    road_actions = ActionProvider.regional_place_actions(road_view)
    assert not any(action.value == "survey_route" for action in road_actions)

    controller.handle(TravelRegional("haven"))
    gate_view = build_regional_map_view(controller.company, controller.definitions)
    gate_actions = ActionProvider.regional_place_actions(gate_view)
    gate_values = [action.value for action in gate_actions]
    assert gate_values.index("shallow_cave") < gate_values.index("old_road")
    assert gate_values.index("old_road") < gate_values.index("survey_route")
    assert gate_values.index("survey_route") < gate_values.index("enter_haven")
    roadbook_action = next(action for action in gate_actions if action.value == "survey_route")
    assert roadbook_action.label == "Open Roadbook"
    assert next(action for action in gate_actions if action.value == "old_road").label == (
        "Leave by Old Road"
    )
    charted_action = next(action for action in gate_actions if action.value == "shallow_cave")
    assert charted_action.label == "Take Charted Road to Shallow Cave"
    assert charted_action.default is True
    assert "Route: charted road." in charted_action.preview
    assert "No new discoveries on this route." in charted_action.result_hint
    gate_view_uncharted = replace(gate_view, route_charted=False, travel_available=False)
    assert not ActionProvider.regional_map_travel_actions(gate_view_uncharted)


def test_regional_map_walk_updates_exits_without_leaving_map_context() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    _mark_charted_route_at_cave_entrance(controller)
    controller.handle(TravelRegional("haven"))
    assert controller.company is not None
    controller.company.town_state["regional_node_id"] = "town_gate"

    first = controller.handle(MoveRegional("old_road"))
    assert first.success, first.error
    assert first.value.current_node_id == "old_road"
    assert any(exit_view.node_id == "town_gate" for exit_view in first.value.exits)

    second = controller.handle(MoveRegional("town_gate"))
    assert second.success, second.error
    assert second.value.current_node_id == "town_gate"
    assert any(exit_view.node_id == "old_road" for exit_view in second.value.exits)


def test_move_regional_to_uncleared_encounter_starts_combat() -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "abandoned_toll_post",
        "old_road",
        "town_gate",
    )
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    controller.company.known_route_ids.add("shallow_cave")
    controller.company.town_state["regional_node_id"] = "abandoned_toll_post"
    memory = controller.company.dungeon_memory.setdefault(
        "shallow_cave",
        DungeonMemoryState(dungeon_id="shallow_cave"),
    )
    memory.visited_node_ids.extend(["town_gate", "old_road", "abandoned_toll_post"])
    memory.cleared_node_ids.extend(["town_gate", "old_road", "abandoned_toll_post"])

    result = controller.handle(MoveRegional("bandit_camp"))

    assert result.success, result.error
    assert isinstance(result.value, CombatView)
    assert controller.manual_combat is not None
    assert controller.company.town_state["pending_regional_combat_node_id"] == "bandit_camp"
    assert controller.company.town_state["regional_node_id"] == "bandit_camp"
    assert any(isinstance(event, EncounterStartedEvent) for event in result.events)


def test_use_regional_action_reveals_bramble_shortcut() -> None:
    controller = _started_interactive_controller()
    _move_along(controller, "old_road", "bramble_shrine", "old_road", "town_gate")
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    assert controller.company.active_expedition is None
    controller.company.known_route_ids.add("shallow_cave")
    controller.company.town_state["regional_node_id"] = "bramble_shrine"
    memory = controller.company.dungeon_memory["shallow_cave"]
    assert "bramble_shrine" in memory.visited_node_ids

    result = controller.handle(UseRegionalAction("clear_bramble_path"))
    assert result.success, result.error
    assert "bramble_shrine->hidden_deer_path" in memory.revealed_exit_ids
    assert any(
        exit_view.node_id == "hidden_deer_path"
        for exit_view in result.value.exits
    )


def test_regional_node_id_round_trips_through_save(tmp_path: Path) -> None:
    controller = _started_interactive_controller()
    _move_along(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
    )
    controller.handle(ReturnFromDungeon())
    assert controller.company is not None
    controller.company.town_state["regional_node_id"] = "shallow_cave_entrance"

    save_path = tmp_path / "company.json"
    save_company(controller.company, save_path)
    loaded, _ = load_company(save_path)

    assert loaded.town_state["regional_node_id"] == "shallow_cave_entrance"


def _reach_boss(controller: AppController) -> None:
    _reach_cache(controller)
    _win_active_manual_combat(controller)
    result = controller.handle(UseDungeonAction("recover_gate_key"))
    assert result.success, result.error
    _move_along(controller, "fungus_chamber", "stone_gate")
    result = controller.handle(UseDungeonAction("unlock_black_gate"))
    assert result.success, result.error
    _move_along(controller, "maze_touched_lair")


def _event_index(events: list[object], event_type: type[object]) -> int:
    return next(index for index, event in enumerate(events) if isinstance(event, event_type))


def _win_active_manual_combat(controller: AppController) -> None:
    if controller.manual_combat is not None:
        for enemy in controller.manual_combat.state.enemies.values():
            enemy.hp = 0
            enemy.life_state = LifeState.DEAD
    while controller.manual_combat is not None:
        session = controller.manual_combat
        if session.pending_enemy_intent is not None:
            result = controller.handle(ResolveCombatReaction(None))
            assert result.success
            continue
        skill_ids = legal_skill_ids(session, controller.definitions)
        if not skill_ids:
            result = controller.handle(PassCombatTurn())
            assert result.success
            continue
        skill_id = skill_ids[0]
        target_id = legal_target_ids(session, controller.definitions, skill_id)[0]
        result = controller.handle(ResolveCombatAction(skill_id, target_id))
        assert result.success
