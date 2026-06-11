from pathlib import Path

from game.campaign.company import STARTING_COIN, create_new_company
from game.campaign.save_load import load_company, save_company
from game.core.events import ExpeditionEvent
from game.core.rng import GameRng
from game.expedition.expedition import SHALLOW_CAVE_BREACH_ID, run_opening_route
from tests.conftest import get_definitions


def test_deterministic_opening_vertical_slice(tmp_path: Path) -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)

    events = run_opening_route(company, definitions, GameRng(7), enter_maze=True)

    expected_route = [
        "town_gate",
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
        "fungus_chamber",
        "old_works_cache",
        "old_works_cache",
        "fungus_chamber",
        "stone_gate",
        "maze_touched_lair",
        "cave_mini_boss",
        "maze_breach",
        "maze_depth_1",
        "maze_depth_1_room_1",
        "maze_depth_1_room_3",
        "maze_retreat",
        "haven_return",
    ]
    authored_node_ids = {node.id for node in definitions.expeditions["opening"].nodes}
    event_node_ids = [
        event.node_id
        for event in events
        if isinstance(event, ExpeditionEvent) and event.node_id in authored_node_ids
    ]
    assert event_node_ids == expected_route
    assert SHALLOW_CAVE_BREACH_ID in company.known_breaches
    assert "shallow_cave" in company.known_route_ids
    assert company.reputation == 2
    assert company.coin == STARTING_COIN + 10
    assert "shallow_cave_boss_defeated" in company.expedition_history
    assert "maze_depth_1_scouted" in company.expedition_history
    assert company.town_state["location"] == "Haven Town"
    assert company.inventory["cave_relic"] == 1
    assert company.inventory["maze_glass"] == 1

    save_path = tmp_path / "vertical_slice.json"
    save_company(company, save_path)
    loaded, _event = load_company(save_path)

    assert SHALLOW_CAVE_BREACH_ID in loaded.known_breaches
    assert loaded.reputation == 2
    assert loaded.coin == STARTING_COIN + 10
    assert loaded.town_state["location"] == "Haven Town"
    assert "shallow_cave" in loaded.known_route_ids
    assert "shallow_cave_boss_defeated" in loaded.expedition_history
    assert "maze_depth_1_scouted" in loaded.expedition_history
    assert loaded.inventory["cave_relic"] == 1
    assert loaded.inventory["maze_glass"] == 1


def test_repeating_opening_route_does_not_duplicate_completion_rewards() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)

    run_opening_route(company, definitions, GameRng(7), enter_maze=True)
    run_opening_route(company, definitions, GameRng(7), enter_maze=True)

    assert company.reputation == 2
    assert company.coin == STARTING_COIN + 10
    assert company.inventory["cave_relic"] == 1
    assert company.inventory["maze_glass"] == 1
    assert company.expedition_history.count("shallow_cave_boss_defeated") == 1
    assert company.expedition_history.count("maze_depth_1_scouted") == 1
