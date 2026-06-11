from game.campaign.company import create_new_company
from game.core.events import ContractCompletedEvent, LoreDiscoveredEvent
from game.core.rng import GameRng
from game.expedition.expedition import run_opening_route
from tests.conftest import get_definitions


def test_world_scaffolding_loads_contracts_locations_and_rumors() -> None:
    definitions = get_definitions()

    assert definitions.world.maze_name == "Pandora's Maze"
    assert definitions.world.starting_settlement == "haven"
    assert "blackwood_road_charter" in definitions.contracts
    assert definitions.contracts["blackwood_road_charter"].available_at_start
    assert definitions.contracts["blackwood_road_charter"].board_order == 0
    assert "shallow_cave_breach_scout" in definitions.contracts
    assert definitions.contracts["shallow_cave_breach_scout"].location_id == (
        "shallow_cave_breach"
    )
    assert "shallow_cave_breach_hunt" in definitions.contracts
    assert definitions.contracts["shallow_cave_breach_hunt"].difficulty == 5
    assert "shallow_cave_breach_scout_posting" in definitions.contracts
    assert "repeatable" in definitions.contracts["shallow_cave_breach_scout_posting"].tags
    assert "shallow_cave_breach_hunt_posting" in definitions.contracts
    assert "repeatable" in definitions.contracts["shallow_cave_breach_hunt_posting"].tags
    assert set(definitions.gear) >= {
        "reinforced_vest",
        "field_satchel",
        "balanced_blade",
        "sighting_charm",
        "maze_glass_talisman",
    }
    assert all(gear.slot == "kit" for gear in definitions.gear.values())
    assert not any(
        "defense" in gear.effects.model_dump()
        for gear in definitions.gear.values()
    )
    assert definitions.contracts["shallow_cave_breach_hunt"].reward_gear == {
        "maze_glass_talisman": 1
    }
    assert "roads_that_repeat" in definitions.rumors
    assert "shallow_cave_breach" in definitions.locations


def test_opening_route_records_rumors_flags_and_contract_completion() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)

    events = run_opening_route(company, definitions, GameRng(7), stop_at_breach=True)

    assert "blackwood_road_charter" in company.completed_contract_ids
    assert "blackwood_road_charter" not in company.active_contract_ids
    assert "roads_that_repeat" in company.known_lore_entries
    assert "maze_profit_invites" in company.known_lore_entries
    assert company.flags["act_1_frontier_company_started"]
    assert company.flags["maze_leak_confirmed"]
    assert any(isinstance(event, ContractCompletedEvent) for event in events)
    assert any(isinstance(event, LoreDiscoveredEvent) for event in events)


def test_recruits_keep_grounded_backgrounds() -> None:
    company = create_new_company(get_definitions())

    assert company.roster[0].background == "failed road warden"
    assert "creditors" in company.roster[0].motive
