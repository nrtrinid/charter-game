from __future__ import annotations

import game.expedition.travel as travel_module
from game.campaign.company import create_new_company
from game.core.events import (
    BreachDiscoveredEvent,
    EncounterEndedEvent,
    EncounterStartedEvent,
    ExpeditionReturnedEvent,
    LootGainedEvent,
    RoundEndedEvent,
    RoundStartedEvent,
)
from game.core.rng import GameRng
from game.expedition.cave import create_shallow_cave_combat
from game.expedition.expedition import run_opening_route
from game.expedition.travel import run_combat_to_end
from tests.conftest import get_definitions


def test_auto_combat_emits_structured_encounter_and_round_events() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    combat = create_shallow_cave_combat(company, definitions)

    events = run_combat_to_end(
        combat,
        definitions,
        GameRng(7),
        encounter_id="shallow_cave",
        encounter_name="Guarded Hollow",
    )

    assert any(isinstance(event, EncounterStartedEvent) for event in events)
    assert any(isinstance(event, RoundStartedEvent) for event in events)
    assert any(isinstance(event, RoundEndedEvent) for event in events)
    assert any(isinstance(event, EncounterEndedEvent) for event in events)


def test_auto_combat_passes_selected_enemy_ai_mode(monkeypatch) -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    combat = create_shallow_cave_combat(company, definitions)
    modes: list[str] = []
    wait_modes: list[str] = []
    move_modes: list[str] = []
    real_factory = travel_module.production_enemy_decision_policy
    real_wait_reason = travel_module.enemy_wait_reason
    real_proactive_move = travel_module.enemy_proactive_move

    def policy_spy(mode: str = "learned_static"):
        modes.append(mode)
        return real_factory(mode)

    def wait_reason_spy(*args, **kwargs):
        wait_modes.append(args[4])
        return real_wait_reason(*args, **kwargs)

    def proactive_move_spy(*args, **kwargs):
        move_modes.append(args[3])
        return real_proactive_move(*args, **kwargs)

    monkeypatch.setattr(travel_module, "production_enemy_decision_policy", policy_spy)
    monkeypatch.setattr(travel_module, "enemy_wait_reason", wait_reason_spy)
    monkeypatch.setattr(travel_module, "enemy_proactive_move", proactive_move_spy)

    run_combat_to_end(
        combat,
        definitions,
        GameRng(7),
        max_rounds=1,
        enemy_ai_mode="heuristic",
        enemy_wait_mode="package_only",
        enemy_movement_mode="package_only",
    )

    assert "heuristic" in modes
    assert "package_only" in wait_modes
    assert "package_only" in move_modes


def test_expedition_emits_loot_breach_and_return_events() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)

    events = run_opening_route(company, definitions, GameRng(7), enter_maze=True)

    assert any(isinstance(event, LootGainedEvent) for event in events)
    assert any(isinstance(event, BreachDiscoveredEvent) for event in events)
    assert any(isinstance(event, ExpeditionReturnedEvent) for event in events)
