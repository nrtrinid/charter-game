from __future__ import annotations

from pathlib import Path

from game.app.commands import (
    AcceptContract,
    MoveDungeon,
    PassCombatTurn,
    ResolveCombatAction,
    ResolveCombatReaction,
    StartExpedition,
    StartNewCompany,
    TurnInLoot,
    UseDungeonAction,
    ViewDungeon,
)
from game.app.contracts import contract_board_state
from game.app.controller import AppController
from game.app.manual_combat import legal_skill_ids, legal_target_ids
from game.campaign.company import create_new_company
from game.campaign.save_load import load_company, save_company
from game.combat.combat_state import LifeState
from game.core.events import ContractCompletedEvent, EncounterStartedEvent, LootGainedEvent
from tests.conftest import get_definitions


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
def _controller_with_missing_carters_active() -> AppController:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.flags["cave_relic_filed"] = True
    controller.company.inventory["cave_relic"] = 1
    controller.handle(TurnInLoot("cave_relic"))
    accept = controller.handle(AcceptContract("missing_carters"))
    assert accept.success, accept.error
    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))
    assert result.success, result.error
    return controller


def _reach_hunters_trail(controller: AppController) -> None:
    _move_along(controller, "old_road", "hunters_trail")


def test_missing_carters_locked_until_cave_relic_filed() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.add("blackwood_road_charter")

    state, reason = contract_board_state(
        controller.company,
        controller.definitions,
        "missing_carters",
    )
    assert state == "locked", reason

    controller.company.flags["cave_relic_filed"] = True
    state, reason = contract_board_state(
        controller.company,
        controller.definitions,
        "missing_carters",
    )
    assert state == "available", reason


def test_hunters_trail_hides_wagon_action_without_active_contract() -> None:
    controller = _started_interactive_controller()
    _reach_hunters_trail(controller)

    view = controller.handle(ViewDungeon()).value
    action_ids = {action.action_id for action in view.room_actions}
    assert "follow_wagon_scars" not in action_ids


def test_hunters_trail_shows_wagon_action_with_active_contract() -> None:
    controller = _controller_with_missing_carters_active()
    _reach_hunters_trail(controller)

    view = controller.handle(ViewDungeon()).value
    action = next(
        candidate for candidate in view.room_actions if candidate.action_id == "follow_wagon_scars"
    )
    assert action.state == "available"


def test_follow_wagon_scars_reveals_wagon_cut() -> None:
    controller = _controller_with_missing_carters_active()
    _reach_hunters_trail(controller)

    result = controller.handle(UseDungeonAction("follow_wagon_scars"))
    assert result.success, result.error
    session = controller.company.active_expedition
    assert "hunters_trail->wagon_cut" in session.revealed_exit_ids

    view = controller.handle(ViewDungeon()).value
    assert any(action.value == "wagon_cut" and action.enabled for action in view.actions)


def test_carter_wreck_branch_combat_and_survivor_recovery() -> None:
    controller = _controller_with_missing_carters_active()
    _reach_hunters_trail(controller)
    controller.handle(UseDungeonAction("follow_wagon_scars"))
    _move_along(controller, "wagon_cut")

    move_result = controller.handle(MoveDungeon("carter_wreck"))
    assert move_result.success, move_result.error
    assert any(isinstance(event, EncounterStartedEvent) for event in move_result.events)
    assert controller.company.active_expedition.pending_combat_node_id == "carter_wreck"

    blocked = controller.handle(UseDungeonAction("recover_lost_carter"))
    assert not blocked.success

    coin_before = controller.company.coin
    rep_before = controller.company.reputation
    _win_active_manual_combat(controller)

    result = controller.handle(UseDungeonAction("recover_lost_carter"))
    assert result.success, result.error
    assert controller.company.flags["missing_carters_survivor_found"] is True
    assert controller.company.inventory["carter_brass_tag"] == 1
    assert controller.company.inventory["carter_ledger_page"] == 1
    assert "missing_carters_resolved" in controller.company.expedition_history
    assert "missing_carters" in controller.company.completed_contract_ids
    assert "missing_carters" not in controller.company.active_contract_ids
    assert controller.company.coin == coin_before + 8
    assert controller.company.reputation == rep_before + 3
    assert any(isinstance(event, ContractCompletedEvent) for event in result.events)
    assert any(isinstance(event, LootGainedEvent) and event.coin == 8 for event in result.events)

    repeat = controller.handle(UseDungeonAction("recover_lost_carter"))
    assert not repeat.success


def test_missing_carters_branch_state_survives_save_load(tmp_path: Path) -> None:
    controller = _controller_with_missing_carters_active()
    _reach_hunters_trail(controller)
    controller.handle(UseDungeonAction("follow_wagon_scars"))

    save_path = tmp_path / "company.json"
    save_company(controller.company, save_path)
    loaded, _event = load_company(save_path)
    loaded_controller = AppController(definitions=get_definitions(), company=loaded)

    session = loaded_controller.company.active_expedition
    assert session is not None
    assert "hunters_trail->wagon_cut" in session.revealed_exit_ids
    assert "missing_carters" in loaded_controller.company.active_contract_ids

    _reach_hunters_trail(loaded_controller)
    loaded_controller.handle(UseDungeonAction("follow_wagon_scars"))
    _move_along(loaded_controller, "wagon_cut", "carter_wreck")
    _win_active_manual_combat(loaded_controller)
    result = loaded_controller.handle(UseDungeonAction("recover_lost_carter"))
    assert result.success, result.error

    save_company(loaded_controller.company, save_path)
    final, _event = load_company(save_path)
    assert final.flags["missing_carters_survivor_found"] is True
    assert final.inventory["carter_brass_tag"] == 1
    assert "missing_carters" in final.completed_contract_ids
