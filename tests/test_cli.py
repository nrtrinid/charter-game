from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from game.app.commands import (
    MoveDungeon,
    PassCombatTurn,
    ResolveCombatAction,
    ResolveCombatReaction,
    StartExpedition,
    UseDungeonAction,
)
from game.app.controller import AppController
from game.app.manual_combat import legal_skill_ids, legal_target_ids
from game.combat.combat_state import LifeState
from game.combat.formation import FormationSlot
from game.content.definitions import GameDefinitions
from game.ui.cli import Cli, CliExpeditionFlow
from tests.conftest import OPENING_DUNGEON_TO_WORKS_CACHE, OPENING_POST_COMBAT


def _fast_manual_combat_loop(self: CliExpeditionFlow) -> None:
    while self.controller.manual_combat is not None and not self._stop_playback:
        for enemy in self.controller.manual_combat.state.enemies.values():
            enemy.hp = 0
            enemy.life_state = LifeState.DEAD
        session = self.controller.manual_combat
        if session.pending_enemy_intent is not None:
            result = self.controller.handle(ResolveCombatReaction(None))
            assert result.success, result.error
            continue
        skill_ids = legal_skill_ids(session, self.controller.definitions)
        if not skill_ids:
            result = self.controller.handle(PassCombatTurn())
            assert result.success, result.error
            continue
        skill_id = skill_ids[0]
        target_id = legal_target_ids(session, self.controller.definitions, skill_id)[0]
        result = self.controller.handle(ResolveCombatAction(skill_id, target_id))
        assert result.success, result.error


def _fast_dungeon_action(self: CliExpeditionFlow, value: str) -> None:
    if value.startswith("action:"):
        result = self.controller.handle(UseDungeonAction(value.removeprefix("action:")))
    else:
        result = self.controller.handle(MoveDungeon(value))
    assert result.success, result.error
    _fast_manual_combat_loop(self)


def _fast_begin_opening_expedition(self: CliExpeditionFlow) -> None:
    self._auto_play = False
    self._stop_playback = False
    if self._breach_pending():
        self._resolve_breach()
        return
    result = self.controller.handle(
        StartExpedition(
            stop_at_breach=True,
            manual_combat=True,
            interactive_dungeon=True,
        )
    )
    if not result.success:
        self._show_screen(
            "Expedition",
            "Expedition failed.",
        )
        self._pause()
        return
    self._play_events(result.events, result.hci)
    if self._stop_playback:
        return
    for node_id in (*OPENING_DUNGEON_TO_WORKS_CACHE, "old_works_cache", *OPENING_POST_COMBAT):
        _fast_dungeon_action(self, node_id)
    if self._stop_playback:
        return
    if (
        self.controller.company is not None
        and self.controller.company.active_expedition is None
    ):
        return
    self._resolve_breach()


@pytest.fixture
def fast_opening_combat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CliExpeditionFlow, "_manual_combat_loop", _fast_manual_combat_loop)
    monkeypatch.setattr(
        CliExpeditionFlow,
        "_begin_opening_expedition",
        _fast_begin_opening_expedition,
    )


class FakeInput:
    def __init__(self, values: list[str], *, auto_combat: bool = False) -> None:
        self.values = values
        self.auto_combat = auto_combat
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("Continue"):
            if self.values and self.values[0] == "":
                return self.values.pop(0)
            return self._optional_default(prompt)
        if prompt.startswith(("Command", "Skill", "Target", "Move", "Reaction", "Log")):
            return self._optional_default(prompt)
        if prompt.startswith(("Regional Map", "Record")):
            if self.values and self.values[0] == "":
                return self.values.pop(0)
            return ""
        if not self.values:
            return self._optional_default(prompt)
        return self.values.pop(0)

    def _optional_default(self, prompt: str) -> str:
        if prompt.startswith("Continue"):
            return ""
        if prompt.startswith("Reaction"):
            return "skip"
        if prompt.startswith("Log"):
            return ""
        if prompt.startswith("Expedition"):
            return "back"
        if self.auto_combat:
            if prompt.startswith("Move"):
                return "1"
            if prompt.startswith(("Command", "Skill", "Target")):
                return "1"
        raise AssertionError(f"Fake input exhausted at prompt: {prompt!r}")


def make_cli(
    inputs: list[str],
    save_path: Path,
    definitions: GameDefinitions,
    *,
    auto_combat: bool = False,
) -> tuple[Cli, StringIO, FakeInput]:
    output = StringIO()
    fake_input = FakeInput(inputs, auto_combat=auto_combat)
    cli = Cli(
        controller=AppController(definitions=definitions),
        console=Console(file=output, force_terminal=False, width=120),
        input_fn=fake_input,
        save_path=save_path,
        clear_screen=False,
    )
    return cli, output, fake_input


def opening_manual_inputs(breach_inputs: list[str]) -> list[str]:
    return ["1"] + list(breach_inputs)


def test_number_and_alias_dispatch_start_and_roster(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    cli, output, _fake_input = make_cli(
        ["1", "Amber Gate", "", "roster", "", "back"],
        tmp_path / "company.json",
        definitions,
    )

    cli._handle_choice("1")

    assert cli.controller.company is not None
    assert cli.controller.company.name == "Amber Gate"
    assert "Roster" in output.getvalue()
    assert "Mara Vell" in output.getvalue()


def test_company_name_defaults_to_haven_charter(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    cli, _output, _fake_input = make_cli(["", ""], tmp_path / "company.json", definitions)

    cli._handle_choice("new")

    assert cli.controller.company is not None
    assert cli.controller.company.name == "Haven Charter"


def test_replacing_existing_company_requires_confirmation(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    save_path = tmp_path / "company.json"
    cli, output, fake_input = make_cli(["First Charter", ""], save_path, definitions)
    cli._handle_choice("start")

    fake_input.values.extend(["", ""])
    cli._handle_choice("start")

    assert cli.controller.company is not None
    assert cli.controller.company.name == "First Charter"
    assert "Kept the current company" in output.getvalue()


def test_save_overwrite_requires_confirmation(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    save_path = tmp_path / "company.json"
    save_path.write_text("existing", encoding="utf-8")
    cli, output, fake_input = make_cli(["Haven Charter", ""], save_path, definitions)
    cli._handle_choice("start")

    fake_input.values.extend(["", ""])
    cli._handle_choice("save")

    assert save_path.read_text(encoding="utf-8") == "existing"
    assert "Save cancelled" in output.getvalue()


def test_missing_load_reports_error_without_raising(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    cli, output, _fake_input = make_cli([""], tmp_path / "missing.json", definitions)

    cli._handle_choice("load")

    assert "Could not load save" in output.getvalue()


def test_town_services_work_through_numbered_submenu(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    save_path = tmp_path / "company.json"
    cli, output, fake_input = make_cli(["Haven Charter", ""], save_path, definitions)
    cli._handle_choice("start")
    assert cli.controller.company is not None
    cli.controller.company.coin = 11

    fake_input.values.extend(["recruit", "1", "", "recover", "1", "", "buy", "1", "", "back"])
    cli._town_menu()

    assert cli.controller.company is not None
    assert len(cli.controller.company.roster) == 5
    assert cli.controller.company.supplies["rations"] == 5
    assert cli.controller.company.coin == 0
    assert "Haven Town" in output.getvalue()


def test_disabled_town_actions_are_rejected(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    save_path = tmp_path / "company.json"
    cli, output, fake_input = make_cli(["Haven Charter", ""], save_path, definitions)
    cli._handle_choice("start")
    assert cli.controller.company is not None
    cli.controller.company.coin = 0

    fake_input.values.extend(["recruit", "back"])
    cli._town_menu()

    assert len(cli.controller.company.roster) == 4
    assert "unavailable" in output.getvalue()


def test_formation_can_empty_a_slot_through_cli(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    save_path = tmp_path / "company.json"
    cli, _output, fake_input = make_cli(["Haven Charter", ""], save_path, definitions)
    cli._handle_choice("start")
    assert cli.controller.company is not None

    fake_input.values.extend(["3", "1", ""])
    cli._assign_formation()

    assert cli.controller.company.active_party_slots[FormationSlot.FRONT_LEFT] is None


def test_expedition_start_blocks_empty_active_party(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    save_path = tmp_path / "company.json"
    cli, output, fake_input = make_cli(["Haven Charter", ""], save_path, definitions)
    cli._handle_choice("start")
    assert cli.controller.company is not None
    for slot in FormationSlot:
        cli.controller.company.active_party_slots[slot] = None

    fake_input.values.extend(["1", "", "2"])
    cli._handle_choice("expedition")

    assert "Assign at least one living hero" in output.getvalue()


def test_known_route_start_skips_interlude_and_opens_cave_mouth(
    definitions: GameDefinitions, tmp_path: Path
) -> None:
    save_path = tmp_path / "company.json"
    cli, output, _fake_input = make_cli(["Haven Charter", ""], save_path, definitions)
    cli._handle_choice("start")
    assert cli.controller.company is not None
    cli.controller.company.known_route_ids.add("shallow_cave")

    result = cli.controller.handle(
        StartExpedition(
            stop_at_breach=True,
            manual_combat=True,
            interactive_dungeon=True,
        )
    )
    assert result.success, result.error
    cli._play_events(result.events, result.hci)

    text = output.getvalue()
    assert "Charted Approach: Shallow Cave" not in text
    assert "Incident: Clear" not in text
    assert "Cave Mouth" in text


@pytest.mark.slow
def test_opening_expedition_stops_at_breach_room_and_can_return(
    definitions: GameDefinitions,
    tmp_path: Path,
    fast_opening_combat: None,
) -> None:
    cli, output, fake_input = make_cli(
        ["Haven Charter", ""],
        tmp_path / "company.json",
        definitions,
    )
    cli._handle_choice("start")

    fake_input.values.extend(opening_manual_inputs(["1", ""]))
    cli._handle_choice("expedition")

    assert cli.controller.company is not None
    assert cli.controller.company.town_state["location_id"] == "haven"
    assert cli.controller.company.town_state["location"] == "Haven Town"
    assert "shallow_cave_breach" in cli.controller.company.known_breaches
    assert "maze_depth_1_scouted" not in cli.controller.company.expedition_history
    assert "Breach" in output.getvalue()
    assert cli.controller.company.last_expedition_report is not None


@pytest.mark.slow
def test_opening_breach_exposes_generated_maze_entry(
    definitions: GameDefinitions,
    tmp_path: Path,
    fast_opening_combat: None,
) -> None:
    cli, output, fake_input = make_cli(
        ["Haven Charter", ""],
        tmp_path / "company.json",
        definitions,
    )
    cli._handle_choice("start")

    fake_input.values.extend(opening_manual_inputs(["1", ""]))
    cli._handle_choice("expedition")

    assert cli.controller.company is not None
    assert "Shallow Cave Breach discovered" in output.getvalue()
    assert "maze_depth_1_scouted" not in cli.controller.company.expedition_history


@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "CLI opening flow resolves the breach menu before dungeon enter_generated_maze "
        "is reachable; see tests/test_dungeon.py generated maze coverage."
    )
)
def test_opening_breach_can_enter_and_withdraw_generated_maze(
    definitions: GameDefinitions,
    tmp_path: Path,
    fast_opening_combat: None,
) -> None:
    cli, output, fake_input = make_cli(
        ["Haven Charter", ""],
        tmp_path / "company.json",
        definitions,
    )
    cli._handle_choice("start")

    fake_input.values.extend(
        opening_manual_inputs(["1", ""])
    )
    cli._handle_choice("expedition")

    assert cli.controller.company is not None
    assert "maze_depth_1_scouted" not in cli.controller.company.expedition_history
    assert cli.controller.company.active_expedition is None
    assert cli.controller.company.last_expedition_report is not None
    assert any(
        history.startswith("generated_maze_route_")
        for history in cli.controller.company.expedition_history
    )
    text = output.getvalue()
    assert "Withdraw to Shallow Cave" in text
    assert "Return to Haven" in text
    assert "Regional Map — Act 1" in text
    assert "Filed Company Record" in text
