from __future__ import annotations

import sys
from types import ModuleType

import pytest

import game.main as main_module
from game.app.controller import AppController


def test_controller_defaults_to_learned_static_enemy_ai() -> None:
    assert AppController().enemy_ai_mode == "learned_static"


def test_demo_mode_remains_noninteractive(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_module.main(["--demo"]) == 0

    output = capsys.readouterr().out
    assert "Haven Charter receives its charter." in output
    assert "The company returns to Haven." in output


def test_demo_uses_deterministic_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int | None] = []
    real_controller = main_module.AppController

    def controller_spy(*args, **kwargs):
        seeds.append(kwargs["rng"].seed)
        return real_controller(*args, **kwargs)

    monkeypatch.setattr(main_module, "AppController", controller_spy)

    assert main_module.run_demo() == 0
    assert seeds == [7]


def test_tui_mode_launches_textual_app(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[bool] = []

    def fake_run_tui(*, enemy_ai_mode: str = "learned_static") -> int:
        launched.append(enemy_ai_mode == "learned_static")
        return 0

    monkeypatch.setattr(main_module, "run_tui", fake_run_tui)

    assert main_module.main(["--tui"]) == 0
    assert launched == [True]


def test_tui_is_default_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[bool] = []

    def fake_run_tui(*, enemy_ai_mode: str = "learned_static") -> int:
        launched.append(enemy_ai_mode == "learned_static")
        return 0

    monkeypatch.setattr(main_module, "run_tui", fake_run_tui)

    assert main_module.main([]) == 0
    assert launched == [True]


def test_cli_mode_launches_legacy_rich_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[bool] = []

    def fake_run_cli(*, enemy_ai_mode: str = "learned_static") -> int:
        launched.append(enemy_ai_mode == "learned_static")
        return 0

    monkeypatch.setattr(main_module, "run_cli", fake_run_cli)

    assert main_module.main(["--cli"]) == 0
    assert launched == [True]


def test_player_facing_modes_use_unseeded_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    controllers: list[tuple[int | None, str]] = []

    class FakeApp:
        def __init__(self, *, controller):
            controllers.append((controller.rng.seed, controller.enemy_ai_mode))

        def run(self) -> None:
            return None

    class FakeCli:
        def __init__(self, *, controller):
            controllers.append((controller.rng.seed, controller.enemy_ai_mode))

        def run(self) -> None:
            return None

    monkeypatch.setattr(main_module, "Cli", FakeCli)
    fake_tui = ModuleType("game.ui.tui")
    fake_tui.CharterApp = FakeApp
    monkeypatch.setitem(sys.modules, "game.ui.tui", fake_tui)

    assert main_module.run_cli() == 0
    assert main_module.run_tui() == 0
    assert controllers == [(None, "learned_static"), (None, "learned_static")]


def test_enemy_ai_mode_flag_passes_to_cli_and_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    modes: list[str] = []

    def fake_run_cli(*, enemy_ai_mode: str = "learned_static") -> int:
        modes.append(enemy_ai_mode)
        return 0

    def fake_run_tui(*, enemy_ai_mode: str = "learned_static") -> int:
        modes.append(enemy_ai_mode)
        return 0

    monkeypatch.setattr(main_module, "run_cli", fake_run_cli)
    monkeypatch.setattr(main_module, "run_tui", fake_run_tui)

    assert main_module.main(["--cli", "--enemy-ai-mode", "heuristic"]) == 0
    assert main_module.main(["--tui", "--enemy-ai-mode", "heuristic"]) == 0
    assert modes == ["heuristic", "heuristic"]


def test_run_demo_accepts_enemy_ai_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    modes: list[str] = []
    real_controller = main_module.AppController

    def controller_spy(*args, **kwargs):
        modes.append(kwargs["enemy_ai_mode"])
        return real_controller(*args, **kwargs)

    monkeypatch.setattr(main_module, "AppController", controller_spy)

    assert main_module.run_demo(enemy_ai_mode="heuristic") == 0
    assert modes == ["heuristic"]


def test_demo_and_tui_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main_module.main(["--demo", "--tui"])


def test_demo_and_cli_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main_module.main(["--demo", "--cli"])
