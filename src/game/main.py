"""Entrypoint for the terminal application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from game.app.commands import SaveGame, StartExpedition, StartNewCompany
from game.app.controller import AppController
from game.combat.enemy_decision import SUPPORTED_PRODUCTION_ENEMY_AI_MODES
from game.core.rng import GameRng
from game.ui.cli import Cli
from game.ui.screens import render_events_text


def run_demo(save_path: Path | None = None, *, enemy_ai_mode: str = "learned_static") -> int:
    controller = AppController(rng=GameRng(7), enemy_ai_mode=enemy_ai_mode)
    events = []
    events.extend(controller.handle(StartNewCompany()).events)
    events.extend(controller.handle(StartExpedition(enter_maze=True)).events)
    if save_path is not None:
        events.extend(controller.handle(SaveGame(save_path)).events)
    print(render_events_text(events))
    return 0


def run_tui(*, enemy_ai_mode: str = "learned_static") -> int:
    from game.ui.tui import CharterApp

    CharterApp(controller=AppController(rng=GameRng(None), enemy_ai_mode=enemy_ai_mode)).run()
    return 0


def run_cli(*, enemy_ai_mode: str = "learned_static") -> int:
    Cli(controller=AppController(rng=GameRng(None), enemy_ai_mode=enemy_ai_mode)).run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The Charter v0.1")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--demo", action="store_true", help="run deterministic vertical slice")
    mode_group.add_argument("--cli", action="store_true", help="run the legacy Rich CLI")
    mode_group.add_argument(
        "--tui",
        action="store_true",
        help="run the fullscreen Textual interface (default)",
    )
    parser.add_argument("--save", type=Path, default=None, help="save path for demo mode")
    parser.add_argument(
        "--enemy-ai-mode",
        choices=SUPPORTED_PRODUCTION_ENEMY_AI_MODES,
        default="learned_static",
        help="enemy decision policy for playtest launches",
    )
    args = parser.parse_args(argv)

    if args.demo:
        return run_demo(args.save, enemy_ai_mode=args.enemy_ai_mode)
    if args.cli:
        return run_cli(enemy_ai_mode=args.enemy_ai_mode)

    return run_tui(enemy_ai_mode=args.enemy_ai_mode)


if __name__ == "__main__":
    raise SystemExit(main())
