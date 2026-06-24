"""Main menu, system, save/load, and confirm action handlers for the TUI."""

from __future__ import annotations

from dataclasses import dataclass

from game.app.commands import Quit, TakeExpeditionChoice
from game.combat.enemy_decision import (
    PRODUCTION_ENEMY_AI_MODE_LABELS,
    SUPPORTED_PRODUCTION_ENEMY_AI_MODES,
)
from game.ui.tui_handlers.protocol import TuiHandlerHost


def _enemy_ai_mode_label(mode: str) -> str:
    return PRODUCTION_ENEMY_AI_MODE_LABELS.get(mode, mode.replace("_", " ").title())


def _next_enemy_ai_mode(mode: str) -> str:
    modes = SUPPORTED_PRODUCTION_ENEMY_AI_MODES
    try:
        index = modes.index(mode)
    except ValueError:
        return modes[0]
    return modes[(index + 1) % len(modes)]


@dataclass
class ShellHandlers:
    app: TuiHandlerHost

    def handle_main_action(self, value: str) -> None:
        if value == "start":
            if self.app.controller.company is not None:
                self.app._show_confirm(
                    "replace_company",
                    "Replace Company",
                    "Starting a new company will replace the current in-memory company.",
                    confirm_label="Replace Company",
                    cancel_label="Keep Current",
                    irreversible=True,
                )
            else:
                self.app._show_name_prompt()
        elif value == "continue":
            self.app._show_current_place()
        elif value == "saves":
            self.app._show_save_load()
        elif value == "gear":
            self.app._show_gear_locker(return_to="main")
        elif value == "load":
            self.app._request_load_company()
        elif value == "help":
            self.app._show_help()
        elif value == "quit":
            self.app._show_confirm("quit", "Quit", "Close the charter desk?", confirm_label="Quit")

    def handle_system_action(self, value: str) -> None:
        if value == "save":
            if self.app.save_path.exists():
                self.app._show_confirm(
                    "overwrite_save",
                    "Overwrite Save",
                    f"Overwrite the save slot at {self.app.save_path}?",
                    confirm_label="Overwrite",
                    irreversible=True,
                )
            else:
                self.app._save_company()
        elif value == "load":
            self.app._request_load_company()
        elif value == "toggle_enemy_ai":
            mode = self.app.controller.enemy_ai_mode
            self.app.controller.enemy_ai_mode = _next_enemy_ai_mode(mode)
            self.app._show_system(
                f"Enemy AI set to {_enemy_ai_mode_label(self.app.controller.enemy_ai_mode)}."
            )
        elif value == "help":
            self.app._show_help()
        elif value == "quit":
            self.app._show_confirm("quit", "Quit", "Close the charter desk?", confirm_label="Quit")

    def handle_company_action(self, value: str) -> None:
        if value == "start":
            if self.app.controller.company is not None:
                self.app._show_confirm(
                    "replace_company",
                    "Replace Company",
                    "Starting a new company will replace the current in-memory company.",
                    confirm_label="Replace Company",
                    cancel_label="Keep Current",
                    irreversible=True,
                )
            else:
                self.app._show_name_prompt()
        elif value == "town":
            self.app._show_town()
        elif value == "roster":
            self.app._show_roster()
        elif value == "supplies":
            self.app._show_supplies()
        elif value == "ledger":
            self.app._show_ledger()

    def handle_save_action(self, value: str) -> None:
        if value == "save":
            if self.app.save_path.exists():
                self.app._show_confirm(
                    "overwrite_save",
                    "Overwrite Save",
                    f"Overwrite the save slot at {self.app.save_path}?",
                    confirm_label="Overwrite",
                    irreversible=True,
                )
            else:
                self.app._save_company()
        elif value == "load":
            self.app._request_load_company()

    def handle_confirm_action(self, value: str) -> None:
        if value == "cancel":
            self.app._cancel()
            return
        if self.app.pending_confirm == "quit":
            result = self.app.controller.handle(Quit())
            self.app._record_events(result.events)
            self.app.exit()
        elif self.app.pending_confirm == "replace_company":
            self.app.pending_confirm = None
            self.app._show_name_prompt()
        elif self.app.pending_confirm == "overwrite_save":
            self.app.pending_confirm = None
            self.app._save_company()
        elif self.app.pending_confirm == "load_company":
            self.app.pending_confirm = None
            self.app._load_company()
        elif self.app.pending_confirm == "descend_maze_depth_1":
            self.app.pending_confirm = None
            result = self.app.controller.handle(TakeExpeditionChoice("descend_maze_depth_1"))
            if result.success:
                self.app._record_events(result.events)
                self.app._start_playback(result.events, result.hci)
            else:
                self.app._show_breach(result.error or "Could not descend.")

    def back_from_help(self) -> None:
        if self.app.pending_help_return_state == "current_place":
            self.app.pending_help_return_state = "system"
            self.app._show_current_place()
            return
        if self.app.controller.company is None:
            self.app._show_main()
        else:
            self.app._show_system()
