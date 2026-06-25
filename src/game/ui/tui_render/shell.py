"""TUI screen rendering for shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from textual.widgets import Input

from game.app.actions import (
    ActionProvider,
)
from game.app.commands import (
    ViewRegionalMap,
)
from game.app.views import (
    RegionalMapView,
    ScreenAction,
    ScreenActionKind,
    ScreenActionRisk,
)
from game.core.hci import HciResultAnalysis
from game.ui.tui_constants import (
    DEFAULT_COMPANY_NAME,
    _enemy_ai_controls_text,
    _enemy_ai_mode_label,
    _next_enemy_ai_mode,
)
from game.ui.tui_render.protocol import TuiRenderHost


@dataclass
class ShellRender:
    app: TuiRenderHost

    def show_main(self, message: str = "") -> None:
        company = self.app.controller.company
        actions: tuple[ScreenAction, ...]
        if company is None:
            actions = (
                ScreenAction(
                    "1",
                    "Start Company",
                    "start",
                    ("n",),
                    default=True,
                    kind=ScreenActionKind.TOWN,
                    preview="Create a new charter company.",
                    result_hint="Opens the company name prompt.",
                ),
                ScreenAction(
                    "2",
                    "Load Save",
                    "load",
                    ("l",),
                    kind=ScreenActionKind.SYSTEM,
                    preview="Load the save slot if one exists.",
                ),
                ScreenAction(
                    "3",
                    "Help",
                    "help",
                    ("h", "?"),
                    kind=ScreenActionKind.INSPECT,
                    preview="Review keyboard controls.",
                ),
                ScreenAction(
                    "4",
                    "Quit",
                    "quit",
                    ("q",),
                    kind=ScreenActionKind.SYSTEM,
                    risk=ScreenActionRisk.IRREVERSIBLE,
                    preview="Close the TUI.",
                    result_hint="Requires confirmation.",
                ),
            )
            body = "Charter Desk\n\nNo company is loaded."
        else:
            actions = (
                ScreenAction(
                    "1",
                    "Continue",
                    "continue",
                    ("c",),
                    default=True,
                    description="Resume at the current place or active expedition.",
                    kind=ScreenActionKind.NAVIGATE,
                    preview="Return to the current company context.",
                ),
                ScreenAction(
                    "2",
                    "Start New Company",
                    "start",
                    ("n",),
                    kind=ScreenActionKind.TOWN,
                    risk=ScreenActionRisk.IRREVERSIBLE,
                    preview="Replace the current in-memory company after confirmation.",
                ),
                ScreenAction(
                    "3",
                    "Save / Load",
                    "saves",
                    ("s",),
                    kind=ScreenActionKind.SYSTEM,
                    preview="Manage the single save slot.",
                ),
                ScreenAction(
                    "4",
                    "Armory",
                    "gear",
                    ("i", "gear", "inventory", "armory"),
                    kind=ScreenActionKind.INSPECT,
                    preview="Open the company armory.",
                ),
                ScreenAction(
                    "5",
                    "Help",
                    "help",
                    ("h", "?"),
                    kind=ScreenActionKind.INSPECT,
                    preview="Review keyboard controls.",
                ),
                ScreenAction(
                    "6",
                    "Quit",
                    "quit",
                    ("q",),
                    kind=ScreenActionKind.SYSTEM,
                    risk=ScreenActionRisk.IRREVERSIBLE,
                    preview="Close the TUI.",
                    result_hint="Requires confirmation.",
                ),
            )
            body = (
                "Charter Desk\n\n"
                f"Company: {company.name}\n"
                f"Location: {company.town_state.get('location', 'Haven Town')}\n"
                f"Reputation: {company.reputation}\n"
                f"Coin: {company.coin}"
            )
        self.app._show_screen("main", "Charter Desk", body, actions, message=message)

    def show_company(self, message: str = "") -> None:
        has_company = self.app.controller.company is not None
        actions = (
            ScreenAction("1", "Start New Company", "start", ("n",), default=not has_company),
            ScreenAction(
                "2",
                "Haven Town",
                "town",
                ("t",),
                enabled=has_company,
                default=has_company,
                description="Town dashboard, records, and services",
            ),
            ScreenAction("3", "Roster", "roster", ("r",), enabled=has_company),
            ScreenAction("4", "Supplies", "supplies", ("u",), enabled=has_company),
            ScreenAction("5", "Ledger", "ledger", ("l",), enabled=has_company),
            ScreenAction("6", "Back", "back", ("b",)),
        )
        if has_company:
            company = self.app.controller.company
            assert company is not None
            roster_line = (
                f"Roster: {len(company.roster)} living, {len(company.deceased_heroes)} memorialized"
            )
            active_slots = [
                hero_id or "empty"
                for _slot, hero_id in sorted(
                    company.active_party_slots.items(),
                    key=lambda item: item[0].value,
                )
            ]
            supplies = ", ".join(
                f"{supply_id} {quantity}"
                for supply_id, quantity in sorted(company.supplies.items())
            )
            body = (
                "Charter\n"
                f"{company.name}\n\n"
                "Status\n"
                f"Reputation: {company.reputation}\n"
                f"Coin: {company.coin}\n"
                f"{roster_line}\n"
                f"Known breaches: {', '.join(sorted(company.known_breaches)) or 'none'}\n\n"
                "Active Slots\n"
                f"{', '.join(active_slots)}\n\n"
                "Supplies\n"
                f"{supplies or 'none'}"
            )
        else:
            body = (
                "No company is loaded.\n\n"
                "Start a new charter or load the save slot to open Haven services."
            )
        self.app._show_screen("company", "Charter", body, actions, message=message)

    def show_save_load(self, message: str = "") -> None:
        has_company = self.app.controller.company is not None
        slot = "present" if self.app.save_path.exists() else "empty"
        actions = (
            ScreenAction(
                "1",
                "Save Company",
                "save",
                ("s",),
                enabled=has_company,
                kind=ScreenActionKind.SYSTEM,
                risk=ScreenActionRisk.LOW,
                unavailable_reason="Start or load a company before saving.",
                preview="Write the current company to the save slot.",
                result_hint="Existing saves ask for confirmation before overwrite.",
            ),
            ScreenAction(
                "2",
                "Load Company",
                "load",
                ("l",),
                default=not has_company,
                kind=ScreenActionKind.SYSTEM,
                risk=ScreenActionRisk.RISKY if has_company else ScreenActionRisk.LOW,
                preview="Load from the save slot.",
                result_hint="When a company is active, loading asks for confirmation.",
            ),
            ScreenAction("3", "Back", "back", ("b",), kind=ScreenActionKind.NAVIGATE),
        )
        body = f"Single save slot\n{self.app.save_path}\n\nSlot state: {slot}"
        self.app._show_screen("save_load", "Save / Load", body, actions, message=message)

    def show_help(self, *, return_to: str = "system") -> None:
        self.app.pending_help_return_state = return_to
        body = (
            "Textual controls\n"
            "Up/Down changes the focused command.\n"
            "Enter activates the focused command.\n"
            "Number keys activate visible commands.\n"
            "Single-key hotkeys activate visible commands.\n"
            "Esc or Backspace backs out where a Back or Cancel command is available.\n\n"
            "The legacy Rich CLI remains available with --cli."
        )
        self.app._show_screen(
            "help",
            "Help",
            body,
            (ScreenAction("1", "Back", "back", ("b",), default=True),),
        )

    def show_current_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        company = self.app.controller.company
        if company is None:
            self.show_main(message or "Start or load a company first.")
            return
        if self.app.controller.manual_combat is not None:
            if self.app._show_opening_enemy_response_if_needed():
                return
            self.app._show_combat_command(message)
            return
        if self.app._breach_pending():
            self.app._show_breach(message)
            return
        if company.active_expedition is not None:
            self.app._show_dungeon(message, hci)
            return
        result = self.app.controller.handle(ViewRegionalMap())
        if not result.success:
            self.show_main(result.error or "Regional map unavailable.")
            return
        view = cast(RegionalMapView, result.value)
        if company.town_state.get("location_id") == "haven":
            self.app._show_town(message, hci)
            return
        self.app._show_regional_place(message, hci, view=view)

    def show_system(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        has_company = self.app.controller.company is not None
        current_ai_label = _enemy_ai_mode_label(self.app.controller.enemy_ai_mode)
        next_ai_mode = _next_enemy_ai_mode(self.app.controller.enemy_ai_mode)
        enemy_controls_text = _enemy_ai_controls_text(ai_mode=self.app.controller.enemy_ai_mode)
        actions = (
            ScreenAction(
                "1",
                "Save Company",
                "save",
                ("s",),
                enabled=has_company,
                kind=ScreenActionKind.SYSTEM,
                risk=ScreenActionRisk.LOW,
                unavailable_reason="Start or load a company before saving.",
                preview="Write the current company to the save slot.",
                result_hint="Existing saves ask for confirmation before overwrite.",
            ),
            ScreenAction(
                "2",
                "Load Company",
                "load",
                ("l",),
                kind=ScreenActionKind.SYSTEM,
                risk=ScreenActionRisk.RISKY if has_company else ScreenActionRisk.LOW,
                preview="Load the company from the save slot.",
                result_hint="Replaces the in-memory company if a save exists.",
            ),
            ScreenAction(
                "3",
                f"Enemy AI: {current_ai_label}",
                "toggle_enemy_ai",
                ("a", "ai"),
                kind=ScreenActionKind.SYSTEM,
                risk=ScreenActionRisk.LOW,
                preview=(
                    "Switch between learned-static package timing and heuristic "
                    "immediate-action enemy policies."
                ),
                result_hint=f"Next mode: {_enemy_ai_mode_label(next_ai_mode)}.",
            ),
            ScreenAction(
                "4",
                "Help",
                "help",
                ("h", "?"),
                kind=ScreenActionKind.INSPECT,
                preview="Review controls and interface conventions.",
            ),
            ScreenAction(
                "5",
                "Quit",
                "quit",
                ("q",),
                kind=ScreenActionKind.SYSTEM,
                risk=ScreenActionRisk.IRREVERSIBLE,
                preview="Close the charter desk.",
                result_hint="Requires confirmation before quitting.",
            ),
            ScreenAction(
                "6",
                "Back",
                "back",
                ("b",),
                default=True,
                kind=ScreenActionKind.NAVIGATE,
            ),
        )
        if has_company:
            company = self.app.controller.company
            assert company is not None
            body = (
                "System\n\n"
                f"Company: {company.name}\n"
                f"Location: {company.town_state.get('location', 'Haven Town')}\n"
                f"Save slot: {self.app.save_path}\n\n"
                f"{enemy_controls_text}"
            )
        else:
            body = (
                f"System\n\nNo company is loaded.\nSave slot: {self.app.save_path}\n\n"
                f"{enemy_controls_text}"
            )
        self.app._show_screen(
            "system",
            "System",
            body,
            actions,
            message=message,
            log=self.app._result_log_text(self.app.recent_events, hci)
            if hci is not None
            else self.app._events_text(self.app.recent_events),
        )

    def show_confirm(
        self,
        confirm_id: str,
        title: str,
        body: str,
        *,
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        irreversible: bool = False,
    ) -> None:
        self.app.pending_confirm = confirm_id
        actions = ActionProvider.confirmation_actions(
            confirm_label,
            cancel_label,
            consequence=body,
            irreversible=irreversible,
        )
        self.app._show_screen("confirm", title, body, actions)

    def show_name_prompt(self) -> None:
        actions = (ScreenAction("1", "Cancel", "cancel", ("b",)),)
        self.app._show_screen(
            "name_company",
            "Start New Company",
            "Enter a company name, or press Enter for the default.",
            actions,
        )
        self.app.input_mode = "company_name"
        name_input = self.app.query_one("#name-input", Input)
        name_input.value = ""
        name_input.placeholder = DEFAULT_COMPANY_NAME
        name_input.disabled = False
        name_input.can_focus = True
        name_input.display = True
        name_input.focus()

    def _reset_ui_session(self) -> None:
        self.app.visited_screens.clear()
        self.app.current_gear_view = None
        self.app.current_supply_shop_view = None
