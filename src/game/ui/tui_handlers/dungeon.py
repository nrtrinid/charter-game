"""Dungeon, playback, and expedition report action handlers for the TUI."""

from __future__ import annotations

from dataclasses import dataclass

from game.app.commands import (
    EnterGeneratedMaze,
    InspectDungeonRoom,
    MoveDungeon,
    RecoverCompany,
    RetraceGeneratedMaze,
    RetreatGeneratedMaze,
    ReturnFromDungeon,
    TakeExpeditionChoice,
    UseDungeonAction,
    WithdrawGeneratedMaze,
)
from game.app.views import ExpeditionReportView, RegionalMapView
from game.ui.tui_handlers.protocol import TuiHandlerHost


@dataclass
class DungeonHandlers:
    app: TuiHandlerHost

    def handle_playback_action(self, value: str) -> None:
        if value == "continue":
            self.app.playback_index += 1
            self.app._show_playback()

    def handle_dungeon_action(self, value: str) -> None:
        if value == "map":
            self.app._show_dungeon_map()
            return
        if value == "interact":
            self.app._show_dungeon_interactions()
            return
        if value == "inspect":
            result = self.app.controller.handle(InspectDungeonRoom())
            if result.success:
                self.app._record_events(result.events)
                self.app._show_dungeon("Room inspected.", result.hci)
            else:
                self.app._show_dungeon(result.error or "Inspection failed.")
            return
        if value == "return":
            result = self.app.controller.handle(ReturnFromDungeon())
            if result.success:
                self.app._record_events(result.events)
                if isinstance(result.value, RegionalMapView):
                    self.app._show_regional_place(hci=result.hci, view=result.value)
                elif isinstance(result.value, ExpeditionReportView):
                    self.app._show_report_view(result.value, hci=result.hci)
                else:
                    self.app._start_playback(result.events, result.hci)
            else:
                self.app._show_dungeon(result.error or "Return is unavailable.")
            return
        if value == "enter_generated_maze":
            result = self.app.controller.handle(EnterGeneratedMaze())
            if result.success:
                self.app._record_events(result.events)
                self.app._start_playback(result.events, result.hci)
            else:
                self.app._show_dungeon(result.error or "Breach entry failed.")
            return
        if value == "retrace_generated_maze":
            result = self.app.controller.handle(RetraceGeneratedMaze())
            if result.success:
                self.app._record_events(result.events)
                self.app._start_playback(result.events, result.hci)
            else:
                self.app._show_dungeon(result.error or "Maze retrace failed.")
            return
        if value == "withdraw_generated_maze":
            result = self.app.controller.handle(WithdrawGeneratedMaze())
            if result.success:
                self.app._record_events(result.events)
                self.app._start_playback(result.events, result.hci)
            else:
                self.app._show_dungeon(result.error or "Maze withdrawal failed.")
            return
        if value == "retreat_generated_maze":
            result = self.app.controller.handle(RetreatGeneratedMaze())
            if result.success:
                self.app._record_events(result.events)
                self.app._start_playback(result.events, result.hci)
            else:
                self.app._show_dungeon(result.error or "Maze retrace failed.")
            return
        if value.startswith("action:"):
            result = self.app.controller.handle(UseDungeonAction(value.removeprefix("action:")))
            if result.success:
                self.app._record_events(result.events)
                self.app._show_dungeon(self.app._room_action_notice(result.events), result.hci)
            else:
                self.app._show_dungeon(result.error or "Room action failed.")
            return
        result = self.app.controller.handle(MoveDungeon(value))
        if result.success:
            self.app._record_events(result.events)
            self.app._start_playback(result.events, result.hci)
        else:
            self.app._show_dungeon(result.error or "Move failed.")

    def handle_dungeon_interaction_action(self, value: str) -> None:
        if value.startswith("action:"):
            result = self.app.controller.handle(UseDungeonAction(value.removeprefix("action:")))
            if result.success:
                self.app._record_events(result.events)
                self.app._show_dungeon(self.app._room_action_notice(result.events), result.hci)
            else:
                self.app._show_dungeon_interactions(result.error or "Room action failed.")
            return
        self.app._show_dungeon()

    def handle_report_action(self, value: str) -> None:
        if value == "town":
            self.app._show_current_place()
        elif value == "roster":
            self.app._show_roster()
        elif value == "recover":
            result = self.app.controller.handle(RecoverCompany())
            if result.success:
                self.app._record_events(result.events)
                self.app._show_expedition_report("Company recovery funded.", result.hci)
            else:
                self.app._show_expedition_report(result.error or "Recovery is unavailable.")
        elif value == "save":
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


    def handle_breach_action(self, value: str) -> None:
        if value == "return_to_haven":
            result = self.app.controller.handle(TakeExpeditionChoice(value))
            if result.success:
                self.app._record_events(result.events)
                self.app._start_playback(result.events, result.hci)
            else:
                self.app._show_breach(result.error or "Could not return to Haven.")
        elif value == "descend_maze_depth_1":
            self.app._show_confirm(
                "descend_maze_depth_1",
                "Descend",
                "Descend into Maze Depth 1?",
                confirm_label="Descend",
            )

    def blocked_breach_back(self) -> None:
        self.app._show_breach("Resolve Return or Descend before leaving this breach.")

    def blocked_dungeon_back(self) -> None:
        self.app._show_dungeon("Use Return from a safe room to leave the dungeon.")
