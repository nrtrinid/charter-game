"""Protocol for TUI handler hosts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from game.app.controller import AppController
from game.app.views import (
    CombatView,
    DungeonView,
    ExpeditionReportView,
    RecruitOffersView,
    RegionalMapView,
    ScreenAction,
)
from game.core.events import GameEvent
from game.core.hci import HciResultAnalysis

if TYPE_CHECKING:
    from game.ui.tui_handlers.regional import RegionalHandlers


class TuiHandlerHost(Protocol):
    controller: AppController
    screen_state: str
    message: str
    save_path: Path
    playback_index: int
    current_combat_view: CombatView | None
    current_combat_phase: str
    selected_skill_id: str | None
    pending_enemy_events: list[GameEvent]
    pending_enemy_beats: list[list[GameEvent]]
    pending_enemy_view: CombatView | None
    pending_resolution_hci: HciResultAnalysis | None
    opening_enemy_response_session_key: str
    pending_confirm: str | None
    pending_gear_return_state: str
    pending_gear_locker_return_state: str
    pending_formation_return_state: str
    pending_help_return_state: str
    current_regional_view: RegionalMapView | None
    pending_regional_return_state: str
    pending_world_map_return_state: str
    _regional_handlers: RegionalHandlers

    def _show_town(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _record_events(self, events: list[GameEvent]) -> None: ...

    def _show_regional_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
        return_to: str | None = None,
    ) -> None: ...

    def _result_log_text(
        self,
        events: Sequence[GameEvent],
        hci: HciResultAnalysis | None,
    ) -> str: ...

    def _begin_expedition(
        self,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
        direct_to_dungeon: bool = False,
    ) -> None: ...

    def _show_regional_map(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
    ) -> None: ...

    def _show_world_map(
        self,
        message: str = "",
        *,
        return_to: str = "current_place",
    ) -> None: ...

    def _show_system(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _start_playback(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _renumbered_action(
        self,
        action: ScreenAction,
        number: int,
        *,
        label: str | None = None,
        default: bool | None = None,
    ) -> ScreenAction: ...

    def _regional_place_text(self, view: RegionalMapView) -> str: ...

    def _show_screen(
        self,
        state: str,
        title: str,
        body: str,
        actions: Sequence[ScreenAction],
        *,
        message: str = "",
        log: str = "",
    ) -> None: ...

    def _regional_log_text(
        self,
        view: DungeonView,
        hci: HciResultAnalysis | None,
        *,
        actions: Sequence[ScreenAction] | None = None,
    ) -> str: ...

    def _room_action_notice(self, events: Sequence[GameEvent]) -> str: ...

    def _place_name(self, location_id: str) -> str: ...

    def _show_current_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_town_submenu(
        self,
        submenu: str,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_gear_locker(
        self,
        message: str = "",
        *,
        return_to: str = "town_market",
    ) -> None: ...

    def _show_pack(self, message: str = "") -> None: ...

    def _show_formation(
        self,
        message: str = "",
        *,
        return_to: str = "town_yard",
    ) -> None: ...

    def _show_hero_sheet(
        self,
        hero_id: str = "",
        message: str = "",
        *,
        return_to: str = "roster",
    ) -> None: ...

    def _show_company_summary(self, message: str = "") -> None: ...

    def _show_expedition(self, message: str = "") -> None: ...

    def _show_recruiting(self, message: str = "") -> None: ...

    def _show_deep_surgery(self, message: str = "") -> None: ...

    def _show_supply_shop(self, message: str = "") -> None: ...

    def _show_relic_broker(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_memorial(self) -> None: ...

    def _show_roster(self, message: str = "") -> None: ...

    def _show_supplies(self, message: str = "") -> None: ...

    def _show_ledger(self) -> None: ...

    def _show_expedition_report(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_main(self, message: str = "") -> None: ...

    def _show_hero_memories(self) -> None: ...

    def _show_hero_gear(self, message: str = "") -> None: ...

    def _show_recruiting_hire(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_supply_buy(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _current_recruiting_view(self) -> RecruitOffersView | None: ...

    def _render_recruiting_view(
        self,
        view: RecruitOffersView,
        message: str = "",
    ) -> None: ...

    def _show_playback(self) -> None: ...

    def _show_dungeon_map(self) -> None: ...

    def _show_dungeon_interactions(self, message: str = "") -> None: ...

    def _show_dungeon(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_breach(self, message: str = "") -> None: ...

    def _show_report_view(
        self,
        view: ExpeditionReportView,
        *,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_confirm(
        self,
        confirm_id: str,
        title: str,
        body: str,
        *,
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        irreversible: bool = False,
    ) -> None: ...

    def _show_combat_skill(self, message: str = "") -> None: ...

    def _show_combat_view(
        self,
        view: CombatView,
        *,
        phase: str,
        message: str = "",
    ) -> None: ...

    def _show_combat_command(self, message: str = "") -> None: ...

    def _show_resolution(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None: ...

    def _show_combat_target(
        self,
        view: CombatView,
        message: str = "",
    ) -> None: ...

    def _show_enemy_turn(self) -> None: ...

    def _finish_playback(self) -> None: ...

    def _split_turn_events(
        self,
        events: list[GameEvent],
    ) -> tuple[list[GameEvent], list[GameEvent]]: ...

    def _enemy_response_beats(
        self,
        events: list[GameEvent],
    ) -> list[list[GameEvent]]: ...

    def _render(self) -> None: ...

    def _show_name_prompt(self) -> None: ...

    def _show_save_load(self) -> None: ...

    def _request_load_company(self) -> None: ...

    def _show_help(self) -> None: ...

    def _cancel(self) -> None: ...

    def _save_company(self) -> None: ...

    def _load_company(self) -> None: ...

    def exit(self) -> None: ...
