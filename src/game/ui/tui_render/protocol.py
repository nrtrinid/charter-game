"""Protocol for TUI render host (CharterApp surface used by render modules)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from textual.widgets import Input

from game.app.controller import AppController
from game.app.views import (
    CombatView,
    DungeonView,
    ExpeditionReportView,
    FormationView,
    GearInventoryView,
    RegionalMapView,
    RelicBrokerView,
    ScreenAction,
    SupplyShopView,
)
from game.core.events import GameEvent
from game.core.hci import HciResultAnalysis

if TYPE_CHECKING:
    from game.ui.tui_handlers.combat import CombatHandlers
    from game.ui.tui_handlers.regional import RegionalHandlers


class TuiRenderHost(Protocol):
    controller: AppController
    screen_state: str
    screen_title: str
    body_text: str
    message: str
    save_path: Path
    actions: tuple[ScreenAction, ...]
    focused_command_index: int
    playback_index: int
    playback_hci: HciResultAnalysis | None
    current_combat_view: CombatView | None
    current_combat_phase: str
    selected_skill_id: str | None
    pending_enemy_events: list[GameEvent]
    pending_enemy_beats: list[list[GameEvent]]
    pending_enemy_view: CombatView | None
    pending_resolution_hci: HciResultAnalysis | None
    opening_enemy_response_session_key: str
    pending_confirm: str | None
    pending_slot_label: str
    pending_gear_return_state: str
    pending_gear_locker_return_state: str
    pending_formation_return_state: str
    pending_help_return_state: str
    pending_regional_return_state: str
    pending_world_map_return_state: str
    current_regional_view: RegionalMapView | None
    current_dungeon_view: DungeonView | None
    current_report_view: ExpeditionReportView | None
    current_gear_view: GearInventoryView | None
    current_supply_shop_view: SupplyShopView | None
    current_relic_broker_view: RelicBrokerView | None
    idle_animation_frame: int
    beat_animation_frame: int
    current_beat_events: list[GameEvent]
    current_beat_deferred_events: list[GameEvent]
    current_beat_view: CombatView | None
    current_beat_title: str
    visited_screens: set[str]
    input_mode: str | None
    _current_formation_view: FormationView | None
    _combat_handlers: CombatHandlers
    _regional_handlers: RegionalHandlers
    _company_summary_objective: str
    _company_summary_roster_lines: list[str]

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

    def _record_events(self, events: list[GameEvent]) -> None: ...
    def _hide_name_input(self) -> None: ...
    def query_one(self, selector: str, expect_type: type[Input]) -> Input: ...

    def _show_main(self, message: str = "") -> None: ...
    def _show_town(self, message: str = "", hci: HciResultAnalysis | None = None) -> None: ...
    def _show_system(self, message: str = "", hci: HciResultAnalysis | None = None) -> None: ...
    def _show_current_place(
        self, message: str = "", hci: HciResultAnalysis | None = None
    ) -> None: ...
    def _show_regional_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
        return_to: str | None = None,
    ) -> None: ...
    def _show_dungeon(self, message: str = "", hci: HciResultAnalysis | None = None) -> None: ...
    def _show_combat_view(
        self,
        view: CombatView,
        *,
        phase: str = "command",
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _show_resolution(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _show_enemy_turn(self) -> None: ...
    def _show_expedition_report(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _begin_expedition(
        self,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
        direct_to_dungeon: bool = False,
    ) -> None: ...
    def _start_playback(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _finish_playback(self, message: str = "") -> None: ...
    def _renumbered_action(
        self,
        action: ScreenAction,
        number: int,
        *,
        label: str | None = None,
        default: bool | None = None,
    ) -> ScreenAction: ...
    def _result_log_text(
        self,
        events: Sequence[GameEvent],
        hci: HciResultAnalysis | None,
    ) -> str: ...
    def _events_text(self, events: Sequence[GameEvent]) -> str: ...
    def _party_watch_text(self) -> str: ...
    def _hero_lines(self, heroes: Sequence[Any]) -> str: ...
    def _hero_gear_summary(self, hero: Any) -> str: ...
    def _generic_action_detail(self, action: ScreenAction) -> str: ...
    def _is_safe_default(self, action: ScreenAction) -> bool: ...
    def _render(self) -> None: ...


    last_combat_actor_id: str
    turn_flash_actor_id: str
    turn_flash_frame: int
    pending_gear_hero_id: str
    pending_slot: Any
    playback_beats: list[Any]
    recent_events: list[GameEvent]

    def _breach_pending(self) -> bool: ...
    def _split_turn_events(
        self, events: list[GameEvent]
    ) -> tuple[list[GameEvent], list[GameEvent]]: ...
    def _set_current_beat(
        self,
        title: str,
        events: list[GameEvent],
        view: CombatView | None,
        *,
        deferred_events: list[GameEvent] | None = None,
    ) -> None: ...
    def _is_combat_beat_start(self, event: GameEvent) -> bool: ...
    def _continues_danger_beat(
        self, current: list[GameEvent], event: GameEvent
    ) -> bool: ...
    def _is_combat_footer_event(self, event: GameEvent) -> bool: ...
    def _event_source_actor_ids(self, events: Sequence[GameEvent]) -> set[str]: ...
    def _event_target_intents(self, events: Sequence[GameEvent]) -> dict[str, str]: ...
    def _enemy_response_beats(self, events: list[GameEvent]) -> list[list[GameEvent]]: ...
    def _post_combat_continue_label(self) -> str: ...
    def _post_combat_continue_description(self) -> str: ...
    def _show_opening_enemy_response_if_needed(self) -> bool: ...
    def _show_combat_command(self, message: str = "") -> None: ...
    def _show_expedition(self, message: str = "") -> None: ...
    def _show_breach(self, message: str = "") -> None: ...
    def _arrival_brief_text(self, view: Any) -> str: ...
    def _beat_text(self, beat: Any) -> str: ...

    @property
    def focused_action(self) -> ScreenAction | None: ...
