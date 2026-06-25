"""Fullscreen Textual frontend for the interactive game."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.widgets import Input, Static

from game.app.commands import (
    AssignActiveHero,
    LoadGame,
    SaveGame,
    StartNewCompany,
)
from game.app.controller import AppController
from game.app.views import (
    EMPTY_SLOT_VALUE,
    ArrivalBriefView,
    CombatActorView,
    CombatView,
    DeepSurgeryView,
    DungeonView,
    ExpeditionReportView,
    FormationView,
    GearInventoryView,
    HeroSheetFreshMemoryView,
    HeroSheetMemoryEntryView,
    HeroSheetSignalView,
    HeroSheetTraitView,
    HeroSheetView,
    RecruitOffersView,
    RecruitOfferView,
    RegionalMapView,
    RelicBrokerView,
    RosterSectionView,
    ScreenAction,
    SupplyShopView,
    TownDashboardView,
    WorldView,
    build_formation_view,
    build_regional_render_view,
    build_shell_status,
)
from game.core.events import (
    CombatEffectEvent,
    DamageEvent,
    DeathEvent,
    DownedEvent,
    EnemyIntentEvent,
    EventType,
    GameEvent,
    HealingEvent,
    MissEvent,
    MoveEvent,
    ReactionSkippedEvent,
    ReactionUsedEvent,
    RoundEndedEvent,
    RoundStartedEvent,
    SkillUsedEvent,
    StatusChangedEvent,
    TurnDelayedEvent,
    TurnPassedEvent,
)
from game.core.hci import EventBeat, HciResultAnalysis
from game.ui.hci_text import (
    event_messages_text,
    format_formation_slot,
    format_meta_line,
    format_party_watch,
    generic_action_detail,
    kind_label,
    primary_hotkey,
    result_log_text,
    unavailable_message,
)
from game.ui.tui_constants import (
    BEAT_ANIMATION_START_FRAME,
    BEAT_IDLE_CYCLE,
    BREACH_PENDING_FLAG,
    DEFAULT_COMPANY_NAME,
    DEFAULT_SAVE_PATH,
    GLOBAL_SHORTCUT_SCREENS,
    GLOBAL_SHORTCUT_TEXT,
    TURN_FLASH_LAST_FRAME,
    UNSAFE_DEFAULT_RISKS,
)
from game.ui.tui_handlers import (
    CombatHandlers,
    DungeonHandlers,
    RegionalHandlers,
    ShellHandlers,
    TownHandlers,
)
from game.ui.tui_models import ScreenDescriptor, TuiScreenModel
from game.ui.tui_render import (
    CombatRender,
    DungeonRender,
    RegionalRender,
    ShellRender,
    TownRender,
)
from game.ui.tui_render.protocol import TuiRenderHost
from game.ui.tui_widgets import (
    BodyPane,
    CombatPanel,
    CommandDock,
    DetailPane,
    DungeonMapPanel,
    LogPane,
    StatusHeader,
)


class CharterApp(App):
    """Keyboard-first Textual app that runs beside the Rich CLI."""

    BINDINGS = [Binding(key, "noop", show=False) for key in "0123456789abcdefghijklmnopqrstuvwxyz"]

    CSS = """
    Screen {
        layout: vertical;
        background: #0d1117;
        color: #e6edf3;
    }

    #header {
        height: 3;
        padding: 0 1;
        background: #111827;
        color: #dbeafe;
    }

    #middle {
        height: 1fr;
    }

    #body {
        width: 2fr;
        padding: 1 2;
        border: round cyan;
        overflow-y: auto;
    }

    #side {
        width: 1fr;
    }

    #detail {
        height: 1fr;
        padding: 1;
        border: round yellow;
        overflow-y: auto;
    }

    #log {
        height: 1fr;
        padding: 1;
        border: round gray;
        color: #9ca3af;
        overflow-y: auto;
    }

    #dock {
        height: auto;
        max-height: 13;
        padding: 0 1;
        border: heavy yellow;
        background: #17120a;
        color: #f8e7b4;
    }

    #name-input {
        height: 3;
        margin: 0 1;
    }

    #footer {
        height: 1;
        padding: 0 1;
        color: #8b949e;
    }
    """

    def __init__(
        self,
        *,
        controller: AppController | None = None,
        save_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller or AppController()
        self.save_path = save_path or DEFAULT_SAVE_PATH
        self.screen_state = "main"
        self.screen_title = "Main"
        self.body_text = ""
        self.detail_text = ""
        self.log_text = ""
        self.message = ""
        self.actions: tuple[ScreenAction, ...] = ()
        self.focused_command_index = 0
        self.recent_events: list[GameEvent] = []
        self.playback_beats: list[EventBeat] = []
        self.playback_index = 0
        self.playback_hci: HciResultAnalysis | None = None
        self.current_combat_view: CombatView | None = None
        self.current_dungeon_view: DungeonView | None = None
        self.current_regional_view: RegionalMapView | None = None
        self.pending_regional_return_state = "current_place"
        self.pending_world_map_return_state = "current_place"
        self.current_report_view: ExpeditionReportView | None = None
        self.current_combat_phase = "command"
        self.selected_skill_id: str | None = None
        self.pending_enemy_events: list[GameEvent] = []
        self.pending_enemy_beats: list[list[GameEvent]] = []
        self.pending_enemy_view: CombatView | None = None
        self.pending_resolution_hci: HciResultAnalysis | None = None
        self.idle_animation_frame = 0
        self.beat_animation_frame = BEAT_ANIMATION_START_FRAME
        self.current_beat_title = ""
        self.current_beat_events: list[GameEvent] = []
        self.current_beat_deferred_events: list[GameEvent] = []
        self.current_beat_view: CombatView | None = None
        self.last_combat_actor_id = ""
        self.turn_flash_actor_id = ""
        self.turn_flash_frame = 0
        self.opening_enemy_response_session_key = ""
        self.pending_confirm: str | None = None
        self.pending_slot: Any | None = None
        self.pending_slot_label = ""
        self._current_formation_view: FormationView | None = None
        self.pending_gear_hero_id = ""
        self.pending_gear_return_state = "roster"
        self.pending_gear_locker_return_state = "town_market"
        self.pending_formation_return_state = "town_yard"
        self.pending_help_return_state = "system"
        self.visited_screens: set[str] = set()
        self.current_gear_view: GearInventoryView | None = None
        self.current_supply_shop_view: SupplyShopView | None = None
        self.current_relic_broker_view: RelicBrokerView | None = None
        self._company_summary_objective = ""
        self._company_summary_roster_lines: list[str] = []
        self.input_mode: str | None = None
        self._regional_handlers = RegionalHandlers(self)
        self._town_handlers = TownHandlers(self)
        self._dungeon_handlers = DungeonHandlers(self)
        self._combat_handlers = CombatHandlers(self)
        self._shell_handlers = ShellHandlers(self)
        render_host = cast(TuiRenderHost, self)
        self._shell_render = ShellRender(render_host)
        self._regional_render = RegionalRender(render_host)
        self._town_render = TownRender(render_host)
        self._dungeon_render = DungeonRender(render_host)
        self._combat_render = CombatRender(render_host)

    def compose(self) -> ComposeResult:
        yield StatusHeader(id="header")
        with Horizontal(id="middle"):
            yield BodyPane(id="body")
            with Vertical(id="side"):
                yield DetailPane(id="detail")
                yield LogPane(id="log")
        yield CommandDock(id="dock")
        yield Input(placeholder=DEFAULT_COMPANY_NAME, id="name-input")
        yield Static("Up/Down focus  |  Enter activate  |  1-9 shortcuts  |  Esc back", id="footer")

    def on_mount(self) -> None:
        self.set_interval(0.8, self._tick_idle_animation)
        self.set_interval(0.24, self._tick_beat_animation)
        self.set_interval(0.24, self._tick_turn_flash_animation)
        self._hide_name_input()
        self._show_main()

    def _tick_idle_animation(self) -> None:
        if self.screen_state == "combat" and self.current_combat_view is not None:
            self.idle_animation_frame = (self.idle_animation_frame + 1) % 4
            self._render_animation_frame()
            return
        if self.screen_state in {
            "formation",
            "assign_hero",
            "roster",
            "company_summary",
        }:
            self.idle_animation_frame = (self.idle_animation_frame + 1) % 4
            self._render_animation_frame()

    def _tick_beat_animation(self) -> None:
        if self.screen_state not in {"resolution", "enemy_turn"}:
            return
        if not self.current_beat_events:
            return
        last_frame = self._current_beat_animation_last_frame()
        if self.beat_animation_frame >= last_frame:
            self.beat_animation_frame = (
                last_frame + 1 + ((self.beat_animation_frame - last_frame) % BEAT_IDLE_CYCLE)
            )
        else:
            self.beat_animation_frame += 1
        self._render_animation_frame()

    def _tick_turn_flash_animation(self) -> None:
        if self.screen_state != "combat" or not self.turn_flash_actor_id:
            return
        if self.turn_flash_frame >= TURN_FLASH_LAST_FRAME:
            self.turn_flash_actor_id = ""
            self.turn_flash_frame = 0
        else:
            self.turn_flash_frame += 1
        self._render_animation_frame()

    def _render_animation_frame(self) -> None:
        try:
            self._render()
        except NoMatches:
            return

    def action_noop(self) -> None:
        pass

    @property
    def focused_action(self) -> ScreenAction | None:
        if not self.actions:
            return None
        return self.actions[self.focused_command_index]

    def on_key(self, event: Key) -> None:
        if self.input_mode is not None:
            if event.key == "escape":
                self.input_mode = None
                self._hide_name_input()
                if self.controller.company is None:
                    self._show_main("Company creation cancelled.")
                else:
                    self._show_current_place("Company creation cancelled.")
                event.stop()
            return

        key = event.key.lower()
        if key == "question_mark":
            key = "?"
        if key == "up":
            self._move_focus(-1)
            event.stop()
            return
        if key == "down":
            self._move_focus(1)
            event.stop()
            return
        if key == "enter":
            action = self.focused_action
            if action is not None:
                self._activate_action(action)
            event.stop()
            return
        if key in {"escape", "backspace"}:
            self._activate_back()
            event.stop()
            return
        if key.isdigit():
            action = self._action_for_number(key)
            if action is not None:
                self._activate_action(action)
                event.stop()
            return
        if len(key) == 1:
            action = self._action_for_hotkey(key)
            if action is not None:
                self._activate_action(action)
                event.stop()
                return
            if self._activate_global_shortcut(key):
                event.stop()
                return
            if key == "i" and self.controller.company is not None:
                self._show_gear_locker(return_to="town_market")
                event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.input_mode != "company_name":
            return
        name = event.value.strip() or DEFAULT_COMPANY_NAME
        self.input_mode = None
        self._hide_name_input()
        result = self.controller.handle(StartNewCompany(name))
        if result.success:
            self._reset_ui_session()
            self._record_events(result.events)
            self._show_current_place(f"{name} receives its charter.")
        else:
            self._show_main(result.error or "Could not start company.")
        event.stop()

    def _handle_deep_surgery_action(self, value: str) -> None:
        self._town_handlers.handle_deep_surgery_action(value)

    def _show_main(self, message: str = "") -> None:
        return self._shell_render.show_main(message)

    def _show_company(self, message: str = "") -> None:
        return self._shell_render.show_company(message)

    def _show_save_load(self, message: str = "") -> None:
        return self._shell_render.show_save_load(message)

    def _show_help(self, *, return_to: str = "system") -> None:
        return self._shell_render.show_help(return_to=return_to)

    def _show_current_place(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        return self._shell_render.show_current_place(message, hci)

    def _show_regional_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
        return_to: str | None = None,
    ) -> None:
        return self._regional_render.show_regional_place(
            message, hci, view=view, return_to=return_to
        )

    def _show_regional_map(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
    ) -> None:
        return self._regional_render.show_regional_map(message, hci, view=view)

    def _show_world_map(self, message: str = "", *, return_to: str = "current_place") -> None:
        return self._regional_render.show_world_map(message, return_to=return_to)

    def _show_system(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        return self._shell_render.show_system(message, hci)

    def _show_town(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        return self._town_render.show_town(message, hci)

    def _render_town_view(
        self,
        view: TownDashboardView,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        arrival_brief: ArrivalBriefView | None = None,
    ) -> None:
        return self._town_render._render_town_view(view, message, hci, arrival_brief)

    def _show_arrival_brief(
        self, view: ArrivalBriefView, message: str = "", hci: HciResultAnalysis | None = None
    ) -> None:
        return self._regional_render.show_arrival_brief(view, message, hci)

    def _show_town_submenu(
        self, submenu: str, message: str = "", hci: HciResultAnalysis | None = None
    ) -> None:
        return self._town_render.show_town_submenu(submenu, message, hci)

    def _town_hub_actions(self) -> tuple[ScreenAction, ...]:
        return self._town_render._town_hub_actions()

    def _town_submenu_actions(
        self, view: TownDashboardView, submenu: str
    ) -> tuple[ScreenAction, ...]:
        return self._town_render._town_submenu_actions(view, submenu)

    def _town_service_action(self, view: TownDashboardView, value: str) -> ScreenAction:
        return self._town_render._town_service_action(view, value)

    def _town_back_label(self, submenu: str) -> str:
        return self._town_render._town_back_label(submenu)

    def _brief_text(self, title: str, sections: Sequence[tuple[str, Sequence[str]]]) -> str:
        return self._town_render._brief_text(title, sections)

    def _reset_ui_session(self) -> None:
        return self._shell_render._reset_ui_session()

    def _first_visit_hint(self, screen_id: str, text: str) -> str:
        return self._town_render._first_visit_hint(screen_id, text)

    def _town_hub_body(self, view: TownDashboardView) -> str:
        return self._town_render._town_hub_body(view)

    def _town_yard_body(self, view: TownDashboardView) -> str:
        return self._town_render._town_yard_body(view)

    def _town_gate_text(self, view: TownDashboardView) -> str:
        return self._town_render._town_gate_text(view)

    def _town_market_text(self, view: TownDashboardView) -> str:
        return self._town_render._town_market_text(view)

    def _town_recovery_text(self, view: TownDashboardView) -> str:
        return self._town_render._town_recovery_text(view)

    def _deep_surgery_text(self, view: DeepSurgeryView) -> str:
        return self._town_render._deep_surgery_text(view)

    def _town_quartermaster_text(self, view: TownDashboardView) -> str:
        return self._town_render._town_quartermaster_text(view)

    def _town_recruitment_text(self, view: TownDashboardView) -> str:
        return self._town_render._town_recruitment_text(view)

    def _town_records_text(self, view: TownDashboardView) -> str:
        return self._town_render._town_records_text(view)

    def _contract_board_text(self, view: TownDashboardView) -> str:
        return self._town_render._contract_board_text(view)

    def _contract_reward_summary(self, entry: Any) -> str:
        return self._town_render._contract_reward_summary(entry)

    def _upgrade_board_text(self, view: TownDashboardView) -> str:
        return self._town_render._upgrade_board_text(view)

    def _show_roster(self, message: str = "") -> None:
        return self._town_render.show_roster(message)

    def _roster_hero_actions(
        self, sections: tuple[RosterSectionView, ...]
    ) -> tuple[ScreenAction, ...]:
        return self._town_render._roster_hero_actions(sections)

    def _show_supplies(self, message: str = "") -> None:
        return self._town_render.show_supplies(message)

    def _show_pack(self, message: str = "") -> None:
        return self._town_render.show_pack(message)

    def _pack_actions(self, gear: GearInventoryView) -> tuple[ScreenAction, ...]:
        return self._town_render._pack_actions(gear)

    def _pack_text(self, supplies: dict[str, int], gear: GearInventoryView) -> str:
        return self._town_render._pack_text(supplies, gear)

    def _show_company_summary(self, message: str = "") -> None:
        return self._town_render.show_company_summary(message)

    def _company_summary_actions(
        self, sections: tuple[RosterSectionView, ...]
    ) -> tuple[ScreenAction, ...]:
        return self._town_render._company_summary_actions(sections)

    def _show_gear_locker(self, message: str = "", *, return_to: str = "town_market") -> None:
        return self._town_render.show_gear_locker(message, return_to=return_to)

    def _gear_locker_actions(self, actions: tuple[ScreenAction, ...]) -> tuple[ScreenAction, ...]:
        return self._town_render._gear_locker_actions(actions)

    def _show_hero_sheet(
        self, hero_id: str = "", message: str = "", *, return_to: str = "roster"
    ) -> None:
        return self._town_render.show_hero_sheet(hero_id, message, return_to=return_to)

    def _show_hero_memories(self, message: str = "") -> None:
        return self._town_render.show_hero_memories(message)

    def _show_hero_gear(self, message: str = "") -> None:
        return self._town_render.show_hero_gear(message)

    def _hero_sheet_actions(self, view: HeroSheetView) -> tuple[ScreenAction, ...]:
        return self._town_render._hero_sheet_actions(view)

    def _hero_gear_actions(self, view: HeroSheetView) -> tuple[ScreenAction, ...]:
        return self._town_render._hero_gear_actions(view)

    def _gear_locker_text(self, view: GearInventoryView) -> str:
        return self._town_render._gear_locker_text(view)

    def _hero_sheet_text(self, view: HeroSheetView) -> str:
        return self._town_render._hero_sheet_text(view)

    def _hero_gear_text(self, view: HeroSheetView) -> str:
        return self._town_render._hero_gear_text(view)

    def _hero_quirks_memories_text(self, view: HeroSheetView) -> str:
        return self._town_render._hero_quirks_memories_text(view)

    def _sheet_trait_lines(self, trait: HeroSheetTraitView, *, prefix: str) -> list[str]:
        return self._town_render._sheet_trait_lines(trait, prefix=prefix)

    def _fresh_memory_lines(self, memory: HeroSheetFreshMemoryView) -> list[str]:
        return self._town_render._fresh_memory_lines(memory)

    def _permanent_memory_lines(self, memory: HeroSheetMemoryEntryView) -> list[str]:
        return self._town_render._permanent_memory_lines(memory)

    def _career_signal_lines(self, signals: Sequence[HeroSheetSignalView]) -> list[str]:
        return self._town_render._career_signal_lines(signals)

    def _hero_portrait_actor(self, hero_id: str, *, slot: str = "") -> CombatActorView | None:
        return self._town_render._hero_portrait_actor(hero_id, slot=slot)

    def _formation_portrait_actors(self, view: FormationView) -> dict[str, CombatActorView]:
        return self._town_render._formation_portrait_actors(view)

    def _formation_board_text(
        self, view: FormationView, *, focus_slot: str = "", focus_hero_id: str = ""
    ) -> str:
        return self._town_render._formation_board_text(
            view, focus_slot=focus_slot, focus_hero_id=focus_hero_id
        )

    def _company_summary_body(self, company_name: str) -> str:
        return self._town_render._company_summary_body(company_name)

    def _formation_detail_text(self, action: ScreenAction) -> str:
        return self._town_render._formation_detail_text(action)

    def _hero_sheet_preview_detail(self, hero_id: str) -> str:
        return self._town_render._hero_sheet_preview_detail(hero_id)

    def _hero_sheet_section_detail(self, action: ScreenAction) -> str:
        return self._town_render._hero_sheet_section_detail(action)

    def _hero_sheet_detail_view(self, hero_id: str) -> HeroSheetView | None:
        return self._town_render._hero_sheet_detail_view(hero_id)

    def _hero_sheet_quirk_line(self, view: HeroSheetView) -> str:
        return self._town_render._hero_sheet_quirk_line(view)

    def _hero_sheet_memory_line(self, view: HeroSheetView) -> str:
        return self._town_render._hero_sheet_memory_line(view)

    def _show_ledger(self, message: str = "") -> None:
        return self._town_render.show_ledger(message)

    def _show_memorial(self, message: str = "") -> None:
        return self._town_render.show_memorial(message)

    def _show_recruiting(self, message: str = "") -> None:
        return self._town_render.show_recruiting(message)

    def _render_recruiting_view(self, view: RecruitOffersView, message: str = "") -> None:
        return self._town_render._render_recruiting_view(view, message)

    def _show_recruiting_hire(
        self, message: str = "", hci: HciResultAnalysis | None = None
    ) -> None:
        return self._town_render.show_recruiting_hire(message, hci)

    def _current_recruiting_view(self) -> RecruitOffersView | None:
        return self._town_render._current_recruiting_view()

    def _recruiting_text(self, view: RecruitOffersView) -> str:
        return self._town_render._recruiting_text(view)

    def _recruit_offer_detail(self, action: ScreenAction) -> str:
        return self._town_render._recruit_offer_detail(action)

    def _recruit_fit_line(self, view: RecruitOffersView, offer: RecruitOfferView) -> str:
        return self._town_render._recruit_fit_line(view, offer)

    def _show_deep_surgery(self, message: str = "") -> None:
        return self._town_render.show_deep_surgery(message)

    def _show_supply_shop(self, message: str = "") -> None:
        return self._town_render.show_supply_shop(message)

    def _render_supply_shop_view(self, view: SupplyShopView, message: str = "") -> None:
        return self._town_render._render_supply_shop_view(view, message)

    def _show_supply_buy(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        return self._town_render.show_supply_buy(message, hci)

    def _supply_shop_text(self, view: SupplyShopView, *, screen_id: str = "supply_shop") -> str:
        return self._town_render._supply_shop_text(view, screen_id=screen_id)

    def _show_relic_broker(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        return self._town_render.show_relic_broker(message, hci)

    def _selection_actions(
        self, actions: tuple[ScreenAction, ...], *, back_label: str = "Back"
    ) -> tuple[ScreenAction, ...]:
        return self._town_render._selection_actions(actions, back_label=back_label)

    def _show_formation(self, message: str = "", *, return_to: str = "town_yard") -> None:
        return self._town_render.show_formation(message, return_to=return_to)

    def _render_formation_view(self, view: FormationView, message: str = "") -> None:
        return self._town_render._render_formation_view(view, message)

    def _show_assign_hero(self, slot_value: str) -> None:
        return self._town_render.show_assign_hero(slot_value)

    def _show_expedition(self, message: str = "") -> None:
        return self._dungeon_render.show_expedition(message)

    def _begin_expedition(
        self,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
        direct_to_dungeon: bool = False,
    ) -> None:
        return self._dungeon_render._begin_expedition(
            use_known_route=use_known_route,
            skip_known_route_playback=skip_known_route_playback,
            direct_to_dungeon=direct_to_dungeon,
        )

    def _start_playback(
        self, events: list[GameEvent], hci: HciResultAnalysis | None = None
    ) -> None:
        return self._dungeon_render._start_playback(events, hci)

    def _meaningful_playback_events(self, events: list[GameEvent]) -> list[GameEvent]:
        return self._dungeon_render._meaningful_playback_events(events)

    def _opening_enemy_response_playback_event_ids(self, events: list[GameEvent]) -> set[int]:
        return self._dungeon_render._opening_enemy_response_playback_event_ids(events)

    def _show_playback(self, message: str = "") -> None:
        return self._dungeon_render.show_playback(message)

    def _finish_playback(self, message: str = "") -> None:
        return self._dungeon_render._finish_playback(message)

    def _show_dungeon(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        return self._dungeon_render.show_dungeon(message, hci)

    def _show_dungeon_map(self, message: str = "") -> None:
        return self._dungeon_render.show_dungeon_map(message)

    def _show_dungeon_interactions(self, message: str = "") -> None:
        return self._dungeon_render.show_dungeon_interactions(message)

    def _dungeon_navigation_actions(self, view: DungeonView) -> tuple[ScreenAction, ...]:
        return self._dungeon_render._dungeon_navigation_actions(view)

    def _available_room_action_commands(self, view: DungeonView) -> tuple[ScreenAction, ...]:
        return self._dungeon_render._available_room_action_commands(view)

    def _blocked_room_actions(self, view: DungeonView) -> tuple[Any, ...]:
        return self._dungeon_render._blocked_room_actions(view)

    def _blocked_room_actions_reason(self, view: DungeonView) -> str:
        return self._dungeon_render._blocked_room_actions_reason(view)

    def _renumbered_action(
        self,
        action: ScreenAction,
        number: int,
        *,
        label: str | None = None,
        default: bool | None = None,
    ) -> ScreenAction:
        return self._dungeon_render._renumbered_action(action, number, label=label, default=default)

    def _show_expedition_report(
        self, message: str = "", hci: HciResultAnalysis | None = None
    ) -> None:
        return self._dungeon_render.show_expedition_report(message, hci)

    def _show_report_view(
        self, view: ExpeditionReportView, message: str = "", hci: HciResultAnalysis | None = None
    ) -> None:
        return self._dungeon_render.show_report_view(view, message, hci)

    def _show_combat_command(self, message: str = "") -> None:
        return self._combat_render.show_combat_command(message)

    def _show_combat_skill(self, message: str = "") -> None:
        return self._combat_render.show_combat_skill(message)

    def _show_combat_view(self, view: CombatView, *, phase: str, message: str = "") -> None:
        return self._combat_render.show_combat_view(view, phase=phase, message=message)

    def _track_combat_turn_handoff(self, view: CombatView, phase: str) -> None:
        return self._combat_render._track_combat_turn_handoff(view, phase)

    def _show_combat_target(self, view: CombatView, message: str = "") -> None:
        return self._combat_render.show_combat_target(view, message)

    def _show_resolution(
        self, events: list[GameEvent], hci: HciResultAnalysis | None = None
    ) -> None:
        return self._combat_render.show_resolution(events, hci)

    def _show_enemy_turn(self) -> None:
        return self._combat_render.show_enemy_turn()

    def _post_combat_continue_label(self) -> str:
        return self._dungeon_render._post_combat_continue_label()

    def _post_combat_continue_description(self) -> str:
        return self._dungeon_render._post_combat_continue_description()

    def _show_breach(self, message: str = "") -> None:
        return self._dungeon_render.show_breach(message)

    def _show_confirm(
        self,
        confirm_id: str,
        title: str,
        body: str,
        *,
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        irreversible: bool = False,
    ) -> None:
        return self._shell_render.show_confirm(
            confirm_id,
            title,
            body,
            confirm_label=confirm_label,
            cancel_label=cancel_label,
            irreversible=irreversible,
        )

    def _show_name_prompt(self) -> None:
        return self._shell_render.show_name_prompt()

    def _show_opening_enemy_response_if_needed(self) -> bool:
        return self._combat_render._show_opening_enemy_response_if_needed()

    def _regional_move_lands_on_place(
        self, view: RegionalMapView, events: Sequence[GameEvent]
    ) -> bool:
        return self._regional_render._regional_move_lands_on_place(view, events)

    def _available_regional_room_action_commands(
        self, view: RegionalMapView
    ) -> tuple[ScreenAction, ...]:
        return self._regional_render._available_regional_room_action_commands(view)

    def _show_regional_interactions(self, message: str = "") -> None:
        return self._regional_render.show_regional_interactions(message)

    def _regional_navigation_actions(self, view: RegionalMapView) -> tuple[ScreenAction, ...]:
        return self._regional_render._regional_navigation_actions(view)

    def _regional_map_display_actions(
        self, actions: Sequence[ScreenAction]
    ) -> tuple[ScreenAction, ...]:
        return self._regional_render._regional_map_display_actions(actions)

    def _focused_regional_node_id(self) -> str:
        return self._regional_render._focused_regional_node_id()

    def _contract_dock_help_text(self, action: ScreenAction) -> str:
        return self._town_render._contract_dock_help_text(action)

    def _route_dock_help_text(self, action: ScreenAction) -> str:
        return self._dungeon_render._route_dock_help_text(action)

    def _focused_dungeon_node_id(self) -> str:
        return self._dungeon_render._focused_dungeon_node_id()

    def _playback_log_text(self, events: Sequence[GameEvent]) -> str:
        return self._dungeon_render._playback_log_text(events)

    def _dungeon_log_text(
        self,
        view: DungeonView,
        hci: HciResultAnalysis | None,
        *,
        actions: Sequence[ScreenAction] | None = None,
    ) -> str:
        return self._dungeon_render._dungeon_log_text(view, hci, actions=actions)

    def _regional_log_text(
        self,
        view: DungeonView,
        hci: HciResultAnalysis | None,
        *,
        actions: Sequence[ScreenAction] | None = None,
    ) -> str:
        return self._regional_render._regional_log_text(view, hci, actions=actions)

    def _regional_place_text(self, view: RegionalMapView) -> str:
        return self._regional_render._regional_place_text(view)

    def _regional_map_text(
        self,
        view: RegionalMapView,
        render_view: DungeonView,
        *,
        actions: Sequence[ScreenAction],
        map_actions: Sequence[ScreenAction],
    ) -> str:
        return self._regional_render._regional_map_text(
            view, render_view, actions=actions, map_actions=map_actions
        )

    def _arrival_brief_text(self, view: ArrivalBriefView) -> str:
        return self._regional_render._arrival_brief_text(view)

    def _world_map_text(self, view: WorldView) -> str:
        return self._regional_render._world_map_text(view)

    def _travel_destination_label(self, view: WorldView, destination_id: str) -> str:
        return self._regional_render._travel_destination_label(view, destination_id)

    def _place_name(self, location_id: str) -> str:
        return self._regional_render._place_name(location_id)

    def _active_dungeon_minimap_text(self) -> str:
        return self._dungeon_render._active_dungeon_minimap_text()

    def _room_action_notice(self, events: Sequence[GameEvent]) -> str:
        return self._dungeon_render._room_action_notice(events)

    def _dungeon_place_text(self, view: DungeonView) -> str:
        return self._dungeon_render._dungeon_place_text(view)

    def _beat_text(self, beat: EventBeat) -> str:
        return self._combat_render._beat_text(beat)

    def _expedition_progress_strip(self) -> str:
        return self._dungeon_render._expedition_progress_strip()

    def _roster_text(self, sections: tuple[RosterSectionView, ...]) -> str:
        return self._town_render._roster_text(sections)

    def _combat_detail_text(self, action: ScreenAction) -> str:
        return self._combat_render._combat_detail_text(action)

    def _dungeon_detail_text(self, action: ScreenAction) -> str:
        return self._dungeon_render._dungeon_detail_text(action)

    def _dungeon_interact_detail_text(self, view: DungeonView) -> str:
        return self._dungeon_render._dungeon_interact_detail_text(view)

    def _dungeon_route_detail_text(self, action: ScreenAction, exit_node: Any) -> str:
        return self._dungeon_render._dungeon_route_detail_text(action, exit_node)

    def _dungeon_room_action_detail_text(self, action: ScreenAction, room_action: Any) -> str:
        return self._dungeon_render._dungeon_room_action_detail_text(action, room_action)

    def _room_action_state_reason(self, room_action: Any) -> str:
        return self._dungeon_render._room_action_state_reason(room_action)

    def _quantity_list(self, values: Sequence[tuple[str, int]]) -> str:
        return self._dungeon_render._quantity_list(values)

    def _quantity_label(self, item_id: str, quantity: int) -> str:
        return self._dungeon_render._quantity_label(item_id, quantity)

    def _sentence_case(self, text: str) -> str:
        return self._dungeon_render._sentence_case(text)

    def _gear_action_detail(self, action: ScreenAction) -> str:
        return self._town_render._gear_action_detail(action)

    def _supply_action_detail(self, action: ScreenAction) -> str:
        return self._town_render._supply_action_detail(action)

    def _pack_gear_detail(self, action: ScreenAction) -> str:
        return self._town_render._pack_gear_detail(action)

    def _contract_action_detail(self, action: ScreenAction) -> str:
        return self._town_render._contract_action_detail(action)

    def _combat_text(self, view: CombatView, *, mode: str, idle_frame: int = 0) -> str:
        return self._combat_render._combat_text(view, mode=mode, idle_frame=idle_frame)

    def _current_beat_text(self) -> str:
        return self._combat_render._current_beat_text()

    def _current_beat_animation_last_frame(self) -> int:
        return self._combat_render._current_beat_animation_last_frame()

    def _combat_beat_text(
        self,
        title: str,
        events: list[GameEvent],
        *,
        view: CombatView | None = None,
        animation_frame: int = 0,
        deferred_events: list[GameEvent] | None = None,
    ) -> str:
        return self._combat_render._combat_beat_text(
            title,
            events,
            view=view,
            animation_frame=animation_frame,
            deferred_events=deferred_events,
        )

    def _combat_event_beats(self, events: list[GameEvent]) -> list[list[GameEvent]]:
        return self._combat_render._combat_event_beats(events)

    def _combatant_lines(self, actors: Sequence[Any]) -> str:
        return self._combat_render._combatant_lines(actors)

    def _screen_descriptors(self) -> dict[str, ScreenDescriptor]:
        return {
            "main": ScreenDescriptor("main", self._handle_main_action, self._show_main),
            "system": ScreenDescriptor(
                "system",
                self._handle_system_action,
                self._show_current_place,
            ),
            "company": ScreenDescriptor("company", self._handle_company_action, self._show_main),
            "save_load": ScreenDescriptor(
                "save_load",
                self._handle_save_action,
                self._show_main,
            ),
            "help": ScreenDescriptor(
                "help",
                lambda _value: self._activate_back(),
                self._back_from_help,
            ),
            "town": ScreenDescriptor("town", self._handle_town_action, self._show_system),
            "town_gate": ScreenDescriptor(
                "town_gate",
                self._handle_town_submenu_action,
                self._show_town,
            ),
            "regional_place": ScreenDescriptor(
                "regional_place",
                self._handle_regional_place_action,
                self._back_from_regional_place,
            ),
            "regional_map": ScreenDescriptor(
                "regional_map",
                self._handle_regional_map_action,
                self._fold_roadbook,
            ),
            "regional_interact": ScreenDescriptor(
                "regional_interact",
                self._handle_regional_interaction_action,
                self._show_regional_place,
            ),
            "world_map": ScreenDescriptor(
                "world_map",
                lambda _value: self._activate_back(),
                self._back_from_world_map,
            ),
            "pack": ScreenDescriptor(
                "pack",
                self._handle_pack_action,
                self._show_current_place,
            ),
            "company_summary": ScreenDescriptor(
                "company_summary",
                self._handle_company_summary_action,
                self._show_current_place,
            ),
            "town_charter": ScreenDescriptor(
                "town_charter",
                self._handle_town_submenu_action,
                self._show_town,
            ),
            "town_market": ScreenDescriptor(
                "town_market",
                self._handle_town_submenu_action,
                self._show_town,
            ),
            "town_recovery": ScreenDescriptor(
                "town_recovery",
                self._handle_town_submenu_action,
                self._show_town,
            ),
            "deep_surgery": ScreenDescriptor(
                "deep_surgery",
                self._handle_deep_surgery_action,
                lambda: self._show_town_submenu("town_recovery"),
            ),
            "town_quartermaster": ScreenDescriptor(
                "town_quartermaster",
                self._handle_town_submenu_action,
                lambda: self._show_town_submenu("town_market"),
            ),
            "town_recruitment": ScreenDescriptor(
                "town_recruitment",
                self._handle_town_submenu_action,
                lambda: self._show_town_submenu("town_market"),
            ),
            "town_yard": ScreenDescriptor(
                "town_yard",
                self._handle_town_submenu_action,
                self._show_town,
            ),
            "town_upgrades": ScreenDescriptor(
                "town_upgrades",
                self._handle_town_submenu_action,
                lambda: self._show_town_submenu("town_charter"),
            ),
            "town_records": ScreenDescriptor(
                "town_records",
                self._handle_town_submenu_action,
                lambda: self._show_town_submenu("town_charter"),
            ),
            "roster": ScreenDescriptor(
                "roster",
                self._handle_roster_action,
                lambda: self._show_town_submenu("town_yard"),
            ),
            "hero_sheet": ScreenDescriptor(
                "hero_sheet",
                self._handle_hero_sheet_action,
                self._back_from_hero_sheet,
            ),
            "hero_memories": ScreenDescriptor(
                "hero_memories",
                lambda _value: self._back_from_hero_memories(),
                self._back_from_hero_memories,
            ),
            "hero_gear": ScreenDescriptor(
                "hero_gear",
                self._handle_hero_gear_action,
                self._back_from_hero_gear,
            ),
            "supplies": ScreenDescriptor(
                "supplies",
                lambda _value: self._activate_back(),
                self._show_town,
            ),
            "ledger": ScreenDescriptor(
                "ledger",
                lambda _value: self._activate_back(),
                lambda: self._show_town_submenu("town_records"),
            ),
            "memorial": ScreenDescriptor(
                "memorial",
                lambda _value: self._activate_back(),
                lambda: self._show_town_submenu("town_records"),
            ),
            "gear": ScreenDescriptor(
                "gear",
                self._handle_gear_action,
                self._back_from_gear_locker,
            ),
            "recruiting": ScreenDescriptor(
                "recruiting",
                self._handle_recruiting_action,
                lambda: self._show_town_submenu("town_market"),
            ),
            "recruiting_hire": ScreenDescriptor(
                "recruiting_hire",
                self._handle_recruiting_hire_action,
                self._back_from_recruiting_hire,
            ),
            "supply_shop": ScreenDescriptor(
                "supply_shop",
                self._handle_supply_action,
                lambda: self._show_town_submenu("town_market"),
            ),
            "supply_buy": ScreenDescriptor(
                "supply_buy",
                self._handle_supply_buy_action,
                self._show_supply_shop,
            ),
            "relic_broker": ScreenDescriptor(
                "relic_broker",
                self._handle_relic_broker_action,
                lambda: self._show_town_submenu("town_charter"),
            ),
            "formation": ScreenDescriptor(
                "formation",
                self._show_assign_hero,
                self._back_from_formation,
            ),
            "assign_hero": ScreenDescriptor(
                "assign_hero",
                self._assign_hero,
                self._back_from_assign_hero,
            ),
            "expedition": ScreenDescriptor(
                "expedition",
                lambda _value: self._begin_expedition(),
                self._show_main,
            ),
            "playback": ScreenDescriptor(
                "playback",
                self._handle_playback_action,
                self._finish_playback,
            ),
            "dungeon": ScreenDescriptor(
                "dungeon",
                self._handle_dungeon_action,
                self._blocked_dungeon_back,
                locked=True,
            ),
            "dungeon_interact": ScreenDescriptor(
                "dungeon_interact",
                self._handle_dungeon_interaction_action,
                self._show_dungeon,
            ),
            "dungeon_map": ScreenDescriptor(
                "dungeon_map",
                lambda _value: self._activate_back(),
                self._show_dungeon,
            ),
            "combat": ScreenDescriptor(
                "combat",
                self._activate_combat_action,
                self._back_from_combat,
                locked=True,
            ),
            "resolution": ScreenDescriptor(
                "resolution",
                lambda _value: self._combat_handlers.advance_resolution(),
                self._blocked_resolution_back,
                locked=True,
            ),
            "enemy_turn": ScreenDescriptor(
                "enemy_turn",
                lambda _value: self._combat_handlers.advance_enemy_turn(),
                self._blocked_resolution_back,
                locked=True,
            ),
            "breach": ScreenDescriptor(
                "breach",
                self._handle_breach_action,
                self._blocked_breach_back,
                locked=True,
            ),
            "expedition_report": ScreenDescriptor(
                "expedition_report",
                self._handle_report_action,
                self._show_current_place,
            ),
            "confirm": ScreenDescriptor("confirm", self._handle_confirm_action, self._cancel),
            "name_company": ScreenDescriptor(
                "name_company",
                lambda _value: self._cancel(),
                self._cancel,
            ),
        }

    def _activate_action(self, action: ScreenAction) -> None:
        if not action.enabled:
            self.message = self._unavailable_message(action)
            self._render()
            return

        value = action.value
        if value == "back":
            self._activate_back()
            return
        if value == "cancel":
            self._cancel()
            return

        descriptor = self._screen_descriptors().get(self.screen_state)
        if descriptor is None:
            self._show_main()
            return
        descriptor.activate(value)

    def _activate_combat_action(self, value: str) -> None:
        self._combat_handlers.activate_combat_action(value)

    def _advance_resolution(self) -> None:
        self._combat_handlers.advance_resolution()

    def _advance_enemy_turn(self) -> None:
        self._combat_handlers.advance_enemy_turn()

    def _handle_main_action(self, value: str) -> None:
        self._shell_handlers.handle_main_action(value)

    def _handle_system_action(self, value: str) -> None:
        self._shell_handlers.handle_system_action(value)

    def _handle_company_action(self, value: str) -> None:
        self._shell_handlers.handle_company_action(value)

    def _handle_save_action(self, value: str) -> None:
        self._shell_handlers.handle_save_action(value)

    def _handle_town_action(self, value: str) -> None:
        self._town_handlers.handle_town_action(value)

    def _handle_regional_place_action(self, value: str) -> None:
        self._regional_handlers.handle_place_action(value)

    def _handle_regional_map_action(self, value: str) -> None:
        self._regional_handlers.handle_map_action(value)

    def _handle_regional_walk(self, node_id: str) -> None:
        self._regional_handlers.handle_walk(node_id)

    def _resolve_regional_action(self, action_id: str) -> None:
        self._regional_handlers.resolve_action(action_id)

    def _handle_regional_interaction_action(self, value: str) -> None:
        self._regional_handlers.handle_interaction_action(value)

    def _handle_regional_travel(self, destination_id: str) -> None:
        self._regional_handlers.handle_travel(destination_id)

    def _back_from_regional_place(self) -> None:
        self._regional_handlers.back_from_place()

    def _fold_roadbook(self) -> None:
        self._regional_handlers.fold_roadbook()

    def _back_from_world_map(self) -> None:
        self._regional_handlers.back_from_world_map()

    def _handle_pack_action(self, value: str) -> None:
        self._town_handlers.handle_pack_action(value)

    def _handle_company_summary_action(self, value: str) -> None:
        self._town_handlers.handle_company_summary_action(value)

    def _handle_town_submenu_action(self, value: str) -> None:
        self._town_handlers.handle_town_submenu_action(value)

    def _handle_roster_action(self, value: str) -> None:
        self._town_handlers.handle_roster_action(value)

    def _handle_gear_action(self, value: str) -> None:
        self._town_handlers.handle_gear_action(value)

    def _handle_hero_sheet_action(self, value: str) -> None:
        self._town_handlers.handle_hero_sheet_action(value)

    def _handle_hero_gear_action(self, value: str) -> None:
        self._town_handlers.handle_hero_gear_action(value)

    def _back_from_hero_sheet(self) -> None:
        self._town_handlers.back_from_hero_sheet()

    def _back_from_hero_memories(self) -> None:
        self._town_handlers.back_from_hero_memories()

    def _back_from_hero_gear(self) -> None:
        self._town_handlers.back_from_hero_gear()

    def _back_from_gear_locker(self) -> None:
        self._town_handlers.back_from_gear_locker()

    def _back_from_formation(self) -> None:
        self._town_handlers.back_from_formation()

    def _back_from_assign_hero(self) -> None:
        self._town_handlers.back_from_assign_hero()

    def _handle_recruiting_action(self, value: str) -> None:
        self._town_handlers.handle_recruiting_action(value)

    def _handle_recruiting_hire_action(self, value: str) -> None:
        self._town_handlers.handle_recruiting_hire_action(value)

    def _handle_supply_action(self, value: str) -> None:
        self._town_handlers.handle_supply_action(value)

    def _handle_supply_buy_action(self, value: str) -> None:
        self._town_handlers.handle_supply_buy_action(value)

    def _handle_relic_broker_action(self, value: str) -> None:
        self._town_handlers.handle_relic_broker_action(value)

    def _handle_playback_action(self, value: str) -> None:
        self._dungeon_handlers.handle_playback_action(value)

    def _handle_dungeon_action(self, value: str) -> None:
        self._dungeon_handlers.handle_dungeon_action(value)

    def _handle_dungeon_interaction_action(self, value: str) -> None:
        self._dungeon_handlers.handle_dungeon_interaction_action(value)

    def _handle_report_action(self, value: str) -> None:
        self._dungeon_handlers.handle_report_action(value)

    def _handle_breach_action(self, value: str) -> None:
        self._dungeon_handlers.handle_breach_action(value)

    def _handle_confirm_action(self, value: str) -> None:
        self._shell_handlers.handle_confirm_action(value)

    def _handle_combat_command(self, value: str) -> None:
        self._combat_handlers.handle_combat_command(value)

    def _choose_skill(self, skill_id: str) -> None:
        self._combat_handlers.choose_skill(skill_id)

    def _choose_move(self, slot_id: str) -> None:
        self._combat_handlers.choose_move(slot_id)

    def _choose_reaction(self, value: str) -> None:
        self._combat_handlers.choose_reaction(value)

    def _choose_target(self, target_id: str) -> None:
        self._combat_handlers.choose_target(target_id)

    def _assign_hero(self, hero_id: str) -> None:
        if self.pending_slot is None:
            self._show_formation(
                "Choose a formation slot first.",
                return_to=self.pending_formation_return_state,
            )
            return
        result = self.controller.handle(AssignActiveHero(hero_id, self.pending_slot))
        if result.success:
            self._record_events(result.events)
            self._show_formation(
                "Formation updated.",
                return_to=self.pending_formation_return_state,
            )
        else:
            self._show_formation(
                result.error or "Formation update failed.",
                return_to=self.pending_formation_return_state,
            )
        self.pending_slot = None
        self.pending_slot_label = ""

    def _save_company(self) -> None:
        result = self.controller.handle(SaveGame(self.save_path))
        if result.success:
            self._record_events(result.events)
            self._show_system("Company saved.", result.hci)
        else:
            self._show_system(result.error or "Could not save company.")

    def _request_load_company(self) -> None:
        if self.controller.company is None:
            self._load_company()
            return
        self._show_confirm(
            "load_company",
            "Load Company",
            "Loading will replace the current in-memory company with the save slot.",
            confirm_label="Load Company",
            cancel_label="Keep Current",
            irreversible=True,
        )

    def _load_company(self) -> None:
        result = self.controller.handle(LoadGame(self.save_path))
        if result.success:
            self._reset_ui_session()
            self._record_events(result.events)
            self._show_current_place("Company loaded.", result.hci)
        else:
            if self.screen_state == "system":
                self._show_system(result.error or "Could not load save.")
            else:
                self._show_main(result.error or "Could not load save.")

    def _cancel(self) -> None:
        if self.screen_state == "name_company":
            self.input_mode = None
            self._hide_name_input()
            if self.controller.company is None:
                self._show_main("Company creation cancelled.")
            else:
                self._show_current_place("Company creation cancelled.")
        elif self.screen_state == "confirm":
            confirm = self.pending_confirm
            self.pending_confirm = None
            if confirm == "replace_company":
                self._show_current_place("Kept the current company.")
            elif confirm == "overwrite_save":
                if self.controller.company is None:
                    self._show_main("Save cancelled.")
                else:
                    self._show_system("Save cancelled.")
            elif confirm == "load_company":
                self._show_system("Load cancelled.")
            elif confirm == "descend_maze_depth_1":
                self._show_breach("Descent cancelled.")
            elif confirm == "quit":
                if self.controller.company is None:
                    self._show_main("Quit cancelled.")
                else:
                    self._show_current_place("Quit cancelled.")
            else:
                self._show_main()
        else:
            self._activate_back()

    def _activate_back(self) -> None:
        descriptor = self._screen_descriptors().get(self.screen_state)
        if descriptor is None:
            self._show_main()
            return
        descriptor.back()

    def _breach_pending(self) -> bool:
        company = self.controller.company
        return bool(
            company is not None
            and company.active_expedition is None
            and company.flags.get(BREACH_PENDING_FLAG, False)
        )

    def _back_from_help(self) -> None:
        self._shell_handlers.back_from_help()

    def _back_from_recruiting_hire(self) -> None:
        self._town_handlers.back_from_recruiting_hire()

    def _blocked_dungeon_back(self) -> None:
        self._dungeon_handlers.blocked_dungeon_back()

    def _blocked_breach_back(self) -> None:
        self._dungeon_handlers.blocked_breach_back()

    def _blocked_resolution_back(self) -> None:
        self._combat_handlers.blocked_resolution_back()

    def _back_from_combat(self) -> None:
        self._combat_handlers.back_from_combat()

    def _show_screen(
        self,
        state: str,
        title: str,
        body: str,
        actions: Sequence[ScreenAction],
        *,
        message: str = "",
        log: str = "",
    ) -> None:
        self._apply_screen_model(
            TuiScreenModel(
                state_id=state,
                title=title,
                body=body,
                actions=actions,
                message=message,
                log=log,
            )
        )

    def _apply_screen_model(self, model: TuiScreenModel) -> None:
        self.input_mode = None
        self._hide_name_input()
        self.screen_state = model.state_id
        self.screen_title = model.title
        self.body_text = model.body
        self.detail_text = model.detail
        self.log_text = model.log
        self.message = model.message
        self.actions = tuple(model.actions)
        self.focused_command_index = self._default_focus_index()
        self.visited_screens.add(model.state_id)
        self._render()

    def _render(self) -> None:
        status = build_shell_status(
            self.controller.company,
            str(self.save_path),
            save_exists=self.save_path.exists(),
            definitions=self.controller.definitions,
        )
        self.query_one("#header", StatusHeader).update_status(status)
        self.query_one("#body", BodyPane).update_screen(self.screen_title, self._live_body_text())
        self.query_one("#detail", DetailPane).update_detail(self._detail_text())
        log_pane = self.query_one("#log", LogPane)
        if self.screen_state == "town_charter":
            log_pane.display = False
        else:
            log_pane.display = True
            log_pane.update_log(self._live_log_content())
        self.query_one("#dock", CommandDock).update_actions(
            self.actions,
            self.focused_command_index,
            self._dock_help_text(),
            self._global_shortcut_text(),
        )

    def _live_body_text(self) -> str:
        if self.screen_state == "combat" and self.current_combat_view is not None:
            return self._combat_text(
                self.current_combat_view,
                mode=self.current_combat_phase,
                idle_frame=self.idle_animation_frame,
            )
        if self.screen_state in {"resolution", "enemy_turn"} and self.current_beat_events:
            return self._current_beat_text()
        if self.screen_state == "formation" and self._current_formation_view is not None:
            focus_slot = ""
            focus_hero_id = ""
            action = self.focused_action
            if action is not None and action.value != "back":
                focus_slot = action.value
                slot = next(
                    (
                        entry
                        for entry in self._current_formation_view.slots
                        if entry.slot_label == action.value
                    ),
                    None,
                )
                if slot is not None and slot.hero_id is not None:
                    focus_hero_id = slot.hero_id
            return "\n".join(
                (
                    "Formation",
                    "",
                    self._formation_board_text(
                        self._current_formation_view,
                        focus_slot=focus_slot,
                        focus_hero_id=focus_hero_id,
                    ),
                )
            )
        if self.screen_state == "company_summary" and self.controller.company is not None:
            return self._company_summary_body(self.controller.company.name)
        if self.screen_state == "assign_hero" and self._current_formation_view is not None:
            selected_slot = next(
                (
                    entry
                    for entry in self._current_formation_view.slots
                    if entry.slot_label == self.pending_slot_label
                ),
                None,
            )
            slot_name = format_formation_slot(self.pending_slot_label)
            current_name = selected_slot.hero_name if selected_slot is not None else "empty"
            focus_hero_id = ""
            action = self.focused_action
            if action is not None and action.value not in {"back", EMPTY_SLOT_VALUE}:
                focus_hero_id = action.value
            return "\n".join(
                (
                    "Assign Formation Slot",
                    "",
                    f"Slot: {slot_name}  |  Current: {current_name}",
                    "",
                    self._formation_board_text(
                        self._current_formation_view,
                        focus_slot=self.pending_slot_label,
                        focus_hero_id=focus_hero_id,
                    ),
                )
            )
        return self.body_text

    def _live_log_content(self) -> object:
        if (
            self.screen_state in {"regional_place", "regional_map"}
            and self.current_regional_view is not None
        ):
            render_view = build_regional_render_view(self.current_regional_view)
            return DungeonMapPanel.render_minimap(
                render_view,
                highlighted_node_id=self._focused_regional_node_id(),
                actions=self._active_regional_minimap_actions(),
            )
        if (
            self.screen_state in {"dungeon", "dungeon_interact", "dungeon_map"}
            and self.current_dungeon_view is not None
        ):
            return DungeonMapPanel.render_minimap(
                self.current_dungeon_view,
                highlighted_node_id=self._focused_dungeon_node_id(),
                actions=self._active_minimap_actions(),
            )
        return self.log_text

    def _active_minimap_actions(self) -> tuple[ScreenAction, ...]:
        if self.current_regional_view is not None and self.screen_state in {
            "regional_place",
            "regional_map",
        }:
            return self._active_regional_minimap_actions()
        if self.current_dungeon_view is None:
            return tuple(self.actions)
        if self.screen_state == "dungeon_map":
            return self._dungeon_navigation_actions(self.current_dungeon_view)
        return tuple(self.actions)

    def _active_regional_minimap_actions(self) -> tuple[ScreenAction, ...]:
        if self.current_regional_view is None:
            return tuple(self.actions)
        if self.screen_state == "regional_map":
            return self._regional_map_display_actions(
                self._regional_navigation_actions(self.current_regional_view)
            )
        return tuple(self.actions)

    def _detail_text(self) -> str:
        roadbook_opening_notice = self.screen_state == "regional_map" and self.message.startswith(
            "The company roadbook is unrolled."
        )
        if self.message and not roadbook_opening_notice:
            return f"Notice\n\n{self.message}"
        action = self.focused_action
        if action is None:
            return "No command selected."
        if self.screen_state == "combat":
            return self._combat_detail_text(action)
        if self.screen_state in {"dungeon", "dungeon_interact"}:
            return self._dungeon_detail_text(action)
        if self.screen_state == "town_charter":
            return self._contract_action_detail(action)
        if self.screen_state == "gear":
            return self._gear_action_detail(action)
        if self.screen_state in {"supply_shop", "supply_buy"}:
            return self._supply_action_detail(action)
        if self.screen_state == "pack" and action.value == "gear":
            return self._pack_gear_detail(action)
        if self.screen_state in {"roster", "company_summary", "pack"}:
            if action.value.startswith("hero:"):
                hero_id = action.value.removeprefix("hero:")
                return self._hero_sheet_preview_detail(hero_id)
        if self.screen_state == "hero_sheet":
            return self._hero_sheet_section_detail(action)
        if self.screen_state == "recruiting_hire":
            return self._recruit_offer_detail(action)
        if self.screen_state in {"formation", "assign_hero"}:
            return self._formation_detail_text(action)
        return self._generic_action_detail(action)

    def _dock_help_text(self) -> str:
        action = self.focused_action
        if action is None:
            return ""
        if self.screen_state == "combat" and self.current_combat_view is not None:
            return CombatPanel.command_help_text(
                self.current_combat_view,
                self.current_combat_phase,
                action,
            )
        if self.screen_state == "town_charter":
            return self._contract_dock_help_text(action)
        if not action.enabled:
            reason = action.unavailable_reason or action.description or "Not available right now."
            return "\n".join(("Locked", reason))
        if self.screen_state in {
            "dungeon",
            "dungeon_interact",
            "regional_place",
            "regional_map",
        }:
            return self._route_dock_help_text(action)
        if action.value == "back":
            return "Return to the previous screen."
        if action.value == "cancel":
            return action.preview or "Cancel and keep the current state."
        if action.value == "confirm":
            return action.result_hint or "Confirm the pending choice."
        lines = [
            action.result_hint
            or action.preview
            or action.description
            or "Enter activates the focused command."
        ]
        if action.cost:
            lines.append(f"Cost: {action.cost}")
        return "\n".join(lines)

    def _move_focus(self, offset: int) -> None:
        if not self.actions:
            return
        self.focused_command_index = (self.focused_command_index + offset) % len(self.actions)
        self.message = ""
        self._render()

    def _default_focus_index(self) -> int:
        for index, action in enumerate(self.actions):
            if action.default and action.enabled and self._is_safe_default(action):
                return index
        for index, action in enumerate(self.actions):
            if action.enabled and self._is_safe_default(action):
                return index
        for index, action in enumerate(self.actions):
            if action.enabled:
                return index
        return 0

    def _is_safe_default(self, action: ScreenAction) -> bool:
        return action.risk not in UNSAFE_DEFAULT_RISKS

    def _kind_label(self, action: ScreenAction) -> str:
        return kind_label(action)

    def _unavailable_message(self, action: ScreenAction) -> str:
        return unavailable_message(action)

    def _action_for_number(self, number: str) -> ScreenAction | None:
        return next((action for action in self.actions if action.number == number), None)

    def _action_for_hotkey(self, hotkey: str) -> ScreenAction | None:
        for action in self.actions:
            if hotkey in {alias.lower() for alias in action.aliases if len(alias) == 1}:
                return action
        return None

    def _global_shortcut_text(self) -> str:
        if self.screen_state in GLOBAL_SHORTCUT_SCREENS:
            return GLOBAL_SHORTCUT_TEXT
        return ""

    def _activate_global_shortcut(self, key: str) -> bool:
        if self.screen_state not in GLOBAL_SHORTCUT_SCREENS:
            return False
        if key == "m":
            if self.screen_state in {"world_map", "dungeon_map", "regional_map"}:
                self._activate_back()
            elif self.screen_state in {"dungeon", "dungeon_interact"}:
                self._show_dungeon_map()
            elif self.screen_state in {"regional_place", "regional_interact"}:
                if self.screen_state == "regional_interact":
                    self._show_regional_place()
                elif (
                    self.current_regional_view is not None
                    and self.current_regional_view.anchor_kind in {"east_gate", "shallow_cave"}
                ):
                    self._show_regional_map()
            else:
                self._show_world_map()
            return True
        if key == "p":
            if self.screen_state == "pack":
                self._activate_back()
            else:
                self._show_pack()
            return True
        if key == "c":
            if self.screen_state == "company_summary":
                self._activate_back()
            else:
                self._show_company_summary()
            return True
        if key == "?":
            if self.screen_state == "help":
                self._activate_back()
            else:
                self._show_help(return_to="current_place")
            return True
        return False

    def _primary_hotkey(self, action: ScreenAction) -> str:
        return primary_hotkey(action)

    def _hide_name_input(self) -> None:
        name_input = self.query_one("#name-input", Input)
        name_input.display = False
        name_input.disabled = True
        name_input.can_focus = False
        name_input.value = ""
        self.set_focus(None)

    def _record_events(self, events: list[GameEvent]) -> None:
        self.recent_events = list(events[-8:])

    def _events_text(self, events: Sequence[GameEvent]) -> str:
        return event_messages_text(events)

    def _result_log_text(
        self,
        events: Sequence[GameEvent],
        hci: HciResultAnalysis | None,
    ) -> str:
        return result_log_text(events, hci)

    def _party_watch_text(self) -> str:
        company = self.controller.company
        if company is None:
            return ""
        formation = build_formation_view(company, self.controller.definitions)
        return format_party_watch(formation)

    def _hero_lines(self, heroes: Sequence[Any]) -> str:
        if not heroes:
            return "none"
        lines = []
        for hero in heroes:
            status = ", ".join(hero.statuses)
            memory = f"Memory: {hero.latest_memory}" if getattr(hero, "latest_memory", "") else ""
            detail = format_meta_line(
                hero.name,
                hero.class_id,
                f"{hero.hp}/{hero.max_hp} HP",
                f"{hero.effort}/{hero.max_effort} Effort",
                f"Strain {getattr(hero, 'strain', '')}",
                self._hero_gear_summary(hero),
                status,
                memory,
            )
            lines.append(f"{hero.slot}: {detail}")
        return "\n".join(lines)

    def _hero_gear_summary(self, hero: Any) -> str:
        equipped = getattr(hero, "equipped_gear", "")
        if not equipped:
            return ""
        bonus = getattr(hero, "stat_bonus", "")
        if bonus:
            return f"Gear: {equipped} ({bonus})"
        return f"Gear: {equipped}"

    def _generic_action_detail(self, action: ScreenAction) -> str:
        return generic_action_detail(action, safe_default=self._is_safe_default(action))

    def _set_current_beat(
        self,
        title: str,
        events: list[GameEvent],
        view: CombatView | None,
        *,
        deferred_events: list[GameEvent] | None = None,
    ) -> None:
        self.beat_animation_frame = BEAT_ANIMATION_START_FRAME
        self.current_beat_title = title
        self.current_beat_events = list(events)
        self.current_beat_deferred_events = list(deferred_events or [])
        self.current_beat_view = view

    def _is_combat_beat_start(self, event: GameEvent) -> bool:
        return isinstance(
            event,
            SkillUsedEvent | EnemyIntentEvent | MoveEvent | TurnDelayedEvent | TurnPassedEvent,
        )

    def _continues_danger_beat(
        self,
        current: list[GameEvent],
        event: GameEvent,
    ) -> bool:
        if not current or not isinstance(event, SkillUsedEvent):
            return False
        first = current[0]
        return (
            isinstance(first, EnemyIntentEvent)
            and first.enemy_id == event.actor_id
            and first.skill_id == event.skill_id
        )

    def _is_combat_footer_event(self, event: GameEvent) -> bool:
        return event.event_type in {
            EventType.ROUND_STARTED,
            EventType.ROUND_ENDED,
            EventType.ENCOUNTER_ENDED,
            EventType.COMBAT_ENDED,
            EventType.COMBAT_RETREAT_DECLARED,
            EventType.COMBAT_RETREATED,
            EventType.EXPEDITION,
            EventType.LOOT_GAINED,
            EventType.BREACH_DISCOVERED,
            EventType.EXPEDITION_RETURNED,
        }

    def _track_combat_active_side(
        self,
        event: GameEvent,
        active_side: str,
        *,
        enemy_side_seen: bool,
    ) -> tuple[str, bool]:
        if isinstance(event, SkillUsedEvent):
            active_side = self._actor_side(event.actor_id) or active_side
            enemy_side_seen = enemy_side_seen or active_side == "enemy"
        elif isinstance(event, EnemyIntentEvent):
            active_side = "enemy"
            enemy_side_seen = True
        elif isinstance(event, MoveEvent):
            side = self._actor_side(event.actor_id)
            if side and (side == "enemy" or side == active_side):
                active_side = side
                enemy_side_seen = enemy_side_seen or active_side == "enemy"
        elif isinstance(event, TurnDelayedEvent | TurnPassedEvent):
            side = self._actor_side(event.actor_id)
            if side:
                active_side = side
                enemy_side_seen = enemy_side_seen or active_side == "enemy"
        return active_side, enemy_side_seen

    def _enemy_response_beats(
        self,
        enemy_events: list[GameEvent],
    ) -> list[list[GameEvent]]:
        return [
            beat
            for beat in self._combat_event_beats(enemy_events)
            if any(
                isinstance(
                    event,
                    SkillUsedEvent
                    | EnemyIntentEvent
                    | MoveEvent
                    | TurnDelayedEvent
                    | TurnPassedEvent,
                )
                for event in beat
            )
        ]

    def _split_turn_events(
        self,
        events: list[GameEvent],
    ) -> tuple[list[GameEvent], list[GameEvent]]:
        hero_events: list[GameEvent] = []
        enemy_events: list[GameEvent] = []
        active_side = "hero"
        enemy_side_seen = False
        for event in events:
            active_side, enemy_side_seen = self._track_combat_active_side(
                event,
                active_side,
                enemy_side_seen=enemy_side_seen,
            )
            target = enemy_events if active_side == "enemy" else hero_events
            if isinstance(event, RoundEndedEvent | RoundStartedEvent):
                target = enemy_events if enemy_side_seen else hero_events
            target.append(event)
        return hero_events, enemy_events

    def _actor_side(self, actor_id: str) -> str:
        for view in (
            self.current_beat_view,
            self.current_combat_view,
            self.pending_enemy_view,
        ):
            if view is None:
                continue
            for actor in (*view.party, *view.enemies):
                if actor.actor_id == actor_id:
                    return actor.team
        session = self.controller.manual_combat
        if session is not None:
            if actor_id in session.state.heroes:
                return "hero"
            if actor_id in session.state.enemies:
                return "enemy"
        if actor_id.startswith("hero_"):
            return "hero"
        return ""

    def _event_source_actor_ids(self, events: Sequence[GameEvent]) -> set[str]:
        actor_ids: set[str] = set()
        for event in events:
            if isinstance(event, SkillUsedEvent):
                actor_ids.add(event.actor_id)
            elif isinstance(event, EnemyIntentEvent):
                actor_ids.add(event.enemy_id)
            elif isinstance(event, DamageEvent):
                actor_ids.add(event.source_id)
            elif isinstance(event, HealingEvent):
                actor_ids.add(event.source_id)
            elif isinstance(event, CombatEffectEvent):
                actor_ids.add(event.source_id or event.actor_id)
            elif isinstance(event, MissEvent):
                actor_ids.add(event.actor_id)
            elif isinstance(event, ReactionUsedEvent):
                actor_ids.add(event.actor_id)
                actor_ids.add(event.enemy_id)
            elif isinstance(event, ReactionSkippedEvent):
                actor_ids.add(event.enemy_id)
            elif isinstance(event, MoveEvent | TurnDelayedEvent | TurnPassedEvent):
                actor_ids.add(event.actor_id)
        return actor_ids

    def _event_target_intents(self, events: Sequence[GameEvent]) -> dict[str, str]:
        intents: dict[str, str] = {}
        active_intent = "attack"
        for event in events:
            if isinstance(event, SkillUsedEvent):
                active_intent = self._skill_intent(event.skill_id)
                if event.target_id is not None:
                    intents[event.target_id] = active_intent
            elif isinstance(event, EnemyIntentEvent):
                intents[event.target_id] = "debuff"
            elif isinstance(event, ReactionUsedEvent | ReactionSkippedEvent):
                intents[event.target_id] = "debuff"
            elif isinstance(event, DamageEvent | MissEvent | HealingEvent):
                intents[event.target_id] = active_intent
            elif isinstance(event, CombatEffectEvent):
                target_id = event.target_id or event.actor_id
                intents[target_id] = "heal" if event.emphasis == "good" else "debuff"
            elif isinstance(event, StatusChangedEvent | DownedEvent | DeathEvent):
                intents[event.actor_id] = active_intent
        return intents

    def _skill_intent(self, skill_id: str) -> str:
        skill = self.controller.definitions.skills.get(skill_id)
        if skill is None:
            return "attack"
        tags = set(skill.tags)
        if "treatment" in tags or "heal" in tags or "support" in tags:
            return "heal"
        if tags & {"debuff", "control", "horror", "status"}:
            return "debuff"
        return "attack"

    def _resolution_text(self, events: list[GameEvent]) -> str:
        if not events:
            return "The turn resolves."
        sections: list[tuple[str, list[str]]] = [
            ("Hero Action", []),
            ("Enemy Response", []),
            ("Round Flow", []),
            ("Outcome", []),
        ]
        hero_action, enemy_response, round_flow, outcome = (section[1] for section in sections)
        active_side = "hero"
        enemy_side_seen = False
        for event in events:
            active_side, enemy_side_seen = self._track_combat_active_side(
                event,
                active_side,
                enemy_side_seen=enemy_side_seen,
            )
            if event.event_type in {
                EventType.SKILL_USED,
                EventType.DAMAGE,
                EventType.HEALING,
                EventType.COMBAT_EFFECT,
                EventType.MISS,
                EventType.MOVE,
                EventType.TURN_DELAYED,
                EventType.TURN_PASSED,
                EventType.DOWNED,
                EventType.DEATH,
            }:
                if active_side == "hero":
                    hero_action.append(event.message)
                else:
                    enemy_response.append(event.message)
            elif event.event_type in {
                EventType.ROUND_STARTED,
                EventType.ROUND_ENDED,
            }:
                round_flow.append(event.message)
            elif event.event_type in {
                EventType.ENCOUNTER_STARTED,
                EventType.ENCOUNTER_ENDED,
                EventType.COMBAT_ENDED,
                EventType.COMBAT_RETREAT_DECLARED,
                EventType.COMBAT_RETREATED,
                EventType.EXPEDITION,
                EventType.LOOT_GAINED,
                EventType.BREACH_DISCOVERED,
                EventType.EXPEDITION_RETURNED,
            }:
                outcome.append(event.message)
            else:
                outcome.append(event.message)
        lines: list[str] = []
        for title, messages in sections:
            if messages:
                lines.append(title)
                lines.extend(f"- {message}" for message in messages)
                lines.append("")
        return "\n".join(lines).strip() or self._events_text(events)
