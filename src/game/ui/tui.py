"""Fullscreen Textual frontend for the interactive game."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.widgets import Input, Static

from game.app.actions import (
    ActionProvider,
    room_action_state_reason,
)
from game.app.commands import (
    AssignActiveHero,
    GenerateRecruitOffers,
    LoadGame,
    SaveGame,
    StartExpedition,
    StartNewCompany,
    ViewCombat,
    ViewDungeon,
    ViewExpeditionReport,
    ViewGear,
    ViewHeroSheet,
    ViewLedger,
    ViewMemorial,
    ViewRegionalMap,
    ViewRoster,
    ViewSupplies,
    ViewTown,
    ViewWorld,
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
    ScreenActionKind,
    ScreenActionRisk,
    SupplyShopView,
    TownDashboardView,
    WorldView,
    build_deep_surgery_view,
    build_formation_view,
    build_hero_portrait_view,
    build_recruit_offers_view,
    build_regional_render_view,
    build_relic_broker_view,
    build_shell_status,
    build_supply_shop_view,
    hero_protection_line,
    preview_assign_hero,
)
from game.combat.enemy_decision import (
    PRODUCTION_ENEMY_AI_MODE_DESCRIPTIONS,
    PRODUCTION_ENEMY_AI_MODE_LABELS,
    SUPPORTED_PRODUCTION_ENEMY_AI_MODES,
    production_enemy_movement_mode,
    production_enemy_wait_mode,
)
from game.core.events import (
    CombatEffectEvent,
    DamageEvent,
    DeathEvent,
    DownedEvent,
    EnemyIntentEvent,
    EventType,
    ExpeditionEvent,
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
from game.core.hci import EventBeat, HciResultAnalysis, build_event_beats
from game.ui.hci_text import (
    event_messages_text,
    format_compact_roster_row,
    format_formation_lane_summary,
    format_formation_slot,
    format_meta_line,
    format_party_watch,
    generic_action_detail,
    kind_label,
    primary_hotkey,
    result_log_text,
    unavailable_message,
)
from game.ui.tui_handlers import (
    CombatHandlers,
    DungeonHandlers,
    RegionalHandlers,
    ShellHandlers,
    TownHandlers,
)
from game.ui.tui_models import ScreenDescriptor, TuiScreenModel
from game.ui.tui_widgets import (
    BodyPane,
    CombatPanel,
    CommandDock,
    CompanyPanel,
    DetailPane,
    DungeonMapPanel,
    DungeonRoomPanel,
    ExpeditionProgressStrip,
    ExpeditionReportPanel,
    FormationBoard,
    GearLockerPanel,
    LogPane,
    PackPanel,
    RelicBrokerPanel,
    StatusHeader,
    SupplyShopPanel,
    TownDashboardPanel,
    YardPanel,
    portrait_detail_lines,
)
from game.ui.wounds import mortal_wound_badge

DEFAULT_SAVE_PATH = Path("saves/company.json")
BREACH_PENDING_FLAG = "opening_breach_pending"
DEFAULT_COMPANY_NAME = "Haven Charter"
BEAT_ANIMATION_LAST_FRAME = 4
BEAT_ANIMATION_START_FRAME = -1
BEAT_IDLE_CYCLE = 16
TURN_FLASH_LAST_FRAME = 1
UNSAFE_DEFAULT_RISKS = {
    ScreenActionRisk.COSTLY,
    ScreenActionRisk.RISKY,
    ScreenActionRisk.IRREVERSIBLE,
    "costly",
    "risky",
    "irreversible",
}
GLOBAL_SHORTCUT_TEXT = "Shortcuts: [M] Map  [P] Pack  [C] Company  [?] Help"
GLOBAL_SHORTCUT_SCREENS = {
    "regional_place",
    "regional_interact",
    "regional_map",
    "world_map",
    "dungeon",
    "dungeon_interact",
    "dungeon_map",
    "pack",
    "company_summary",
    "help",
}


def _route_direction_label(exit_node: Any) -> str:
    direction = str(getattr(exit_node, "direction", "")).strip()
    return direction.title() if direction else "Listed Exit"


def _route_summary_line(exit_node: Any) -> str:
    pieces = [_route_direction_label(exit_node)]
    tag = _route_exception_tag(exit_node)
    if tag:
        pieces.append(tag)
    return " - ".join(piece for piece in pieces if piece)


def _route_exception_tag(exit_node: Any) -> str:
    if bool(getattr(exit_node, "cleared", False)):
        return ""
    node_type = str(getattr(exit_node, "node_type", ""))
    if node_type not in {"boss", "breach", "maze"}:
        return ""
    return node_type.replace("_", " ").title()


def _route_warning_line(exit_node: Any) -> str:
    if bool(getattr(exit_node, "cleared", False)):
        return ""
    node_type = str(getattr(exit_node, "node_type", ""))
    if node_type == "boss":
        return "Warning: serious danger ahead."
    if node_type == "breach":
        return "Warning: breach threshold ahead."
    if node_type == "maze":
        return "Warning: Maze route ahead."
    return ""


def _enemy_ai_mode_label(mode: str) -> str:
    return PRODUCTION_ENEMY_AI_MODE_LABELS.get(mode, mode.replace("_", " ").title())


def _next_enemy_ai_mode(mode: str) -> str:
    modes = SUPPORTED_PRODUCTION_ENEMY_AI_MODES
    try:
        index = modes.index(mode)
    except ValueError:
        return modes[0]
    return modes[(index + 1) % len(modes)]


def _enemy_ai_mode_text(current_mode: str) -> str:
    lines = [
        "Enemy AI",
        f"Current: {_enemy_ai_mode_label(current_mode)}",
        (
            "Timing: wait "
            f"{_enemy_timing_label(production_enemy_wait_mode(current_mode))}, "
            f"move {_enemy_timing_label(production_enemy_movement_mode(current_mode))}"
        ),
        "",
    ]
    for mode in SUPPORTED_PRODUCTION_ENEMY_AI_MODES:
        marker = "*" if mode == current_mode else "-"
        lines.append(
            f"{marker} {_enemy_ai_mode_label(mode)}: "
            f"{PRODUCTION_ENEMY_AI_MODE_DESCRIPTIONS[mode]}"
        )
    return "\n".join(lines)


def _enemy_timing_label(mode: str) -> str:
    return mode.replace("_", " ").title()


def _enemy_ai_controls_text(*, ai_mode: str) -> str:
    return _enemy_ai_mode_text(ai_mode)


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

    def _show_main(self, message: str = "") -> None:
        company = self.controller.company
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
        self._show_screen("main", "Charter Desk", body, actions, message=message)

    def _show_company(self, message: str = "") -> None:
        has_company = self.controller.company is not None
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
            company = self.controller.company
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
        self._show_screen("company", "Charter", body, actions, message=message)

    def _show_save_load(self, message: str = "") -> None:
        has_company = self.controller.company is not None
        slot = "present" if self.save_path.exists() else "empty"
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
        body = f"Single save slot\n{self.save_path}\n\nSlot state: {slot}"
        self._show_screen("save_load", "Save / Load", body, actions, message=message)

    def _show_help(self, *, return_to: str = "system") -> None:
        self.pending_help_return_state = return_to
        body = (
            "Textual controls\n"
            "Up/Down changes the focused command.\n"
            "Enter activates the focused command.\n"
            "Number keys activate visible commands.\n"
            "Single-key hotkeys activate visible commands.\n"
            "Esc or Backspace backs out where a Back or Cancel command is available.\n\n"
            "The legacy Rich CLI remains available with --cli."
        )
        self._show_screen(
            "help",
            "Help",
            body,
            (ScreenAction("1", "Back", "back", ("b",), default=True),),
        )

    def _show_current_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        company = self.controller.company
        if company is None:
            self._show_main(message or "Start or load a company first.")
            return
        if self.controller.manual_combat is not None:
            if self._show_opening_enemy_response_if_needed():
                return
            self._show_combat_command(message)
            return
        if self._breach_pending():
            self._show_breach(message)
            return
        if company.active_expedition is not None:
            self._show_dungeon(message, hci)
            return
        result = self.controller.handle(ViewRegionalMap())
        if not result.success:
            self._show_main(result.error or "Regional map unavailable.")
            return
        view = cast(RegionalMapView, result.value)
        if company.town_state.get("location_id") == "haven":
            self._show_town(message, hci)
            return
        self._show_regional_place(message, hci, view=view)

    def _show_regional_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
        return_to: str | None = None,
    ) -> None:
        if return_to is not None:
            self.pending_regional_return_state = return_to
        cached_arrival = (
            view.arrival_context
            if view is not None
            else (
                self.current_regional_view.arrival_context
                if self.current_regional_view is not None
                else None
            )
        )
        if view is None:
            result = self.controller.handle(ViewRegionalMap())
            if not result.success:
                self._show_main(result.error or "Regional map unavailable.")
                return
            view = cast(RegionalMapView, result.value)
            if cached_arrival is not None and view.arrival_context is None:
                view = replace(view, arrival_context=cached_arrival)
        self.current_regional_view = view
        render_view = build_regional_render_view(view)
        title = view.place_title or view.current_node_name
        actions = ActionProvider.regional_place_actions(view)
        self._show_screen(
            "regional_place",
            title,
            self._regional_place_text(view),
            actions,
            message=message,
            log=self._regional_log_text(render_view, hci, actions=actions),
        )

    def _show_regional_map(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
    ) -> None:
        if view is None:
            if self.current_regional_view is not None:
                view = self.current_regional_view
            else:
                result = self.controller.handle(ViewRegionalMap())
                if not result.success:
                    self._show_main(result.error or "Regional map unavailable.")
                    return
                view = cast(RegionalMapView, result.value)
        self.current_regional_view = view
        render_view = build_regional_render_view(view)
        navigation_actions = self._regional_navigation_actions(view)
        map_actions = self._regional_map_display_actions(navigation_actions)
        self._show_screen(
            "regional_map",
            "Company Roadbook",
            self._regional_map_text(
                view,
                render_view,
                actions=navigation_actions,
                map_actions=map_actions,
            ),
            (
                *navigation_actions,
                ScreenAction(
                    str(len(navigation_actions) + 1),
                    "Fold Roadbook",
                    "back",
                    ("b",),
                    default=True,
                ),
            ),
            message=message
            or (
                "The company roadbook is unrolled. Cleared ground becomes lines, "
                "costs, and destinations."
            ),
            log=self._regional_log_text(render_view, hci, actions=map_actions),
        )

    def _show_world_map(
        self,
        message: str = "",
        *,
        return_to: str = "current_place",
    ) -> None:
        self.pending_world_map_return_state = return_to
        result = self.controller.handle(ViewWorld())
        if not result.success:
            self._show_current_place(result.error or "Map is unavailable.")
            return
        view = cast(WorldView, result.value)
        self._show_screen(
            "world_map",
            "World Map",
            self._world_map_text(view),
            (
                ScreenAction(
                    "1",
                    "Back",
                    "back",
                    ("b",),
                    default=True,
                    kind=ScreenActionKind.NAVIGATE,
                ),
            ),
            message=message,
            log=self._events_text(self.recent_events),
        )

    def _show_system(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        has_company = self.controller.company is not None
        current_ai_label = _enemy_ai_mode_label(self.controller.enemy_ai_mode)
        next_ai_mode = _next_enemy_ai_mode(self.controller.enemy_ai_mode)
        enemy_controls_text = _enemy_ai_controls_text(ai_mode=self.controller.enemy_ai_mode)
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
            company = self.controller.company
            assert company is not None
            body = (
                "System\n\n"
                f"Company: {company.name}\n"
                f"Location: {company.town_state.get('location', 'Haven Town')}\n"
                f"Save slot: {self.save_path}\n\n"
                f"{enemy_controls_text}"
            )
        else:
            body = (
                f"System\n\nNo company is loaded.\nSave slot: {self.save_path}\n\n"
                f"{enemy_controls_text}"
            )
        self._show_screen(
            "system",
            "System",
            body,
            actions,
            message=message,
            log=self._result_log_text(self.recent_events, hci)
            if hci is not None
            else self._events_text(self.recent_events),
        )

    def _show_town(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.controller.handle(ViewTown())
        if not result.success:
            self._show_main(result.error or "Start or load a company first.")
            return
        self._record_events(result.events)
        view = cast(TownDashboardView, result.value)
        self._render_town_view(view, message, hci)

    def _render_town_view(
        self,
        view: TownDashboardView,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        arrival_brief: ArrivalBriefView | None = None,
    ) -> None:
        body = "World > Haven\n\n" + self._town_hub_body(view)
        if arrival_brief is not None:
            body = self._arrival_brief_text(arrival_brief) + "\n\n" + body
        self._show_screen(
            "town",
            "Haven",
            body,
            self._town_hub_actions(),
            message=message,
            log=self._result_log_text(self.recent_events, hci)
            if hci is not None
            else self._events_text(self.recent_events),
        )

    def _show_arrival_brief(
        self,
        view: ArrivalBriefView,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        if view.location_id == "haven":
            world_result = self.controller.handle(ViewRegionalMap())
            if not world_result.success:
                self._show_current_place(world_result.error or "Arrival view unavailable.")
                return
            regional_view = cast(RegionalMapView, world_result.value)
            self._show_regional_place(
                message,
                hci,
                view=replace(regional_view, arrival_context=view),
            )
            return
        world_result = self.controller.handle(ViewRegionalMap())
        if not world_result.success:
            self._show_current_place(world_result.error or "Arrival view unavailable.")
            return
        regional_view = cast(RegionalMapView, world_result.value)
        self._show_regional_place(
            message,
            hci,
            view=replace(regional_view, arrival_context=view),
        )

    def _show_town_submenu(
        self,
        submenu: str,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.controller.handle(ViewTown())
        if not result.success:
            self._show_main(result.error or "Start or load a company first.")
            return
        self._record_events(result.events)
        view = cast(TownDashboardView, result.value)
        titles = {
            "town_gate": "East Gate",
            "town_charter": "Charter Office",
            "town_market": "Market Row",
            "town_recovery": "Recovery Ward",
            "town_quartermaster": "Quartermaster",
            "town_recruitment": "Recruitment Desk",
            "town_yard": "Formation Yard",
            "town_upgrades": "Charter Upgrades",
            "town_records": "Records Room",
        }
        if submenu == "town_gate":
            detail_text = self._town_gate_text(view)
        elif submenu == "town_charter":
            detail_text = self._contract_board_text(view)
        elif submenu == "town_market":
            detail_text = self._town_market_text(view)
        elif submenu == "town_upgrades":
            detail_text = self._upgrade_board_text(view)
        elif submenu == "town_recovery":
            detail_text = self._town_recovery_text(view)
        elif submenu == "town_quartermaster":
            detail_text = self._town_quartermaster_text(view)
        elif submenu == "town_recruitment":
            detail_text = self._town_recruitment_text(view)
        elif submenu == "town_yard":
            detail_text = self._town_yard_body(view)
        elif submenu == "town_records":
            detail_text = self._town_records_text(view)
        else:
            detail_text = self._town_hub_body(view)
        body = f"World > Haven > {titles[submenu]}\n\n" + detail_text
        self._show_screen(
            submenu,
            titles[submenu],
            body,
            self._town_submenu_actions(view, submenu),
            message=message,
            log=self._result_log_text(self.recent_events, hci)
            if hci is not None
            else self._events_text(self.recent_events),
        )

    def _town_hub_actions(self) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        company = self.controller.company
        route_charted = (
            company is not None and "shallow_cave" in company.known_route_ids
        )
        if route_charted:
            actions.append(
                ScreenAction(
                    "1",
                    "Take Charted Road to Shallow Cave",
                    "shallow_cave",
                    ("c", "cave", "travel"),
                    default=True,
                    description="Fast travel along the charted road to the cave mouth.",
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    cost="1 ration when available",
                    preview="Follow the charted road to Shallow Cave.",
                    result_hint="Arrives at the cave mouth, skipping cleared Old Road beats.",
                )
            )
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "East Gate",
                "east_gate",
                ("l", "travel"),
                default=not route_charted,
                description="Roads and regional travel.",
                kind=ScreenActionKind.TRAVEL,
                risk=ScreenActionRisk.LOW,
                preview="Step out to East Gate and survey the Haven-Cave route.",
                result_hint="Opens the regional staging ground at East Gate.",
            )
        )
        for label, value, aliases, description, kind, preview, result_hint, risk in (
            (
                "Charter Office",
                "town_charter",
                ("c", "charter", "contracts"),
                "Contracts, upgrades, and records.",
                ScreenActionKind.TOWN,
                "Review postings, filed records, and company infrastructure.",
                "Accepting contracts, records, and upgrades happen inside.",
                None,
            ),
            (
                "Company Yard",
                "town_yard",
                ("f", "formation", "party", "yard"),
                "Formation and roster.",
                ScreenActionKind.TOWN,
                "Assign the active party and inspect heroes.",
                "No resources are spent from the yard.",
                None,
            ),
            (
                "Market Row",
                "town_market",
                ("m", "market", "supplies", "recruit"),
                "Quartermaster, recruitment, and armory.",
                ScreenActionKind.TOWN,
                "Buy supplies, review recruits, or visit the armory.",
                "Spending happens at individual counters.",
                ScreenActionRisk.COSTLY,
            ),
            (
                "Recovery Ward",
                "town_recovery",
                ("w", "ward", "recover"),
                "Treatment and memorial.",
                ScreenActionKind.TOWN,
                "Fund recovery or visit the memorial wall.",
                "Recovery restores HP and Effort; Mortal Wounds remain.",
                ScreenActionRisk.COSTLY,
            ),
            (
                "System",
                "system",
                ("s",),
                "Save, load, help, or quit.",
                ScreenActionKind.SYSTEM,
                "Manage save/load and application commands.",
                "",
                None,
            ),
        ):
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    label,
                    value,
                    aliases,
                    description=description,
                    kind=kind,
                    risk=risk or ScreenActionRisk.LOW,
                    preview=preview,
                    result_hint=result_hint or "",
                )
            )
        return tuple(actions)

    def _town_submenu_actions(
        self,
        view: TownDashboardView,
        submenu: str,
    ) -> tuple[ScreenAction, ...]:
        values = {
            "town_gate": ("travel", "map"),
            "town_charter": (),
            "town_market": ("buy", "recruit"),
            "town_recovery": ("recover", "deep_surgery"),
            "town_quartermaster": ("buy",),
            "town_recruitment": ("recruit",),
            "town_yard": ("formation", "roster"),
            "town_upgrades": (),
            "town_records": ("ledger", "memorial"),
        }[submenu]
        if submenu == "town_gate":
            return (
                ScreenAction(
                    "1",
                    "East Gate",
                    "regional_map",
                    ("l", "travel", "map"),
                    default=True,
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    preview="Go to the Haven East Gate staging ground.",
                    result_hint="Opens the local gate view; Open Roadbook handles charted roads.",
                ),
                ScreenAction(
                    "2",
                    "Map Board",
                    "map",
                    ("m",),
                    kind=ScreenActionKind.INSPECT,
                    preview="Review known places and charted approaches.",
                ),
                ScreenAction("3", "Back to Haven", "back", ("b",), default=False),
            )
        if submenu == "town_charter":
            contract_actions = tuple(
                action
                for action in ActionProvider.contract_board_actions(view.contract_board)
                if action.value != "back"
            )
            actions = (
                *contract_actions,
                ScreenAction(
                    str(len(contract_actions) + 1),
                    "Relic Clerk",
                    "relic_broker",
                    ("relic", "broker"),
                    kind=ScreenActionKind.TOWN,
                    preview="Sell Maze salvage or file charter proof for new road work.",
                    result_hint="Consumes inventory loot and may post new contracts.",
                ),
                ScreenAction(
                    str(len(contract_actions) + 2),
                    "Upgrade Desk",
                    "town_upgrades",
                    ("u", "upgrades"),
                    kind=ScreenActionKind.TOWN,
                    risk=ScreenActionRisk.COSTLY,
                    preview="Review permanent company improvements.",
                ),
                ScreenAction(
                    str(len(contract_actions) + 3),
                    "Records Room",
                    "town_records",
                    ("r", "records"),
                    kind=ScreenActionKind.INSPECT,
                    preview="Review ledger, filed company records, and memorial names.",
                ),
                ScreenAction(
                    str(len(contract_actions) + 4),
                    "Back to Haven",
                    "back",
                    ("b", "back"),
                    kind=ScreenActionKind.NAVIGATE,
                ),
            )
            return tuple(
                self._renumbered_action(action, index, default=False)
                for index, action in enumerate(actions, start=1)
            )
        if submenu == "town_upgrades":
            upgrade_actions = tuple(
                action
                for action in view.upgrade_actions
                if action.value != "back" or action.label.lower().startswith("back")
            )
            return tuple(
                self._renumbered_action(
                    action,
                    index,
                    default=False,
                    label="Back to Charter Office" if action.value == "back" else None,
                )
                for index, action in enumerate(upgrade_actions, start=1)
            )
        actions = tuple(
            self._renumbered_action(
                action,
                index,
                default=index == 1 and action.enabled and self._is_safe_default(action),
            )
            for index, action in enumerate(
                (self._town_service_action(view, value) for value in values),
                start=1,
            )
        )
        if submenu == "town_market":
            actions = (
                *actions,
                ScreenAction(
                    str(len(actions) + 1),
                    "Armory",
                    "gear",
                    ("i", "gear", "inventory", "armory"),
                    kind=ScreenActionKind.TOWN,
                    risk=ScreenActionRisk.COSTLY,
                    preview="Buy kits and review company equipment.",
                ),
            )
        if submenu == "town_records" and self.controller.company is not None:
            if self.controller.company.last_expedition_report is not None:
                actions = (
                    *actions,
                    ScreenAction(
                        str(len(actions) + 1),
                        "Latest Filed Record",
                        "latest_record",
                        ("record", "report"),
                        kind=ScreenActionKind.INSPECT,
                        preview="Open the latest filed company record.",
                    ),
                )
        return (
            *actions,
            ScreenAction(
                str(len(actions) + 1),
                self._town_back_label(submenu),
                "back",
                ("b",),
                default=False,
            ),
        )

    def _town_service_action(self, view: TownDashboardView, value: str) -> ScreenAction:
        return next(action for action in view.services if action.value == value)

    def _town_back_label(self, submenu: str) -> str:
        return {
            "town_quartermaster": "Back to Market Row",
            "town_recruitment": "Back to Market Row",
            "town_upgrades": "Back to Charter Office",
            "town_records": "Back to Charter Office",
        }.get(submenu, "Back to Haven")

    def _brief_text(self, title: str, sections: Sequence[tuple[str, Sequence[str]]]) -> str:
        lines = [title]
        for heading, entries in sections:
            clean_entries = [entry for entry in entries if entry]
            if not clean_entries:
                continue
            lines.extend(("", heading, *clean_entries))
        return "\n".join(lines)

    def _reset_ui_session(self) -> None:
        self.visited_screens.clear()
        self.current_gear_view = None
        self.current_supply_shop_view = None

    def _first_visit_hint(self, screen_id: str, text: str) -> str:
        if screen_id in self.visited_screens:
            return ""
        return text

    def _town_hub_body(self, view: TownDashboardView) -> str:
        hero_lines = "\n".join(
            f"- {format_compact_roster_row(hero)}" for hero in view.active_party
        ) or "none"
        reserve_lines = "\n".join(
            f"- {format_compact_roster_row(hero)}" for hero in view.reserves
        ) or "none"
        body = TownDashboardPanel.render_text(
            view,
            hero_lines=hero_lines,
            reserve_lines=reserve_lines,
        )
        hint = self._first_visit_hint(
            "town",
            "Districts: East Gate, Charter Office, Company Yard, Market Row, Recovery Ward.",
        )
        if hint:
            return f"{body}\n\n{hint}"
        return body

    def _town_yard_body(self, view: TownDashboardView) -> str:
        formation_line = ""
        company = self.controller.company
        if company is not None:
            formation = build_formation_view(company, self.controller.definitions)
            slots = [
                (slot.slot_label, slot.hero_name)
                for slot in formation.slots
                if slot.hero_name != "empty"
            ]
            if slots:
                formation_line = format_formation_lane_summary(
                    slots,
                    slot_order=("FRONT_LEFT", "FRONT_RIGHT", "BACK_LEFT", "BACK_RIGHT"),
                )
        return YardPanel.render_text(
            view,
            formation_text=formation_line,
            hint=self._first_visit_hint(
                "town_yard",
                "Focus Formation or Roster to manage the active party.",
            ),
        )

    def _town_gate_text(self, view: TownDashboardView) -> str:
        return self._brief_text(
            "East Gate",
            (
                ("Purpose", ("Go to the East Gate for local routes and charted roads.",)),
                (
                    "Status",
                    (f"- Current objective: {view.objective.title} - {view.objective.next_step}",),
                ),
                (
                    "Available",
                    (
                        "- East Gate opens the Haven edge for local movement.",
                        "- Open Roadbook zooms out to charted travel.",
                        "- Map Board reviews known places and charted approaches.",
                    ),
                ),
                ("Next", ("Pick East Gate for movement or Map Board for orientation.",)),
            ),
        )

    def _town_market_text(self, view: TownDashboardView) -> str:
        buy = self._town_service_action(view, "buy")
        recruit = self._town_service_action(view, "recruit")
        recruit_status = recruit.unavailable_reason if not recruit.enabled else "offers available"
        return self._brief_text(
            "Market Row",
            (
                ("Purpose", ("Buy route supplies, review recruits, or purchase kits.",)),
                ("Status", (f"- Coin: {view.coin}", f"- Recruitment: {recruit_status}")),
                (
                    "Counters",
                    (
                        "- Quartermaster: "
                        f"{buy.preview or buy.result_hint or 'Buy route supplies.'}",
                        f"- Recruitment Desk: {recruit.cost or recruit.description}",
                        "- Armory: buy kits and review company equipment.",
                    ),
                ),
                ("Next", ("Choose a counter; spending happens one step deeper.",)),
            ),
        )

    def _town_recovery_text(self, view: TownDashboardView) -> str:
        recover = self._town_service_action(view, "recover")
        surgery = self._town_service_action(view, "deep_surgery")
        status = [
            f"- Wounded: {view.wounded_count}",
            f"- Downed: {view.downed_count}",
            f"- Memorial: {view.deceased_count}",
        ]
        if not recover.enabled and recover.unavailable_reason:
            status.append(f"- Recovery unavailable: {recover.unavailable_reason}")
        if not surgery.enabled and surgery.unavailable_reason:
            status.append(f"- Deep Surgery unavailable: {surgery.unavailable_reason}")
        return self._brief_text(
            "Recovery Ward",
            (
                ("Purpose", ("Restore HP and Effort for living company members.",)),
                ("Status", tuple(status)),
                (
                    "Treatment",
                    (
                        f"- Recovery cost: {recover.cost or recover.description or 'free'}",
                        "- Mortal Wounds remain after recovery.",
                        f"- Deep Surgery cost: {surgery.cost or surgery.description or 'free'}",
                        "- Deep Surgery removes one Mortal Wound; hero sits out next expedition.",
                    ),
                ),
                ("Next", ("Fund recovery, schedule deep surgery, or open the memorial wall.",)),
            ),
        )

    def _deep_surgery_text(self, view: DeepSurgeryView) -> str:
        if not view.candidates:
            return self._brief_text(
                "Deep Surgery",
                (
                    ("Purpose", ("Remove one Mortal Wound from a living hero.",)),
                    ("Status", ("- No eligible heroes need deep surgery.",)),
                    ("Next", ("Return to the Recovery Ward.",)),
                ),
            )
        candidate_lines = tuple(
            f"- {candidate.name}: {candidate.mortal_wounds} Mortal Wound(s)"
            for candidate in view.candidates
        )
        return self._brief_text(
            "Deep Surgery",
            (
                ("Purpose", ("Remove one Mortal Wound. Hero is In Surgery until next return.",)),
                (
                    "Cost",
                    (
                        f"- {view.surgery_cost} Coin per treatment",
                        f"- Coin on hand: {view.coin}",
                    ),
                ),
                ("Candidates", candidate_lines),
                ("Next", ("Choose a hero to treat.",)),
            ),
        )

    def _town_quartermaster_text(self, view: TownDashboardView) -> str:
        buy = self._town_service_action(view, "buy")
        return self._brief_text(
            "Quartermaster",
            (
                ("Purpose", ("Turn Coin into supplies used on roads and in dangerous places.",)),
                ("Status", (f"- Coin: {view.coin}",)),
                ("Stock", (f"- {buy.result_hint or buy.preview or 'Buy route supplies.'}",)),
                ("Next", ("Open Buy Supplies to inspect stock and affordability.",)),
            ),
        )

    def _town_recruitment_text(self, view: TownDashboardView) -> str:
        recruit = self._town_service_action(view, "recruit")
        status = [
            f"- Roster: {view.active_count + view.reserve_count}/{view.roster_cap}",
            f"- Coin: {view.coin}",
            f"- Hiring cost: {recruit.cost or recruit.description}",
        ]
        if not recruit.enabled and recruit.unavailable_reason:
            status.append(f"- Unavailable: {recruit.unavailable_reason}")
        return self._brief_text(
            "Recruitment Desk",
            (
                ("Purpose", ("Review and hire available heroes for the company roster.",)),
                ("Status", tuple(status)),
                ("Available", ("Recruit offers refresh from this desk.",)),
                ("Next", ("Open Recruit to see names, classes, backgrounds, and costs.",)),
            ),
        )

    def _town_records_text(self, view: TownDashboardView) -> str:
        company = self.controller.company
        report_count = len(company.expedition_reports) if company is not None else 0
        latest = (
            "- Latest filed record available."
            if company is not None and company.last_expedition_report is not None
            else "- No company record filed yet."
        )
        return self._brief_text(
            "Records Room",
            (
                ("Purpose", ("Review company ledger, filed records, and memorial names.",)),
                (
                    "Status",
                    (
                        f"- Filed records: {report_count}",
                        f"- Memorial names: {view.deceased_count}",
                        f"- Current objective: {view.objective.title}",
                    ),
                ),
                ("Records", (latest, "- Ledger and memorial are available from this room.")),
                ("Next", ("Open the ledger, memorial, or latest filed record.",)),
            ),
        )

    def _contract_board_text(self, view: TownDashboardView) -> str:
        lines = [
            "Charter Office",
            "",
            "Purpose",
            "Turn dangerous places into posted work and paid proof.",
        ]
        active_contract_count = sum(
            1 for entry in view.contract_board if entry.state == "active"
        )
        lines.extend(
            (
                "",
                "Status",
                f"- Reputation: {view.reputation}",
                f"- Active contracts: {active_contract_count}",
                "",
                "Contract Board",
            )
        )
        if not view.contract_board:
            return "\n".join(
                (
                    "Charter Office",
                    "",
                    "Purpose",
                    "Turn dangerous places into posted work and paid proof.",
                    "",
                    "Status",
                    f"- Reputation: {view.reputation}",
                    "",
                    "Contract Board",
                    "No breach contracts are posted yet.",
                    "",
                    "Next",
                    "Finish the current charter to draw new Haven work.",
                )
            )
        entries_by_state = {
            state: [entry for entry in view.contract_board if entry.state == state]
            for state in ("available", "active")
        }
        for state, title in (
            ("available", "Available"),
            ("active", "Active"),
        ):
            entries = entries_by_state[state]
            if not entries:
                continue
            lines.extend(("", title))
            for entry in entries:
                reward = self._contract_reward_summary(entry)
                lines.append(f"  {entry.name:<32} D{entry.difficulty}   {reward}")
        lines.extend(("", "Next", "Choose a contract, visit upgrades, or open records."))
        return "\n".join(lines)

    def _contract_reward_summary(self, entry: Any) -> str:
        pieces: list[str] = []
        if getattr(entry, "reward_reputation", 0):
            pieces.append(f"+{entry.reward_reputation} rep")
        if getattr(entry, "coin_reward", 0):
            pieces.append(f"+{entry.coin_reward} Coin")
        return ", ".join(pieces) or "no payout"

    def _upgrade_board_text(self, view: TownDashboardView) -> str:
        lines = [
            "Charter Upgrades",
            "",
            "Purpose",
            "Install permanent company improvements.",
            "",
            "Status",
            f"- Reputation: {view.reputation}",
            f"- Coin: {view.coin}",
            "",
            "Upgrades",
        ]
        if not view.upgrades:
            return "\n".join(
                (
                    "Charter Upgrades",
                    "",
                    "Purpose",
                    "Install permanent company improvements.",
                    "",
                    "Available",
                    "No company upgrades are authored yet.",
                )
            )
        entries_by_state = {
            state: [entry for entry in view.upgrades if entry.state == state]
            for state in ("available", "unavailable", "locked", "installed")
        }
        for state, title in (
            ("available", "Available"),
            ("unavailable", "Unavailable"),
            ("locked", "Locked"),
            ("installed", "Installed"),
        ):
            entries = entries_by_state[state]
            if not entries:
                continue
            lines.extend(("", title))
            for entry in entries:
                lines.append(f"  {entry.name:<28} cost {entry.cost}")
                if entry.effect_summary:
                    lines.append(f"    {entry.effect_summary}")
                if entry.unavailable_reason and entry.state != "installed":
                    lines.append(f"    {entry.unavailable_reason}")
        return "\n".join(lines)

    def _show_roster(self, message: str = "") -> None:
        result = self.controller.handle(ViewRoster())
        if not result.success:
            self._show_main(result.error or "Start or load a company first.")
            return
        sections = cast(tuple[RosterSectionView, ...], result.value)
        hero_actions = self._roster_hero_actions(sections)
        actions = hero_actions + (
            ScreenAction(
                str(len(hero_actions) + 1),
                "Back to Company Yard",
                "back",
                ("b",),
                default=not hero_actions,
                kind=ScreenActionKind.NAVIGATE,
            ),
        )
        self._show_screen(
            "roster",
            "Roster",
            self._roster_text(sections),
            actions,
            message=message,
        )

    def _roster_hero_actions(
        self,
        sections: tuple[RosterSectionView, ...],
    ) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        for section in sections:
            if section.title.lower().startswith("memorial"):
                continue
            for hero in section.heroes:
                hero_id = str(getattr(hero, "hero_id", ""))
                if not hero_id:
                    continue
                actions.append(
                    ScreenAction(
                        str(len(actions) + 1),
                        str(hero.name),
                        f"hero:{hero_id}",
                        (hero_id, str(hero.name).lower().replace(" ", "_")),
                        default=len(actions) == 0,
                        description=self._hero_gear_summary(hero) or "No kit equipped",
                        kind=ScreenActionKind.INSPECT,
                        preview="Open this hero's character sheet.",
                    )
                )
        return tuple(actions)

    def _show_supplies(self, message: str = "") -> None:
        result = self.controller.handle(ViewSupplies())
        if not result.success:
            self._show_main(result.error or "Start or load a company first.")
            return
        supplies = cast(dict[str, int], result.value)
        lines = ["Current supplies"]
        for supply_id, quantity in sorted(supplies.items()):
            lines.append(f"{supply_id}: {quantity}")
        self._show_screen(
            "supplies",
            "Supplies",
            "\n".join(lines),
            (ScreenAction("1", "Back", "back", ("b",), default=True),),
            message=message,
        )

    def _show_pack(self, message: str = "") -> None:
        supplies_result = self.controller.handle(ViewSupplies())
        gear_result = self.controller.handle(ViewGear())
        if not supplies_result.success or not gear_result.success:
            self._show_current_place(
                supplies_result.error or gear_result.error or "Company inventory is unavailable."
            )
            return
        self._record_events([*supplies_result.events, *gear_result.events])
        supplies = cast(dict[str, int], supplies_result.value)
        gear = cast(GearInventoryView, gear_result.value)
        actions = self._pack_actions(gear)
        self._show_screen(
            "pack",
            "Pack",
            self._pack_text(supplies, gear),
            actions,
            message=message,
            log=self._events_text(self.recent_events),
        )

    def _pack_actions(
        self,
        gear: GearInventoryView,
    ) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = [
            ScreenAction(
                "1",
                "Armory",
                "gear",
                ("i", "gear", "inventory"),
                kind=ScreenActionKind.INSPECT,
                preview="Inspect company kits; purchases are only available in Haven.",
            )
        ]
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back to Current Place",
                "back",
                ("b", "back"),
                default=not gear.heroes,
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    def _pack_text(
        self,
        supplies: dict[str, int],
        gear: GearInventoryView,
    ) -> str:
        company = self.controller.company
        items = company.inventory if company is not None else {}
        return PackPanel.render_text(
            supplies,
            items,
            gear,
            hint=self._first_visit_hint(
                "pack",
                "Focus Armory to inspect kits. Use Company for hero sheets.",
            ),
        )

    def _show_company_summary(self, message: str = "") -> None:
        company = self.controller.company
        if company is None:
            self._show_main("Start or load a company first.")
            return
        formation = build_formation_view(company, self.controller.definitions)
        roster_result = self.controller.handle(ViewRoster())
        town_result = self.controller.handle(ViewTown())
        sections = (
            cast(tuple[RosterSectionView, ...], roster_result.value)
            if roster_result.success
            else ()
        )
        town_view = (
            cast(TownDashboardView, town_result.value) if town_result.success else None
        )
        objective_line = (
            f"{town_view.objective.title}: {town_view.objective.next_step}"
            if town_view is not None
            else "No active objective."
        )
        roster_lines = [
            f"- {format_compact_roster_row(hero)}"
            for section in sections
            if "memorial" not in section.title.lower()
            for hero in section.heroes
        ]
        self._current_formation_view = formation
        self._company_summary_objective = objective_line
        self._company_summary_roster_lines = roster_lines
        body = self._company_summary_body(company.name)
        self._show_screen(
            "company_summary",
            "Company",
            body,
            self._company_summary_actions(sections),
            message=message,
            log=self._events_text(self.recent_events),
        )

    def _company_summary_actions(
        self,
        sections: tuple[RosterSectionView, ...],
    ) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = [
            ScreenAction(
                "1",
                "Formation",
                "formation",
                ("f", "formation"),
                default=True,
                kind=ScreenActionKind.TOWN,
                preview="Change active party slots.",
            ),
            ScreenAction(
                "2",
                "Armory",
                "gear",
                ("i", "gear", "inventory"),
                kind=ScreenActionKind.TOWN,
                preview="Inspect kits and buy only when the company is in Haven.",
            ),
        ]
        for section in sections:
            if section.title.lower().startswith("memorial"):
                continue
            for hero in section.heroes:
                actions.append(
                    ScreenAction(
                        str(len(actions) + 1),
                        hero.name,
                        f"hero:{hero.hero_id}",
                        (hero.hero_id, hero.name.lower().replace(" ", "_")),
                        kind=ScreenActionKind.INSPECT,
                        description=self._hero_gear_summary(hero) or "Gear: none",
                        preview="Open this hero's character sheet.",
                    )
                )
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back to Current Place",
                "back",
                ("b", "back"),
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    def _show_gear_locker(self, message: str = "", *, return_to: str = "town_market") -> None:
        self.pending_gear_locker_return_state = return_to
        result = self.controller.handle(ViewGear())
        if not result.success:
            self._show_main(result.error or "Start or load a company first.")
            return
        self._record_events(result.events)
        view = cast(GearInventoryView, result.value)
        self.current_gear_view = view
        self._show_screen(
            "gear",
            "Armory",
            self._gear_locker_text(view),
            self._gear_locker_actions(view.actions),
            message=message,
        )

    def _gear_locker_actions(
        self,
        actions: tuple[ScreenAction, ...],
    ) -> tuple[ScreenAction, ...]:
        back_label = {
            "company": "Back to Company",
            "pack": "Back to Pack",
            "main": "Back to Charter Desk",
        }.get(self.pending_gear_locker_return_state, "Back to Market Row")
        return tuple(
            self._renumbered_action(
                action,
                index,
                label=back_label if action.value == "back" else None,
            )
            for index, action in enumerate(actions, start=1)
        )

    def _show_hero_sheet(
        self,
        hero_id: str = "",
        message: str = "",
        *,
        return_to: str = "roster",
    ) -> None:
        selected_id = hero_id or self.pending_gear_hero_id
        result = self.controller.handle(ViewHeroSheet(selected_id))
        if not result.success:
            if return_to == "pack":
                self._show_pack(result.error or "Hero sheet unavailable.")
            elif return_to == "company":
                self._show_company_summary(result.error or "Hero sheet unavailable.")
            else:
                self._show_roster(result.error or "Hero sheet unavailable.")
            return
        self._record_events(result.events)
        view = cast(HeroSheetView, result.value)
        if not view.hero_id:
            self.pending_gear_hero_id = ""
            if return_to == "pack":
                self._show_pack("Choose a hero before opening a character sheet.")
            elif return_to == "company":
                self._show_company_summary("Choose a hero before opening a character sheet.")
            else:
                self._show_roster("Choose a hero before opening a character sheet.")
            return
        self.pending_gear_hero_id = view.hero_id
        self.pending_gear_return_state = return_to
        self._show_screen(
            "hero_sheet",
            f"{view.name} Sheet",
            self._hero_sheet_text(view),
            self._hero_sheet_actions(view),
            message=message,
        )

    def _show_hero_memories(self, message: str = "") -> None:
        if not self.pending_gear_hero_id:
            if self.pending_gear_return_state == "company":
                self._show_company_summary("Choose a hero before opening memories.")
            else:
                self._show_roster("Choose a hero before opening memories.")
            return
        result = self.controller.handle(ViewHeroSheet(self.pending_gear_hero_id))
        if not result.success:
            if self.pending_gear_return_state == "company":
                self._show_company_summary(result.error or "Hero memories unavailable.")
            else:
                self._show_roster(result.error or "Hero memories unavailable.")
            return
        view = cast(HeroSheetView, result.value)
        self._show_screen(
            "hero_memories",
            f"{view.name} Quirks / Memories",
            self._hero_quirks_memories_text(view),
            (ScreenAction("1", "Back to Sheet", "back", ("b",), default=True),),
            message=message,
        )

    def _show_hero_gear(self, message: str = "") -> None:
        if not self.pending_gear_hero_id:
            self._show_roster("Choose a hero before opening gear.")
            return
        result = self.controller.handle(ViewHeroSheet(self.pending_gear_hero_id))
        if not result.success:
            self._show_roster(result.error or "Hero gear unavailable.")
            return
        view = cast(HeroSheetView, result.value)
        self._show_screen(
            "hero_gear",
            f"{view.name} Gear",
            self._hero_gear_text(view),
            self._hero_gear_actions(view),
            message=message,
        )

    def _hero_sheet_actions(self, view: HeroSheetView) -> tuple[ScreenAction, ...]:
        return (
            ScreenAction(
                "1",
                "Quirks / Memories",
                "memories",
                ("m",),
                kind=ScreenActionKind.INSPECT,
                preview="Review quirks, fresh memories, permanent records, and career patterns.",
            ),
            ScreenAction(
                "2",
                "Gear",
                "gear",
                ("g", "i"),
                kind=ScreenActionKind.TOWN,
                preview="Review equipped kit and choose available company gear.",
            ),
            ScreenAction("3", "Back", "back", ("b",), default=True),
        )

    def _hero_gear_actions(self, view: HeroSheetView) -> tuple[ScreenAction, ...]:
        if self.controller.company is None:
            return (ScreenAction("1", "Back to Sheet", "back", ("b",), default=True),)
        gear_actions = list(
            ActionProvider.hero_gear_actions(
                self.controller.company,
                self.controller.definitions,
                view.hero_id,
                can_manage=view.can_manage_gear,
                manage_reason=view.gear_manage_reason,
            )
        )
        if gear_actions and gear_actions[-1].value == "back":
            gear_actions[-1] = ScreenAction("", "Back to Sheet", "back", ("b",), default=True)
        else:
            gear_actions.append(ScreenAction("", "Back to Sheet", "back", ("b",), default=True))
        return tuple(
            self._renumbered_action(action, index)
            for index, action in enumerate(gear_actions, start=1)
        )

    def _gear_locker_text(self, view: GearInventoryView) -> str:
        return GearLockerPanel.render_text(
            view,
            hint=self._first_visit_hint(
                "gear",
                "Focus a kit purchase for costs and effects. Equip kits from a hero sheet.",
            ),
        )

    def _hero_sheet_text(self, view: HeroSheetView) -> str:
        lines = [
            view.name,
            format_meta_line(view.class_name, view.roster_state, view.slot),
            "",
            "Character Sheet",
            "Choose a section below. Focus previews it in the side pane.",
            "",
            "Identity",
            f"- Class: {view.class_name} ({view.class_id})",
        ]
        if view.background:
            lines.append(f"- Background: {view.background}")
        if view.motive:
            lines.append(f"- Motive: {view.motive}")
        lines.extend(
            (
                "",
                "Vitals",
                f"- HP: {view.hp}/{view.max_hp}",
                f"- Effort: {view.effort}/{view.max_effort}",
                f"- Morale: {view.morale}",
                f"- Strain: {view.strain}",
                f"- Wounds: {mortal_wound_badge(view.mortal_wounds, markup_safe=True)}",
                f"- State: {', '.join(view.statuses)}",
                (
                    f"- Stats: SPD {view.speed}, ACC {view.accuracy}, "
                    f"DEF {view.defense}, DMG {view.damage}"
                ),
            )
        )
        quirk_count = len(view.earned_quirks) + (1 if view.personal_quirk is not None else 0)
        lines.extend(
            (
                "",
                "At a Glance",
                f"- Quirks: {quirk_count or 'none'}",
                f"- Fresh memories: {len(view.fresh_memories)}",
                f"- Permanent records: {len(view.permanent_memories)}",
                f"- Gear: {view.equipped_gear or 'none'}",
            )
        )
        if view.latest_memory:
            lines.append(f"- Latest record: {view.latest_memory}")
        return "\n".join(lines)

    def _hero_gear_text(self, view: HeroSheetView) -> str:
        lines = [
            view.name,
            format_meta_line(view.class_name, view.roster_state),
            "",
            "Gear",
            f"- Equipped: {view.equipped_gear or 'none'}",
        ]
        if view.stat_bonus:
            lines.append(f"- Bonus: {view.stat_bonus}")
        if view.equipped_gear_description:
            lines.append(f"- Note: {view.equipped_gear_description}")
        lines.extend(("", "Available Kits"))
        if not view.available_kits:
            lines.append("- none")
        for item in view.available_kits:
            state = f"available {item.available_count}, owned {item.owned_count}"
            lines.append(f"- {item.name}: {state}")
            if item.effect_summary:
                lines.append(f"  {item.effect_summary}")
            if item.description:
                lines.append(f"  {item.description}")
        if not view.can_manage_gear:
            lines.extend(("", view.gear_manage_reason))
        return "\n".join(lines)

    def _hero_quirks_memories_text(self, view: HeroSheetView) -> str:
        lines = [view.name, format_meta_line(view.class_name, view.roster_state), ""]
        lines.extend(("Quirks",))
        if view.personal_quirk is not None:
            lines.extend(self._sheet_trait_lines(view.personal_quirk, prefix="Personal"))
        if view.earned_quirks:
            for trait in view.earned_quirks:
                lines.extend(self._sheet_trait_lines(trait, prefix="Earned"))
        elif view.personal_quirk is None:
            lines.append("- none")
        if view.strain_marks:
            marks = ", ".join(trait.name for trait in view.strain_marks)
            lines.append(f"- Strain marks: {marks}")
        lines.extend(("", "Active Memory Pressure"))
        if view.fresh_memories:
            for memory in view.fresh_memories:
                lines.extend(self._fresh_memory_lines(memory))
        else:
            lines.append("- none")
        lines.extend(("", "Recent Records"))
        if view.permanent_memories:
            for record in view.permanent_memories[:5]:
                lines.extend(self._permanent_memory_lines(record))
        else:
            lines.append("- none")
        lines.extend(("", "Career Patterns"))
        if view.career_signals:
            lines.extend(self._career_signal_lines(view.career_signals[:6]))
        else:
            lines.append("- none")
        return "\n".join(lines)

    def _sheet_trait_lines(self, trait: HeroSheetTraitView, *, prefix: str) -> list[str]:
        stability = f" [{trait.stability.title()}]" if trait.stability else ""
        lines = [f"- {prefix}: {trait.name}{stability}"]
        effect = trait.positive_text or trait.description
        if effect:
            lines.append(f"  {effect}")
        return lines

    def _fresh_memory_lines(self, memory: HeroSheetFreshMemoryView) -> list[str]:
        meter = "#" * memory.intensity + "-" * max(0, 3 - memory.intensity)
        pending = " ready to manifest" if memory.pending_manifestation else ""
        lines = [f"- {memory.name}: [{meter}]{pending}"]
        if memory.source_summary:
            lines.append(f"  {memory.source_summary}")
        if memory.tags:
            lines.append("  Tags: " + ", ".join(memory.tags))
        return lines

    def _permanent_memory_lines(self, memory: HeroSheetMemoryEntryView) -> list[str]:
        where = format_meta_line(memory.expedition_id, memory.dungeon_id, memory.node_id)
        lines = [f"- {memory.summary}"]
        if where:
            lines.append(f"  {where}")
        return lines

    def _career_signal_lines(self, signals: Sequence[HeroSheetSignalView]) -> list[str]:
        if not signals:
            return ["- none"]
        strongest = signals[0]
        lines = [f"- Strongest pattern: {strongest.label} ({strongest.score})"]
        if len(signals) > 1:
            others = ", ".join(signal.label for signal in signals[1:6])
            lines.append(f"- Also shaped by: {others}")
        return lines

    def _hero_portrait_actor(self, hero_id: str, *, slot: str = "") -> CombatActorView | None:
        company = self.controller.company
        if company is None or not hero_id:
            return None
        hero = next((entry for entry in company.roster if entry.hero_id == hero_id), None)
        if hero is None:
            return None
        return build_hero_portrait_view(
            hero,
            self.controller.definitions,
            slot=slot,
        )

    def _formation_portrait_actors(self, view: FormationView) -> dict[str, CombatActorView]:
        company = self.controller.company
        if company is None:
            return {}
        roster_by_id = {hero.hero_id: hero for hero in company.roster}
        actors: dict[str, CombatActorView] = {}
        for slot in view.slots:
            if slot.hero_id is None or slot.hero_id not in roster_by_id:
                continue
            actors[slot.slot_label] = build_hero_portrait_view(
                roster_by_id[slot.hero_id],
                self.controller.definitions,
                slot=slot.slot_label,
            )
        return actors

    def _formation_board_text(
        self,
        view: FormationView,
        *,
        focus_slot: str = "",
        focus_hero_id: str = "",
    ) -> str:
        return FormationBoard.render_mini_text(
            self._formation_portrait_actors(view),
            focus_slot=focus_slot,
            focus_hero_id=focus_hero_id,
            idle_frame=self.idle_animation_frame,
            inward_facing=False,
        )

    def _company_summary_body(self, company_name: str) -> str:
        formation = self._current_formation_view
        if formation is None:
            return self.body_text
        focus_hero_id = ""
        action = self.focused_action
        if action is not None and action.value.startswith("hero:"):
            focus_hero_id = action.value.removeprefix("hero:")
        formation_text = self._formation_board_text(
            formation,
            focus_hero_id=focus_hero_id,
        )
        return CompanyPanel.render_text(
            company_name,
            self._company_summary_objective,
            formation_text,
            self._company_summary_roster_lines,
            hint=self._first_visit_hint(
                "company_summary",
                "Focus a hero for their sheet, or open Formation or Armory.",
            ),
        )

    def _formation_detail_text(self, action: ScreenAction) -> str:
        view = self._current_formation_view
        if self.screen_state == "formation" and view is not None:
            if action.value == "back":
                return self._generic_action_detail(action)
            slot = next(
                (entry for entry in view.slots if entry.slot_label == action.value),
                None,
            )
            if slot is not None:
                portrait_lines: list[str] = []
                if slot.hero_id is not None:
                    portrait_lines = portrait_detail_lines(
                        self._hero_portrait_actor(slot.hero_id, slot=slot.slot_label),
                        idle_frame=self.idle_animation_frame,
                    )
                lines = ["Formation Slot", ""]
                if portrait_lines:
                    lines.extend((*portrait_lines, ""))
                lines.extend(
                    (
                        format_formation_slot(slot.slot_label),
                        f"Current: {slot.hero_name}",
                    )
                )
                if slot.class_name:
                    lines.append(slot.class_name)
                if slot.vitals_line:
                    lines.append(slot.vitals_line)
                if slot.protection_line:
                    lines.append(slot.protection_line)
                if slot.abnormal_status:
                    lines.append(slot.abnormal_status)
                if action.preview:
                    lines.extend(("", action.preview))
                if action.result_hint:
                    lines.extend(("", "Result", action.result_hint))
                return "\n".join(lines)
        if self.screen_state == "assign_hero":
            if action.value == "back":
                return self._generic_action_detail(action)
            if action.value == EMPTY_SLOT_VALUE:
                lines = ["Empty Slot", "", "Leave this formation slot empty."]
                if action.result_hint:
                    lines.extend(("", "Result", action.result_hint))
                return "\n".join(lines)
            portrait_lines = portrait_detail_lines(
                self._hero_portrait_actor(action.value, slot=self.pending_slot_label),
                idle_frame=self.idle_animation_frame,
            )
            lines = ["Assign Preview", ""]
            if portrait_lines:
                lines.extend((*portrait_lines, ""))
            if action.preview:
                lines.append(action.preview)
            if action.description:
                lines.extend(("", action.description))
            if action.result_hint:
                lines.extend(("", "Result", action.result_hint))
            return "\n".join(lines)
        return self._generic_action_detail(action)

    def _hero_sheet_preview_detail(self, hero_id: str) -> str:
        view = self._hero_sheet_detail_view(hero_id)
        if view is None:
            return "Character sheet unavailable."
        slot_label = (
            format_formation_slot(view.slot)
            if view.roster_state == "Active" and view.slot
            else view.roster_state
        )
        active_slot = view.slot if view.roster_state == "Active" else ""
        portrait_lines = portrait_detail_lines(
            self._hero_portrait_actor(hero_id, slot=active_slot),
            idle_frame=self.idle_animation_frame,
        )
        lines: list[str] = []
        if portrait_lines:
            lines.extend((*portrait_lines, ""))
        lines.extend(
            [
                view.name,
                format_meta_line(view.class_name, view.roster_state, slot_label),
                "",
                "Vitals",
                f"HP {view.hp}/{view.max_hp} | Effort {view.effort}/{view.max_effort}",
                f"Morale {view.morale} | Strain {view.strain}",
                (
                    f"Wounds: {mortal_wound_badge(view.mortal_wounds)}"
                    if view.mortal_wounds
                    else "Wounds: none"
                ),
            ]
        )
        if self.controller.company is not None and view.roster_state == "Active":
            protection = hero_protection_line(self.controller.company, hero_id)
            if protection:
                lines.extend(("", "Combat Role", protection))
        quirk_line = self._hero_sheet_quirk_line(view)
        memory_line = self._hero_sheet_memory_line(view)
        if quirk_line:
            lines.extend(("", quirk_line))
        if memory_line:
            lines.append(memory_line)
        lines.extend(("", "Enter opens Quirks / Memories, Gear, and Back."))
        return "\n".join(lines)

    def _hero_sheet_section_detail(self, action: ScreenAction) -> str:
        if action.value == "back":
            return self._generic_action_detail(action)
        view = self._hero_sheet_detail_view(self.pending_gear_hero_id)
        if view is None:
            return action.preview or self._generic_action_detail(action)
        if action.value == "memories":
            return self._hero_quirks_memories_text(view)
        if action.value == "gear":
            return self._hero_gear_text(view)
        return self._generic_action_detail(action)

    def _hero_sheet_detail_view(self, hero_id: str) -> HeroSheetView | None:
        if not hero_id:
            return None
        result = self.controller.handle(ViewHeroSheet(hero_id))
        if not result.success or not isinstance(result.value, HeroSheetView):
            return None
        return result.value

    def _hero_sheet_quirk_line(self, view: HeroSheetView) -> str:
        if view.personal_quirk is not None:
            return f"Quirk: {view.personal_quirk.name}"
        if view.earned_quirks:
            names = ", ".join(quirk.name for quirk in view.earned_quirks[:2])
            suffix = (
                f" (+{len(view.earned_quirks) - 2} more)"
                if len(view.earned_quirks) > 2
                else ""
            )
            return f"Quirk: {names}{suffix}"
        return ""

    def _hero_sheet_memory_line(self, view: HeroSheetView) -> str:
        fresh_count = len(view.fresh_memories)
        if fresh_count == 0 and not view.latest_memory:
            return ""
        fresh_label = f"{fresh_count} fresh" if fresh_count else "none fresh"
        if view.latest_memory:
            return f"Memory: {fresh_label} | Latest: {view.latest_memory}"
        return f"Memory: {fresh_label}"

    def _show_ledger(self, message: str = "") -> None:
        result = self.controller.handle(ViewLedger())
        if not result.success:
            self._show_main(result.error or "Start or load a company first.")
            return
        self._record_events(result.events)
        ledger = cast(dict[str, object], result.value)
        lines = [
            "Company Ledger",
            "",
            "Purpose",
            "Track the company totals that persist between journeys.",
            "",
            "Records",
        ]
        lines.extend(f"- {key}: {value}" for key, value in ledger.items())
        lines.extend(("", "Next", "Return to Records Room for filed records or memorial names."))
        self._show_screen(
            "ledger",
            "Company Ledger",
            "\n".join(lines),
            (ScreenAction("1", "Back to Records Room", "back", ("b",), default=True),),
            message=message,
        )

    def _show_memorial(self, message: str = "") -> None:
        result = self.controller.handle(ViewMemorial())
        if not result.success:
            self._show_town(result.error or "Start or load a company first.")
            return
        self._record_events(result.events)
        heroes = cast(Sequence[Any], result.value)
        if heroes:
            names = [
                f"- {hero.name} ({hero.class_id})"
                if not getattr(hero, "final_memory", "")
                else f"- {hero.name} ({hero.class_id}) - {hero.final_memory}"
                for hero in heroes
            ]
        else:
            names = ["- No heroes are listed in the memorial."]
        body = self._brief_text(
            "Memorial",
            (
                ("Purpose", ("Keep the names and final memories of fallen company members.",)),
                ("Names", tuple(names)),
                ("Next", ("Return to Records Room when finished.",)),
            ),
        )
        self._show_screen(
            "memorial",
            "Memorial",
            body,
            (ScreenAction("1", "Back to Records Room", "back", ("b",), default=True),),
            message=message,
        )

    def _show_recruiting(self, message: str = "") -> None:
        result = self.controller.handle(GenerateRecruitOffers())
        if not result.success:
            self._show_town_submenu("town_market", result.error or "Recruiting is unavailable.")
            return
        self._record_events(result.events)
        view = cast(RecruitOffersView, result.value)
        self._render_recruiting_view(view, message)

    def _render_recruiting_view(self, view: RecruitOffersView, message: str = "") -> None:
        actions = (
            ScreenAction(
                "1",
                "Hire Recruit",
                "hire",
                ("h",),
                enabled=any(action.enabled for action in view.actions if action.value != "back"),
                default=any(action.enabled for action in view.actions if action.value != "back"),
                description="Choose from the available recruits.",
            ),
            ScreenAction("2", "Back to Market Row", "back", ("b",)),
        )
        self._show_screen(
            "recruiting",
            "Recruiting",
            self._recruiting_text(view),
            actions,
            message=message,
            log=self._events_text(self.recent_events),
        )

    def _show_recruiting_hire(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        view = self._current_recruiting_view()
        if view is None:
            self._show_town_submenu("town_market", "Start or load a company first.")
            return
        actions = self._selection_actions(view.actions, back_label="Back to Recruitment Desk")
        self._show_screen(
            "recruiting_hire",
            "Hire Recruit",
            self._recruiting_text(view),
            actions,
            message=message,
            log=self._result_log_text(self.recent_events, hci)
            if hci is not None
            else self._events_text(self.recent_events),
        )

    def _current_recruiting_view(self) -> RecruitOffersView | None:
        if self.controller.company is None:
            return None
        return build_recruit_offers_view(
            self.controller.company,
            self.controller.definitions,
            self.controller.recruit_offers,
        )

    def _recruiting_text(self, view: RecruitOffersView) -> str:
        can_hire = view.coin >= next((offer.cost for offer in view.offers), 0)
        lines = [
            "Recruitment Desk",
            "",
            "Purpose",
            "Review a short slate of distinct candidates and hire one into the roster.",
            "",
            "Status",
            f"- Reputation: {view.reputation}",
            f"- Coin: {view.coin}",
            f"- Roster: {view.roster_count}/{view.roster_cap}",
            f"- Hiring: {'available' if can_hire else 'needs coin'}",
            "",
            "Candidates",
        ]
        if not view.offers:
            lines.append("- none")
        for index, offer in enumerate(view.offers, start=1):
            affordable = "ready" if view.coin >= offer.cost else "short coin"
            lines.append(f"{index}. {offer.name} - {offer.class_name} ({affordable})")
            if offer.background:
                lines.append(f"   {offer.background}")
            if offer.motive:
                lines.append(f"   Wants: {offer.motive}")
        lines.extend(("", "Next", "Open Hire Recruit, then focus a candidate for details."))
        return "\n".join(lines)

    def _recruit_offer_detail(self, action: ScreenAction) -> str:
        if action.value == "back":
            return self._generic_action_detail(action)
        view = self._current_recruiting_view()
        if view is None:
            return self._generic_action_detail(action)
        try:
            index = int(action.value)
        except ValueError:
            return self._generic_action_detail(action)
        if index < 0 or index >= len(view.offers):
            return self._generic_action_detail(action)
        offer = view.offers[index]
        lines = [
            "Candidate",
            "",
            offer.name,
            format_meta_line(offer.class_name, f"Cost {offer.cost} Coin"),
            "",
            "Background",
            offer.background or "unknown",
            "",
            "Motive",
            offer.motive or "not recorded",
            "",
            "Roster Fit",
            self._recruit_fit_line(view, offer),
        ]
        if action.unavailable_reason:
            lines.extend(("", "Unavailable", action.unavailable_reason))
        elif action.result_hint:
            lines.extend(("", "Hiring Result", action.result_hint))
        return "\n".join(lines)

    def _recruit_fit_line(self, view: RecruitOffersView, offer: RecruitOfferView) -> str:
        class_count = sum(1 for candidate in view.offers if candidate.class_id == offer.class_id)
        if class_count > 1:
            return f"One of {class_count} {offer.class_name} candidates on this slate."
        return f"Only {offer.class_name} on this slate."

    def _show_deep_surgery(self, message: str = "") -> None:
        if self.controller.company is None:
            self._show_town_submenu("town_recovery", "Start or load a company first.")
            return
        view = build_deep_surgery_view(self.controller.company, self.controller.definitions)
        self._show_screen(
            "deep_surgery",
            "Deep Surgery",
            self._deep_surgery_text(view),
            self._selection_actions(view.actions, back_label="Back to Recovery Ward"),
            message=message,
        )

    def _handle_deep_surgery_action(self, value: str) -> None:
        self._town_handlers.handle_deep_surgery_action(value)

    def _show_supply_shop(self, message: str = "") -> None:
        if self.controller.company is None:
            self._show_town("Start or load a company first.")
            return
        view = build_supply_shop_view(self.controller.company, self.controller.definitions)
        self.current_supply_shop_view = view
        self._render_supply_shop_view(view, message)

    def _render_supply_shop_view(self, view: SupplyShopView, message: str = "") -> None:
        self.current_supply_shop_view = view
        actions = (
            ScreenAction(
                "1",
                "Buy Supplies",
                "buy_supplies",
                ("b",),
                enabled=any(action.enabled for action in view.actions if action.value != "back"),
                default=any(action.enabled for action in view.actions if action.value != "back"),
                description="Choose from quartermaster stock.",
            ),
            ScreenAction("2", "Back to Market Row", "back", ("back",)),
        )
        self._show_screen(
            "supply_shop",
            "Quartermaster",
            self._supply_shop_text(view),
            actions,
            message=message,
            log=self._events_text(self.recent_events),
        )

    def _show_supply_buy(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        if self.controller.company is None:
            self._show_town_submenu("town_market", "Start or load a company first.")
            return
        view = build_supply_shop_view(self.controller.company, self.controller.definitions)
        self.current_supply_shop_view = view
        self._show_screen(
            "supply_buy",
            "Buy Supplies",
            self._supply_shop_text(view, screen_id="supply_buy"),
            self._selection_actions(view.actions, back_label="Back to Quartermaster"),
            message=message,
            log=self._result_log_text(self.recent_events, hci)
            if hci is not None
            else self._events_text(self.recent_events),
        )

    def _supply_shop_text(self, view: SupplyShopView, *, screen_id: str = "supply_shop") -> str:
        hints = {
            "supply_shop": "Focus a supply for cost and owned count. Purchases spend Coin.",
            "supply_buy": "Focus a supply row, then confirm the purchase.",
        }
        return SupplyShopPanel.render_text(
            view,
            hint=self._first_visit_hint(screen_id, hints.get(screen_id, "")),
        )

    def _show_relic_broker(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        if self.controller.company is None:
            self._show_town_submenu("town_charter", "Start or load a company first.")
            return
        view = build_relic_broker_view(self.controller.company, self.controller.definitions)
        self.current_relic_broker_view = view
        self._show_screen(
            "relic_broker",
            "Relic Clerk",
            RelicBrokerPanel.render_text(
                view,
                hint=self._first_visit_hint(
                    "relic_broker",
                    "Sell salvage for Coin or file proof to post new charter work.",
                ),
            ),
            self._selection_actions(view.actions, back_label="Back to Charter Office"),
            message=message,
            log=self._result_log_text(self.recent_events, hci)
            if hci is not None
            else self._events_text(self.recent_events),
        )

    def _selection_actions(
        self,
        actions: tuple[ScreenAction, ...],
        *,
        back_label: str = "Back",
    ) -> tuple[ScreenAction, ...]:
        selectable = [action for action in actions if action.value != "back"]
        enabled_indexes = [index for index, action in enumerate(selectable) if action.enabled]
        selection_actions = tuple(
            self._renumbered_action(
                action,
                index,
                default=len(enabled_indexes) == 1 and index - 1 == enabled_indexes[0],
            )
            for index, action in enumerate(selectable, start=1)
        )
        return (
            *selection_actions,
            ScreenAction(
                str(len(selection_actions) + 1),
                back_label,
                "back",
                ("b",),
                default=False,
            ),
        )

    def _show_formation(self, message: str = "", *, return_to: str = "town_yard") -> None:
        self.pending_formation_return_state = return_to
        if self.controller.company is None:
            self._show_town("Start or load a company first.")
            return
        view = build_formation_view(self.controller.company, self.controller.definitions)
        self._render_formation_view(view, message)

    def _render_formation_view(self, view: FormationView, message: str = "") -> None:
        self._current_formation_view = view
        body = "\n".join(("Formation", "", self._formation_board_text(view)))
        self._show_screen(
            "formation",
            "Formation",
            body,
            view.actions,
            message=message,
            log=self._events_text(self.recent_events),
        )

    def _show_assign_hero(self, slot_value: str) -> None:
        if self.controller.company is None:
            self._show_town("Start or load a company first.")
            return
        view = build_formation_view(self.controller.company, self.controller.definitions)
        selected_slot = next(
            (slot for slot in view.slots if slot.slot.value == slot_value),
            None,
        )
        if selected_slot is None:
            self._show_formation(
                "Choose a listed formation slot.",
                return_to=self.pending_formation_return_state,
            )
            return
        self.pending_slot = selected_slot.slot
        self.pending_slot_label = selected_slot.slot_label
        self._current_formation_view = view
        roster_by_id = {
            roster_hero.hero_id: roster_hero
            for roster_hero in self.controller.company.roster
        }
        slot_name = format_formation_slot(selected_slot.slot_label)
        actions = []
        for index, hero in enumerate(view.assignable_heroes, start=1):
            before, after = preview_assign_hero(
                self.controller.company.active_party_slots,
                roster_by_id,
                hero.hero_id,
                selected_slot.slot,
            )
            actions.append(
                ScreenAction(
                    str(index),
                    hero.name,
                    hero.hero_id,
                    (hero.name[:1].lower(),),
                    description="\n".join(
                        (
                            f"Before: {format_formation_lane_summary(before)}",
                            f"After: {format_formation_lane_summary(after)}",
                        )
                    ),
                    preview=f"Put {hero.name} in {slot_name}",
                    result_hint="Protection lane changes before the next fight.",
                )
            )
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Empty Slot",
                EMPTY_SLOT_VALUE,
                ("e",),
                description="Leave this formation slot empty.",
            )
        )
        actions.append(
            ScreenAction(str(len(actions) + 1), "Back to Formation", "back", ("b",))
        )
        body = "\n".join(
            (
                "Assign Formation Slot",
                "",
                f"Slot: {slot_name}  |  Current: {selected_slot.hero_name}",
                "",
                self._formation_board_text(
                    view,
                    focus_slot=selected_slot.slot_label,
                    focus_hero_id=selected_slot.hero_id or "",
                ),
            )
        )
        self._show_screen(
            "assign_hero",
            "Assign Formation Slot",
            body,
            tuple(actions),
            log=self._events_text(self.recent_events),
        )

    def _show_expedition(self, message: str = "") -> None:
        has_company = self.controller.company is not None
        if self._breach_pending():
            body = "The opening breach is pending. Resolve Return or Descend before beginning."
        elif (
            has_company
            and self.controller.company is not None
            and self.controller.company.active_expedition
        ):
            body = "An opening expedition is active inside the Shallow Cave dungeon."
        else:
            body = "Begin the opening expedition. Travel is paced; dungeon rooms are interactive."
        actions = (
            ScreenAction(
                "1",
                "Begin / Resume Opening Expedition",
                "begin",
                ("x",),
                enabled=has_company,
                default=has_company,
            ),
            ScreenAction("2", "Back", "back", ("b",), default=not has_company),
        )
        self._show_screen("expedition", "Expedition", body, actions, message=message)

    def _begin_expedition(
        self,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
        direct_to_dungeon: bool = False,
    ) -> None:
        if self._breach_pending():
            self._show_breach()
            return
        result = self.controller.handle(
            StartExpedition(
                stop_at_breach=True,
                manual_combat=True,
                interactive_dungeon=True,
                use_known_route=use_known_route,
                skip_known_route_playback=skip_known_route_playback,
            )
        )
        if not result.success:
            self._show_expedition(result.error or "The expedition cannot begin.")
            return
        self._record_events(result.events)
        if direct_to_dungeon:
            if (
                self.controller.company is not None
                and self.controller.company.active_expedition is not None
            ):
                self._show_dungeon(hci=result.hci)
            return
        if not result.events and self.controller.company is not None:
            if self.controller.company.active_expedition is not None:
                self._show_dungeon()
                return
        self._start_playback(result.events, result.hci)

    def _start_playback(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None:
        self.playback_beats = build_event_beats(self._meaningful_playback_events(events))
        self.playback_index = 0
        self.playback_hci = hci
        if not self.playback_beats:
            if self.controller.company is not None and self.controller.company.active_expedition:
                self._show_dungeon()
                return
            self._finish_playback("Nothing happens.")
            return
        self._show_playback()

    def _meaningful_playback_events(self, events: list[GameEvent]) -> list[GameEvent]:
        hidden_event_ids = self._opening_enemy_response_playback_event_ids(events)
        return [
            event
            for event in events
            if id(event) not in hidden_event_ids
            if not isinstance(event, ExpeditionEvent) or event.first_visit or event.major_beat
        ]

    def _opening_enemy_response_playback_event_ids(
        self,
        events: list[GameEvent],
    ) -> set[int]:
        if self.controller.manual_combat is None:
            return set()
        _hero_events, enemy_events = self._split_turn_events(events)
        enemy_beats = self._enemy_response_beats(enemy_events)
        hidden_types = (
            EnemyIntentEvent,
            SkillUsedEvent,
            MoveEvent,
            TurnDelayedEvent,
            TurnPassedEvent,
            DamageEvent,
            HealingEvent,
            MissEvent,
            ReactionUsedEvent,
            ReactionSkippedEvent,
            DownedEvent,
            DeathEvent,
            StatusChangedEvent,
        )
        return {
            id(event) for beat in enemy_beats for event in beat if isinstance(event, hidden_types)
        }

    def _show_playback(self, message: str = "") -> None:
        if self.playback_index >= len(self.playback_beats):
            self._finish_playback()
            return
        beat = self.playback_beats[self.playback_index]
        actions = (
            ScreenAction("1", "Continue", "continue", ("c",), default=True),
        )
        progress = ExpeditionProgressStrip.render_text(self.playback_beats, self.playback_index)
        self._show_screen(
            "playback",
            "Expedition Playback",
            f"{progress}\n\nCurrent Beat\n{self._beat_text(beat)}",
            actions,
            message=message,
            log=self._playback_log_text(beat.events),
        )

    def _finish_playback(self, message: str = "") -> None:
        if self.controller.manual_combat is not None:
            if self._show_opening_enemy_response_if_needed():
                return
            self._show_combat_command(message)
            return
        if self._breach_pending():
            self._show_breach(message)
            return
        company = self.controller.company
        if company is not None and company.active_expedition is not None:
            self._show_dungeon(message)
            return
        if company is not None and company.last_expedition_report is not None:
            self._show_current_place(message or "Company record filed.")
            return
        if self.controller.return_to_regional_place:
            self.controller.return_to_regional_place = False
            self._show_regional_place(message)
            return
        self._show_expedition(message or "Expedition section complete.")

    def _show_dungeon(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.controller.handle(ViewDungeon())
        if not result.success:
            self._show_expedition(result.error or "No active dungeon expedition.")
            return
        view = cast(DungeonView, result.value)
        self.current_dungeon_view = view
        title = "Wilderness" if view.current_map_id == "old_road_wilderness" else "Dungeon"
        actions = self._dungeon_navigation_actions(view)
        self._show_screen(
            "dungeon",
            title,
            self._dungeon_place_text(view),
            actions,
            message=message,
            log=self._dungeon_log_text(view, hci, actions=actions),
        )

    def _show_dungeon_map(self, message: str = "") -> None:
        result = self.controller.handle(ViewDungeon())
        if not result.success:
            self._show_expedition(result.error or "No active dungeon expedition.")
            return
        view = cast(DungeonView, result.value)
        self.current_dungeon_view = view
        navigation_actions = self._dungeon_navigation_actions(view)
        self._show_screen(
            "dungeon_map",
            "Map",
            DungeonMapPanel.render_text(view, actions=navigation_actions),
            (ScreenAction("1", "Back to Place", "back", ("b",), default=True),),
            message=message,
            log=DungeonMapPanel.render_minimap_text(view, actions=navigation_actions),
        )

    def _show_dungeon_interactions(self, message: str = "") -> None:
        result = self.controller.handle(ViewDungeon())
        if not result.success:
            self._show_expedition(result.error or "No active dungeon expedition.")
            return
        view = cast(DungeonView, result.value)
        self.current_dungeon_view = view
        commands = self._available_room_action_commands(view)
        if not commands:
            self._show_dungeon(message or "Nothing here needs handling.")
            return
        actions = tuple(
            self._renumbered_action(command, index, default=False)
            for index, command in enumerate(commands, start=1)
        )
        actions = (
            *actions,
            ScreenAction(str(len(actions) + 1), "Back to Room", "back", ("b",)),
        )
        self._show_screen(
            "dungeon_interact",
            "Interact",
            self._dungeon_place_text(view),
            actions,
            message=message,
            log=DungeonMapPanel.render_minimap_text(view, actions=actions),
        )

    def _dungeon_navigation_actions(
        self,
        view: DungeonView,
    ) -> tuple[ScreenAction, ...]:
        action_by_value = {action.value: action for action in view.actions}
        action_list: list[ScreenAction] = []
        for exit_view in view.exits:
            source = action_by_value.get(exit_view.node_id)
            if source is None:
                continue
            action_list.append(
                self._renumbered_action(
                    source,
                    len(action_list) + 1,
                    default=False,
                )
            )
        room_action_commands = self._available_room_action_commands(view)
        blocked_room_actions = self._blocked_room_actions(view)
        if room_action_commands or blocked_room_actions:
            action_list.append(
                ScreenAction(
                    str(len(action_list) + 1),
                    "Interact",
                    "interact",
                    ("i",),
                    enabled=bool(room_action_commands),
                    default=bool(room_action_commands)
                    and not any(action.enabled for action in action_list),
                    description="Handle something in this room."
                    if room_action_commands
                    else "Room actions are blocked.",
                    kind=ScreenActionKind.DUNGEON,
                    risk=ScreenActionRisk.COSTLY
                    if any(command.cost for command in room_action_commands)
                    else ScreenActionRisk.LOW,
                    unavailable_reason=self._blocked_room_actions_reason(view),
                    preview="Open room-specific actions and blocked requirements.",
                    result_hint="Room actions can reveal routes, spend supplies, or claim loot.",
                )
            )
        primary_indexes = [
            index
            for index, action in enumerate(action_list)
            if action.enabled and action.value != "interact"
        ]
        if len(primary_indexes) == 1:
            index = primary_indexes[0]
            action_list[index] = self._renumbered_action(
                action_list[index],
                int(action_list[index].number),
                default=True,
            )
        for value in (
            "enter_generated_maze",
            "retrace_generated_maze",
            "withdraw_generated_maze",
            "retreat_generated_maze",
        ):
            generated_action = action_by_value.get(value)
            if generated_action is not None:
                action_list.append(self._renumbered_action(generated_action, len(action_list) + 1))
        return_action = action_by_value.get("return")
        if return_action is not None and return_action.enabled:
            action_list.append(self._renumbered_action(return_action, len(action_list) + 1))
        return tuple(action_list)

    def _available_room_action_commands(self, view: DungeonView) -> tuple[ScreenAction, ...]:
        command_by_action_id = {
            action.value.removeprefix("action:"): action
            for action in view.actions
            if action.value.startswith("action:")
        }
        return tuple(
            command
            for room_action in view.room_actions
            if (command := command_by_action_id.get(room_action.action_id)) is not None
            and command.enabled
        )

    def _blocked_room_actions(self, view: DungeonView) -> tuple[Any, ...]:
        return tuple(
            room_action
            for room_action in view.room_actions
            if room_action.state not in {"available", "completed"}
        )

    def _blocked_room_actions_reason(self, view: DungeonView) -> str:
        blocked = self._blocked_room_actions(view)
        if not blocked:
            return ""
        return "; ".join(
            f"{room_action.label}: {self._room_action_state_reason(room_action)}"
            for room_action in blocked
        )

    def _renumbered_action(
        self,
        action: ScreenAction,
        number: int,
        *,
        label: str | None = None,
        default: bool | None = None,
    ) -> ScreenAction:
        return ScreenAction(
            str(number),
            label or action.label,
            action.value,
            action.aliases,
            enabled=action.enabled,
            default=action.default if default is None else default,
            description=action.description,
            kind=action.kind,
            risk=action.risk,
            cost=action.cost,
            unavailable_reason=action.unavailable_reason,
            preview=action.preview,
            result_hint=action.result_hint,
            confirm=action.confirm,
            route_warning=action.route_warning,
        )

    def _show_expedition_report(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.controller.handle(ViewExpeditionReport())
        if not result.success:
            self._show_expedition(result.error or "No expedition report is available.")
            return
        view = cast(ExpeditionReportView, result.value)
        self._show_report_view(view, message=message, hci=hci)

    def _show_report_view(
        self,
        view: ExpeditionReportView,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        self.current_report_view = view
        self._show_screen(
            "expedition_report",
            "Filed Company Record",
            ExpeditionReportPanel.render_text(view),
            view.actions,
            message=message,
            log=self._result_log_text(self.recent_events, hci)
            if hci is not None
            else self._events_text(self.recent_events),
        )

    def _show_combat_command(self, message: str = "") -> None:
        result = self.controller.handle(ViewCombat())
        if not result.success:
            company = self.controller.company
            if company is not None and company.active_expedition is not None:
                self._show_dungeon(result.error or "No combat is active.")
            else:
                self._show_expedition(result.error or "No combat is active.")
            return
        view = cast(CombatView, result.value)
        phase = "reaction" if view.pending_enemy_intent is not None else "command"
        self._show_combat_view(view, phase=phase, message=message)

    def _show_combat_skill(self, message: str = "") -> None:
        result = self.controller.handle(ViewCombat())
        if not result.success:
            company = self.controller.company
            if company is not None and company.active_expedition is not None:
                self._show_dungeon(result.error or "No combat is active.")
            else:
                self._show_expedition(result.error or "No combat is active.")
            return
        view = cast(CombatView, result.value)
        self._show_combat_view(view, phase="skill", message=message)

    def _show_combat_view(
        self,
        view: CombatView,
        *,
        phase: str,
        message: str = "",
    ) -> None:
        self._track_combat_turn_handoff(view, phase)
        self.current_combat_view = view
        self.current_combat_phase = phase
        self.selected_skill_id = view.selected_skill_id if phase == "target" else None
        if phase == "command":
            actions = view.commands
        elif phase == "skill":
            actions = tuple(skill.action for skill in view.skills)
        elif phase == "move":
            actions = tuple(move.action for move in view.moves)
        elif phase == "reaction":
            actions = tuple(option.action for option in view.reaction_options)
        else:
            actions = tuple(target.action for target in view.targets)
        if not actions:
            actions = (ScreenAction("1", "Continue", "continue", ("c",), default=True),)
        subtitles = {
            "command": "Choose Command",
            "skill": "Choose Skill",
            "target": "Choose Target",
            "move": "Choose Move",
            "reaction": "Reaction",
        }
        subtitle = subtitles.get(phase, "Choose Command")
        self._show_screen(
            "combat",
            f"Combat - {subtitle}",
            self._combat_text(view, mode=phase),
            actions,
            message=message,
            log=self._events_text(view.recent_events),
        )

    def _track_combat_turn_handoff(self, view: CombatView, phase: str) -> None:
        if phase != "command" or view.current_actor is None:
            return
        actor_id = view.current_actor.actor_id
        if self.last_combat_actor_id and actor_id != self.last_combat_actor_id:
            self.turn_flash_actor_id = actor_id
            self.turn_flash_frame = 0
        self.last_combat_actor_id = actor_id

    def _show_combat_target(self, view: CombatView, message: str = "") -> None:
        self._show_combat_view(view, phase="target", message=message)

    def _show_resolution(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None:
        hero_events, enemy_events = self._split_turn_events(events)
        self.pending_enemy_events = enemy_events
        self.pending_enemy_beats = self._combat_event_beats(enemy_events)
        self.pending_enemy_view = self.current_combat_view
        self.pending_resolution_hci = hci
        display_events = hero_events or events
        self._record_events(display_events)
        title = "Combat Complete" if self.controller.manual_combat is None else "Hero Action"
        self._set_current_beat(
            title,
            display_events,
            self.current_combat_view,
            deferred_events=enemy_events,
        )
        continue_label = self._post_combat_continue_label()
        actions = (
            ScreenAction(
                "1",
                "Enemy Response" if enemy_events else continue_label,
                "enemy_response" if enemy_events else "continue",
                ("c",),
                default=True,
                description=(
                    "Resolve enemy actions."
                    if enemy_events
                    else self._post_combat_continue_description()
                ),
            ),
        )
        self._show_screen(
            "resolution",
            title,
            self._combat_beat_text(
                title,
                display_events,
                view=self.current_combat_view,
                animation_frame=self.beat_animation_frame,
                deferred_events=enemy_events,
            ),
            actions,
            log=self._result_log_text(display_events, hci),
        )

    def _show_enemy_turn(self) -> None:
        if not self.pending_enemy_beats:
            self.pending_enemy_events = []
            self.pending_enemy_view = None
            self.pending_resolution_hci = None
            self._finish_playback()
            return

        events = self.pending_enemy_beats.pop(0)
        remaining_events = [event for beat in self.pending_enemy_beats for event in beat]
        self.pending_enemy_events = [*events, *remaining_events]
        self._record_events(events)
        self._set_current_beat(
            "Enemy Response",
            events,
            self.pending_enemy_view,
            deferred_events=remaining_events,
        )
        continue_label = self._post_combat_continue_label()
        has_more = bool(self.pending_enemy_beats)
        actions = (
            ScreenAction(
                "1",
                "Next Enemy Action" if has_more else continue_label,
                "enemy_response" if has_more else "continue",
                ("c",),
                default=True,
                description=(
                    "Resolve the next enemy action."
                    if has_more
                    else self._post_combat_continue_description()
                ),
            ),
        )
        self._show_screen(
            "enemy_turn",
            "Enemy Response",
            CombatPanel.render_enemy_turn(
                self.pending_enemy_view,
                events,
                source_actor_ids=self._event_source_actor_ids(events),
                target_intents=self._event_target_intents(events),
                animation_frame=self.beat_animation_frame,
                deferred_events=remaining_events,
            ),
            actions,
            log=self._result_log_text(events, self.pending_resolution_hci),
        )

    def _post_combat_continue_label(self) -> str:
        if self.controller.manual_combat is not None:
            return "Next Command"
        if self._breach_pending():
            return "Resolve Breach"
        company = self.controller.company
        if company is not None and company.active_expedition is not None:
            return "Return to Dungeon"
        if company is not None and company.last_expedition_report is not None:
            return "Return to Place"
        if self.controller.return_to_regional_place:
            return "Return to Place"
        return "Continue"

    def _post_combat_continue_description(self) -> str:
        if self.controller.manual_combat is not None:
            return "Return to the next hero command."
        if self._breach_pending():
            return "Open the breach decision."
        company = self.controller.company
        if company is not None and company.active_expedition is not None:
            return "Commit combat results and return to room actions."
        if company is not None and company.last_expedition_report is not None:
            return "Return to the current place with the company record filed."
        return "Continue."


    def _show_breach(self, message: str = "") -> None:
        company = self.controller.company
        if company is None:
            self._show_expedition("Start or load a company first.")
            return
        body = (
            "The Shallow Cave Breach is open.\n\n"
            f"Company: {company.name}\n"
            f"Reputation: {company.reputation}\n"
            f"Coin: {company.coin}\n"
            f"Location: {company.town_state.get('location', 'Haven Town')}\n"
            f"Active party: {len([slot for slot in company.active_party_slots.values() if slot])}"
        )
        actions = (
            ScreenAction(
                "1",
                "Return to Haven",
                "return_to_haven",
                ("r",),
                default=True,
                kind=ScreenActionKind.TRAVEL,
                risk=ScreenActionRisk.LOW,
                preview="Report the breach and bring the company back to Haven.",
                result_hint="Ends the expedition with proof and survivors.",
            ),
            ScreenAction(
                "2",
                "Descend to Maze Depth 1",
                "descend_maze_depth_1",
                ("d",),
                kind=ScreenActionKind.TRAVEL,
                risk=ScreenActionRisk.RISKY,
                preview="Push beyond the cave into the first impossible rooms.",
                result_hint="Requires confirmation before the company descends.",
                confirm="Descend into Maze Depth 1?",
            ),
        )
        self._show_screen(
            "breach",
            "Breach",
            body,
            actions,
            message=message,
            log=self._events_text(self.recent_events),
        )

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
        self.pending_confirm = confirm_id
        actions = ActionProvider.confirmation_actions(
            confirm_label,
            cancel_label,
            consequence=body,
            irreversible=irreversible,
        )
        self._show_screen("confirm", title, body, actions)

    def _show_name_prompt(self) -> None:
        actions = (ScreenAction("1", "Cancel", "cancel", ("b",)),)
        self._show_screen(
            "name_company",
            "Start New Company",
            "Enter a company name, or press Enter for the default.",
            actions,
        )
        self.input_mode = "company_name"
        name_input = self.query_one("#name-input", Input)
        name_input.value = ""
        name_input.placeholder = DEFAULT_COMPANY_NAME
        name_input.disabled = False
        name_input.can_focus = True
        name_input.display = True
        name_input.focus()

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

    def _show_opening_enemy_response_if_needed(self) -> bool:
        return self._combat_handlers.show_opening_enemy_response_if_needed()

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

    def _regional_move_lands_on_place(
        self,
        view: RegionalMapView,
        events: Sequence[GameEvent],
    ) -> bool:
        return self._regional_handlers.move_lands_on_place(view, events)

    def _available_regional_room_action_commands(
        self,
        view: RegionalMapView,
    ) -> tuple[ScreenAction, ...]:
        return self._regional_handlers.available_room_action_commands(view)

    def _show_regional_interactions(self, message: str = "") -> None:
        self._regional_handlers.show_interactions(message)

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
            if (
                action is not None
                and action.value not in {"back", EMPTY_SLOT_VALUE}
            ):
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

    def _regional_navigation_actions(
        self,
        view: RegionalMapView,
    ) -> tuple[ScreenAction, ...]:
        travel_actions = ActionProvider.regional_map_travel_actions(view)
        return tuple(
            self._renumbered_action(action, index + 1, default=index == 0)
            for index, action in enumerate(travel_actions)
        )

    def _regional_map_display_actions(
        self,
        actions: Sequence[ScreenAction],
    ) -> tuple[ScreenAction, ...]:
        node_by_location_id = {
            node.location_id: node.node_id
            for node in (self.current_regional_view.map_nodes if self.current_regional_view else ())
        }
        return tuple(
            replace(action, value=node_by_location_id.get(action.value, action.value))
            for action in actions
        )

    def _focused_regional_node_id(self) -> str:
        action = self.focused_action
        if action is None or self.current_regional_view is None:
            return ""
        if action.value in {
            self.current_regional_view.current_node_id,
            "back",
            "survey_route",
            "mark_route",
            "map",
            "world_map",
            "system",
        }:
            return self.current_regional_view.current_node_id
        from game.expedition.travel import REGIONAL_OVERWORLD_NODE_IDS

        if action.value in REGIONAL_OVERWORLD_NODE_IDS:
            return action.value
        if action.value in {"haven", "shallow_cave"}:
            node_by_location_id = {
                node.location_id: node.node_id
                for node in self.current_regional_view.map_nodes
            }
            return node_by_location_id.get(action.value, action.value)
        if action.value == "begin_opening_route":
            return self.current_regional_view.other_node_id
        return self.current_regional_view.current_node_id

    def _detail_text(self) -> str:
        roadbook_opening_notice = (
            self.screen_state == "regional_map"
            and self.message.startswith("The company roadbook is unrolled.")
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

    def _contract_dock_help_text(self, action: ScreenAction) -> str:
        if action.value == "back":
            return "Return to the Charter Office."
        if not action.enabled:
            reason = action.unavailable_reason or action.result_hint or action.description
            heading = "Locked until" if action.unavailable_reason else "Status"
            return "\n".join((heading, reason or "Not available right now."))
        return "\n".join(
            (
                action.result_hint or "Accept this posting.",
                "Charter only; no route starts here.",
            )
        )

    def _route_dock_help_text(self, action: ScreenAction) -> str:
        if action.value == "map":
            return "Open the full map.\nNo route is taken."
        if action.value == "interact":
            return "Open room actions.\nBlocked actions explain their requirements."
        if action.value == "back":
            return "Return to the current place."
        if action.value == "return":
            return "Follow the road back to Haven.\nArrives with the latest company record filed."
        if action.value == "enter_generated_maze":
            return (
                "Enter a breach route that extends as you travel.\n"
                "The Maze does not end until the company withdraws."
            )
        if action.value == "retrace_generated_maze":
            return "Follow marks back to the Maze threshold.\nKeeps the route active."
        if action.value == "withdraw_generated_maze":
            return (
                "Company withdraws to the Shallow Cave breach.\n"
                "This run collapses. The breach remains exploitable."
            )
        if action.value == "retreat_generated_maze":
            return "Follow marks back to the Maze threshold.\nKeeps the route active."
        if action.value.startswith("action:"):
            return action.result_hint or "Resolve this room action."
        if str(action.kind) == "travel":
            return "Take the focused route.\nUse Up/Down to compare exits. M opens the map."
        return action.result_hint or action.preview or action.description

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

    def _focused_dungeon_node_id(self) -> str:
        action = self.focused_action
        view = self.current_dungeon_view
        if action is None or view is None:
            return ""
        node_ids = {node.node_id for node in view.map_nodes}
        return action.value if action.value in node_ids else ""

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
                    and self.current_regional_view.anchor_kind
                    in {"east_gate", "shallow_cave"}
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

    def _playback_log_text(self, events: Sequence[GameEvent]) -> str:
        pieces = []
        if self.playback_hci is not None:
            pieces.append(self._result_log_text(events, self.playback_hci))
        map_text = self._active_dungeon_minimap_text()
        if map_text:
            pieces.append(map_text)
        if not pieces:
            pieces.append(self._events_text(self.recent_events))
        return "\n\n".join(piece for piece in pieces if piece)

    def _dungeon_log_text(
        self,
        view: DungeonView,
        hci: HciResultAnalysis | None,
        *,
        actions: Sequence[ScreenAction] | None = None,
    ) -> str:
        map_text = DungeonMapPanel.render_minimap_text(view, actions=actions)
        if hci is None:
            return map_text
        brief = self._result_log_text([], hci)
        return "\n\n".join(piece for piece in (brief, map_text) if piece)

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

    def _regional_log_text(
        self,
        view: DungeonView,
        hci: HciResultAnalysis | None,
        *,
        actions: Sequence[ScreenAction] | None = None,
    ) -> str:
        map_text = DungeonMapPanel.render_minimap_text(view, actions=actions)
        if hci is None:
            return map_text
        brief = self._result_log_text([], hci)
        return "\n\n".join(piece for piece in (brief, map_text) if piece)

    def _regional_place_text(self, view: RegionalMapView) -> str:
        lines = [
            view.breadcrumb,
            "",
            view.place_text or view.location_description,
        ]
        if view.arrival_context is not None:
            lines.extend(["", self._arrival_brief_text(view.arrival_context)])
        lines.extend(
            (
                "",
                "Current Objective",
                f"- {view.objective.title}: {view.objective.next_step}",
            )
        )
        if view.objective.progress:
            lines.append(f"  Progress: {view.objective.progress}")
        if view.travel_flavor:
            lines.extend(("", view.travel_flavor))
        if view.route_charted:
            lines.extend(("", f"Roadbook route available. Cost: {view.travel_cost}."))
        elif view.anchor_kind == "east_gate" and not view.route_charted:
            lines.extend(
                (
                    "",
                    (
                        "The road to Shallow Cave is uncharted. Take the Old Road "
                        "to begin the opening expedition."
                    ),
                )
            )
        if view.active_contracts:
            lines.extend(("", "Active Contracts"))
            lines.extend(
                f"- {name}: {summary}" if summary else f"- {name}"
                for _contract_id, name, summary in view.active_contracts
            )
        party_watch = self._party_watch_text()
        if party_watch:
            lines.extend(("", party_watch))
        return "\n".join(lines)

    def _regional_map_text(
        self,
        view: RegionalMapView,
        render_view: DungeonView,
        *,
        actions: Sequence[ScreenAction],
        map_actions: Sequence[ScreenAction],
    ) -> str:
        lines = [
            "Company Roadbook",
            "Old Road Wilderness",
            "",
            "Charted Route Survey",
            "known roads - cleared ground - fast travel",
            "",
            "ZOOMED OUT: choose a known destination. No new discoveries on charted travel.",
            "",
            "The company roadbook is unrolled across a crate.",
            "Cleared ground collapses into lines, costs, and destinations.",
        ]
        if view.travel_flavor:
            lines.extend(("", view.travel_flavor))
        lines.extend(
            (
                "",
                "Current Objective",
                f"- {view.objective.title}: {view.objective.next_step}",
            )
        )
        if view.objective.progress:
            lines.append(f"  Progress: {view.objective.progress}")
        lines.extend(("", "Route Commands"))
        if actions:
            for action in actions:
                destination = action.label.removeprefix("Take Charted Road to ")
                lines.append(f"{action.number} = {destination}")
        else:
            lines.append("- No charted road is available from here.")
        lines.extend(
            (
                "",
                DungeonMapPanel.render_text(
                    replace(render_view, exits=()),
                    actions=map_actions,
                    title="Roadbook Map",
                    survey_label="Charted Route Survey",
                    legend_line=(
                        "@ current  |  o visited  |  ? known  |  "
                        "number = charted route command"
                    ),
                ),
            )
        )
        return "\n".join(lines)

    def _arrival_brief_text(self, view: ArrivalBriefView) -> str:
        lines = [
            view.title,
            "",
            " -- ".join(f"[{label}]" for label in view.path),
            "",
            f"From: {view.origin_name}",
            f"Arrived: {view.location_name}",
            "",
        ]
        lines.extend(view.flavor_lines)
        lines.extend(("", "What Changed"))
        lines.extend(f"- {line}" for line in view.what_changed)
        lines.extend(("", "Next"))
        lines.append(view.next_objective or "Choose the company's next route.")
        lines.extend(("", "Records"))
        lines.append(f"{view.record_label} is available in Haven Records.")
        return "\n".join(lines)

    def _world_map_text(self, view: WorldView) -> str:
        lines = [
            "World Map",
            view.breadcrumb,
            "",
            "Current Objective",
            f"- {view.objective.title}: {view.objective.next_step}",
        ]
        if view.objective.progress:
            lines.append(f"  Progress: {view.objective.progress}")
        lines.extend(("", "Known Places"))
        if view.known_locations:
            for location in view.known_locations:
                if not location.known:
                    continue
                lines.append(f"- {location.name} ({location.kind})")
                if location.memory_summary:
                    lines.append(f"  {location.memory_summary}")
                for contract_line in location.related_contracts:
                    lines.append(f"  {contract_line}")
        else:
            lines.append("- none")
        lines.extend(("", "Charted Approaches"))
        if view.known_routes:
            lines.extend(
                f"- {route_id.replace('_', ' ').title()}" for route_id in view.known_routes
            )
        else:
            lines.append("- none")
        lines.extend(("", "Active Contracts"))
        if view.active_contracts:
            lines.extend(
                f"- {name}: {summary}" if summary else f"- {name}"
                for _contract_id, name, summary in view.active_contracts
            )
        else:
            lines.append("- none")
        return "\n".join(lines)

    def _travel_destination_label(self, view: WorldView, destination_id: str) -> str:
        destination = next(
            (
                candidate
                for candidate in view.travel_destinations
                if candidate.destination_id == destination_id
            ),
            None,
        )
        if destination is None:
            return destination_id.replace("_", " ").title()
        if destination.location_id == "haven" and view.current_location_id != "haven":
            return "Enter Haven"
        return f"Travel {destination.label}"

    def _place_name(self, location_id: str) -> str:
        location = self.controller.definitions.locations.get(location_id)
        if location is not None:
            return location.name
        return location_id.replace("_", " ").title()

    def _active_dungeon_minimap_text(self) -> str:
        company = self.controller.company
        if company is None or company.active_expedition is None:
            return ""
        result = self.controller.handle(ViewDungeon())
        if not result.success:
            return ""
        view = cast(DungeonView, result.value)
        self.current_dungeon_view = view
        return DungeonMapPanel.render_minimap_text(view)

    def _room_action_notice(self, events: Sequence[GameEvent]) -> str:
        return events[-1].message if events else "Room action resolved."

    def _dungeon_place_text(self, view: DungeonView) -> str:
        if view.current_map_id == "old_road_wilderness":
            breadcrumb = "World > Old Road Wilds"
        elif view.current_map_id == "shallow_cave":
            breadcrumb = "World > Shallow Cave > Dungeon"
        elif view.current_map_id == "shallow_cave_breach":
            breadcrumb = "World > Shallow Cave Breach"
        elif view.current_map_id == "maze_depth_1":
            breadcrumb = "World > Pandora's Maze Depth 1"
        else:
            breadcrumb = f"World > {view.current_map_id.replace('_', ' ').title()}"
        body = f"{breadcrumb}\n\n{DungeonRoomPanel.render_text(view)}"
        party_watch = self._party_watch_text()
        if party_watch:
            body = f"{body}\n\n{party_watch}"
        return body

    def _beat_text(self, beat: EventBeat) -> str:
        return f"{beat.title}\n" + "\n".join(event.message for event in beat.events)

    def _expedition_progress_strip(self) -> str:
        if not self.playback_beats:
            return "Expedition Progress\n(no route)"
        lines = ["Expedition Progress"]
        for index, beat in enumerate(self.playback_beats):
            if index < self.playback_index:
                marker = "[x]"
            elif index == self.playback_index:
                marker = "[>]"
            else:
                marker = "[ ]"
            kind = "combat" if beat.combat else beat.title.lower()
            lines.append(f"{marker} {index + 1:02}. {beat.title} ({kind})")
        return "\n".join(lines)

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

    def _roster_text(self, sections: tuple[RosterSectionView, ...]) -> str:
        living_count = sum(
            len(section.heroes)
            for section in sections
            if "memorial" not in section.title.lower()
        )
        memorial_count = sum(
            len(section.heroes)
            for section in sections
            if "memorial" in section.title.lower()
        )
        lines: list[str] = [
            "Roster",
            "",
            f"Living {living_count}  |  Memorial {memorial_count}",
        ]
        for section in sections:
            lines.append("")
            lines.append(section.title)
            if not section.heroes:
                lines.append("- none")
            else:
                memorial = "memorial" in section.title.lower()
                for hero in section.heroes:
                    lines.append(
                        f"- {format_compact_roster_row(hero, memorial=memorial)}"
                    )
        return "\n".join(lines).strip()

    def _combat_detail_text(self, action: ScreenAction) -> str:
        view = self.current_combat_view
        if view is None:
            return "Combat detail unavailable."
        return CombatPanel.detail_text(
            view,
            self.current_combat_phase,
            action,
            idle_frame=self.idle_animation_frame,
        )

    def _dungeon_detail_text(self, action: ScreenAction) -> str:
        view = self.current_dungeon_view
        if view is None:
            return action.description or action.label
        if action.value == "interact":
            return self._dungeon_interact_detail_text(view)
        exit_node = next((node for node in view.exits if node.node_id == action.value), None)
        if exit_node is not None:
            return self._dungeon_route_detail_text(action, exit_node)
        if action.value.startswith("action:"):
            action_id = action.value.removeprefix("action:")
            room_action = next(
                (candidate for candidate in view.room_actions if candidate.action_id == action_id),
                None,
            )
            if room_action is not None:
                return self._dungeon_room_action_detail_text(action, room_action)
        return self._generic_action_detail(action)

    def _dungeon_interact_detail_text(self, view: DungeonView) -> str:
        lines = ["Interact"]
        available = [
            room_action for room_action in view.room_actions if room_action.state == "available"
        ]
        blocked = self._blocked_room_actions(view)
        entries = [room_action.label for room_action in available]
        entries.extend(
            f"{room_action.label} - {self._room_action_state_reason(room_action)}"
            for room_action in blocked
        )
        if entries:
            lines.extend(("", *entries))
        else:
            lines.extend(("", "Nothing here needs handling."))
        return "\n".join(lines)

    def _dungeon_route_detail_text(self, action: ScreenAction, exit_node: Any) -> str:
        view = self.current_dungeon_view
        lines = [
            exit_node.name,
            "",
            _route_summary_line(exit_node),
        ]
        if view is not None and exit_node.map_id != view.current_map_id:
            lines.append(f"Area: {str(exit_node.map_id).replace('_', ' ').title()}")
        if not action.enabled:
            lines.extend(("", action.unavailable_reason or "Clear this room before moving."))
            if "Clear this room" not in (action.unavailable_reason or ""):
                lines.append("Clear this room before moving.")
        else:
            detail_lines = [
                line for line in (_route_warning_line(exit_node),) if line
            ]
            if detail_lines:
                lines.extend(("", *detail_lines))
        if view is not None and view.current_room.safe_return:
            lines.append("Safe return is available here.")
        elif exit_node.safe_return:
            lines.append("Destination has safe return.")
        return "\n".join(lines)

    def _dungeon_room_action_detail_text(self, action: ScreenAction, room_action: Any) -> str:
        lines = [room_action.label]
        if room_action.description:
            lines.extend(("", room_action.description))
        reason = self._room_action_state_reason(room_action)
        if reason and room_action.state != "available":
            lines.extend(("", self._sentence_case(reason)))
        if room_action.cost:
            lines.append(f"Cost: {self._quantity_list(room_action.cost)}")
        if room_action.reward:
            lines.append("Reward: " + ", ".join(room_action.reward))
        if action.result_hint:
            lines.append("Expected: " + action.result_hint)
        return "\n".join(lines)

    def _room_action_state_reason(self, room_action: Any) -> str:
        return room_action_state_reason(room_action)

    def _quantity_list(self, values: Sequence[tuple[str, int]]) -> str:
        return ", ".join(self._quantity_label(item_id, quantity) for item_id, quantity in values)

    def _quantity_label(self, item_id: str, quantity: int) -> str:
        label = item_id.replace("_", " ")
        if quantity == 1:
            return label
        return f"{quantity} {label}"

    def _sentence_case(self, text: str) -> str:
        if not text:
            return text
        return f"{text[0].upper()}{text[1:]}."

    def _generic_action_detail(self, action: ScreenAction) -> str:
        return generic_action_detail(action, safe_default=self._is_safe_default(action))

    def _gear_action_detail(self, action: ScreenAction) -> str:
        if action.value == "back":
            return self._generic_action_detail(action)
        view = self.current_gear_view
        if view is not None and action.value.startswith("gear:buy:"):
            gear_id = action.value.removeprefix("gear:buy:")
            item = next((entry for entry in view.items if entry.gear_id == gear_id), None)
            if item is not None:
                lines = ["Kit Focus", "", item.name, item.effect_summary]
                if item.description:
                    lines.extend(("", "Description", item.description))
                lines.extend(
                    (
                        "",
                        "Stock",
                        format_meta_line(
                            f"Owned {item.owned_count}",
                            f"Available {item.available_count}",
                            f"State {item.state}",
                        ),
                    )
                )
                if item.unavailable_reason and not action.enabled:
                    lines.extend(("", "Why unavailable", item.unavailable_reason))
                elif action.result_hint:
                    lines.extend(("", "Expected Result", action.result_hint))
                return "\n".join(lines)
        return self._generic_action_detail(action)

    def _supply_action_detail(self, action: ScreenAction) -> str:
        if action.value == "back":
            return self._generic_action_detail(action)
        if action.value == "buy_supplies":
            return self._generic_action_detail(action)
        lines = ["Supply Focus", "", action.label]
        if action.description:
            lines.append(action.description)
        if action.cost:
            lines.append(f"Cost: {action.cost}")
        if not action.enabled and action.unavailable_reason:
            lines.extend(("", "Why unavailable", action.unavailable_reason))
        if action.preview:
            lines.extend(("", "Preview", action.preview))
        if action.result_hint:
            lines.extend(("", "Expected Result", action.result_hint))
        return "\n".join(lines)

    def _pack_gear_detail(self, action: ScreenAction) -> str:
        lines = ["Armory", "", action.label]
        if action.preview:
            lines.extend(("", "Preview", action.preview))
        if action.description:
            lines.extend(("", "Detail", action.description))
        return "\n".join(lines)

    def _contract_action_detail(self, action: ScreenAction) -> str:
        if action.value == "back":
            return self._generic_action_detail(action)
        hotkey = primary_hotkey(action)
        lines = ["Contract Focus", "", action.label]
        if action.description:
            lines.append(action.description)
        lines.append("Charter only")
        if hotkey:
            lines.append(f"Hotkey: {hotkey}")
        if not action.enabled:
            reason = action.unavailable_reason or action.result_hint
            if reason:
                lines.extend(("", "Why unavailable", reason))
        if action.preview:
            lines.extend(("", "Objective", action.preview))
        if action.result_hint:
            lines.extend(("", "Next", action.result_hint))
        return "\n".join(lines)

    def _combat_text(self, view: CombatView, *, mode: str, idle_frame: int = 0) -> str:
        return CombatPanel.render_text(
            view,
            phase=mode,
            focused_action=self.focused_action,
            idle_frame=idle_frame,
            turn_flash_actor_id=self.turn_flash_actor_id,
            turn_flash_frame=self.turn_flash_frame,
        )

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

    def _current_beat_text(self) -> str:
        if self.screen_state == "enemy_turn":
            return CombatPanel.render_enemy_turn(
                self.current_beat_view,
                self.current_beat_events,
                source_actor_ids=self._event_source_actor_ids(self.current_beat_events),
                target_intents=self._event_target_intents(self.current_beat_events),
                animation_frame=self.beat_animation_frame,
                deferred_events=self.current_beat_deferred_events,
            )
        return self._combat_beat_text(
            self.current_beat_title,
            self.current_beat_events,
            view=self.current_beat_view,
            animation_frame=self.beat_animation_frame,
            deferred_events=self.current_beat_deferred_events,
        )

    def _current_beat_animation_last_frame(self) -> int:
        return CombatPanel.beat_animation_last_frame(
            self.current_beat_view,
            self.current_beat_events,
            source_actor_ids=self._event_source_actor_ids(self.current_beat_events),
            target_intents=self._event_target_intents(self.current_beat_events),
        )

    def _combat_beat_text(
        self,
        title: str,
        events: list[GameEvent],
        *,
        view: CombatView | None = None,
        animation_frame: int = 0,
        deferred_events: list[GameEvent] | None = None,
    ) -> str:
        if not events:
            return "The turn resolves."
        return CombatPanel.render_combat_beat(
            view if view is not None else self.current_combat_view,
            events,
            title=title,
            source_actor_ids=self._event_source_actor_ids(events),
            target_intents=self._event_target_intents(events),
            animation_frame=animation_frame,
            deferred_events=deferred_events or [],
        )

    def _combat_event_beats(self, events: list[GameEvent]) -> list[list[GameEvent]]:
        beats: list[list[GameEvent]] = []
        current: list[GameEvent] = []
        for event in events:
            if self._is_combat_beat_start(event):
                if self._continues_danger_beat(current, event):
                    current.append(event)
                    continue
                if current:
                    beats.append(current)
                current = [event]
                continue
            if current:
                current.append(event)
            else:
                current = [event]
                if self._is_combat_footer_event(event):
                    beats.append(current)
                    current = []
        if current:
            beats.append(current)
        return beats

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

    def _combatant_lines(self, actors: Sequence[Any]) -> str:
        if not actors:
            return "none"
        lines = []
        for actor in actors:
            marker = ">" if actor.acting else " "
            status = ", ".join(actor.statuses)
            marks = ", ".join(getattr(actor, "strain_marks", ()))
            detail = format_meta_line(
                actor.name,
                f"{actor.hp}/{actor.max_hp} HP",
                f"{actor.effort}/{actor.max_effort} Effort",
                f"Strain {getattr(actor, 'strain', '')}",
                f"Marks {marks}" if marks else "",
                status,
            )
            lines.append(f"{marker} {actor.slot}: {detail}")
        return "\n".join(lines)
