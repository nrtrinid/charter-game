"""Rich terminal CLI wrapper."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console

from game.app.actions import ActionProvider, ScreenActionKind, ScreenActionRisk
from game.app.commands import (
    AcceptContract,
    AssignActiveHero,
    BuySupply,
    ChooseCombatSkill,
    DelayCombatTurn,
    EnterGeneratedMaze,
    EquipGear,
    GenerateRecruitOffers,
    HireRecruit,
    LoadGame,
    MoveCombatActor,
    MoveDungeon,
    PassCombatTurn,
    PerformDeepSurgery,
    PurchaseGear,
    PurchaseUpgrade,
    Quit,
    RecoverCompany,
    ResolveCombatAction,
    ResolveCombatReaction,
    RetraceGeneratedMaze,
    RetreatCombat,
    RetreatGeneratedMaze,
    ReturnFromDungeon,
    SaveGame,
    StartExpedition,
    StartNewCompany,
    TakeExpeditionChoice,
    UnequipGear,
    UseDungeonAction,
    ViewCombat,
    ViewDungeon,
    ViewExpeditionReport,
    ViewGear,
    ViewLedger,
    ViewMemorial,
    ViewRoster,
    ViewSupplies,
    ViewTown,
    WithdrawGeneratedMaze,
)
from game.app.controller import AppController
from game.app.views import (
    EMPTY_SLOT_VALUE,
    ArrivalBriefView,
    CombatView,
    DungeonView,
    ExpeditionReportView,
    GearInventoryView,
    MemorialEntryView,
    RecruitOffersView,
    RegionalMapView,
    RosterSectionView,
    ScreenAction,
    TownDashboardView,
    build_formation_view,
    build_supply_shop_view,
)
from game.combat.formation import FormationSlot
from game.core.events import GameEvent
from game.core.hci import HciResultAnalysis
from game.ui.hci_text import format_party_watch, result_log_text, unavailable_message
from game.ui.screens import (
    build_event_beats,
    render_breach_prompt,
    render_combat_view,
    render_command_dock,
    render_event_beat,
    render_expedition_report,
    render_formation,
    render_gear_inventory,
    render_hci_summary,
    render_help,
    render_ledger,
    render_memorial,
    render_notice,
    render_recent_log,
    render_recruit_offers,
    render_resolution_card,
    render_roster_sections,
    render_save_slot,
    render_screen,
    render_supplies,
    render_supply_shop,
    render_town,
)

DEFAULT_SAVE_PATH = Path("saves/company.json")
BREACH_PENDING_FLAG = "opening_breach_pending"

InputFn = Callable[[str], str]
ChoiceSpec = tuple[str, str, str, tuple[str, ...]]


def _contract_reward_summary(entry: Any) -> str:
    pieces: list[str] = []
    if getattr(entry, "reward_reputation", 0):
        pieces.append(f"+{entry.reward_reputation} reputation")
    if getattr(entry, "coin_reward", 0):
        pieces.append(f"+{entry.coin_reward} Coin")
    return ", ".join(pieces) or "no payout"


MAIN_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Haven Town", "town", ("town", "t", "company", "c")),
    ("2", "Expedition", "expedition", ("expedition", "x", "begin")),
    ("3", "Armory", "gear", ("gear", "inventory", "i", "armory")),
    ("4", "Save / Load", "saves", ("saves", "slot")),
    ("5", "Help", "help", ("help", "?")),
    ("6", "Quit", "quit", ("quit", "q")),
)

CHARTER_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Start New Company", "start", ("start", "new")),
    ("2", "Back to Main", "back", ("back", "b")),
)

EXPEDITION_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Begin / Resume Opening Expedition", "begin", ("begin", "expedition", "x")),
    ("2", "Back", "back", ("back", "b")),
)

SAVE_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Save", "save", ("save",)),
    ("2", "Load", "load", ("load",)),
    ("3", "Back", "back", ("back", "b")),
)

BREACH_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Return to Haven", "return", ("return", "r")),
    ("2", "Descend into Maze Depth 1", "descend", ("descend", "d")),
)

PLAYBACK_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Continue", "continue", ("continue", "c")),
)

REPLACE_COMPANY_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Replace Current Company", "replace", ("replace", "yes")),
    ("2", "Keep Current Company", "cancel", ("keep", "cancel", "no")),
)

OVERWRITE_SAVE_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Overwrite Save", "overwrite", ("overwrite", "yes")),
    ("2", "Cancel", "cancel", ("cancel", "no")),
)

QUIT_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Quit to Terminal", "quit", ("quit", "yes")),
    ("2", "Stay", "stay", ("stay", "cancel", "no")),
)

DESCEND_CONFIRM_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Descend", "descend", ("descend", "yes")),
    ("2", "Return to Haven", "return", ("return", "cancel", "no")),
)


def _regional_map_text(view: RegionalMapView) -> str:
    marker_line = (
        "     @                     "
        if view.current_node_id == "haven"
        else "                           @"
    )
    lines = [
        "Company Roadbook",
        "Old Road Wilderness",
        "",
        "Charted Route Survey",
        "known roads - cleared ground - fast travel",
        "",
        "ZOOMED OUT: choose a known destination. No new discoveries on charted travel.",
        "",
        "  [ Haven Town ] ——— [ Shallow Cave ]",
        marker_line,
        "",
        f"You are at: {view.current_node_name}",
    ]
    if view.travel_flavor:
        lines.extend(("", view.travel_flavor))
    if view.arrival_context is not None:
        lines.extend(
            (
                "",
                view.arrival_context.title,
                *(view.arrival_context.flavor_lines or ()),
                "",
                "What Changed",
                *(f"- {line}" for line in view.arrival_context.what_changed),
            )
        )
    lines.extend(
        (
            "",
            "Current Objective",
            f"- {view.objective.title}: {view.objective.next_step}",
        )
    )
    if view.route_charted:
        lines.extend(("", "Route", f"Charted road available. Cost: {view.travel_cost}."))
    else:
        lines.extend(
            (
                "",
                "Route",
                "The road to Shallow Cave is uncharted. Travel starts the opening expedition.",
            )
        )
    return "\n".join(lines)


RECOVERY_CHOICES: tuple[ChoiceSpec, ...] = (
    ("1", "Fund Recovery", "recover", ("recover", "fund")),
    ("2", "Back", "back", ("back", "b")),
)


class Cli:
    def __init__(
        self,
        controller: AppController | None = None,
        *,
        console: Console | None = None,
        input_fn: InputFn | None = None,
        save_path: Path = DEFAULT_SAVE_PATH,
        clear_screen: bool = True,
    ) -> None:
        self.controller = controller or AppController()
        self.console = console or Console()
        self.input_fn = input_fn
        self.save_path = save_path
        self.clear_screen = clear_screen
        self._auto_play = False
        self._stop_playback = False
        self._last_frame: tuple[str, object, str, object | None] | None = None

    def run(self) -> None:
        while not self.controller.should_quit:
            self._show_main_menu()
            self._handle_choice(self._ask_choice(MAIN_CHOICES, show_options=False, prompt="Main"))

    def _handle_choice(self, choice: str) -> None:
        action = _normalize_main_choice(choice)
        if action in {"town", "company"}:
            self._town_menu()
        elif action == "expedition":
            self._expedition_menu()
        elif action == "saves":
            self._save_menu()
        elif action == "help":
            self._show_screen("Help", render_help())
            self._pause()
        elif action == "quit":
            self._quit()
        elif action == "start":
            self._start_company()
        elif action == "roster":
            self._view_roster()
        elif action == "supplies":
            self._view_supplies()
        elif action == "gear":
            self._gear_locker()
        elif action == "ledger":
            self._view_ledger()
        elif action == "save":
            self._save()
        elif action == "load":
            self._load()
        else:
            self._show_screen(
                "Unknown Choice",
                render_notice("Unknown choice. Type 'help' for options.", style="yellow"),
            )
            self._pause()

    def _town_menu(self) -> None:
        while not self.controller.should_quit:
            result = self.controller.handle(ViewTown())
            if not result.success:
                self._show_screen(
                    "Charter Desk",
                    "No company is active. Start a charter before using Haven services.",
                    command_dock=_command_dock_from_choices(
                        CHARTER_CHOICES,
                        prompt="Charter",
                        guidance="Start a company or return to the main menu.",
                    ),
                )
                action = self._ask_choice(
                    CHARTER_CHOICES,
                    show_options=False,
                    prompt="Charter",
                )
                if action == "back":
                    return
                if action == "start":
                    self._start_company()
                continue
            body: object = render_notice(
                result.error or "Start or load a company first.",
                style="red",
            )
            if result.success and isinstance(result.value, TownDashboardView):
                body = render_town(result.value)
            command_dock = (
                render_command_dock(
                    result.value.services,
                    prompt="Town",
                    guidance="Choose a Haven service.",
                )
                if result.success and isinstance(result.value, TownDashboardView)
                else None
            )
            self._show_screen("Haven Town", body, command_dock=command_dock)
            if not isinstance(result.value, TownDashboardView):
                self._pause()
                return
            action = self._ask_action(result.value.services, prompt="Town")
            if action == "back":
                return
            if action == "expedition":
                self._expedition_menu()
            elif action == "recruit":
                self._recruiting()
            elif action == "recover":
                self._recover_company()
            elif action == "deep_surgery":
                self._deep_surgery_menu()
            elif action == "buy":
                self._buy_supplies()
            elif action == "contracts":
                self._contract_board()
            elif action == "upgrades":
                self._upgrade_board()
            elif action == "gear":
                self._gear_locker()
            elif action == "formation":
                self._assign_formation()
            elif action == "memorial":
                self._view_memorial()
            elif action == "roster":
                self._view_roster()
            elif action == "ledger":
                self._view_ledger()
            else:
                self._show_screen(
                    "Haven Town",
                    render_notice("Unknown town service.", style="yellow"),
                )
                self._pause()

    def _contract_board(self) -> None:
        while not self.controller.should_quit:
            result = self.controller.handle(ViewTown())
            if not result.success or not isinstance(result.value, TownDashboardView):
                self._show_screen(
                    "Contract Board",
                    render_notice(result.error or "Contract board unavailable.", style="red"),
                )
                self._pause()
                return
            view = result.value
            lines = ["Posted Contracts", ""]
            if not view.contract_board:
                lines.append("No breach contracts are posted yet.")
            for entry in view.contract_board:
                reward = _contract_reward_summary(entry)
                lines.append(f"{entry.name} [{entry.state}] D{entry.difficulty}, {reward}")
                if entry.unavailable_reason:
                    lines.append(f"  {entry.unavailable_reason}")
                lines.append("")
            actions = ActionProvider.contract_board_actions(view.contract_board)
            self._show_screen(
                "Contract Board",
                "\n".join(lines).strip(),
                command_dock=render_command_dock(
                    actions,
                    prompt="Contract",
                    guidance="Accept an available contract or go back.",
                ),
            )
            choice = self._ask_action(actions, prompt="Contract")
            if choice == "back":
                return
            if choice.startswith("accept:"):
                contract_id = choice.removeprefix("accept:")
                accept = self.controller.handle(AcceptContract(contract_id))
                self._play_events(accept.events, accept.hci)
                if not accept.success:
                    self._show_screen(
                        "Contract Board",
                        render_notice(
                            accept.error or "Contract is unavailable.",
                            style="yellow",
                        ),
                    )
                    self._pause()

    def _upgrade_board(self) -> None:
        while not self.controller.should_quit:
            result = self.controller.handle(ViewTown())
            if not result.success or not isinstance(result.value, TownDashboardView):
                self._show_screen(
                    "Company Upgrades",
                    render_notice(result.error or "Upgrades unavailable.", style="red"),
                )
                self._pause()
                return
            view = result.value
            lines = ["Company Upgrades", ""]
            if not view.upgrades:
                lines.append("No company upgrades are authored yet.")
            for entry in view.upgrades:
                line = f"{entry.name} [{entry.state}] cost {entry.cost}"
                if entry.effect_summary:
                    line += f" - {entry.effect_summary}"
                lines.append(line)
                if entry.unavailable_reason and entry.state != "installed":
                    lines.append(f"  {entry.unavailable_reason}")
                lines.append("")
            self._show_screen(
                "Company Upgrades",
                "\n".join(lines).strip(),
                command_dock=render_command_dock(
                    view.upgrade_actions,
                    prompt="Upgrade",
                    guidance="Install an available upgrade or go back.",
                ),
            )
            choice = self._ask_action(view.upgrade_actions, prompt="Upgrade")
            if choice == "back":
                return
            if choice.startswith("upgrade:"):
                upgrade_id = choice.removeprefix("upgrade:")
                result = self.controller.handle(PurchaseUpgrade(upgrade_id))
                self._play_events(result.events, result.hci)
                if not result.success:
                    self._show_screen(
                        "Company Upgrades",
                        render_notice(result.error or "Upgrade unavailable.", style="yellow"),
                    )
                    self._pause()

    def _expedition_menu(self) -> None:
        CliExpeditionFlow(self)._expedition_menu()

    def _save_menu(self) -> None:
        while not self.controller.should_quit:
            self._show_screen(
                "Save / Load",
                render_save_slot(self.save_path),
                command_dock=_command_dock_from_choices(
                    SAVE_CHOICES,
                    prompt="Saves",
                    guidance="Manage the save slot.",
                ),
            )
            action = self._ask_choice(SAVE_CHOICES, show_options=False, prompt="Saves")
            if action == "back":
                return
            if action == "save":
                self._save()
            elif action == "load":
                self._load()
            else:
                self._show_screen(
                    "Save / Load",
                    render_notice("Unknown save option.", style="yellow"),
                )
                self._pause()

    def _show_main_menu(self) -> None:
        self._show_screen(
            "Main Menu",
            "Choose where the company spends its attention.",
            command_dock=_command_dock_from_choices(
                MAIN_CHOICES,
                prompt="Main",
                guidance="Choose a section.",
            ),
        )

    def _show_screen(
        self,
        title: str,
        body: object,
        *,
        hint: str = "",
        command_dock: object | None = None,
        log: object | None = None,
    ) -> None:
        if self.clear_screen:
            self.console.clear()
        self.console.print(
            render_screen(
                self.controller.company,
                self.save_path,
                title,
                body,
                hint=hint,
                command_dock=command_dock,
                log=log,
                console=self.console,
                enable_spacer=self._use_soft_regions(),
            )
        )
        if command_dock is None:
            self._last_frame = (title, body, hint, log)

    def _start_company(self) -> None:
        self._show_screen(
            "New Company",
            render_notice("Name the charter. Leave blank to use Haven Charter."),
        )
        if self.controller.company is not None:
            if (
                self._ask_confirmation(
                    "Replace Company",
                    "A company is already loaded.",
                    prompt="Replace",
                    confirm_label="Replace Company",
                    cancel_label="Keep Current",
                    confirm_value="replace",
                    consequence=(
                        "Starting a new company will replace the current in-memory company."
                    ),
                    irreversible=True,
                )
                != "replace"
            ):
                self._show_screen(
                    "New Company",
                    render_notice("Kept the current company.", style="yellow"),
                )
                self._pause()
                return
        name = self._ask("Company name [Haven Charter]: ").strip() or "Haven Charter"
        result = self.controller.handle(StartNewCompany(name=name))
        self._show_result_screen("New Company", result.events, result.error, hci=result.hci)

    def _view_roster(self) -> None:
        result = self.controller.handle(ViewRoster())
        if result.success and isinstance(result.value, tuple):
            sections = tuple(
                section for section in result.value if isinstance(section, RosterSectionView)
            )
            self._show_screen("Roster", render_roster_sections(sections))
        else:
            self._show_screen("Roster", render_notice(result.error or "No roster.", style="red"))
        self._pause()

    def _view_supplies(self) -> None:
        result = self.controller.handle(ViewSupplies())
        if result.success and isinstance(result.value, dict):
            supplies = {str(key): int(value) for key, value in result.value.items()}
            self._show_screen("Supplies", render_supplies(supplies))
        else:
            self._show_screen(
                "Supplies",
                render_notice(result.error or "No supplies.", style="red"),
            )
        self._pause()

    def _gear_locker(self) -> None:
        while not self.controller.should_quit:
            result = self.controller.handle(ViewGear())
            if not result.success or not isinstance(result.value, GearInventoryView):
                self._show_screen(
                    "Armory",
                    render_notice(result.error or "Armory unavailable.", style="red"),
                )
                self._pause()
                return
            view = result.value
            actions = self._gear_locker_actions(view)
            self._show_screen(
                "Armory",
                render_gear_inventory(view),
                command_dock=render_command_dock(
                    actions,
                    prompt="Gear",
                    guidance="Buy company kits or choose Equip Kits for hero outfitting.",
                ),
            )
            action = self._ask_action(actions, prompt="Gear")
            if action == "back":
                return
            if action == "gear:equip_menu":
                self._gear_roster_menu()
                continue
            result = self._handle_gear_action(action)
            self._show_result_screen("Armory", result.events, result.error, hci=result.hci)

    def _gear_locker_actions(self, view: GearInventoryView) -> tuple[ScreenAction, ...]:
        purchase_actions = tuple(action for action in view.actions if action.value != "back")
        return purchase_actions + (
            ScreenAction(
                str(len(purchase_actions) + 1),
                "Equip Kits",
                "gear:equip_menu",
                ("equip", "e"),
                enabled=bool(view.heroes),
                kind=ScreenActionKind.TOWN,
                unavailable_reason="No living heroes can equip gear.",
            ),
            ScreenAction(
                str(len(purchase_actions) + 2),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            ),
        )

    def _handle_gear_action(self, action: str) -> Any:
        if action.startswith("gear:buy:"):
            return self.controller.handle(PurchaseGear(action.removeprefix("gear:buy:")))
        if action.startswith("gear:equip:"):
            _prefix, _equip, hero_id, gear_id = action.split(":", 3)
            return self.controller.handle(EquipGear(hero_id, gear_id))
        if action.startswith("gear:unequip:"):
            return self.controller.handle(UnequipGear(action.removeprefix("gear:unequip:")))
        return self.controller.handle(ViewGear())

    def _gear_roster_menu(self) -> None:
        while not self.controller.should_quit:
            result = self.controller.handle(ViewGear())
            if not result.success or not isinstance(result.value, GearInventoryView):
                self._show_screen(
                    "Equip Kits",
                    render_notice(result.error or "Armory unavailable.", style="red"),
                )
                self._pause()
                return
            view = result.value
            actions = tuple(
                ScreenAction(
                    str(index),
                    hero.name,
                    f"hero:{hero.hero_id}",
                    (hero.hero_id, hero.name.lower().replace(" ", "_")),
                    kind=ScreenActionKind.INSPECT,
                    description=hero.equipped_gear_name or "No kit equipped",
                )
                for index, hero in enumerate(view.heroes, start=1)
            ) + (
                ScreenAction(
                    str(len(view.heroes) + 1),
                    "Back",
                    "back",
                    ("back", "b"),
                    kind=ScreenActionKind.NAVIGATE,
                ),
            )
            lines = ["Choose a hero to outfit.", ""]
            if not view.heroes:
                lines.append("No living heroes can equip gear.")
            for hero in view.heroes:
                equipped = hero.equipped_gear_name or "none"
                lines.append(f"{hero.name} ({hero.class_id}): {equipped}")
                if hero.condition:
                    lines.append(f"  {hero.condition}")
            self._show_screen(
                "Equip Kits",
                "\n".join(lines).strip(),
                command_dock=render_command_dock(
                    actions,
                    prompt="Hero",
                    guidance="Choose which hero to outfit.",
                ),
            )
            action = self._ask_action(actions, prompt="Hero")
            if action == "back":
                return
            if action.startswith("hero:"):
                self._hero_equipment_menu(action.removeprefix("hero:"))

    def _hero_equipment_menu(self, hero_id: str) -> None:
        while not self.controller.should_quit:
            result = self.controller.handle(ViewGear())
            if not result.success or not isinstance(result.value, GearInventoryView):
                self._show_screen(
                    "Hero Equipment",
                    render_notice(result.error or "Armory unavailable.", style="red"),
                )
                self._pause()
                return
            view = result.value
            hero = next((entry for entry in view.heroes if entry.hero_id == hero_id), None)
            if hero is None:
                self._show_screen(
                    "Hero Equipment",
                    render_notice("Choose a living hero before equipping gear.", style="yellow"),
                )
                self._pause()
                return
            if self.controller.company is None:
                return
            actions = ActionProvider.hero_gear_actions(
                self.controller.company,
                self.controller.definitions,
                hero.hero_id,
                can_manage=view.can_manage,
                manage_reason=view.manage_reason,
            )
            self._show_screen(
                f"{hero.name} Equipment",
                self._render_cli_hero_equipment(view, hero),
                command_dock=render_command_dock(
                    actions,
                    prompt="Kit",
                    guidance="Equip or remove this hero's kit.",
                ),
            )
            action = self._ask_action(actions, prompt="Kit")
            if action == "back":
                return
            result = self._handle_gear_action(action)
            self._show_result_screen(
                f"{hero.name} Equipment",
                result.events,
                result.error,
                hci=result.hci,
            )

    def _render_cli_hero_equipment(self, view: GearInventoryView, hero: Any) -> str:
        equipped = hero.equipped_gear_name or "none"
        lines = [
            f"{hero.name} ({hero.class_id})",
            f"Equipped: {equipped}",
            "",
            "Available Kits",
        ]
        owned_items = [item for item in view.items if item.owned_count > 0]
        if not owned_items:
            lines.append("none")
        for item in owned_items:
            lines.append(
                f"- {item.name}: owned {item.owned_count}, available {item.available_count}"
            )
            if item.effect_summary:
                lines.append(f"  {item.effect_summary}")
        if not view.can_manage:
            lines.extend(("", view.manage_reason))
        return "\n".join(lines)

    def _view_ledger(self) -> None:
        result = self.controller.handle(ViewLedger())
        if result.success and isinstance(result.value, dict):
            self._show_screen("Ledger", render_ledger(result.value))
        else:
            self._show_screen("Ledger", render_notice(result.error or "No ledger.", style="red"))
        self._pause()

    def _view_memorial(self) -> None:
        result = self.controller.handle(ViewMemorial())
        if result.success and isinstance(result.value, tuple):
            heroes = [hero for hero in result.value if isinstance(hero, MemorialEntryView)]
            self._show_screen("Memorial", render_memorial(tuple(heroes)))
        else:
            self._show_screen(
                "Memorial",
                render_notice(result.error or "No memorial.", style="red"),
            )
        self._pause()

    def _recruiting(self) -> None:
        result = self.controller.handle(GenerateRecruitOffers())
        if not result.success:
            self._show_screen(
                "Recruiting",
                render_notice(result.error or "No offers.", style="red"),
            )
            self._pause()
            return
        if not isinstance(result.value, RecruitOffersView):
            self._show_screen("Recruiting", render_notice("No offers.", style="yellow"))
            self._pause()
            return
        self._show_screen(
            "Recruiting",
            render_recruit_offers(result.value),
            command_dock=render_command_dock(
                result.value.actions,
                prompt="Recruit",
                guidance="Choose a recruit or go back.",
            ),
        )
        choice = self._ask_action(result.value.actions, prompt="Recruit")
        if choice == "back":
            return
        hire_result = self.controller.handle(HireRecruit(int(choice)))
        self._show_result_screen(
            "Recruiting",
            hire_result.events,
            hire_result.error,
            hci=hire_result.hci,
        )

    def _deep_surgery_menu(self) -> None:
        if self.controller.company is None:
            self._show_screen(
                "Deep Surgery",
                render_notice("Start or load a company first.", style="red"),
            )
            self._pause()
            return
        from game.app.views import build_deep_surgery_view

        view = build_deep_surgery_view(self.controller.company, self.controller.definitions)
        if not view.candidates:
            self._show_screen(
                "Deep Surgery",
                render_notice("No wounded heroes need deep surgery.", style="yellow"),
            )
            self._pause()
            return
        self._show_screen(
            "Deep Surgery",
            (
                f"Cost: {view.surgery_cost} Coin. "
                "Removes 1 Mortal Wound; hero is In Surgery until the next expedition returns."
            ),
            command_dock=render_command_dock(
                view.actions,
                prompt="Deep Surgery",
                guidance="Choose a hero to treat.",
            ),
        )
        choice = self._ask_action(view.actions, prompt="Deep Surgery")
        if choice == "back" or not choice.startswith("surgery:"):
            return
        hero_id = choice.removeprefix("surgery:")
        result = self.controller.handle(PerformDeepSurgery(hero_id))
        self._show_result_screen(
            "Deep Surgery",
            result.events,
            result.error,
            hci=result.hci,
        )

    def _recover_company(self) -> None:
        if self.controller.company is None:
            self._show_screen(
                "Recovery Ward",
                render_notice("Start or load a company first.", style="red"),
            )
            self._pause()
            return
        cost = self.controller.definitions.town.recovery_cost
        can_recover = self.controller.company.coin >= cost
        actions = (
            ScreenAction(
                "1",
                "Fund Recovery",
                "recover",
                ("recover", "fund"),
                enabled=can_recover,
                description="Restore HP and Effort, and clear Downed.",
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                cost=f"{cost} Coin",
                unavailable_reason="" if can_recover else f"Need {cost} Coin.",
                preview=f"Budget: {self.controller.company.coin} Coin.",
                result_hint="Recovered heroes return to fighting condition.",
            ),
            ScreenAction(
                "2",
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            ),
        )
        self._show_screen(
            "Recovery Ward",
            f"Recovery restores HP and Effort, and clears Downed. Cost: {cost} Coin.",
            command_dock=render_command_dock(
                actions,
                prompt="Recovery",
                guidance="Fund recovery or go back.",
            ),
        )
        if self._ask_action(actions, prompt="Recovery") != "recover":
            return
        result = self.controller.handle(RecoverCompany())
        self._show_result_screen(
            "Recovery Ward",
            result.events,
            result.error,
            hci=result.hci,
        )

    def _buy_supplies(self) -> None:
        if self.controller.company is None:
            self._show_screen(
                "Supplies",
                render_notice("Start or load a company first.", style="red"),
            )
            self._pause()
            return
        view = build_supply_shop_view(self.controller.company, self.controller.definitions)
        self._show_screen(
            "Quartermaster",
            render_supply_shop(view),
            command_dock=render_command_dock(
                view.actions,
                prompt="Supply",
                guidance="Buy one supply item or go back.",
            ),
        )
        choice = self._ask_action(view.actions, prompt="Supply")
        if choice == "back":
            return
        result = self.controller.handle(BuySupply(choice, quantity=1))
        self._show_result_screen(
            "Quartermaster",
            result.events,
            result.error,
            hci=result.hci,
        )

    def _assign_formation(self) -> None:
        if self.controller.company is None:
            self._show_screen(
                "Formation",
                render_notice("Start or load a company first.", style="red"),
            )
            self._pause()
            return
        view = build_formation_view(self.controller.company, self.controller.definitions)
        self._show_screen(
            "Formation",
            render_formation(view),
            command_dock=render_command_dock(
                view.actions,
                prompt="Slot",
                guidance="Choose the slot to edit.",
            ),
        )
        slot_choice = self._ask_action(view.actions, prompt="Slot")
        if slot_choice == "back":
            return
        slot_view = next(
            candidate for candidate in view.slots if candidate.slot.value == slot_choice
        )
        hero_actions = (
            ScreenAction(
                "1",
                "Empty Slot",
                EMPTY_SLOT_VALUE,
                ("empty", "clear"),
                description=f"Current: {slot_view.hero_name}",
            ),
            *(
                ScreenAction(
                    str(index),
                    hero.name,
                    hero.hero_id,
                    (hero.hero_id,),
                    description=_format_meta(hero.class_id, f"{hero.hp}/{hero.max_hp} HP"),
                )
                for index, hero in enumerate(view.assignable_heroes, start=2)
            ),
            ScreenAction(str(len(view.assignable_heroes) + 2), "Back", "back", ("back", "b")),
        )
        self._show_screen(
            "Assign Hero",
            f"Assign {slot_choice}. Current: {slot_view.hero_name}",
            command_dock=render_command_dock(
                hero_actions,
                prompt="Hero",
                guidance="Choose who occupies the selected slot.",
            ),
        )
        hero_choice = self._ask_action(hero_actions, prompt="Hero")
        if hero_choice == "back":
            return
        result = self.controller.handle(AssignActiveHero(hero_choice, FormationSlot(slot_choice)))
        self._show_result_screen("Formation", result.events, result.error, hci=result.hci)

    def _begin_opening_expedition(self) -> None:
        CliExpeditionFlow(self)._begin_opening_expedition()

    def _dungeon_loop(self) -> None:
        CliExpeditionFlow(self)._dungeon_loop()

    def _manual_combat_loop(self) -> None:
        CliExpeditionFlow(self)._manual_combat_loop()

    def _resolve_breach(self) -> None:
        CliExpeditionFlow(self)._resolve_breach()

    def _save(self) -> None:
        self._show_screen("Save", render_save_slot(self.save_path))
        if self.save_path.exists():
            if (
                self._ask_confirmation(
                    "Overwrite Save",
                    render_save_slot(self.save_path),
                    prompt="Overwrite",
                    confirm_label="Overwrite",
                    confirm_value="overwrite",
                    consequence=f"Overwrite the save slot at {self.save_path}.",
                    irreversible=True,
                )
                != "overwrite"
            ):
                self._show_screen("Save", render_notice("Save cancelled.", style="yellow"))
                self._pause()
                return
        result = self.controller.handle(SaveGame(self.save_path))
        self._show_result_screen("Save", result.events, result.error, hci=result.hci)

    def _load(self) -> None:
        self._show_screen("Load", render_save_slot(self.save_path))
        result = self.controller.handle(LoadGame(self.save_path))
        self._show_result_screen("Load", result.events, result.error, hci=result.hci)

    def _quit(self) -> None:
        if (
            self._ask_confirmation(
                "Quit",
                "Close the charter desk?",
                prompt="Quit",
                confirm_label="Quit",
                cancel_label="Stay",
                confirm_value="quit",
                cancel_value="stay",
                consequence="Close the charter desk.",
            )
            != "quit"
        ):
            self._show_screen("Quit", render_notice("Still at the charter desk.", style="yellow"))
            self._pause()
            return
        result = self.controller.handle(Quit())
        self._show_result_screen(
            "Quit",
            result.events,
            result.error,
            hci=result.hci,
            pause=False,
        )

    def _play_events(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None:
        beats = build_event_beats(events)
        summary = result_log_text(events, hci) if hci is not None else ""
        if not beats:
            if summary:
                self._show_screen(
                    "Outcome",
                    render_notice(summary, title="Outcome", style="cyan"),
                )
                self._pause()
            return
        if self._auto_play:
            self._show_screen(
                "Expedition Log",
                "The section resolves.",
                log=render_notice(summary, title="Outcome", style="cyan")
                if summary
                else render_recent_log(events, title="Expedition Log"),
            )
            self._pause()
            return
        index = 0
        while index < len(beats):
            beat = beats[index]
            self._show_screen(
                beat.title,
                render_event_beat(beat, show_title=False),
                log=render_notice(
                    "\n\n".join(
                        text for text in (f"Beat {index + 1} of {len(beats)}", summary) if text
                    ),
                    title="Expedition Log",
                    style="dim",
                ),
                command_dock=_command_dock_from_choices(
                    PLAYBACK_CHOICES,
                    prompt="Log",
                    default="continue",
                    guidance="Continue through the expedition log.",
                ),
            )
            response = self._ask_choice(
                PLAYBACK_CHOICES,
                default="continue",
                show_options=False,
                prompt="Log",
            )
            if response == "auto":
                self._auto_play = True
                remaining_events = [
                    event for remaining in beats[index + 1 :] for event in remaining.events
                ]
                self._show_screen(
                    "Expedition Log",
                    "The section resolves.",
                    log=render_notice(summary, title="Outcome", style="cyan")
                    if summary
                    else render_recent_log(remaining_events, title="Expedition Log"),
                )
                self._pause()
                return
            if response == "stop":
                self._stop_playback = True
                return
            index += 1

    def _show_result_screen(
        self,
        title: str,
        events: list[GameEvent],
        error: str | None,
        *,
        hci: HciResultAnalysis | None = None,
        pause: bool = True,
    ) -> None:
        log: object | None = None
        if error:
            body = render_notice(error, title="Error", style="red")
        elif hci is not None and hci.summary:
            body = render_hci_summary(hci, title="Result")
            remaining_events = [
                event
                for beat in hci.beats
                for event in beat.events
                if event.message not in hci.summary
            ]
            if remaining_events:
                log = render_recent_log(remaining_events, title="Result Log")
        elif events:
            beats = build_event_beats(events)
            if beats:
                body = render_event_beat(beats[0], show_title=False)
                remaining_events = [event for beat in beats[1:] for event in beat.events]
                if remaining_events:
                    log = render_recent_log(remaining_events, title="Result Log")
            else:
                body = render_notice("Done.")
        else:
            body = render_notice("Done.")
        self._show_screen(title, body, log=log)
        if pause:
            self._pause()

    def _pause(
        self,
        *,
        label: str = "Continue",
        prompt: str = "Continue",
    ) -> None:
        continue_choices = (("1", label, "continue", ("continue", "c")),)
        continue_dock = _command_dock_from_choices(
            continue_choices,
            prompt=prompt,
            default="continue",
            guidance="Continue when ready.",
        )
        if self._last_frame is None:
            self.console.print(continue_dock)
        else:
            title, body, hint, log = self._last_frame
            self._show_screen(
                title,
                body,
                hint=hint,
                log=log,
                command_dock=continue_dock,
            )
        self._ask_choice(
            continue_choices,
            default="continue",
            show_options=False,
            prompt=prompt,
        )

    def _ask(self, prompt: str) -> str:
        if self.input_fn is not None:
            return self.input_fn(prompt)
        return self.console.input(f"[bold bright_yellow]{prompt}[/]")

    def _ask_choice(
        self,
        choices: tuple[ChoiceSpec, ...],
        *,
        default: str | None = None,
        show_options: bool = True,
        prompt: str = "Choice",
    ) -> str:
        if show_options:
            self.console.print(_command_dock_from_choices(choices, prompt=prompt, default=default))
        lookup: dict[str, str] = {}
        for number, _label, value, aliases in choices:
            lookup[number] = value
            lookup[value] = value
            lookup[value.lower()] = value
            for alias in aliases:
                lookup[alias] = value
                lookup[alias.lower()] = value
        while True:
            raw = self._ask(f"{prompt} > ").strip().lower()
            if not raw and default is not None:
                return default
            if raw in lookup:
                return lookup[raw]
            self.console.print(render_notice("Choose one of the listed options.", style="yellow"))

    def _ask_action(self, actions: tuple[ScreenAction, ...], *, prompt: str = "Action") -> str:
        enabled_lookup: dict[str, str] = {}
        disabled_lookup: dict[str, ScreenAction] = {}
        default_value: str | None = None
        for action in actions:
            keys = {action.number, action.value, action.value.lower()}
            keys.update(action.aliases)
            keys.update(alias.lower() for alias in action.aliases)
            if action.default and action.enabled:
                default_value = action.value
            if action.enabled:
                for key in keys:
                    enabled_lookup[key] = action.value
            else:
                for key in keys:
                    disabled_lookup[key] = action

        while True:
            raw = self._ask(f"{prompt} > ").strip().lower()
            if not raw and default_value is not None:
                return default_value
            if raw in enabled_lookup:
                return enabled_lookup[raw]
            if raw in disabled_lookup:
                self.console.print(
                    render_notice(unavailable_message(disabled_lookup[raw]), style="yellow")
                )
                continue
            self.console.print(render_notice("Choose one of the listed options.", style="yellow"))

    def _ask_confirmation(
        self,
        title: str,
        body: object,
        *,
        prompt: str,
        confirm_label: str,
        cancel_label: str = "Cancel",
        confirm_value: str = "confirm",
        cancel_value: str = "cancel",
        consequence: str = "",
        irreversible: bool = False,
        confirm_aliases: tuple[str, ...] = (),
        guidance: str = "Cancel is the default; confirm explicitly to proceed.",
    ) -> str:
        actions = ActionProvider.confirmation_actions(
            confirm_label,
            cancel_label,
            confirm_value=confirm_value,
            cancel_value=cancel_value,
            consequence=consequence,
            irreversible=irreversible,
        )
        if confirm_aliases:
            confirm_action = actions[1]
            actions = (
                actions[0],
                replace(
                    confirm_action,
                    aliases=tuple(dict.fromkeys((*confirm_action.aliases, *confirm_aliases))),
                ),
            )
        self._show_screen(
            title,
            body,
            command_dock=render_command_dock(
                actions,
                prompt=prompt,
                guidance=guidance,
            ),
        )
        return self._ask_action(actions, prompt=prompt)

    def _breach_pending(self) -> bool:
        return bool(
            self.controller.company is not None
            and self.controller.company.active_expedition is None
            and self.controller.company.flags.get(BREACH_PENDING_FLAG, False)
        )

    def _use_soft_regions(self) -> bool:
        return self.clear_screen and self.input_fn is None


class CliExpeditionFlow:
    def __init__(self, cli: Cli) -> None:
        self.cli = cli

    def __getattr__(self, name: str) -> Any:
        return getattr(self.cli, name)

    @property
    def _auto_play(self) -> bool:
        return self.cli._auto_play

    @_auto_play.setter
    def _auto_play(self, value: bool) -> None:
        self.cli._auto_play = value

    @property
    def _stop_playback(self) -> bool:
        return self.cli._stop_playback

    @_stop_playback.setter
    def _stop_playback(self, value: bool) -> None:
        self.cli._stop_playback = value

    def _expedition_menu(self) -> None:
        while not self.controller.should_quit:
            self._show_screen(
                "Expedition",
                "A discovered breach is waiting."
                if self._breach_pending()
                else "The opening route is ready.",
                command_dock=_command_dock_from_choices(
                    EXPEDITION_CHOICES,
                    prompt="Expedition",
                    guidance="Choose an expedition action.",
                ),
            )
            action = self._ask_choice(EXPEDITION_CHOICES, show_options=False, prompt="Expedition")
            if action == "back":
                return
            if action == "begin":
                self._begin_opening_expedition()
            else:
                self._show_screen(
                    "Expedition",
                    render_notice("Unknown expedition option.", style="yellow"),
                )
                self._pause()

    def _begin_opening_expedition(self) -> None:
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
                render_notice(result.error or "Expedition failed.", style="red"),
            )
            self._pause()
            return
        self._play_events(result.events, result.hci)
        if self._stop_playback:
            return
        self._dungeon_loop()
        if self._stop_playback:
            return
        if (
            self.controller.company is not None
            and self.controller.company.active_expedition is None
        ):
            return
        self._resolve_breach()

    def _show_regional_map(self, view: RegionalMapView) -> None:
        self._show_screen("Company Roadbook", _regional_map_text(view))
        self._pause(label="Continue", prompt="Company Roadbook")

    def _show_arrival_brief(self, view: ArrivalBriefView) -> None:
        lines = [
            view.title,
            "",
            " -- ".join(f"[{label}]" for label in view.path),
            "",
            *view.flavor_lines,
            "",
            "What Changed",
            *(f"- {line}" for line in view.what_changed),
            "",
            "Next",
            view.next_objective or "Choose the company's next route.",
        ]
        self._show_screen(view.title, "\n".join(lines))
        self._pause(label="Continue to Filed Record", prompt="Record")

    def _dungeon_loop(self) -> None:
        while (
            self.controller.company is not None
            and self.controller.company.active_expedition is not None
            and not self._breach_pending()
            and not self._stop_playback
        ):
            view_result = self.controller.handle(ViewDungeon())
            if not view_result.success or not isinstance(view_result.value, DungeonView):
                self._show_screen(
                    "Expedition",
                    render_notice(view_result.error or "Expedition view unavailable.", style="red"),
                )
                self._pause()
                return
            view = view_result.value
            self._show_screen(
                "Expedition",
                _render_dungeon_text(view, party_watch=_party_watch_text(self.controller)),
                log=render_recent_log(view.recent_events),
                command_dock=render_command_dock(
                    view.actions,
                    prompt="Expedition",
                    guidance="Choose a route, room action, or return option.",
                ),
            )
            action = self._ask_action(view.actions, prompt="Expedition")
            if action == "return":
                result = self.controller.handle(ReturnFromDungeon())
                if not result.success:
                    self._show_screen(
                        "Expedition",
                        render_notice(result.error or "Return failed.", style="red"),
                    )
                    self._pause()
                elif isinstance(result.value, RegionalMapView):
                    self._show_regional_map(result.value)
                    report_result = self.controller.handle(ViewExpeditionReport())
                    if report_result.success and isinstance(
                        report_result.value,
                        ExpeditionReportView,
                    ):
                        self._show_screen(
                            "Filed Company Record",
                            render_expedition_report(report_result.value),
                        )
                        self._pause()
                elif isinstance(result.value, ExpeditionReportView):
                    self._show_screen(
                        "Filed Company Record",
                        render_expedition_report(result.value),
                    )
                    self._pause()
                return
            if action == "enter_generated_maze":
                result = self.controller.handle(EnterGeneratedMaze())
            elif action == "retrace_generated_maze":
                result = self.controller.handle(RetraceGeneratedMaze())
            elif action == "withdraw_generated_maze":
                result = self.controller.handle(WithdrawGeneratedMaze())
            elif action == "retreat_generated_maze":
                result = self.controller.handle(RetreatGeneratedMaze())
            elif action.startswith("action:"):
                result = self.controller.handle(UseDungeonAction(action.removeprefix("action:")))
            else:
                result = self.controller.handle(MoveDungeon(action))
            self._play_events(result.events, result.hci)
            if not result.success:
                self._show_screen(
                    "Expedition",
                    render_notice(result.error or "Expedition action failed.", style="red"),
                )
                self._pause()
                continue
            self._manual_combat_loop()

    def _manual_combat_loop(self) -> None:
        while self.controller.manual_combat is not None and not self._stop_playback:
            view_result = self.controller.handle(ViewCombat())
            if not view_result.success or not isinstance(view_result.value, CombatView):
                self._show_screen(
                    "Combat",
                    render_notice(view_result.error or "Combat view unavailable.", style="red"),
                )
                self._pause()
                return
            view = view_result.value
            if view.pending_enemy_intent is not None:
                reaction_actions = tuple(option.action for option in view.reaction_options)
                self._show_screen(
                    "Combat",
                    render_combat_view(view),
                    log=render_recent_log(view.recent_events),
                    command_dock=render_command_dock(
                        reaction_actions,
                        prompt="Reaction",
                        guidance="Choose a class reaction or skip.",
                    ),
                )
                reaction_value = self._ask_action(reaction_actions, prompt="Reaction")
                reaction_id = None if reaction_value == "skip" else reaction_value
                result = self.controller.handle(ResolveCombatReaction(reaction_id))
                self._show_screen(
                    "Combat Resolution",
                    render_resolution_card(result.events, result.hci),
                )
                self._pause()
                if not result.success:
                    self._show_screen(
                        "Combat",
                        render_notice(result.error or "Reaction failed.", style="red"),
                    )
                    self._pause()
                continue
            if not any(action.enabled for action in view.commands):
                self._show_screen(
                    "Combat",
                    render_notice("No legal hero actions are available.", style="yellow"),
                )
                self._pause()
                return
            self._show_screen(
                "Combat",
                render_combat_view(view),
                log=render_recent_log(view.recent_events),
                command_dock=render_command_dock(
                    view.commands,
                    prompt="Command",
                    guidance="Choose how this hero acts.",
                ),
            )
            command = self._ask_action(view.commands, prompt="Command")
            if command == "move":
                self._show_screen(
                    "Combat",
                    render_combat_view(view),
                    log=render_recent_log(view.recent_events),
                    command_dock=render_command_dock(
                        tuple(move.action for move in view.moves),
                        prompt="Move",
                        guidance="Choose an adjacent slot.",
                    ),
                )
                slot_id = self._ask_action(tuple(move.action for move in view.moves), prompt="Move")
                result = self.controller.handle(MoveCombatActor(FormationSlot(slot_id)))
                self._show_screen(
                    "Combat Resolution",
                    render_resolution_card(result.events, result.hci),
                )
                self._pause()
                if not result.success:
                    self._show_screen(
                        "Combat",
                        render_notice(result.error or "Move failed.", style="red"),
                    )
                    self._pause()
                continue
            if command == "pass":
                result = self.controller.handle(PassCombatTurn())
                self._show_screen(
                    "Combat Resolution",
                    render_resolution_card(result.events, result.hci),
                )
                self._pause()
                if not result.success:
                    self._show_screen(
                        "Combat",
                        render_notice(result.error or "Pass failed.", style="red"),
                    )
                    self._pause()
                continue
            if command == "delay":
                result = self.controller.handle(DelayCombatTurn())
                self._show_screen(
                    "Combat Resolution",
                    render_resolution_card(result.events, result.hci),
                )
                self._pause()
                if not result.success:
                    self._show_screen(
                        "Combat",
                        render_notice(result.error or "Delay failed.", style="red"),
                    )
                    self._pause()
                continue
            if command == "retreat":
                confirm = self._ask_confirmation(
                    "Retreat",
                    render_notice(
                        "Begin retreat? Enemies with remaining turns may still act.",
                        style="yellow",
                    ),
                    prompt="Retreat",
                    confirm_label="Retreat",
                    cancel_label="Stand Ground",
                    confirm_value="retreat",
                    consequence="Escape resolves at the end of the current round.",
                    confirm_aliases=("r",),
                    guidance="Stand ground is the default; confirm explicitly to retreat.",
                )
                if confirm != "retreat":
                    continue
                result = self.controller.handle(RetreatCombat())
                self._show_screen(
                    "Combat Resolution",
                    render_resolution_card(result.events, result.hci),
                )
                self._pause()
                if not result.success:
                    self._show_screen(
                        "Combat",
                        render_notice(result.error or "Retreat failed.", style="red"),
                    )
                    self._pause()
                    continue
                return

            self._show_screen(
                "Combat",
                render_combat_view(view),
                log=render_recent_log(view.recent_events),
                command_dock=render_command_dock(
                    tuple(option.action for option in view.skills),
                    prompt="Skill",
                    guidance="Choose a skill. Enter selects the default.",
                ),
            )
            if not any(option.action.enabled for option in view.skills):
                self._pause()
                continue
            skill_id = self._ask_action(
                tuple(option.action for option in view.skills),
                prompt="Skill",
            )
            skill_result = self.controller.handle(ChooseCombatSkill(skill_id))
            if not skill_result.success or not isinstance(skill_result.value, CombatView):
                self._show_screen(
                    "Combat",
                    render_notice(skill_result.error or "Skill failed.", style="red"),
                )
                self._pause()
                continue
            view = skill_result.value

            self._show_screen(
                "Combat",
                render_combat_view(view),
                log=render_recent_log(view.recent_events),
                command_dock=render_command_dock(
                    tuple(option.action for option in view.targets),
                    prompt="Target",
                    guidance="Choose a target. Enter selects the default.",
                ),
            )
            target_id = self._ask_action(
                tuple(option.action for option in view.targets),
                prompt="Target",
            )
            result = self.controller.handle(ResolveCombatAction(skill_id, target_id))
            self._show_screen(
                "Combat Resolution",
                render_resolution_card(result.events, result.hci),
            )
            self._pause()
            if not result.success:
                self._show_screen(
                    "Combat",
                    render_notice(result.error or "Combat action failed.", style="red"),
                )
                self._pause()

    def _resolve_breach(self) -> None:
        if self.controller.company is None:
            self._show_screen(
                "Breach",
                render_notice("Start or load a company first.", style="red"),
            )
            self._pause()
            return
        self._show_screen(
            "Breach",
            render_breach_prompt(self.controller.company),
            command_dock=_command_dock_from_choices(
                BREACH_CHOICES,
                prompt="Breach",
                guidance="Choose what the company does at the breach.",
            ),
        )
        choice = self._ask_choice(BREACH_CHOICES, show_options=False, prompt="Breach")
        if choice == "return":
            result = self.controller.handle(TakeExpeditionChoice("return_to_haven"))
        else:
            if (
                self._ask_confirmation(
                    "Confirm Descent",
                    "The Maze descent is risky and cannot be undone this expedition.",
                    prompt="Descent",
                    confirm_label="Descend",
                    cancel_label="Return to Haven",
                    confirm_value="descend",
                    cancel_value="return",
                    consequence="Descend into Maze Depth 1.",
                    confirm_aliases=("d",),
                )
                != "descend"
            ):
                result = self.controller.handle(TakeExpeditionChoice("return_to_haven"))
            else:
                result = self.controller.handle(TakeExpeditionChoice("descend_maze_depth_1"))
        self._play_events(result.events, result.hci)
        if not result.success:
            self._show_screen(
                "Breach",
                render_notice(result.error or "Choice failed.", style="red"),
            )
            self._pause()


def _normalize_main_choice(choice: str) -> str | None:
    aliases: dict[str, str] = {
        "1": "town",
        "company": "company",
        "c": "company",
        "2": "expedition",
        "expedition": "expedition",
        "x": "expedition",
        "begin": "expedition",
        "3": "saves",
        "saves": "saves",
        "slot": "saves",
        "4": "help",
        "help": "help",
        "?": "help",
        "5": "quit",
        "quit": "quit",
        "q": "quit",
        "start": "start",
        "new": "start",
        "roster": "roster",
        "r": "roster",
        "supplies": "supplies",
        "s": "supplies",
        "ledger": "ledger",
        "town": "town",
        "t": "town",
        "save": "save",
        "load": "load",
    }
    return aliases.get(choice)


def _party_watch_text(controller: AppController) -> str:
    company = controller.company
    if company is None:
        return ""
    formation = build_formation_view(company, controller.definitions)
    return format_party_watch(formation)


def _render_dungeon_text(view: DungeonView, *, party_watch: str = "") -> str:
    room = view.current_room
    art_lines = _compact_art_lines(room.art_lines, max_lines=10, max_width=72)
    lines = [
        *art_lines,
        "" if art_lines else None,
        room.name,
        "",
        room.text,
    ]
    impact_lines = [
        line for line in (room.scene_state, room.route_hint, room.party_hint) if line.strip()
    ][:3]
    if impact_lines:
        lines.extend(("", *impact_lines))
    if party_watch:
        lines.extend(("", party_watch))
    return "\n".join(line for line in lines if line is not None)


def _compact_art_lines(
    art_lines: tuple[str, ...],
    *,
    max_lines: int,
    max_width: int,
) -> list[str]:
    return [line[:max_width].rstrip() for line in art_lines[:max_lines]]


def _format_meta(*parts: str) -> str:
    return "  |  ".join(part.strip() for part in parts if part.strip())


def _normalize_company_choice(choice: str) -> str | None:
    aliases = {
        "1": "start",
        "start": "start",
        "new": "start",
        "2": "town",
        "town": "town",
        "t": "town",
        "3": "roster",
        "roster": "roster",
        "r": "roster",
        "4": "supplies",
        "supplies": "supplies",
        "s": "supplies",
        "5": "ledger",
        "ledger": "ledger",
        "6": "back",
        "back": "back",
        "b": "back",
    }
    return aliases.get(choice)


def _normalize_expedition_choice(choice: str) -> str | None:
    aliases = {
        "1": "begin",
        "begin": "begin",
        "expedition": "begin",
        "x": "begin",
        "2": "back",
        "back": "back",
        "b": "back",
    }
    return aliases.get(choice)


def _normalize_save_choice(choice: str) -> str | None:
    aliases = {
        "1": "save",
        "save": "save",
        "2": "load",
        "load": "load",
        "3": "back",
        "back": "back",
        "b": "back",
    }
    return aliases.get(choice)


def _command_dock_from_choices(
    choices: tuple[ChoiceSpec, ...],
    *,
    prompt: str,
    guidance: str = "",
    default: str | None = None,
):
    return render_command_dock(
        _choice_actions(choices, default=default),
        prompt=prompt,
        guidance=guidance,
    )


def _choice_actions(
    choices: tuple[ChoiceSpec, ...],
    *,
    default: str | None = None,
) -> tuple[ScreenAction, ...]:
    return tuple(
        ScreenAction(
            number,
            label,
            value,
            aliases,
            default=default in {number, value} if default is not None else False,
        )
        for number, label, value, aliases in choices
    )


def _visible_options(choices: tuple[ChoiceSpec, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple((number, label, ", ".join(aliases)) for number, label, _value, aliases in choices)


def _choice_specs_from_labels(
    values: list[tuple[str, str | int]],
    *,
    back: bool = False,
) -> tuple[ChoiceSpec, ...]:
    choices: list[ChoiceSpec] = []
    for index, (label, value) in enumerate(values, start=1):
        normalized_value = str(value)
        aliases = (normalized_value, label.lower().replace(" ", "_"))
        choices.append((str(index), label, normalized_value, aliases))
    if back:
        number = str(len(choices) + 1)
        choices.append((number, "Back", "back", ("back", "b")))
    return tuple(choices)
