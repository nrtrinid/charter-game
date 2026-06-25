"""TUI screen rendering for town."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from game.app.actions import (
    ActionProvider,
)
from game.app.commands import (
    GenerateRecruitOffers,
    ViewGear,
    ViewHeroSheet,
    ViewLedger,
    ViewMemorial,
    ViewRoster,
    ViewSupplies,
    ViewTown,
)
from game.app.views import (
    EMPTY_SLOT_VALUE,
    ArrivalBriefView,
    CombatActorView,
    DeepSurgeryView,
    FormationView,
    GearInventoryView,
    HeroSheetFreshMemoryView,
    HeroSheetMemoryEntryView,
    HeroSheetSignalView,
    HeroSheetTraitView,
    HeroSheetView,
    RecruitOffersView,
    RecruitOfferView,
    RosterSectionView,
    ScreenAction,
    ScreenActionKind,
    ScreenActionRisk,
    SupplyShopView,
    TownDashboardView,
    build_deep_surgery_view,
    build_formation_view,
    build_hero_portrait_view,
    build_recruit_offers_view,
    build_relic_broker_view,
    build_supply_shop_view,
    hero_protection_line,
    preview_assign_hero,
)
from game.core.hci import HciResultAnalysis
from game.ui.hci_text import (
    format_compact_roster_row,
    format_formation_lane_summary,
    format_formation_slot,
    format_meta_line,
    primary_hotkey,
)
from game.ui.tui_render.protocol import TuiRenderHost
from game.ui.tui_widgets import (
    CompanyPanel,
    FormationBoard,
    GearLockerPanel,
    PackPanel,
    RelicBrokerPanel,
    SupplyShopPanel,
    TownDashboardPanel,
    YardPanel,
    portrait_detail_lines,
)
from game.ui.wounds import mortal_wound_badge


@dataclass
class TownRender:
    app: TuiRenderHost

    def show_town(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.app.controller.handle(ViewTown())
        if not result.success:
            self.app._show_main(result.error or "Start or load a company first.")
            return
        self.app._record_events(result.events)
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
            body = self.app._arrival_brief_text(arrival_brief) + "\n\n" + body
        self.app._show_screen(
            "town",
            "Haven",
            body,
            self._town_hub_actions(),
            message=message,
            log=self.app._result_log_text(self.app.recent_events, hci)
            if hci is not None
            else self.app._events_text(self.app.recent_events),
        )

    def show_town_submenu(
        self,
        submenu: str,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.app.controller.handle(ViewTown())
        if not result.success:
            self.app._show_main(result.error or "Start or load a company first.")
            return
        self.app._record_events(result.events)
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
        self.app._show_screen(
            submenu,
            titles[submenu],
            body,
            self._town_submenu_actions(view, submenu),
            message=message,
            log=self.app._result_log_text(self.app.recent_events, hci)
            if hci is not None
            else self.app._events_text(self.app.recent_events),
        )

    def _town_hub_actions(self) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        company = self.app.controller.company
        route_charted = company is not None and "shallow_cave" in company.known_route_ids
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
                self.app._renumbered_action(action, index, default=False)
                for index, action in enumerate(actions, start=1)
            )
        if submenu == "town_upgrades":
            upgrade_actions = tuple(
                action
                for action in view.upgrade_actions
                if action.value != "back" or action.label.lower().startswith("back")
            )
            return tuple(
                self.app._renumbered_action(
                    action,
                    index,
                    default=False,
                    label="Back to Charter Office" if action.value == "back" else None,
                )
                for index, action in enumerate(upgrade_actions, start=1)
            )
        actions = tuple(
            self.app._renumbered_action(
                action,
                index,
                default=index == 1 and action.enabled and self.app._is_safe_default(action),
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
        if submenu == "town_records" and self.app.controller.company is not None:
            if self.app.controller.company.last_expedition_report is not None:
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

    def _first_visit_hint(self, screen_id: str, text: str) -> str:
        if screen_id in self.app.visited_screens:
            return ""
        return text

    def _town_hub_body(self, view: TownDashboardView) -> str:
        hero_lines = (
            "\n".join(f"- {format_compact_roster_row(hero)}" for hero in view.active_party)
            or "none"
        )
        reserve_lines = (
            "\n".join(f"- {format_compact_roster_row(hero)}" for hero in view.reserves) or "none"
        )
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
        company = self.app.controller.company
        if company is not None:
            formation = build_formation_view(company, self.app.controller.definitions)
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
        company = self.app.controller.company
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
        active_contract_count = sum(1 for entry in view.contract_board if entry.state == "active")
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

    def show_roster(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewRoster())
        if not result.success:
            self.app._show_main(result.error or "Start or load a company first.")
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
        self.app._show_screen(
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
                        description=self.app._hero_gear_summary(hero) or "No kit equipped",
                        kind=ScreenActionKind.INSPECT,
                        preview="Open this hero's character sheet.",
                    )
                )
        return tuple(actions)

    def show_supplies(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewSupplies())
        if not result.success:
            self.app._show_main(result.error or "Start or load a company first.")
            return
        supplies = cast(dict[str, int], result.value)
        lines = ["Current supplies"]
        for supply_id, quantity in sorted(supplies.items()):
            lines.append(f"{supply_id}: {quantity}")
        self.app._show_screen(
            "supplies",
            "Supplies",
            "\n".join(lines),
            (ScreenAction("1", "Back", "back", ("b",), default=True),),
            message=message,
        )

    def show_pack(self, message: str = "") -> None:
        supplies_result = self.app.controller.handle(ViewSupplies())
        gear_result = self.app.controller.handle(ViewGear())
        if not supplies_result.success or not gear_result.success:
            self.app._show_current_place(
                supplies_result.error or gear_result.error or "Company inventory is unavailable."
            )
            return
        self.app._record_events([*supplies_result.events, *gear_result.events])
        supplies = cast(dict[str, int], supplies_result.value)
        gear = cast(GearInventoryView, gear_result.value)
        actions = self._pack_actions(gear)
        self.app._show_screen(
            "pack",
            "Pack",
            self._pack_text(supplies, gear),
            actions,
            message=message,
            log=self.app._events_text(self.app.recent_events),
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
        company = self.app.controller.company
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

    def show_company_summary(self, message: str = "") -> None:
        company = self.app.controller.company
        if company is None:
            self.app._show_main("Start or load a company first.")
            return
        formation = build_formation_view(company, self.app.controller.definitions)
        roster_result = self.app.controller.handle(ViewRoster())
        town_result = self.app.controller.handle(ViewTown())
        sections = (
            cast(tuple[RosterSectionView, ...], roster_result.value)
            if roster_result.success
            else ()
        )
        town_view = cast(TownDashboardView, town_result.value) if town_result.success else None
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
        self.app._current_formation_view = formation
        self.app._company_summary_objective = objective_line
        self.app._company_summary_roster_lines = roster_lines
        body = self._company_summary_body(company.name)
        self.app._show_screen(
            "company_summary",
            "Company",
            body,
            self._company_summary_actions(sections),
            message=message,
            log=self.app._events_text(self.app.recent_events),
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
                        description=self.app._hero_gear_summary(hero) or "Gear: none",
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

    def show_gear_locker(self, message: str = "", *, return_to: str = "town_market") -> None:
        self.app.pending_gear_locker_return_state = return_to
        result = self.app.controller.handle(ViewGear())
        if not result.success:
            self.app._show_main(result.error or "Start or load a company first.")
            return
        self.app._record_events(result.events)
        view = cast(GearInventoryView, result.value)
        self.app.current_gear_view = view
        self.app._show_screen(
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
        }.get(self.app.pending_gear_locker_return_state, "Back to Market Row")
        return tuple(
            self.app._renumbered_action(
                action,
                index,
                label=back_label if action.value == "back" else None,
            )
            for index, action in enumerate(actions, start=1)
        )

    def show_hero_sheet(
        self,
        hero_id: str = "",
        message: str = "",
        *,
        return_to: str = "roster",
    ) -> None:
        selected_id = hero_id or self.app.pending_gear_hero_id
        result = self.app.controller.handle(ViewHeroSheet(selected_id))
        if not result.success:
            if return_to == "pack":
                self.show_pack(result.error or "Hero sheet unavailable.")
            elif return_to == "company":
                self.show_company_summary(result.error or "Hero sheet unavailable.")
            else:
                self.show_roster(result.error or "Hero sheet unavailable.")
            return
        self.app._record_events(result.events)
        view = cast(HeroSheetView, result.value)
        if not view.hero_id:
            self.app.pending_gear_hero_id = ""
            if return_to == "pack":
                self.show_pack("Choose a hero before opening a character sheet.")
            elif return_to == "company":
                self.show_company_summary("Choose a hero before opening a character sheet.")
            else:
                self.show_roster("Choose a hero before opening a character sheet.")
            return
        self.app.pending_gear_hero_id = view.hero_id
        self.app.pending_gear_return_state = return_to
        self.app._show_screen(
            "hero_sheet",
            f"{view.name} Sheet",
            self._hero_sheet_text(view),
            self._hero_sheet_actions(view),
            message=message,
        )

    def show_hero_memories(self, message: str = "") -> None:
        if not self.app.pending_gear_hero_id:
            if self.app.pending_gear_return_state == "company":
                self.show_company_summary("Choose a hero before opening memories.")
            else:
                self.show_roster("Choose a hero before opening memories.")
            return
        result = self.app.controller.handle(ViewHeroSheet(self.app.pending_gear_hero_id))
        if not result.success:
            if self.app.pending_gear_return_state == "company":
                self.show_company_summary(result.error or "Hero memories unavailable.")
            else:
                self.show_roster(result.error or "Hero memories unavailable.")
            return
        view = cast(HeroSheetView, result.value)
        self.app._show_screen(
            "hero_memories",
            f"{view.name} Quirks / Memories",
            self._hero_quirks_memories_text(view),
            (ScreenAction("1", "Back to Sheet", "back", ("b",), default=True),),
            message=message,
        )

    def show_hero_gear(self, message: str = "") -> None:
        if not self.app.pending_gear_hero_id:
            self.show_roster("Choose a hero before opening gear.")
            return
        result = self.app.controller.handle(ViewHeroSheet(self.app.pending_gear_hero_id))
        if not result.success:
            self.show_roster(result.error or "Hero gear unavailable.")
            return
        view = cast(HeroSheetView, result.value)
        self.app._show_screen(
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
        if self.app.controller.company is None:
            return (ScreenAction("1", "Back to Sheet", "back", ("b",), default=True),)
        gear_actions = list(
            ActionProvider.hero_gear_actions(
                self.app.controller.company,
                self.app.controller.definitions,
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
            self.app._renumbered_action(action, index)
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
        company = self.app.controller.company
        if company is None or not hero_id:
            return None
        hero = next((entry for entry in company.roster if entry.hero_id == hero_id), None)
        if hero is None:
            return None
        return build_hero_portrait_view(
            hero,
            self.app.controller.definitions,
            slot=slot,
        )

    def _formation_portrait_actors(self, view: FormationView) -> dict[str, CombatActorView]:
        company = self.app.controller.company
        if company is None:
            return {}
        roster_by_id = {hero.hero_id: hero for hero in company.roster}
        actors: dict[str, CombatActorView] = {}
        for slot in view.slots:
            if slot.hero_id is None or slot.hero_id not in roster_by_id:
                continue
            actors[slot.slot_label] = build_hero_portrait_view(
                roster_by_id[slot.hero_id],
                self.app.controller.definitions,
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
            idle_frame=self.app.idle_animation_frame,
            inward_facing=False,
        )

    def _company_summary_body(self, company_name: str) -> str:
        formation = self.app._current_formation_view
        if formation is None:
            return self.app.body_text
        focus_hero_id = ""
        action = self.app.focused_action
        if action is not None and action.value.startswith("hero:"):
            focus_hero_id = action.value.removeprefix("hero:")
        formation_text = self._formation_board_text(
            formation,
            focus_hero_id=focus_hero_id,
        )
        return CompanyPanel.render_text(
            company_name,
            self.app._company_summary_objective,
            formation_text,
            self.app._company_summary_roster_lines,
            hint=self._first_visit_hint(
                "company_summary",
                "Focus a hero for their sheet, or open Formation or Armory.",
            ),
        )

    def _formation_detail_text(self, action: ScreenAction) -> str:
        view = self.app._current_formation_view
        if self.app.screen_state == "formation" and view is not None:
            if action.value == "back":
                return self.app._generic_action_detail(action)
            slot = next(
                (entry for entry in view.slots if entry.slot_label == action.value),
                None,
            )
            if slot is not None:
                portrait_lines: list[str] = []
                if slot.hero_id is not None:
                    portrait_lines = portrait_detail_lines(
                        self._hero_portrait_actor(slot.hero_id, slot=slot.slot_label),
                        idle_frame=self.app.idle_animation_frame,
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
        if self.app.screen_state == "assign_hero":
            if action.value == "back":
                return self.app._generic_action_detail(action)
            if action.value == EMPTY_SLOT_VALUE:
                lines = ["Empty Slot", "", "Leave this formation slot empty."]
                if action.result_hint:
                    lines.extend(("", "Result", action.result_hint))
                return "\n".join(lines)
            portrait_lines = portrait_detail_lines(
                self._hero_portrait_actor(action.value, slot=self.app.pending_slot_label),
                idle_frame=self.app.idle_animation_frame,
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
        return self.app._generic_action_detail(action)

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
            idle_frame=self.app.idle_animation_frame,
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
        if self.app.controller.company is not None and view.roster_state == "Active":
            protection = hero_protection_line(self.app.controller.company, hero_id)
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
            return self.app._generic_action_detail(action)
        view = self._hero_sheet_detail_view(self.app.pending_gear_hero_id)
        if view is None:
            return action.preview or self.app._generic_action_detail(action)
        if action.value == "memories":
            return self._hero_quirks_memories_text(view)
        if action.value == "gear":
            return self._hero_gear_text(view)
        return self.app._generic_action_detail(action)

    def _hero_sheet_detail_view(self, hero_id: str) -> HeroSheetView | None:
        if not hero_id:
            return None
        result = self.app.controller.handle(ViewHeroSheet(hero_id))
        if not result.success or not isinstance(result.value, HeroSheetView):
            return None
        return result.value

    def _hero_sheet_quirk_line(self, view: HeroSheetView) -> str:
        if view.personal_quirk is not None:
            return f"Quirk: {view.personal_quirk.name}"
        if view.earned_quirks:
            names = ", ".join(quirk.name for quirk in view.earned_quirks[:2])
            suffix = (
                f" (+{len(view.earned_quirks) - 2} more)" if len(view.earned_quirks) > 2 else ""
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

    def show_ledger(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewLedger())
        if not result.success:
            self.app._show_main(result.error or "Start or load a company first.")
            return
        self.app._record_events(result.events)
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
        self.app._show_screen(
            "ledger",
            "Company Ledger",
            "\n".join(lines),
            (ScreenAction("1", "Back to Records Room", "back", ("b",), default=True),),
            message=message,
        )

    def show_memorial(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewMemorial())
        if not result.success:
            self.show_town(result.error or "Start or load a company first.")
            return
        self.app._record_events(result.events)
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
        self.app._show_screen(
            "memorial",
            "Memorial",
            body,
            (ScreenAction("1", "Back to Records Room", "back", ("b",), default=True),),
            message=message,
        )

    def show_recruiting(self, message: str = "") -> None:
        result = self.app.controller.handle(GenerateRecruitOffers())
        if not result.success:
            self.show_town_submenu("town_market", result.error or "Recruiting is unavailable.")
            return
        self.app._record_events(result.events)
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
        self.app._show_screen(
            "recruiting",
            "Recruiting",
            self._recruiting_text(view),
            actions,
            message=message,
            log=self.app._events_text(self.app.recent_events),
        )

    def show_recruiting_hire(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        view = self._current_recruiting_view()
        if view is None:
            self.show_town_submenu("town_market", "Start or load a company first.")
            return
        actions = self._selection_actions(view.actions, back_label="Back to Recruitment Desk")
        self.app._show_screen(
            "recruiting_hire",
            "Hire Recruit",
            self._recruiting_text(view),
            actions,
            message=message,
            log=self.app._result_log_text(self.app.recent_events, hci)
            if hci is not None
            else self.app._events_text(self.app.recent_events),
        )

    def _current_recruiting_view(self) -> RecruitOffersView | None:
        if self.app.controller.company is None:
            return None
        return build_recruit_offers_view(
            self.app.controller.company,
            self.app.controller.definitions,
            self.app.controller.recruit_offers,
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
            return self.app._generic_action_detail(action)
        view = self._current_recruiting_view()
        if view is None:
            return self.app._generic_action_detail(action)
        try:
            index = int(action.value)
        except ValueError:
            return self.app._generic_action_detail(action)
        if index < 0 or index >= len(view.offers):
            return self.app._generic_action_detail(action)
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

    def show_deep_surgery(self, message: str = "") -> None:
        if self.app.controller.company is None:
            self.show_town_submenu("town_recovery", "Start or load a company first.")
            return
        view = build_deep_surgery_view(self.app.controller.company, self.app.controller.definitions)
        self.app._show_screen(
            "deep_surgery",
            "Deep Surgery",
            self._deep_surgery_text(view),
            self._selection_actions(view.actions, back_label="Back to Recovery Ward"),
            message=message,
        )

    def show_supply_shop(self, message: str = "") -> None:
        if self.app.controller.company is None:
            self.show_town("Start or load a company first.")
            return
        view = build_supply_shop_view(self.app.controller.company, self.app.controller.definitions)
        self.app.current_supply_shop_view = view
        self._render_supply_shop_view(view, message)

    def _render_supply_shop_view(self, view: SupplyShopView, message: str = "") -> None:
        self.app.current_supply_shop_view = view
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
        self.app._show_screen(
            "supply_shop",
            "Quartermaster",
            self._supply_shop_text(view),
            actions,
            message=message,
            log=self.app._events_text(self.app.recent_events),
        )

    def show_supply_buy(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        if self.app.controller.company is None:
            self.show_town_submenu("town_market", "Start or load a company first.")
            return
        view = build_supply_shop_view(self.app.controller.company, self.app.controller.definitions)
        self.app.current_supply_shop_view = view
        self.app._show_screen(
            "supply_buy",
            "Buy Supplies",
            self._supply_shop_text(view, screen_id="supply_buy"),
            self._selection_actions(view.actions, back_label="Back to Quartermaster"),
            message=message,
            log=self.app._result_log_text(self.app.recent_events, hci)
            if hci is not None
            else self.app._events_text(self.app.recent_events),
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

    def show_relic_broker(self, message: str = "", hci: HciResultAnalysis | None = None) -> None:
        if self.app.controller.company is None:
            self.show_town_submenu("town_charter", "Start or load a company first.")
            return
        view = build_relic_broker_view(self.app.controller.company, self.app.controller.definitions)
        self.app.current_relic_broker_view = view
        self.app._show_screen(
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
            log=self.app._result_log_text(self.app.recent_events, hci)
            if hci is not None
            else self.app._events_text(self.app.recent_events),
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
            self.app._renumbered_action(
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

    def show_formation(self, message: str = "", *, return_to: str = "town_yard") -> None:
        self.app.pending_formation_return_state = return_to
        if self.app.controller.company is None:
            self.show_town("Start or load a company first.")
            return
        view = build_formation_view(self.app.controller.company, self.app.controller.definitions)
        self._render_formation_view(view, message)

    def _render_formation_view(self, view: FormationView, message: str = "") -> None:
        self.app._current_formation_view = view
        body = "\n".join(("Formation", "", self._formation_board_text(view)))
        self.app._show_screen(
            "formation",
            "Formation",
            body,
            view.actions,
            message=message,
            log=self.app._events_text(self.app.recent_events),
        )

    def show_assign_hero(self, slot_value: str) -> None:
        if self.app.controller.company is None:
            self.show_town("Start or load a company first.")
            return
        view = build_formation_view(self.app.controller.company, self.app.controller.definitions)
        selected_slot = next(
            (slot for slot in view.slots if slot.slot.value == slot_value),
            None,
        )
        if selected_slot is None:
            self.show_formation(
                "Choose a listed formation slot.",
                return_to=self.app.pending_formation_return_state,
            )
            return
        self.app.pending_slot = selected_slot.slot
        self.app.pending_slot_label = selected_slot.slot_label
        self.app._current_formation_view = view
        roster_by_id = {
            roster_hero.hero_id: roster_hero for roster_hero in self.app.controller.company.roster
        }
        slot_name = format_formation_slot(selected_slot.slot_label)
        actions = []
        for index, hero in enumerate(view.assignable_heroes, start=1):
            before, after = preview_assign_hero(
                self.app.controller.company.active_party_slots,
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
        actions.append(ScreenAction(str(len(actions) + 1), "Back to Formation", "back", ("b",)))
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
        self.app._show_screen(
            "assign_hero",
            "Assign Formation Slot",
            body,
            tuple(actions),
            log=self.app._events_text(self.app.recent_events),
        )

    def _roster_text(self, sections: tuple[RosterSectionView, ...]) -> str:
        living_count = sum(
            len(section.heroes) for section in sections if "memorial" not in section.title.lower()
        )
        memorial_count = sum(
            len(section.heroes) for section in sections if "memorial" in section.title.lower()
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
                    lines.append(f"- {format_compact_roster_row(hero, memorial=memorial)}")
        return "\n".join(lines).strip()

    def _gear_action_detail(self, action: ScreenAction) -> str:
        if action.value == "back":
            return self.app._generic_action_detail(action)
        view = self.app.current_gear_view
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
        return self.app._generic_action_detail(action)

    def _supply_action_detail(self, action: ScreenAction) -> str:
        if action.value == "back":
            return self.app._generic_action_detail(action)
        if action.value == "buy_supplies":
            return self.app._generic_action_detail(action)
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
            return self.app._generic_action_detail(action)
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
