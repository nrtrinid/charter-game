"""Town and formation Textual widgets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from textual.widgets import Static

from game.app.views import (
    FormationView,
    GearInventoryView,
    RelicBrokerView,
    SupplyShopView,
    TownDashboardView,
)
from game.ui.hci_text import (
    format_compact_roster_row,
    format_equipped_kit_rows,
    format_fixed_table,
    format_formation_board_cell,
    format_formation_slot,
    format_gear_shop_rows,
    format_gear_stock_rows,
    format_quantity_rows,
    format_scene_body,
    format_supply_stock_rows,
)
from game.ui.screens import EventBeat
from game.ui.tui_widgets.combat import _grid_cell, _grid_text, _lines_or_none, _mini_side_rows
from game.ui.tui_widgets.constants import PARTY_COMBAT_ROWS
from game.ui.tui_widgets.dungeon import (
    _contract_summary_line,
    _objective_lines,
    _upgrade_summary_line,
)
from game.ui.tui_widgets.shell import format_meta_line


class ExpeditionProgressStrip(Static):
    """Compact room/node progress for expedition playback."""

    def update_progress(self, beats: Sequence[EventBeat], playback_index: int) -> None:
        self.update(self.render_text(beats, playback_index))

    @staticmethod
    def render_text(beats: Sequence[EventBeat], playback_index: int) -> str:
        if not beats:
            return "Expedition Progress\n(no route)"
        lines = ["Expedition Progress"]
        for index, beat in enumerate(beats):
            if index < playback_index:
                marker = "[x]"
            elif index == playback_index:
                marker = "[>]"
            else:
                marker = "[ ]"
            kind = "combat" if beat.combat else beat.title.lower()
            lines.append(f"{marker} {index + 1:02}. {beat.title} ({kind})")
        return "\n".join(lines)


class FormationBoard(Static):
    """Reusable 2x2 party formation board."""

    def update_formation(self, view: FormationView, *, focus_slot: str = "") -> None:
        self.update(self.render_text(view, focus_slot=focus_slot))

    @staticmethod
    def render_text(view: FormationView, *, focus_slot: str = "") -> str:
        cells = {}
        for slot in view.slots:
            name_line, detail_line, state_line = format_formation_board_cell(slot)
            cells[slot.slot_label] = _grid_cell(
                name_line,
                detail_line,
                state_line,
                marker=">" if slot.slot_label == focus_slot else " ",
            )
        return "Party Formation\n" + _grid_text(cells)

    @staticmethod
    def render_mini_text(
        portrait_actors_by_slot: Mapping[str, Any],
        *,
        focus_slot: str = "",
        focus_hero_id: str = "",
        idle_frame: int = 0,
        inward_facing: bool = False,
    ) -> str:
        slot_labels = {
            "BACK_LEFT": format_formation_slot("BACK_LEFT"),
            "BACK_RIGHT": format_formation_slot("BACK_RIGHT"),
            "FRONT_LEFT": format_formation_slot("FRONT_LEFT"),
            "FRONT_RIGHT": format_formation_slot("FRONT_RIGHT"),
        }
        lines = ["Party Formation"]
        for row in PARTY_COMBAT_ROWS:
            lines.append(
                format_meta_line(
                    format_formation_slot(row[0]),
                    format_formation_slot(row[1]),
                )
            )
            lines.extend(
                _mini_side_rows(
                    portrait_actors_by_slot,
                    rows=(row,),
                    focus_id=focus_hero_id,
                    legal_ids=set(),
                    highlight_ids={focus_hero_id} if focus_hero_id else set(),
                    intents={},
                    source_ids=set(),
                    slot_annotations=slot_labels,
                    idle_frame=idle_frame,
                    turn_flash_actor_id="",
                    turn_flash_frame=0,
                    inward_facing=inward_facing,
                )
            )
        return "\n".join(lines)


class TownDashboardPanel(Static):
    """Haven town dashboard summary."""

    def update_dashboard(
        self,
        view: TownDashboardView,
        hero_lines: str,
        reserve_lines: str,
    ) -> None:
        self.update(self.render_text(view, hero_lines=hero_lines, reserve_lines=reserve_lines))

    @staticmethod
    def render_text(
        view: TownDashboardView,
        *,
        hero_lines: str,
        reserve_lines: str,
    ) -> str:
        resource_line = format_meta_line(
            f"Reputation: {view.reputation}",
            f"Coin: {view.coin}",
            f"Roster cap: {view.roster_cap}",
        )
        roster_line = format_meta_line(
            f"Active: {view.active_count}",
            f"Reserves: {view.reserve_count}",
        )
        condition_line = format_meta_line(
            f"Wounded: {view.wounded_count}",
            f"Downed: {view.downed_count}",
            f"Memorial: {view.deceased_count}",
        )
        contract_lines = "\n".join(_contract_summary_line(entry) for entry in view.contract_board)
        upgrade_lines = "\n".join(_upgrade_summary_line(entry) for entry in view.upgrades)
        return (
            "Haven Town\n"
            f"{view.company_name} at {view.location}\n\n"
            "Company Status\n"
            f"{resource_line}\n"
            f"{roster_line}\n"
            f"{condition_line}\n\n"
            "Current Objective\n"
            f"{_lines_or_none(_objective_lines(view.objective))}\n\n"
            "Posted Contracts\n"
            f"{contract_lines or 'none'}\n\n"
            "Company Upgrades\n"
            f"{upgrade_lines or 'none'}\n\n"
            "Active Party\n"
            f"{hero_lines}\n\n"
            "Reserves\n"
            f"{reserve_lines}"
        )


class YardPanel(Static):
    """Compact company yard summary."""

    @staticmethod
    def render_text(
        view: TownDashboardView,
        *,
        formation_text: str = "",
        hint: str = "",
    ) -> str:
        roster_lines = tuple(
            format_compact_roster_row(hero) for hero in (*view.active_party, *view.reserves)
        )
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Status",
                (
                    format_meta_line(
                        f"Active {view.active_count}",
                        f"Reserves {view.reserve_count}",
                        f"Cap {view.roster_cap}",
                    ),
                ),
            ),
        ]
        if formation_text:
            sections.append(("Formation", (formation_text,)))
        sections.append(
            (
                "Roster",
                roster_lines or ("none",),
            )
        )
        return format_scene_body("Company Yard", tuple(sections), hint=hint)


class PackPanel(Static):
    """Carried supplies, items, gear, and equipped kits."""

    @staticmethod
    def render_text(
        supplies: dict[str, int],
        inventory: dict[str, int],
        gear: GearInventoryView,
        *,
        hint: str = "",
    ) -> str:
        sections = [
            (
                "Supplies",
                (format_fixed_table(("Name", "Qty"), format_quantity_rows(supplies)),)
                if supplies
                else ("none",),
            ),
            (
                "Items",
                (format_fixed_table(("Name", "Qty"), format_quantity_rows(inventory)),)
                if inventory
                else ("none",),
            ),
            (
                "Gear",
                (
                    format_fixed_table(
                        ("Kit", "Own", "Free", "Eq"),
                        format_gear_stock_rows(gear.items),
                    ),
                )
                if format_gear_stock_rows(gear.items)
                else ("none",),
            ),
            (
                "Equipped",
                (
                    format_fixed_table(
                        ("Hero", "Kit"),
                        format_equipped_kit_rows(gear.heroes),
                    ),
                )
                if gear.heroes
                else ("none",),
            ),
        ]
        if not gear.can_manage and gear.manage_reason:
            sections.append(("Note", (gear.manage_reason,)))
        return format_scene_body("Pack", tuple(sections), hint=hint)


class GearLockerPanel(Static):
    """Armory stock and equipped kits."""

    @staticmethod
    def render_text(view: GearInventoryView, *, hint: str = "") -> str:
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Status",
                (
                    format_meta_line(
                        f"Reputation {view.reputation}",
                        f"Coin {view.coin}",
                    ),
                    "Purchases: " + ("available" if view.can_purchase else view.purchase_reason),
                ),
            ),
            (
                "Company Gear",
                (
                    format_fixed_table(
                        ("Kit", "State", "Cost"),
                        format_gear_shop_rows(view.items),
                    ),
                )
                if view.items
                else ("none",),
            ),
            (
                "Equipped",
                (
                    format_fixed_table(
                        ("Hero", "Kit"),
                        format_equipped_kit_rows(view.heroes),
                    ),
                )
                if view.heroes
                else ("none",),
            ),
        ]
        return format_scene_body("Armory", tuple(sections), hint=hint)


class RelicBrokerPanel(Static):
    """Relic clerk buy/file list."""

    @staticmethod
    def render_text(view: RelicBrokerView, *, hint: str = "") -> str:
        inventory = ", ".join(f"{item_id} x{quantity}" for item_id, quantity in view.inventory)
        offer_rows = [
            (
                action.label.removeprefix("Sell ").removeprefix("File "),
                action.description or action.preview or "",
                "ready" if action.enabled else "blocked",
            )
            for action in view.actions
            if action.value != "back"
        ]
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Ledger",
                (
                    f"Coin {view.coin}",
                    f"Inventory {inventory or 'none'}",
                ),
            ),
            (
                "Offers",
                (format_fixed_table(("Item", "Terms", "State"), offer_rows),)
                if offer_rows
                else ("No sellable or fileable relics are catalogued.",),
            ),
        ]
        return format_scene_body("Relic Clerk", tuple(sections), hint=hint)


class SupplyShopPanel(Static):
    """Quartermaster stock list."""

    @staticmethod
    def render_text(view: SupplyShopView, *, hint: str = "") -> str:
        stock_rows = format_supply_stock_rows(view.actions)
        sections: list[tuple[str, Sequence[str]]] = [
            (
                "Budget",
                (
                    f"Coin {view.coin}",
                    "Purchase quantity: 1",
                ),
            ),
            (
                "Stock",
                (format_fixed_table(("Supply", "Cost", "State"), stock_rows),)
                if stock_rows
                else ("none",),
            ),
        ]
        return format_scene_body("Quartermaster", tuple(sections), hint=hint)


class CompanyPanel(Static):
    """Slim charter company overview."""

    @staticmethod
    def render_text(
        company_name: str,
        objective_line: str,
        formation_text: str,
        roster_lines: Sequence[str],
        *,
        hint: str = "",
    ) -> str:
        sections: list[tuple[str, Sequence[str]]] = [
            ("Objective", (objective_line,)),
            ("Formation", (formation_text,)),
            (
                "Characters",
                tuple(roster_lines) or ("none",),
            ),
        ]
        return format_scene_body(f"Company — {company_name}", tuple(sections), hint=hint)
