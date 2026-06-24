"""Town, roster, gear, and supply action handlers for the TUI."""

from __future__ import annotations

from dataclasses import dataclass

from game.app.commands import (
    AcceptContract,
    BuySupply,
    EquipGear,
    HireRecruit,
    PerformDeepSurgery,
    PurchaseGear,
    PurchaseUpgrade,
    RecoverCompany,
    SellLoot,
    TurnInLoot,
    UnequipGear,
)
from game.ui.tui_handlers.protocol import TuiHandlerHost


@dataclass
class TownHandlers:
    app: TuiHandlerHost

    def handle_town_action(self, value: str) -> None:
        if value == "shallow_cave":
            self.app._regional_handlers.handle_travel(value)
        elif value in {"east_gate", "travel"}:
            self.app._regional_handlers.visit_east_gate_from_town()
        elif value in {
            "town_gate",
            "town_charter",
            "town_market",
            "town_recovery",
            "town_quartermaster",
            "town_recruitment",
            "town_yard",
            "town_upgrades",
            "town_records",
        }:
            self.app._show_town_submenu(value)
        elif value == "map":
            self.app._show_world_map()
        elif value == "gear":
            self.app._show_gear_locker(return_to="town_market")
        elif value == "system":
            self.app._show_system()

    def handle_pack_action(self, value: str) -> None:
        if value == "gear":
            self.app._show_gear_locker(return_to="pack")
        else:
            self.app._show_pack("Choose a listed pack action.")

    def handle_company_summary_action(self, value: str) -> None:
        if value == "formation":
            self.app._show_formation(return_to="company")
        elif value == "gear":
            self.app._show_gear_locker(return_to="company")
        elif value.startswith("hero:"):
            self.app._show_hero_sheet(value.removeprefix("hero:"), return_to="company")
        else:
            self.app._show_company_summary("Choose a listed company action.")

    def handle_town_submenu_action(self, value: str) -> None:
        if value in {
            "town_gate",
            "town_charter",
            "town_market",
            "town_recovery",
            "town_quartermaster",
            "town_recruitment",
            "town_yard",
            "town_upgrades",
            "town_records",
        }:
            self.app._show_town_submenu(value)
            return
        if value == "regional_map":
            self.app._show_regional_place(return_to="town_gate")
            return
        if value == "travel":
            self.app._show_regional_place(return_to="town_gate")
            return
        elif value == "map":
            self.app._show_world_map()
            return
        elif value == "gear":
            self.app._show_gear_locker(return_to="town_market")
            return
        if value == "expedition":
            self.app._show_expedition()
        elif value == "recruit":
            self.app._show_recruiting()
        elif value == "recover":
            result = self.app.controller.handle(RecoverCompany())
            if result.success:
                self.app._record_events(result.events)
                self.app._show_town_submenu(
                    "town_recovery",
                    "Company recovery funded.",
                    result.hci,
                )
            else:
                self.app._show_town_submenu(
                    "town_recovery",
                    result.error or "Recovery is unavailable.",
                )
        elif value == "deep_surgery":
            self.app._show_deep_surgery()
        elif value == "buy":
            self.app._show_supply_shop()
        elif value == "relic_broker":
            self.app._show_relic_broker()
        elif value == "formation":
            self.app._show_formation(return_to="town_yard")
        elif value == "memorial":
            self.app._show_memorial()
        elif value == "roster":
            self.app._show_roster()
        elif value == "ledger":
            self.app._show_ledger()
        elif value == "latest_record":
            self.app._show_expedition_report()
        elif value.startswith("upgrade:"):
            upgrade_id = value.removeprefix("upgrade:")
            result = self.app.controller.handle(PurchaseUpgrade(upgrade_id))
            if result.success:
                self.app._record_events(result.events)
                self.app._show_town_submenu("town_upgrades", "Upgrade installed.", result.hci)
            else:
                self.app._show_town_submenu(
                    "town_upgrades",
                    result.error or "Upgrade unavailable.",
                )
        elif value.startswith("accept:"):
            contract_id = value.removeprefix("accept:")
            result = self.app.controller.handle(AcceptContract(contract_id))
            if result.success:
                self.app._record_events(result.events)
                self.app._show_town_submenu("town_charter", "Contract accepted.", result.hci)
            else:
                self.app._show_town_submenu(
                    "town_charter",
                    result.error or "Contract is unavailable.",
                )

    def handle_deep_surgery_action(self, value: str) -> None:
        if value == "back":
            self.app._show_town_submenu("town_recovery")
            return
        if not value.startswith("surgery:"):
            self.app._show_deep_surgery("Choose a listed hero.")
            return
        hero_id = value.removeprefix("surgery:")
        result = self.app.controller.handle(PerformDeepSurgery(hero_id))
        if result.success:
            self.app._record_events(result.events)
            self.app._show_town_submenu(
                "town_recovery",
                "Deep surgery complete.",
                result.hci,
            )
        else:
            self.app._show_deep_surgery(result.error or "Deep surgery is unavailable.")

    def handle_roster_action(self, value: str) -> None:
        if value.startswith("hero:"):
            hero_id = value.removeprefix("hero:")
            self.app._show_hero_sheet(hero_id, return_to="roster")
        else:
            self.app._show_roster("Choose a listed hero.")

    def handle_gear_action(self, value: str) -> None:
        if value.startswith("gear:buy:"):
            gear_id = value.removeprefix("gear:buy:")
            result = self.app.controller.handle(PurchaseGear(gear_id))
        elif value.startswith("gear:equip:"):
            _prefix, _equip, hero_id, gear_id = value.split(":", 3)
            result = self.app.controller.handle(EquipGear(hero_id, gear_id))
        elif value.startswith("gear:unequip:"):
            hero_id = value.removeprefix("gear:unequip:")
            result = self.app.controller.handle(UnequipGear(hero_id))
        else:
            self.app._show_gear_locker(
                "Choose a listed gear action.",
                return_to=self.app.pending_gear_locker_return_state,
            )
            return
        if result.success:
            self.app._record_events(result.events)
            self.app._show_gear_locker(
                "Armory updated.",
                return_to=self.app.pending_gear_locker_return_state,
            )
        else:
            self.app._show_gear_locker(
                result.error or "Gear action unavailable.",
                return_to=self.app.pending_gear_locker_return_state,
            )

    def handle_hero_sheet_action(self, value: str) -> None:
        if value == "memories":
            self.app._show_hero_memories()
            return
        if value == "gear":
            self.app._show_hero_gear()
            return
        self.app._show_hero_sheet(
            message="Choose a listed sheet section.",
            return_to=self.app.pending_gear_return_state,
        )

    def handle_hero_gear_action(self, value: str) -> None:
        if value.startswith("gear:equip:"):
            _prefix, _equip, hero_id, gear_id = value.split(":", 3)
            result = self.app.controller.handle(EquipGear(hero_id, gear_id))
        elif value.startswith("gear:unequip:"):
            hero_id = value.removeprefix("gear:unequip:")
            result = self.app.controller.handle(UnequipGear(hero_id))
        else:
            self.app._show_hero_gear(message="Choose a listed gear action.")
            return
        if result.success:
            self.app._record_events(result.events)
            self.app._show_hero_gear(
                message="Equipment updated.",
            )
        else:
            self.app._show_hero_gear(
                message=result.error or "Equipment action unavailable.",
            )

    def handle_recruiting_action(self, value: str) -> None:
        if value == "hire":
            self.app._show_recruiting_hire()
            return

    def handle_recruiting_hire_action(self, value: str) -> None:
        result = self.app.controller.handle(HireRecruit(int(value)))
        if result.success:
            self.app._record_events(result.events)
            self.app._show_recruiting_hire("Recruit hired.", result.hci)
            return
        self.app._show_recruiting_hire(result.error or "Recruiting failed.")

    def handle_supply_action(self, value: str) -> None:
        if value == "buy_supplies":
            self.app._show_supply_buy()
            return

    def handle_supply_buy_action(self, value: str) -> None:
        result = self.app.controller.handle(BuySupply(value))
        if result.success:
            self.app._record_events(result.events)
            self.app._show_supply_buy("Supply purchased.", result.hci)
        else:
            self.app._show_supply_buy(result.error or "Purchase failed.")

    def handle_relic_broker_action(self, value: str) -> None:
        if value.startswith("sell_loot:"):
            result = self.app.controller.handle(SellLoot(value.removeprefix("sell_loot:")))
        elif value.startswith("turn_in_loot:"):
            result = self.app.controller.handle(TurnInLoot(value.removeprefix("turn_in_loot:")))
        else:
            self.app._show_relic_broker("Choose a listed relic action.")
            return
        if result.success:
            self.app._record_events(result.events)
            self.app._show_relic_broker(
                result.events[-1].message if result.events else "Relic clerk transaction complete.",
                result.hci,
            )
        else:
            self.app._show_relic_broker(result.error or "Relic clerk transaction failed.")

    def back_from_hero_sheet(self) -> None:
        if self.app.pending_gear_return_state == "pack":
            self.app._show_pack()
        elif self.app.pending_gear_return_state == "company":
            self.app._show_company_summary()
        else:
            self.app._show_roster()

    def back_from_hero_memories(self) -> None:
        self.app._show_hero_sheet(return_to=self.app.pending_gear_return_state)

    def back_from_hero_gear(self) -> None:
        self.app._show_hero_sheet(return_to=self.app.pending_gear_return_state)

    def back_from_gear_locker(self) -> None:
        if self.app.pending_gear_locker_return_state == "pack":
            self.app._show_pack()
        elif self.app.pending_gear_locker_return_state == "company":
            self.app._show_company_summary()
        elif self.app.pending_gear_locker_return_state == "main":
            self.app._show_main()
        else:
            self.app._show_town_submenu("town_market")

    def back_from_formation(self) -> None:
        if self.app.pending_formation_return_state == "company":
            self.app._show_company_summary()
        else:
            self.app._show_town_submenu("town_yard")

    def back_from_assign_hero(self) -> None:
        self.app._show_formation(return_to=self.app.pending_formation_return_state)

    def back_from_recruiting_hire(self) -> None:
        view = self.app._current_recruiting_view()
        if view is None:
            self.app._show_town_submenu("town_market")
        else:
            self.app._render_recruiting_view(view)
