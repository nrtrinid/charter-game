"""Application orchestration flows used by AppController."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game.app.commands import (
    AcceptContract,
    AssignActiveHero,
    BuySupply,
    EquipGear,
    GenerateRecruitOffers,
    HireRecruit,
    MarkRegionalRoute,
    MoveRegional,
    PerformDeepSurgery,
    PurchaseGear,
    PurchaseUpgrade,
    RecoverCompany,
    ReturnToHavenTown,
    SellLoot,
    TravelRegional,
    TurnInLoot,
    UnequipGear,
    UseRegionalAction,
    ViewGear,
    ViewHeroSheet,
    ViewLedger,
    ViewMemorial,
    ViewRegionalMap,
    ViewRoster,
    ViewSupplies,
    ViewTown,
    ViewWorld,
    VisitEastGate,
)
from game.app.contracts import contract_unavailable_reason
from game.app.flows.base import ControllerFlow
from game.app.manual_combat import (
    start_manual_session,
)
from game.app.views import (
    build_combat_view,
    build_gear_inventory_view,
    build_hero_sheet_view,
    build_memorial_entries,
    build_recruit_offers_view,
    build_regional_map_view,
    build_relic_broker_view,
    build_roster_sections,
    build_town_dashboard,
    build_world_view,
)
from game.campaign.company import CompanyState, RecruitmentOfferState, contract_record
from game.campaign.gear import equip_gear, purchase_gear, unequip_gear
from game.campaign.loot import sell_loot, turn_in_loot
from game.campaign.recruitment import RecruitChoice, generate_recruit_choices
from game.campaign.town import (
    assign_active_hero,
    buy_supply,
    deep_surgery,
    hire_recruit,
    ledger,
    purchase_upgrade,
    recover_company,
    town_service_event,
)
from game.core.events import (
    CompanyEvent,
)
from game.core.result import Result
from game.expedition.cave import create_encounter_combat
from game.expedition.dungeon import (
    active_dungeon_nodes,
)
from game.expedition.travel import (
    REGIONAL_CHARTED_HOP_IDS,
    REGIONAL_CHARTED_HOP_NODE_BY_WORLD_ID,
    REGIONAL_EAST_GATE_NODE_ID,
    REGIONAL_OVERWORLD_NODE_IDS,
    mark_regional_charted_route,
    move_regional_node,
    regional_overworld_nodes,
    regional_travel_flavor,
    set_company_location,
    set_company_node_location,
    set_regional_node_id,
    spend_ration,
    use_regional_action,
)


@dataclass
class TownFlow(ControllerFlow):
    def handle(self, command: object) -> Result[Any]:
        if isinstance(command, ViewRoster):
            return self._require_company(
                lambda company: Result.ok(build_roster_sections(company, self.definitions))
            )
        if isinstance(command, ViewSupplies):
            return self._require_company(lambda company: Result.ok(company.supplies))
        if isinstance(command, ViewGear):
            return self._require_company(
                lambda company: Result.ok(
                    build_gear_inventory_view(
                        company,
                        self.definitions,
                        can_manage=self._can_manage_gear(company),
                        can_purchase=self._can_purchase_gear(company),
                        manage_reason=self._gear_manage_reason(company),
                        purchase_reason=self._gear_purchase_reason(company),
                    ),
                    [town_service_event("gear", "The company armory is reviewed.")],
                )
            )
        if isinstance(command, ViewHeroSheet):
            return self._require_company(
                lambda company: self._view_hero_sheet(company, command.hero_id)
            )
        if isinstance(command, ViewLedger):
            return self._require_company(
                lambda company: Result.ok(
                    ledger(company, self.definitions),
                    [town_service_event("ledger", "The company ledger is reviewed.")],
                )
            )
        if isinstance(command, ViewTown):
            return self._require_company(
                lambda company: Result.ok(
                    build_town_dashboard(company, self.definitions),
                    [town_service_event("town", "The Haven services are reviewed.")],
                )
            )
        if isinstance(command, ViewWorld):
            return self._require_company(
                lambda company: Result.ok(build_world_view(company, self.definitions))
            )
        if isinstance(command, ViewRegionalMap):
            return self._require_company(
                lambda company: Result.ok(build_regional_map_view(company, self.definitions))
            )
        if isinstance(command, TravelRegional):
            return self._travel_regional(command.destination_id)
        if isinstance(command, MoveRegional):
            return self._move_regional(command.node_id)
        if isinstance(command, UseRegionalAction):
            return self._use_regional_action(command.action_id)
        if isinstance(command, MarkRegionalRoute):
            return self._mark_regional_route()
        if isinstance(command, VisitEastGate):
            return self._visit_east_gate()
        if isinstance(command, ReturnToHavenTown):
            return self._return_to_haven_town()
        if isinstance(command, ViewMemorial):
            return self._require_company(
                lambda company: Result.ok(
                    build_memorial_entries(company),
                    [town_service_event("memorial", "The memorial ledger is opened.")],
                )
            )
        if isinstance(command, GenerateRecruitOffers):
            return self._generate_recruit_offers()
        if isinstance(command, HireRecruit):
            return self._hire_recruit(command)
        if isinstance(command, RecoverCompany):
            return self._require_company(lambda company: recover_company(company, self.definitions))
        if isinstance(command, PerformDeepSurgery):
            return self._require_company(
                lambda company: deep_surgery(company, self.definitions, command.hero_id)
            )
        if isinstance(command, BuySupply):
            return self._require_company(
                lambda company: buy_supply(
                    company,
                    self.definitions,
                    command.supply_id,
                    command.quantity,
                )
            )
        if isinstance(command, SellLoot):
            return self._require_company(
                lambda company: self._sell_loot(company, command.item_id)
            )
        if isinstance(command, TurnInLoot):
            return self._require_company(
                lambda company: self._turn_in_loot(company, command.item_id)
            )
        if isinstance(command, PurchaseUpgrade):
            return self._require_company(
                lambda company: purchase_upgrade(
                    company,
                    self.definitions,
                    command.upgrade_id,
                )
            )
        if isinstance(command, PurchaseGear):
            return self._purchase_gear(command)
        if isinstance(command, EquipGear):
            return self._equip_gear(command)
        if isinstance(command, UnequipGear):
            return self._unequip_gear(command)
        if isinstance(command, AssignActiveHero):
            return self._require_company(
                lambda company: assign_active_hero(company, command.hero_id, command.slot)
            )
        if isinstance(command, AcceptContract):
            return self._accept_contract(command.contract_id)
        return Result.fail(f"Unsupported town command: {command!r}")

    def _view_hero_sheet(self, company: CompanyState, hero_id: str) -> Result[Any]:
        view = build_hero_sheet_view(
            company,
            self.definitions,
            hero_id,
            can_manage_gear=self._can_manage_gear(company),
            gear_manage_reason=self._gear_manage_reason(company),
        )
        if view is None:
            return Result.fail("Hero not found.")
        return Result.ok(view)

    def _purchase_gear(self, command: PurchaseGear) -> Result[Any]:
        def purchase(company: CompanyState) -> Result[Any]:
            reason = self._gear_purchase_reason(company)
            if reason:
                return Result.fail(reason)
            result = purchase_gear(company, self.definitions, command.gear_id)
            if result.success:
                return Result.ok(
                    build_gear_inventory_view(
                        company,
                        self.definitions,
                        can_manage=self._can_manage_gear(company),
                        can_purchase=self._can_purchase_gear(company),
                        manage_reason=self._gear_manage_reason(company),
                        purchase_reason=self._gear_purchase_reason(company),
                    ),
                    result.events,
                )
            return result

        return self._require_company(purchase)

    def _equip_gear(self, command: EquipGear) -> Result[Any]:
        def equip(company: CompanyState) -> Result[Any]:
            reason = self._gear_manage_reason(company)
            if reason:
                return Result.fail(reason)
            result = equip_gear(company, self.definitions, command.hero_id, command.gear_id)
            if result.success:
                return Result.ok(
                    build_gear_inventory_view(
                        company,
                        self.definitions,
                        can_manage=self._can_manage_gear(company),
                        can_purchase=self._can_purchase_gear(company),
                        manage_reason=self._gear_manage_reason(company),
                        purchase_reason=self._gear_purchase_reason(company),
                    ),
                    result.events,
                )
            return result

        return self._require_company(equip)

    def _unequip_gear(self, command: UnequipGear) -> Result[Any]:
        def unequip(company: CompanyState) -> Result[Any]:
            reason = self._gear_manage_reason(company)
            if reason:
                return Result.fail(reason)
            result = unequip_gear(company, self.definitions, command.hero_id)
            if result.success:
                return Result.ok(
                    build_gear_inventory_view(
                        company,
                        self.definitions,
                        can_manage=self._can_manage_gear(company),
                        can_purchase=self._can_purchase_gear(company),
                        manage_reason=self._gear_manage_reason(company),
                        purchase_reason=self._gear_purchase_reason(company),
                    ),
                    result.events,
                )
            return result

        return self._require_company(unequip)

    def _can_manage_gear(self, company: CompanyState) -> bool:
        return not self._gear_manage_reason(company)

    def _can_purchase_gear(self, company: CompanyState) -> bool:
        return not self._gear_purchase_reason(company)

    def _gear_manage_reason(self, company: CompanyState) -> str:
        if self.manual_combat is not None:
            return "Gear can only be inspected during combat."
        session = company.active_expedition
        if session is None:
            return ""
        nodes = active_dungeon_nodes(self.definitions, session)
        node = nodes.get(session.current_node_id)
        if node is not None and node.safe_return:
            return ""
        return "Gear can only be changed in Haven, on the road, or at safe return rooms."

    def _gear_purchase_reason(self, company: CompanyState) -> str:
        if self.manual_combat is not None:
            return "Gear purchases are only available in Haven."
        if company.active_expedition is not None:
            return "Gear purchases are only available in Haven."
        if str(company.town_state.get("location_id") or "haven") != "haven":
            return "Gear purchases are only available in Haven."
        return ""

    def _sell_loot(self, company: CompanyState, item_id: str) -> Result[Any]:
        result = sell_loot(company, self.definitions, item_id)
        if not result.success:
            return result
        return Result.ok(build_relic_broker_view(company, self.definitions), result.events)

    def _turn_in_loot(self, company: CompanyState, item_id: str) -> Result[Any]:
        result = turn_in_loot(company, self.definitions, item_id)
        if not result.success:
            return result
        return Result.ok(build_relic_broker_view(company, self.definitions), result.events)

    def _accept_contract(self, contract_id: str) -> Result[Any]:
        def accept(company: CompanyState) -> Result[Any]:
            if contract_id not in self.definitions.contracts:
                return Result.fail(f"Unknown contract: {contract_id}")
            unavailable = self._contract_unavailable_reason(company, contract_id)
            if unavailable:
                return Result.fail(unavailable)
            contract = self.definitions.contracts[contract_id]
            company.active_contract_ids.add(contract_id)
            record = contract_record(company, contract_id)
            record.state = "active"
            record.accepted_count += 1
            return Result.ok(
                build_town_dashboard(company, self.definitions),
                [
                    CompanyEvent(
                        message=f"Contract accepted - {contract.name}.",
                    )
                ],
            )

        return self._require_company(accept)

    def _contract_unavailable_reason(
        self,
        company: CompanyState,
        contract_id: str,
    ) -> str:
        return contract_unavailable_reason(company, self.definitions, contract_id)

    def _generate_recruit_offers(self) -> Result[Any]:
        return self._require_company(
            lambda company: self._store_recruit_offers(
                company,
                generate_recruit_choices(
                    self.definitions,
                    self.rng,
                    count=self.definitions.town.recruit_offer_count,
                ),
            )
        )

    def _store_recruit_offers(
        self,
        company: CompanyState,
        offers: list[RecruitChoice],
    ) -> Result[Any]:
        self.recruit_offers = offers
        company.recruitment_state.current_offers = [
            RecruitmentOfferState(
                name=offer.name,
                class_id=offer.class_id,
                background=offer.background,
                motive=offer.motive,
            )
            for offer in offers
        ]
        company.recruitment_state.refresh_count += 1
        return Result.ok(
            build_recruit_offers_view(company, self.definitions, offers),
            [
                town_service_event(
                    "recruitment",
                    f"{len(offers)} recruits are available at the desk.",
                )
            ],
        )

    def _hire_recruit(self, command: HireRecruit) -> Result[Any]:
        if self.company is not None:
            self.recruit_offers = [
                RecruitChoice(
                    name=offer.name,
                    class_id=offer.class_id,
                    background=offer.background,
                    motive=offer.motive,
                )
                for offer in self.company.recruitment_state.current_offers
            ]
        if command.offer_index < 0 or command.offer_index >= len(self.recruit_offers):
            return Result.fail("Choose one of the current recruit offers.")
        recruit = self.recruit_offers[command.offer_index]
        result = self._require_company(
            lambda company: hire_recruit(company, self.definitions, recruit)
        )
        if result.success:
            del self.recruit_offers[command.offer_index]
            if self.company is not None:
                del self.company.recruitment_state.current_offers[command.offer_index]
        return result

    def _travel_regional(self, destination_id: str) -> Result[Any]:
        def travel(company: CompanyState) -> Result[Any]:
            if destination_id not in REGIONAL_CHARTED_HOP_IDS:
                return Result.fail("Choose Haven or Shallow Cave.")
            if "shallow_cave" not in company.known_route_ids:
                return Result.fail(
                    "Chart the route by travelling the opening expedition first."
                )
            anchor_node_id = REGIONAL_CHARTED_HOP_NODE_BY_WORLD_ID[destination_id]
            current_node_id = build_regional_map_view(
                company,
                self.definitions,
            ).current_node_id
            if anchor_node_id == current_node_id:
                return Result.fail("The company is already there.")
            spend_ration(company.supplies)
            location = self.definitions.locations[destination_id]
            set_company_location(company, location.id, location.name)
            set_regional_node_id(company, anchor_node_id)
            flavor = regional_travel_flavor(
                origin_id=current_node_id,
                destination_id=destination_id,
            )
            event = CompanyEvent(message=flavor)
            return Result.ok(
                build_regional_map_view(
                    company,
                    self.definitions,
                    travel_flavor=flavor,
                ),
                [event],
            )

        return self._require_company(travel)

    def _move_regional(self, node_id: str) -> Result[Any]:
        def move(company: CompanyState) -> Result[Any]:
            if node_id not in REGIONAL_OVERWORLD_NODE_IDS:
                return Result.fail("Choose a listed regional exit.")
            try:
                events = move_regional_node(company, self.definitions, node_id)
            except ValueError as exc:
                return Result.fail(str(exc))
            pending_node_id = company.town_state.get("pending_regional_combat_node_id")
            if pending_node_id:
                nodes = regional_overworld_nodes(self.definitions)
                destination = nodes[str(pending_node_id)]
                combat = create_encounter_combat(
                    company,
                    self.definitions,
                    destination.encounter or "",
                )
                self.manual_combat, combat_events = start_manual_session(
                    destination.encounter or "",
                    destination.name,
                    combat,
                    self.definitions,
                    self.rng,
                    enemy_ai_mode=self.controller.enemy_ai_mode,
                )
                self.controller.return_to_regional_place = True
                events.extend(combat_events)
                return Result.ok(
                    build_combat_view(
                        self.manual_combat,
                        self.definitions,
                        retreat_available=False,
                        debug_combat_preview=self.debug_combat_preview,
                    ),
                    events,
                )
            return Result.ok(
                build_regional_map_view(company, self.definitions),
                events,
            )

        return self._require_company(move)

    def _use_regional_action(self, action_id: str) -> Result[Any]:
        def use_action(company: CompanyState) -> Result[Any]:
            try:
                events = use_regional_action(company, self.definitions, action_id)
            except ValueError as exc:
                return Result.fail(str(exc))
            return Result.ok(
                build_regional_map_view(company, self.definitions),
                events,
            )

        return self._require_company(use_action)

    def _mark_regional_route(self) -> Result[Any]:
        def mark_route(company: CompanyState) -> Result[Any]:
            try:
                events = mark_regional_charted_route(company, self.definitions)
            except ValueError as exc:
                return Result.fail(str(exc))
            return Result.ok(
                build_regional_map_view(company, self.definitions),
                events,
            )

        return self._require_company(mark_route)

    def _visit_east_gate(self) -> Result[Any]:
        def visit(company: CompanyState) -> Result[Any]:
            if company.active_expedition is not None:
                return Result.fail("Finish the active expedition first.")
            nodes = regional_overworld_nodes(self.definitions)
            node = nodes[REGIONAL_EAST_GATE_NODE_ID]
            set_regional_node_id(company, REGIONAL_EAST_GATE_NODE_ID)
            set_company_node_location(company, node)
            return Result.ok(
                build_regional_map_view(company, self.definitions),
                [CompanyEvent(message="The company steps out to East Gate.")],
            )

        return self._require_company(visit)

    def _return_to_haven_town(self) -> Result[Any]:
        def return_town(company: CompanyState) -> Result[Any]:
            set_company_location(company, "haven", "Haven Town")
            return Result.ok(
                build_town_dashboard(company, self.definitions),
                [CompanyEvent(message="Returned to Haven Town.")],
            )

        return self._require_company(return_town)
