"""Application orchestration flows used by AppController."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from game.app.commands import (
    AcceptContract,
    AssignActiveHero,
    BuySupply,
    ChooseCombatSkill,
    ChooseCombatTarget,
    ClearExpeditionReport,
    DelayCombatTurn,
    EnterGeneratedMaze,
    EquipGear,
    GenerateRecruitOffers,
    HireRecruit,
    InspectDungeonRoom,
    MarkRegionalRoute,
    MoveCombatActor,
    MoveDungeon,
    MoveRegional,
    PassCombatTurn,
    PerformDeepSurgery,
    PurchaseGear,
    PurchaseUpgrade,
    RecoverCompany,
    ResolveCombatAction,
    ResolveCombatReaction,
    RetraceGeneratedMaze,
    Retreat,
    RetreatCombat,
    RetreatGeneratedMaze,
    ReturnFromDungeon,
    ReturnToHavenTown,
    SelectTarget,
    SellLoot,
    StartExpedition,
    StartManualCombat,
    TakeExpeditionChoice,
    TravelRegional,
    TurnInLoot,
    UnequipGear,
    UseCombatSkill,
    UseDungeonAction,
    UseRegionalAction,
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
    VisitEastGate,
    WithdrawGeneratedMaze,
)
from game.app.contracts import contract_unavailable_reason
from game.app.manual_combat import (
    ManualCombatSession,
    legal_skill_ids,
    legal_target_ids,
    resolve_enemy_reaction,
    resolve_hero_action,
    resolve_hero_delay,
    resolve_hero_move,
    resolve_hero_pass,
    resolve_hero_retreat,
    start_manual_session,
)
from game.app.views import (
    build_combat_view,
    build_dungeon_view,
    build_expedition_report_view,
    build_gear_inventory_view,
    build_hero_sheet_view,
    build_memorial_entries,
    build_recruit_offers_view,
    build_regional_arrival_context,
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
from game.campaign.roster import active_roster, sync_company_from_combat
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
from game.content.definitions import GameDefinitions
from game.core.events import (
    CombatRetreatedEvent,
    CompanyEvent,
    ExpeditionEvent,
    ExpeditionReturnedEvent,
    GameEvent,
)
from game.core.result import Result
from game.core.rng import GameRng
from game.expedition.cave import create_cave_boss_combat, create_encounter_combat
from game.expedition.dungeon import (
    SHALLOW_CAVE_BOSS_NODE_ID,
    active_dungeon_nodes,
    active_report,
    enter_dungeon_node,
    enter_generated_maze,
    finish_report,
    finish_shallow_cave_boss,
    generated_maze_frontier_exit_ids,
    mark_pending_combat_cleared,
    move_generated_maze_if_needed,
    open_opening_breach_room,
    record_events,
    retrace_generated_maze,
    return_from_dungeon,
    revealed_exit_node_ids,
    start_interactive_opening_dungeon,
    use_dungeon_action,
    withdraw_generated_maze,
)
from game.expedition.expedition import (
    OPENING_BREACH_PENDING_FLAG,
    OPENING_EXPEDITION_ID,
    descend_from_breach,
    return_to_haven_from_breach,
    run_opening_route,
)
from game.expedition.travel import (
    REGIONAL_CHARTED_HOP_IDS,
    REGIONAL_CHARTED_HOP_NODE_BY_WORLD_ID,
    REGIONAL_EAST_GATE_NODE_ID,
    REGIONAL_OVERWORLD_NODE_IDS,
    apply_node_rewards,
    event_for_node,
    mark_regional_charted_route,
    mark_regional_combat_cleared,
    move_regional_node,
    opening_nodes,
    regional_overworld_nodes,
    regional_return_flavor,
    regional_travel_flavor,
    set_company_location,
    set_company_node_location,
    set_regional_node_id,
    spend_ration,
    use_regional_action,
)

MANUAL_STAGE_SHALLOW_CAVE = "shallow_cave"
MANUAL_STAGE_CAVE_BOSS = "cave_mini_boss"


@dataclass
class ControllerFlow:
    controller: Any

    @property
    def definitions(self) -> GameDefinitions:
        return self.controller.definitions

    @property
    def rng(self) -> GameRng:
        return self.controller.rng

    @property
    def company(self) -> CompanyState | None:
        return self.controller.company

    @company.setter
    def company(self, value: CompanyState | None) -> None:
        self.controller.company = value

    @property
    def recruit_offers(self) -> list[RecruitChoice]:
        return self.controller.recruit_offers

    @recruit_offers.setter
    def recruit_offers(self, value: list[RecruitChoice]) -> None:
        self.controller.recruit_offers = value

    @property
    def manual_combat(self) -> ManualCombatSession | None:
        return self.controller.manual_combat

    @manual_combat.setter
    def manual_combat(self, value: ManualCombatSession | None) -> None:
        self.controller.manual_combat = value

    @property
    def opening_manual_stage(self) -> str | None:
        return self.controller.opening_manual_stage

    @opening_manual_stage.setter
    def opening_manual_stage(self, value: str | None) -> None:
        self.controller.opening_manual_stage = value

    @property
    def debug_combat_preview(self) -> bool:
        return self.controller.debug_combat_preview

    @property
    def dungeon_first_visit_node_id(self) -> str | None:
        return self.controller.dungeon_first_visit_node_id

    @dungeon_first_visit_node_id.setter
    def dungeon_first_visit_node_id(self, value: str | None) -> None:
        self.controller.dungeon_first_visit_node_id = value

    def _require_company(self, action: Callable[[CompanyState], Result[Any]]) -> Result[Any]:
        return self.controller._require_company(action)

    def _current_first_visit_node_id(self, company: CompanyState) -> str | None:
        session = company.active_expedition
        if session is None:
            return None
        if self.dungeon_first_visit_node_id == session.current_node_id:
            return self.dungeon_first_visit_node_id
        return None

    def _remember_dungeon_entry(self, company: CompanyState, events: list[GameEvent]) -> None:
        session = company.active_expedition
        if session is None:
            self.dungeon_first_visit_node_id = None
            return
        for event in events:
            if isinstance(event, ExpeditionEvent) and event.node_id == session.current_node_id:
                self.dungeon_first_visit_node_id = event.node_id if event.first_visit else None
                return

    def _pending_generated_combat(self) -> bool:
        if self.company is None or self.company.active_expedition is None:
            return False
        session = self.company.active_expedition
        generated = session.generated_dungeon
        if generated is None or generated.collapsed:
            return False
        return session.pending_combat_node_id in {node.id for node in generated.nodes}

    def _latest_safe_return_node_id(self) -> str:
        if self.company is None or self.company.active_expedition is None:
            return "town_gate"
        session = self.company.active_expedition
        nodes = active_dungeon_nodes(self.definitions, session)
        for node_id in reversed(session.visited_node_ids):
            node = nodes.get(node_id)
            if node is not None and node.safe_return:
                return node_id
        node = nodes.get(session.current_node_id)
        if node is not None and node.safe_return:
            return session.current_node_id
        return "town_gate"


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


@dataclass
class ExpeditionFlow(ControllerFlow):
    def handle(self, command: object) -> Result[Any]:
        if isinstance(command, StartExpedition):
            return self._start_expedition(command)
        if isinstance(command, TakeExpeditionChoice):
            return self._take_expedition_choice(command)
        return Result.fail(f"Unsupported expedition command: {command!r}")

    def _start_expedition(self, command: StartExpedition) -> Result[Any]:
        if command.expedition_id != OPENING_EXPEDITION_ID:
            return Result.fail(f"Unknown expedition: {command.expedition_id}")
        if self.company is not None and not active_roster(self.company):
            return Result.fail("Assign at least one living hero to the active party first.")
        if command.interactive_dungeon:
            return self._require_company(
                lambda company: DungeonFlow(self.controller)._start_interactive_opening_dungeon(
                    company,
                    use_known_route=command.use_known_route,
                    skip_known_route_playback=command.skip_known_route_playback,
                )
            )
        if command.manual_combat:
            return self._require_company(self._start_manual_opening_route)
        return self._require_company(
            lambda company: Result.ok(
                company,
                run_opening_route(
                    company,
                    self.definitions,
                    self.rng,
                    enter_maze=command.enter_maze,
                    stop_at_breach=command.stop_at_breach,
                    enemy_ai_mode=self.controller.enemy_ai_mode,
                    enemy_wait_mode=self.controller.enemy_wait_mode,
                    enemy_movement_mode=self.controller.enemy_movement_mode,
                ),
            )
        )

    def _take_expedition_choice(self, command: TakeExpeditionChoice) -> Result[Any]:
        if command.choice_id == "return_to_haven":
            return self._require_company(self._return_from_breach)
        if command.choice_id == "descend_maze_depth_1":
            return self._require_company(self._descend_from_breach)
        return Result.fail(f"Unknown expedition choice: {command.choice_id}")

    def _return_from_breach(self, company: CompanyState) -> Result[Any]:
        report = active_report(company)
        events = return_to_haven_from_breach(company, self.definitions)
        if report is not None:
            record_events(report, events)
            finish_report(company, "returned_to_haven")
        return Result.ok(company, events)

    def _descend_from_breach(self, company: CompanyState) -> Result[Any]:
        report = active_report(company)
        events = descend_from_breach(
            company,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        if report is not None:
            record_events(report, events)
            finish_report(company, "descended_maze_depth_1")
        return Result.ok(company, events)

    def _start_manual_opening_route(self, company: CompanyState) -> Result[Any]:
        nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = []

        set_company_node_location(company, nodes["old_road"])
        spend_ration(company.supplies)
        events.append(event_for_node(nodes["old_road"]))
        events.extend(apply_node_rewards(company, nodes["old_road"], self.definitions))

        events.append(event_for_node(nodes["old_road_cache"]))
        events.extend(apply_node_rewards(company, nodes["old_road_cache"], self.definitions))

        set_company_node_location(company, nodes["blackwood_forest"])
        events.append(event_for_node(nodes["blackwood_forest"]))
        events.extend(apply_node_rewards(company, nodes["blackwood_forest"], self.definitions))

        set_company_node_location(company, nodes["shallow_cave_room_1"])
        events.append(event_for_node(nodes["shallow_cave_room_1"]))
        events.extend(apply_node_rewards(company, nodes["shallow_cave_room_1"], self.definitions))

        set_company_node_location(company, nodes["shallow_cave_room_2"])
        events.append(event_for_node(nodes["shallow_cave_room_2"]))
        combat = create_encounter_combat(company, self.definitions, MANUAL_STAGE_SHALLOW_CAVE)
        self.manual_combat, combat_events = start_manual_session(
            MANUAL_STAGE_SHALLOW_CAVE,
            nodes["shallow_cave_room_2"].name,
            combat,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        self.opening_manual_stage = MANUAL_STAGE_SHALLOW_CAVE
        events.extend(combat_events)
        return Result.ok(company, events)


@dataclass
class DungeonFlow(ControllerFlow):
    def handle(self, command: object) -> Result[Any]:
        if isinstance(command, ViewDungeon):
            return self._view_dungeon()
        if isinstance(command, InspectDungeonRoom):
            return self._inspect_dungeon_room()
        if isinstance(command, MoveDungeon):
            return self._move_dungeon(command.node_id)
        if isinstance(command, UseDungeonAction):
            return self._use_dungeon_action(command.action_id)
        if isinstance(command, EnterGeneratedMaze):
            return self._enter_generated_maze(command.seed)
        if isinstance(command, RetraceGeneratedMaze):
            return self._retrace_generated_maze()
        if isinstance(command, WithdrawGeneratedMaze):
            return self._withdraw_generated_maze()
        if isinstance(command, RetreatGeneratedMaze):
            return self._retrace_generated_maze()
        if isinstance(command, ReturnFromDungeon):
            return self._return_from_dungeon()
        if isinstance(command, ViewExpeditionReport):
            return self._view_expedition_report()
        if isinstance(command, ClearExpeditionReport):
            return self._clear_expedition_report()
        return Result.fail(f"Unsupported dungeon command: {command!r}")

    def _view_dungeon(self) -> Result[Any]:
        def view(company: CompanyState) -> Result[Any]:
            try:
                return Result.ok(
                    build_dungeon_view(
                        company,
                        self.definitions,
                        first_visit_node_id=self._current_first_visit_node_id(company),
                    )
                )
            except ValueError as exc:
                return Result.fail(str(exc))

        return self._require_company(view)

    def _inspect_dungeon_room(self) -> Result[Any]:
        return self._require_company(self._inspect_active_dungeon_room)

    def _move_dungeon(self, node_id: str) -> Result[Any]:
        return self._require_company(lambda company: self._move_active_dungeon(company, node_id))

    def _use_dungeon_action(self, action_id: str) -> Result[Any]:
        return self._require_company(
            lambda company: self._use_active_dungeon_action(company, action_id)
        )

    def _return_from_dungeon(self) -> Result[Any]:
        return self._require_company(self._return_from_active_dungeon)

    def _view_expedition_report(self) -> Result[Any]:
        def view(company: CompanyState) -> Result[Any]:
            try:
                return Result.ok(build_expedition_report_view(company, self.definitions))
            except ValueError as exc:
                return Result.fail(str(exc))

        return self._require_company(view)

    def _clear_expedition_report(self) -> Result[Any]:
        def clear(company: CompanyState) -> Result[Any]:
            company.last_expedition_report = None
            return Result.ok(company)

        return self._require_company(clear)

    def _start_interactive_opening_dungeon(
        self,
        company: CompanyState,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
    ) -> Result[Any]:
        if company.active_expedition is not None:
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                )
            )
        events = start_interactive_opening_dungeon(
            company,
            self.definitions,
            use_known_route=use_known_route,
            skip_known_route_playback=skip_known_route_playback,
        )
        self.manual_combat = None
        self.opening_manual_stage = None
        self._remember_dungeon_entry(company, events)
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _inspect_active_dungeon_room(self, company: CompanyState) -> Result[Any]:
        session = company.active_expedition
        if session is None:
            return Result.fail("No active dungeon expedition.")
        node = active_dungeon_nodes(self.definitions, session)[session.current_node_id]
        events: list[GameEvent] = [event_for_node(node, first_visit=False)]
        self._remember_dungeon_entry(company, events)
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _move_active_dungeon(self, company: CompanyState, node_id: str) -> Result[Any]:
        session = company.active_expedition
        if session is None:
            return Result.fail("No active dungeon expedition.")
        if session.pending_combat_node_id is not None:
            return Result.fail("Resolve the pending room combat first.")
        nodes = active_dungeon_nodes(self.definitions, session)
        current = nodes[session.current_node_id]
        current_exit_ids = [
            *current.exits,
            *revealed_exit_node_ids(session, current.id),
            *generated_maze_frontier_exit_ids(session),
        ]
        if current.id not in session.cleared_node_ids:
            return Result.fail("Clear or inspect this room before moving on.")
        if node_id not in current_exit_ids:
            return Result.fail("Choose a listed dungeon exit.")
        try:
            generated_events = move_generated_maze_if_needed(
                company,
                self.definitions,
                node_id,
            )
        except ValueError as exc:
            return Result.fail(str(exc))
        if generated_events is not None:
            events = generated_events
            nodes = active_dungeon_nodes(self.definitions, session)
            destination = nodes[session.current_node_id]
        else:
            events = enter_dungeon_node(company, self.definitions, node_id)
            destination = nodes[node_id]
        if destination.encounter is not None and node_id not in session.cleared_node_ids:
            combat = create_encounter_combat(company, self.definitions, destination.encounter)
            self.manual_combat, combat_events = start_manual_session(
                destination.encounter,
                destination.name,
                combat,
                self.definitions,
                self.rng,
                enemy_ai_mode=self.controller.enemy_ai_mode,
            )
            self.opening_manual_stage = None
            events.extend(combat_events)
        self._remember_dungeon_entry(company, events)
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _use_active_dungeon_action(
        self,
        company: CompanyState,
        action_id: str,
    ) -> Result[Any]:
        if company.active_expedition is None:
            return Result.fail("No active dungeon expedition.")
        try:
            events = use_dungeon_action(company, self.definitions, action_id)
        except ValueError as exc:
            return Result.fail(str(exc))
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _enter_generated_maze(self, seed: int | None) -> Result[Any]:
        def enter(company: CompanyState) -> Result[Any]:
            try:
                events = enter_generated_maze(
                    company,
                    self.definitions,
                    self.rng,
                    seed=seed,
                )
            except ValueError as exc:
                return Result.fail(str(exc))
            self.manual_combat = None
            self.opening_manual_stage = None
            self._remember_dungeon_entry(company, events)
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    events,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                ),
                events,
            )

        return self._require_company(enter)

    def _retrace_generated_maze(self) -> Result[Any]:
        def retrace(company: CompanyState) -> Result[Any]:
            try:
                events = retrace_generated_maze(company, self.definitions)
            except ValueError as exc:
                return Result.fail(str(exc))
            self.manual_combat = None
            self.opening_manual_stage = None
            self._remember_dungeon_entry(company, events)
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    events,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                ),
                events,
            )

        return self._require_company(retrace)

    def _withdraw_generated_maze(self) -> Result[Any]:
        def withdraw(company: CompanyState) -> Result[Any]:
            try:
                events = withdraw_generated_maze(company, self.definitions)
            except ValueError as exc:
                return Result.fail(str(exc))
            self.manual_combat = None
            self.opening_manual_stage = None
            self._remember_dungeon_entry(company, events)
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    events,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                ),
                events,
            )

        return self._require_company(withdraw)

    def _return_from_active_dungeon(self, company: CompanyState) -> Result[Any]:
        session = company.active_expedition
        if session is None:
            return Result.fail("No active dungeon expedition.")
        node = active_dungeon_nodes(self.definitions, session)[session.current_node_id]
        if not node.safe_return:
            return Result.fail("Return is only available from safe return rooms.")
        events = return_from_dungeon(
            company,
            self.definitions,
            origin_node_id=node.id,
        )
        arrival_context = build_regional_arrival_context(
            company,
            self.definitions,
            origin_name=node.name,
            flavor_line=regional_return_flavor(node.id),
        )
        return Result.ok(
            build_regional_map_view(
                company,
                self.definitions,
                arrival_context=arrival_context,
            ),
            events,
        )


@dataclass
class ManualCombatFlow(ControllerFlow):
    def handle(self, command: object) -> Result[Any]:
        if isinstance(command, StartManualCombat):
            return self._start_manual_combat(command.encounter_id)
        if isinstance(command, ViewCombat):
            if self.manual_combat is None:
                return Result.fail("No manual combat is active.")
            return Result.ok(
                build_combat_view(
                    self.manual_combat,
                    self.definitions,
                    retreat_available=self._combat_retreat_available(),
                    debug_combat_preview=self.debug_combat_preview,
                )
            )
        if isinstance(command, ChooseCombatSkill):
            return self._choose_combat_skill(command)
        if isinstance(command, ChooseCombatTarget):
            return self._choose_combat_target(command)
        if isinstance(command, ResolveCombatAction):
            return self._resolve_combat_action(command)
        if isinstance(command, UseCombatSkill):
            return self._use_combat_skill(command)
        if isinstance(command, SelectTarget):
            return self._select_target(command)
        if isinstance(command, MoveCombatActor):
            return self._move_combat_actor(command)
        if isinstance(command, PassCombatTurn):
            return self._pass_combat_turn()
        if isinstance(command, DelayCombatTurn):
            return self._delay_combat_turn()
        if isinstance(command, RetreatCombat):
            return self._retreat_combat()
        if isinstance(command, Retreat):
            return self._retreat_combat()
        if isinstance(command, ResolveCombatReaction):
            return self._resolve_combat_reaction(command)
        return Result.fail(f"Unsupported combat command: {command!r}")

    def _combat_retreat_available(self) -> bool:
        return bool(
            self.company is not None
            and self.company.active_expedition is not None
            and self.manual_combat is not None
            and self.manual_combat.pending_hero() is not None
        )

    def _start_manual_combat(self, encounter_id: str) -> Result[Any]:
        return self._require_company(
            lambda company: self._create_manual_session(
                company,
                encounter_id,
                encounter_id.replace("_", " ").title(),
            )
        )

    def _choose_combat_skill(self, command: ChooseCombatSkill) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        if command.skill_id not in legal_skill_ids(self.manual_combat, self.definitions):
            return Result.fail("Choose a listed skill.")
        self.manual_combat.selected_skill_id = command.skill_id
        self.manual_combat.selected_target_id = None
        return Result.ok(
            build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            )
        )

    def _choose_combat_target(self, command: ChooseCombatTarget) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        skill_id = self.manual_combat.selected_skill_id
        if skill_id is None:
            return Result.fail("Choose a skill first.")
        if command.target_id not in legal_target_ids(
            self.manual_combat,
            self.definitions,
            skill_id,
        ):
            return Result.fail("Choose a listed target.")
        self.manual_combat.selected_target_id = command.target_id
        return Result.ok(
            build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            )
        )

    def _resolve_combat_action(self, command: ResolveCombatAction) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        if command.skill_id not in legal_skill_ids(self.manual_combat, self.definitions):
            return Result.fail("Choose a listed skill.")
        if command.target_id not in legal_target_ids(
            self.manual_combat,
            self.definitions,
            command.skill_id,
        ):
            return Result.fail("Choose a listed target.")

        events = resolve_hero_action(
            self.manual_combat,
            self.definitions,
            self.rng,
            command.skill_id,
            command.target_id,
        )
        return self._complete_combat_turn(events)

    def _use_combat_skill(self, command: UseCombatSkill) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        actor = self.manual_combat.pending_hero()
        if actor is None or actor.actor_id != command.actor_id:
            return Result.fail("Choose the active hero.")
        return self._resolve_combat_action(ResolveCombatAction(command.skill_id, command.target_id))

    def _select_target(self, command: SelectTarget) -> Result[Any]:
        return self._choose_combat_target(ChooseCombatTarget(command.target_id))

    def _resolve_combat_reaction(self, command: ResolveCombatReaction) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_enemy_reaction(
            self.manual_combat,
            self.definitions,
            self.rng,
            command.reaction_id,
        )
        if not events:
            return Result.fail("Choose a listed reaction.")
        return self._complete_combat_turn(events)

    def _move_combat_actor(self, command: MoveCombatActor) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_hero_move(
            self.manual_combat,
            self.definitions,
            self.rng,
            command.to_slot,
        )
        if not events:
            return Result.fail("Choose a listed movement option.")
        return self._complete_combat_turn(events)

    def _pass_combat_turn(self) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_hero_pass(
            self.manual_combat,
            self.definitions,
            self.rng,
        )
        if not events:
            return Result.fail("No hero can pass right now.")
        return self._complete_combat_turn(events)

    def _delay_combat_turn(self) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_hero_delay(
            self.manual_combat,
            self.definitions,
            self.rng,
        )
        if not events:
            return Result.fail("No later turn slot is available this round.")
        return self._complete_combat_turn(events)

    def _complete_combat_turn(self, events: list[GameEvent]) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        view = build_combat_view(
            self.manual_combat,
            self.definitions,
            retreat_available=self._combat_retreat_available(),
            debug_combat_preview=self.debug_combat_preview,
        )
        if self.manual_combat.ended:
            events.extend(self._finish_manual_encounter(events))
        elif self.manual_combat is not None:
            view = build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            )
        return Result.ok(view, events)

    def _retreat_combat(self) -> Result[Any]:
        if self.company is None:
            return Result.fail("Start or load a company first.")
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        if self.manual_combat.pending_hero() is None:
            return Result.fail("Retreat is only available on a hero command.")
        if self.company.active_expedition is None:
            return Result.fail("Retreat is only available during active dungeon combat.")

        events = resolve_hero_retreat(
            self.manual_combat,
            self.definitions,
            self.rng,
        )
        if not events:
            return Result.fail("Retreat is already pending.")
        return self._complete_combat_turn(events)

    def _finish_dungeon_retreat(self, session: ManualCombatSession) -> list[GameEvent]:
        if self.company is None or self.company.active_expedition is None:
            return []
        active_session = self.company.active_expedition
        nodes = active_dungeon_nodes(self.definitions, active_session)
        to_node_id = self._combat_retreat_destination_node_id(active_session)
        from_node_id = active_session.pending_combat_node_id or active_session.current_node_id
        if from_node_id != to_node_id:
            active_session.previous_node_id = from_node_id
        active_session.current_node_id = to_node_id
        active_session.pending_combat_node_id = None
        if to_node_id not in active_session.visited_node_ids:
            active_session.visited_node_ids.append(to_node_id)
        to_node = nodes[to_node_id]
        set_company_node_location(self.company, to_node)
        actor_id = session.retreat_actor_id or ""
        event = CombatRetreatedEvent(
            message=(
                "You withdraw from the fight. "
                "The enemy still holds this place. "
                "Re-entering will restart the encounter."
            ),
            actor_id=actor_id,
            encounter_id=session.encounter_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
        )
        if active_session.report is not None:
            record_events(
                active_session.report,
                [
                    *self._combat_report_context_events(session, from_node_id),
                    event,
                ],
            )
        events: list[GameEvent] = [event]
        self.manual_combat = None
        self.opening_manual_stage = None
        return events

    def _combat_retreat_destination_node_id(
        self,
        active_session: Any,
    ) -> str:
        generated = active_session.generated_dungeon
        if (
            generated is not None
            and not generated.collapsed
            and active_session.pending_combat_node_id
            in {node.id for node in generated.nodes}
        ):
            return generated.entry_node_id
        return self._latest_safe_return_node_id()

    def _create_manual_session(
        self,
        company: CompanyState,
        encounter_id: str,
        encounter_name: str,
    ) -> Result[Any]:
        combat = create_encounter_combat(company, self.definitions, encounter_id)
        self.manual_combat, events = start_manual_session(
            encounter_id,
            encounter_name,
            combat,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        self.opening_manual_stage = None
        return Result.ok(
            build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            ),
            events,
        )

    def _finish_manual_encounter(
        self,
        resolved_events: list[GameEvent] | None = None,
    ) -> list[GameEvent]:
        if self.company is None or self.manual_combat is None:
            return []

        events: list[GameEvent] = []
        session = self.manual_combat
        resolved_events = resolved_events or []
        sync_company_from_combat(
            self.company,
            session.state.heroes,
            session.state.party_formation,
        )

        if session.outcome == "retreat":
            if self.company.active_expedition is not None:
                events.extend(self._finish_dungeon_retreat(session))
            else:
                self.manual_combat = None
                self.opening_manual_stage = None
            return events

        if self.company.active_expedition is not None:
            events.extend(self._finish_dungeon_manual_encounter(session, resolved_events))
            return events

        if self.opening_manual_stage == MANUAL_STAGE_SHALLOW_CAVE:
            if session.state.is_defeat():
                self.company.expedition_history.append("opening_failed_shallow_cave")
                events.extend(self._manual_return_to_haven())
                return events
            events.extend(self._start_opening_boss_combat())
            return events

        if self.opening_manual_stage == MANUAL_STAGE_CAVE_BOSS:
            if session.state.is_defeat():
                self.company.expedition_history.append("opening_failed_cave_boss")
                events.extend(self._manual_return_to_haven())
                return events
            events.extend(self._finish_manual_opening_to_breach())
            return events

        pending_regional = self.company.town_state.get("pending_regional_combat_node_id")
        if pending_regional:
            if session.state.is_defeat():
                self.company.town_state.pop("pending_regional_combat_node_id", None)
                self.controller.return_to_regional_place = True
                self.manual_combat = None
                return events
            mark_regional_combat_cleared(self.company, self.definitions)
            self.controller.return_to_regional_place = True
            self.manual_combat = None
            return events

        self.manual_combat = None
        return events

    def _finish_dungeon_manual_encounter(
        self,
        session: ManualCombatSession,
        _resolved_events: list[GameEvent],
    ) -> list[GameEvent]:
        if self.company is None or self.company.active_expedition is None:
            return []

        events: list[GameEvent] = []
        active_session = self.company.active_expedition
        pending_node_id = active_session.pending_combat_node_id
        combat_report_events = self._combat_report_context_events(session, pending_node_id)
        if session.state.is_defeat():
            if session.encounter_id == "shallow_cave":
                self.company.expedition_history.append("opening_failed_shallow_cave")
            elif session.encounter_id == "cave_mini_boss":
                self.company.expedition_history.append("opening_failed_cave_boss")
            if active_session.report is not None:
                record_events(active_session.report, combat_report_events)
            events.extend(
                return_from_dungeon(
                    self.company,
                    self.definitions,
                    outcome="defeat",
                    message="The company is carried back to Haven after defeat.",
                )
            )
            self.manual_combat = None
            self.opening_manual_stage = None
            return events

        mark_pending_combat_cleared(
            self.company,
            self.definitions,
            session.encounter_id,
            combat_report_events,
        )
        self.manual_combat = None
        self.opening_manual_stage = None

        if pending_node_id == SHALLOW_CAVE_BOSS_NODE_ID:
            events.extend(finish_shallow_cave_boss(self.company, self.definitions))
        return events

    def _combat_report_context_events(
        self,
        session: ManualCombatSession,
        node_id: str | None,
    ) -> list[GameEvent]:
        if self.company is not None and self.company.active_expedition is not None:
            nodes = active_dungeon_nodes(self.definitions, self.company.active_expedition)
        else:
            nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = []
        if node_id is not None and node_id in nodes:
            events.append(event_for_node(nodes[node_id]))
        events.extend(session.event_log)
        return events

    def _start_opening_boss_combat(self) -> list[GameEvent]:
        if self.company is None:
            return []
        nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = [event_for_node(nodes["shallow_cave_room_3"])]
        combat = create_cave_boss_combat(self.company, self.definitions)
        self.manual_combat, combat_events = start_manual_session(
            MANUAL_STAGE_CAVE_BOSS,
            nodes["shallow_cave_room_3"].name,
            combat,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        self.opening_manual_stage = MANUAL_STAGE_CAVE_BOSS
        events.extend(combat_events)
        return events

    def _finish_manual_opening_to_breach(self) -> list[GameEvent]:
        if self.company is None:
            return []
        nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = [event_for_node(nodes["cave_mini_boss"])]
        events.extend(apply_node_rewards(self.company, nodes["cave_mini_boss"], self.definitions))
        events.extend(open_opening_breach_room(self.company, self.definitions))
        self.manual_combat = None
        self.opening_manual_stage = None
        return events

    def _manual_return_to_haven(self) -> list[GameEvent]:
        if self.company is None:
            return []
        set_company_location(self.company, "haven", "Haven Town")
        self.company.flags[OPENING_BREACH_PENDING_FLAG] = False
        self.manual_combat = None
        self.opening_manual_stage = None
        return [
            ExpeditionReturnedEvent(
                message="The company returns to Haven.",
                expedition_id=OPENING_EXPEDITION_ID,
                location="Haven Town",
            )
        ]
