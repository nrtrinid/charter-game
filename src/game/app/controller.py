"""Thin application controller around engine systems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from game.app.commands import (
    AcceptContract,
    AssignActiveHero,
    BuySupply,
    ChooseCombatSkill,
    ChooseCombatTarget,
    ClearExpeditionReport,
    Command,
    DelayCombatTurn,
    EnterGeneratedMaze,
    EquipGear,
    GenerateRecruitOffers,
    HireRecruit,
    InspectDungeonRoom,
    LoadGame,
    MarkRegionalRoute,
    MoveCombatActor,
    MoveDungeon,
    MoveRegional,
    PassCombatTurn,
    PerformDeepSurgery,
    PurchaseGear,
    PurchaseUpgrade,
    Quit,
    RecoverCompany,
    ResolveCombatAction,
    ResolveCombatReaction,
    RetraceGeneratedMaze,
    Retreat,
    RetreatCombat,
    RetreatGeneratedMaze,
    ReturnFromDungeon,
    SaveGame,
    SelectTarget,
    SellLoot,
    StartExpedition,
    StartManualCombat,
    StartNewCompany,
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
    WithdrawGeneratedMaze,
)
from game.app.flows import DungeonFlow, ExpeditionFlow, ManualCombatFlow, TownFlow
from game.app.hci import analyze_hci_result, capture_hci_state
from game.app.manual_combat import ManualCombatSession
from game.campaign.company import CompanyState, create_new_company
from game.campaign.recruitment import RecruitChoice
from game.campaign.save_load import load_company, save_company
from game.combat.enemy_decision import (
    production_enemy_movement_mode,
    production_enemy_wait_mode,
)
from game.content.definitions import GameDefinitions
from game.core.events import CompanyEvent
from game.core.result import Result
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions


@dataclass
class AppController:
    definitions: GameDefinitions = field(default_factory=load_game_definitions)
    rng: GameRng = field(default_factory=lambda: GameRng(7))
    company: CompanyState | None = None
    should_quit: bool = False
    return_to_regional_place: bool = False
    recruit_offers: list[RecruitChoice] = field(default_factory=list)
    manual_combat: ManualCombatSession | None = None
    opening_manual_stage: str | None = None
    debug_combat_preview: bool = False
    dungeon_first_visit_node_id: str | None = None
    enemy_ai_mode: str = "learned_static"

    @property
    def enemy_wait_mode(self) -> str:
        return production_enemy_wait_mode(self.enemy_ai_mode)

    @property
    def enemy_movement_mode(self) -> str:
        return production_enemy_movement_mode(self.enemy_ai_mode)

    def handle(self, command: Command) -> Result[Any]:
        before = capture_hci_state(self.company, self.manual_combat)
        result = self._handle(command)
        after = capture_hci_state(self.company, self.manual_combat)
        if result.hci is None:
            result.hci = analyze_hci_result(
                before,
                after,
                result.events,
                error=result.error,
            )
        return result

    def _handle(self, command: Command) -> Result[Any]:
        if isinstance(command, StartNewCompany):
            self.company = create_new_company(self.definitions, name=command.name)
            self.recruit_offers = []
            self.manual_combat = None
            self.opening_manual_stage = None
            self.dungeon_first_visit_node_id = None
            return Result.ok(
                self.company,
                [CompanyEvent(message=f"{self.company.name} receives its charter.")],
            )

        if isinstance(
            command,
            (
                ViewRoster,
                ViewSupplies,
                ViewGear,
                ViewHeroSheet,
                ViewLedger,
                ViewTown,
                ViewWorld,
                ViewRegionalMap,
                TravelRegional,
                MoveRegional,
                MarkRegionalRoute,
                UseRegionalAction,
                ViewMemorial,
                GenerateRecruitOffers,
                HireRecruit,
                RecoverCompany,
                PerformDeepSurgery,
                BuySupply,
                SellLoot,
                TurnInLoot,
                PurchaseUpgrade,
                PurchaseGear,
                EquipGear,
                UnequipGear,
                AssignActiveHero,
                AcceptContract,
            ),
        ):
            return TownFlow(self).handle(command)

        if isinstance(command, (StartExpedition, TakeExpeditionChoice)):
            return ExpeditionFlow(self).handle(command)

        if isinstance(
            command,
            (
                ViewDungeon,
                InspectDungeonRoom,
                MoveDungeon,
                UseDungeonAction,
                EnterGeneratedMaze,
                RetraceGeneratedMaze,
                WithdrawGeneratedMaze,
                RetreatGeneratedMaze,
                ReturnFromDungeon,
                ViewExpeditionReport,
                ClearExpeditionReport,
            ),
        ):
            return DungeonFlow(self).handle(command)

        if isinstance(
            command,
            (
                StartManualCombat,
                ViewCombat,
                ChooseCombatSkill,
                ChooseCombatTarget,
                ResolveCombatAction,
                MoveCombatActor,
                PassCombatTurn,
                DelayCombatTurn,
                RetreatCombat,
                ResolveCombatReaction,
                UseCombatSkill,
                SelectTarget,
                Retreat,
            ),
        ):
            return ManualCombatFlow(self).handle(command)

        if isinstance(command, SaveGame):
            return self._require_company(
                lambda company: Result.ok(company, [save_company(company, command.path)])
            )

        if isinstance(command, LoadGame):
            try:
                self.company, event = load_company(command.path)
            except (OSError, ValueError) as exc:
                return Result.fail(f"Could not load save: {exc}")
            self.recruit_offers = [
                RecruitChoice(
                    name=offer.name,
                    class_id=offer.class_id,
                    background=offer.background,
                    motive=offer.motive,
                )
                for offer in self.company.recruitment_state.current_offers
            ]
            self.manual_combat = None
            self.opening_manual_stage = None
            self.dungeon_first_visit_node_id = None
            return Result.ok(self.company, [event])

        if isinstance(command, Quit):
            self.should_quit = True
            return Result.ok(None, [CompanyEvent(message="The charter desk closes.")])

        return Result.fail(f"Unsupported command: {command!r}")

    def _require_company(self, action: Callable[[CompanyState], Result[Any]]) -> Result[Any]:
        if self.company is None:
            return Result.fail("Start or load a company first.")
        return action(self.company)
