"""Structured commands issued by UI or tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from game.combat.formation import FormationSlot


@dataclass(frozen=True)
class StartNewCompany:
    name: str = "Haven Charter"


@dataclass(frozen=True)
class ViewRoster:
    pass


@dataclass(frozen=True)
class ViewSupplies:
    pass


@dataclass(frozen=True)
class ViewGear:
    pass


@dataclass(frozen=True)
class ViewHeroSheet:
    hero_id: str


@dataclass(frozen=True)
class ViewLedger:
    pass


@dataclass(frozen=True)
class ViewTown:
    pass


@dataclass(frozen=True)
class AcceptContract:
    contract_id: str


@dataclass(frozen=True)
class ViewWorld:
    pass


@dataclass(frozen=True)
class ViewRegionalMap:
    pass


@dataclass(frozen=True)
class TravelRegional:
    destination_id: str


@dataclass(frozen=True)
class MoveRegional:
    node_id: str


@dataclass(frozen=True)
class UseRegionalAction:
    action_id: str


@dataclass(frozen=True)
class MarkRegionalRoute:
    pass


@dataclass(frozen=True)
class ViewMemorial:
    pass


@dataclass(frozen=True)
class GenerateRecruitOffers:
    pass


@dataclass(frozen=True)
class HireRecruit:
    offer_index: int


@dataclass(frozen=True)
class RecoverCompany:
    pass


@dataclass(frozen=True)
class PerformDeepSurgery:
    hero_id: str


@dataclass(frozen=True)
class BuySupply:
    supply_id: str
    quantity: int = 1


@dataclass(frozen=True)
class SellLoot:
    item_id: str


@dataclass(frozen=True)
class TurnInLoot:
    item_id: str


@dataclass(frozen=True)
class PurchaseUpgrade:
    upgrade_id: str


@dataclass(frozen=True)
class PurchaseGear:
    gear_id: str


@dataclass(frozen=True)
class EquipGear:
    hero_id: str
    gear_id: str


@dataclass(frozen=True)
class UnequipGear:
    hero_id: str


@dataclass(frozen=True)
class AssignActiveHero:
    hero_id: str
    slot: FormationSlot


@dataclass(frozen=True)
class StartExpedition:
    expedition_id: str = "opening"
    enter_maze: bool = False
    stop_at_breach: bool = False
    manual_combat: bool = False
    interactive_dungeon: bool = False
    use_known_route: bool = True
    skip_known_route_playback: bool = False


@dataclass(frozen=True)
class TakeExpeditionChoice:
    choice_id: str


@dataclass(frozen=True)
class ViewDungeon:
    pass


@dataclass(frozen=True)
class InspectDungeonRoom:
    pass


@dataclass(frozen=True)
class MoveDungeon:
    node_id: str


@dataclass(frozen=True)
class UseDungeonAction:
    action_id: str


@dataclass(frozen=True)
class EnterGeneratedMaze:
    seed: int | None = None


@dataclass(frozen=True)
class RetraceGeneratedMaze:
    pass


@dataclass(frozen=True)
class WithdrawGeneratedMaze:
    pass


@dataclass(frozen=True)
class RetreatGeneratedMaze:
    pass


@dataclass(frozen=True)
class ReturnFromDungeon:
    pass


@dataclass(frozen=True)
class ViewExpeditionReport:
    pass


@dataclass(frozen=True)
class ClearExpeditionReport:
    pass


@dataclass(frozen=True)
class StartManualCombat:
    encounter_id: str


@dataclass(frozen=True)
class ViewCombat:
    pass


@dataclass(frozen=True)
class ChooseCombatSkill:
    skill_id: str


@dataclass(frozen=True)
class ChooseCombatTarget:
    target_id: str


@dataclass(frozen=True)
class ResolveCombatAction:
    skill_id: str
    target_id: str


@dataclass(frozen=True)
class MoveCombatActor:
    to_slot: FormationSlot


@dataclass(frozen=True)
class PassCombatTurn:
    pass


@dataclass(frozen=True)
class DelayCombatTurn:
    pass


@dataclass(frozen=True)
class RetreatCombat:
    pass


@dataclass(frozen=True)
class ResolveCombatReaction:
    reaction_id: str | None


@dataclass(frozen=True)
class UseCombatSkill:
    actor_id: str
    skill_id: str
    target_id: str


@dataclass(frozen=True)
class SelectTarget:
    target_id: str


@dataclass(frozen=True)
class Retreat:
    pass


@dataclass(frozen=True)
class SaveGame:
    path: Path


@dataclass(frozen=True)
class LoadGame:
    path: Path


@dataclass(frozen=True)
class Quit:
    pass


Command = (
    StartNewCompany
    | ViewRoster
    | ViewSupplies
    | ViewGear
    | ViewHeroSheet
    | ViewLedger
    | ViewTown
    | AcceptContract
    | ViewWorld
    | ViewRegionalMap
    | TravelRegional
    | MoveRegional
    | UseRegionalAction
    | MarkRegionalRoute
    | ViewMemorial
    | GenerateRecruitOffers
    | HireRecruit
    | RecoverCompany
    | PerformDeepSurgery
    | BuySupply
    | SellLoot
    | TurnInLoot
    | PurchaseUpgrade
    | PurchaseGear
    | EquipGear
    | UnequipGear
    | AssignActiveHero
    | StartExpedition
    | TakeExpeditionChoice
    | ViewDungeon
    | InspectDungeonRoom
    | MoveDungeon
    | UseDungeonAction
    | EnterGeneratedMaze
    | RetraceGeneratedMaze
    | WithdrawGeneratedMaze
    | RetreatGeneratedMaze
    | ReturnFromDungeon
    | ViewExpeditionReport
    | ClearExpeditionReport
    | StartManualCombat
    | ViewCombat
    | ChooseCombatSkill
    | ChooseCombatTarget
    | ResolveCombatAction
    | MoveCombatActor
    | PassCombatTurn
    | DelayCombatTurn
    | RetreatCombat
    | ResolveCombatReaction
    | UseCombatSkill
    | SelectTarget
    | Retreat
    | SaveGame
    | LoadGame
    | Quit
)
