"""App-facing view models for terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game.app.actions import (
    ActionProvider,
    ScreenAction,
)
from game.app.contracts import contract_board_ids, contract_board_state
from game.app.views.formatting import _join_detail
from game.app.views.hero import (
    HeroListEntry,
    _hero_entry,
    _hero_memories,
    _trait_label,
)
from game.campaign.company import (
    CompanyState,
)
from game.campaign.objectives import (
    CampaignObjectiveView,
    build_campaign_objective,
)
from game.campaign.roster import active_roster, living_roster, reserve_roster
from game.campaign.town import (
    deep_surgery_candidates,
    effective_roster_cap,
    effective_surgery_cost,
    upgrade_unavailable_reason,
)
from game.combat.combat_state import LifeState
from game.content.definitions import GameDefinitions


@dataclass(frozen=True)
class ContractBoardEntryView:
    contract_id: str
    name: str
    summary: str
    reward_reputation: int
    coin_reward: int
    difficulty: int
    state: str
    unavailable_reason: str = ""

@dataclass(frozen=True)
class TownUpgradeView:
    upgrade_id: str
    name: str
    description: str
    cost: int
    state: str
    effect_summary: str
    unavailable_reason: str = ""

@dataclass(frozen=True)
class TownDashboardView:
    company_name: str
    location: str
    reputation: int
    coin: int
    roster_cap: int
    active_count: int
    reserve_count: int
    wounded_count: int
    downed_count: int
    deceased_count: int
    objective: CampaignObjectiveView
    active_party: tuple[HeroListEntry, ...]
    reserves: tuple[HeroListEntry, ...]
    services: tuple[ScreenAction, ...]
    contract_board: tuple[ContractBoardEntryView, ...] = ()
    upgrades: tuple[TownUpgradeView, ...] = ()
    upgrade_actions: tuple[ScreenAction, ...] = ()

@dataclass(frozen=True)
class SupplyShopView:
    reputation: int
    coin: int
    actions: tuple[ScreenAction, ...]

@dataclass(frozen=True)
class RelicBrokerView:
    reputation: int
    coin: int
    inventory: tuple[tuple[str, int], ...]
    actions: tuple[ScreenAction, ...]

@dataclass(frozen=True)
class DeepSurgeryCandidateView:
    hero_id: str
    name: str
    class_name: str
    mortal_wounds: int

@dataclass(frozen=True)
class DeepSurgeryView:
    coin: int
    surgery_cost: int
    candidates: tuple[DeepSurgeryCandidateView, ...]
    actions: tuple[ScreenAction, ...]

def build_town_dashboard(
    company: CompanyState,
    definitions: GameDefinitions,
) -> TownDashboardView:
    active = active_roster(company)
    reserves = reserve_roster(company)
    living = living_roster(company)
    wounded_count = sum(1 for hero in living if hero.hp < hero.max_hp or hero.mortal_wounds > 0)
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    services = ActionProvider.town_services(company, definitions)
    objective = build_campaign_objective(company)
    return TownDashboardView(
        company_name=company.name,
        location=str(company.town_state.get("location", "Haven Town")),
        reputation=company.reputation,
        coin=company.coin,
        roster_cap=effective_roster_cap(company, definitions),
        active_count=len(active),
        reserve_count=len(reserves),
        wounded_count=wounded_count,
        downed_count=downed_count,
        deceased_count=len(company.deceased_heroes),
        objective=objective,
        active_party=tuple(
            _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions) for hero in active
        ),
        reserves=tuple(
            _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
            for hero in reserves
        ),
        services=services,
        contract_board=_contract_board_entries(company, definitions),
        upgrades=_upgrade_entries(company, definitions),
        upgrade_actions=ActionProvider.upgrade_actions(company, definitions),
    )

def _contract_board_entries(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[ContractBoardEntryView, ...]:
    return tuple(
        _contract_board_entry(company, definitions, contract_id)
        for contract_id in contract_board_ids(company, definitions)
        if _contract_board_is_visible(company, definitions, contract_id)
    )

def _contract_board_is_visible(
    company: CompanyState,
    definitions: GameDefinitions,
    contract_id: str,
) -> bool:
    state, _reason = contract_board_state(company, definitions, contract_id)
    return state in {"available", "active"}

def _contract_board_entry(
    company: CompanyState,
    definitions: GameDefinitions,
    contract_id: str,
) -> ContractBoardEntryView:
    contract = definitions.contracts[contract_id]
    state, unavailable_reason = contract_board_state(company, definitions, contract_id)
    return ContractBoardEntryView(
        contract_id=contract.id,
        name=contract.name,
        summary=contract.summary,
        reward_reputation=contract.reward_reputation,
        coin_reward=contract.coin_reward,
        difficulty=contract.difficulty,
        state=state,
        unavailable_reason=unavailable_reason,
    )

def _upgrade_entries(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[TownUpgradeView, ...]:
    entries: list[TownUpgradeView] = []
    for upgrade in definitions.town.upgrades.values():
        unavailable = upgrade_unavailable_reason(company, definitions, upgrade.id)
        if upgrade.id in company.purchased_upgrade_ids:
            state = "installed"
        elif not unavailable:
            state = "available"
        elif unavailable.startswith(("Complete ", "Find ")):
            state = "locked"
        else:
            state = "unavailable"
        entries.append(
            TownUpgradeView(
                upgrade_id=upgrade.id,
                name=upgrade.name,
                description=upgrade.description,
                cost=upgrade.cost,
                state=state,
                effect_summary=_upgrade_effect_summary(upgrade.effects),
                unavailable_reason=unavailable,
            )
        )
    return tuple(entries)

def build_deep_surgery_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> DeepSurgeryView:
    candidates = deep_surgery_candidates(company)
    return DeepSurgeryView(
        coin=company.coin,
        surgery_cost=effective_surgery_cost(company, definitions),
        candidates=tuple(
            DeepSurgeryCandidateView(
                hero_id=hero.hero_id,
                name=hero.name,
                class_name=_trait_label(hero.class_id),
                mortal_wounds=hero.mortal_wounds,
            )
            for hero in candidates
        ),
        actions=ActionProvider.deep_surgery_hero_actions(company, definitions),
    )

def build_supply_shop_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> SupplyShopView:
    return SupplyShopView(
        reputation=company.reputation,
        coin=company.coin,
        actions=ActionProvider.supply_shop_actions(company, definitions),
    )

def build_relic_broker_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> RelicBrokerView:
    return RelicBrokerView(
        reputation=company.reputation,
        coin=company.coin,
        inventory=tuple(sorted(company.inventory.items())),
        actions=ActionProvider.relic_broker_actions(company, definitions),
    )

def _upgrade_effect_summary(effects: Any) -> str:
    pieces: list[str] = []
    if effects.roster_cap_bonus:
        pieces.append(f"Roster cap +{effects.roster_cap_bonus}")
    if effects.recovery_cost_delta:
        pieces.append(f"Recovery cost {effects.recovery_cost_delta:+d}")
    for supply_id, delta in sorted(effects.supply_cost_deltas.items()):
        pieces.append(f"{supply_id.replace('_', ' ').title()} cost {delta:+d}")
    return _join_detail(*pieces)
