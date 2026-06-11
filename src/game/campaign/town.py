"""Haven Town service helpers."""

from __future__ import annotations

from collections.abc import Iterator

from game.campaign.company import CompanyState, HeroState
from game.campaign.economy import spend_coin
from game.campaign.gear import effective_hero_stats
from game.campaign.objectives import build_campaign_objective
from game.campaign.recruitment import RecruitChoice
from game.campaign.roster import active_roster, living_roster, reserve_roster
from game.combat.combat_state import LifeState, StrainTier
from game.combat.formation import FormationSlot
from game.content.definitions import GameDefinitions
from game.core.events import (
    ActivePartyChangedEvent,
    CompanyEvent,
    DeepSurgeryEvent,
    GameEvent,
    HeroRecruitedEvent,
    RecoveryEvent,
    SuppliesPurchasedEvent,
    TownServiceEvent,
)
from game.core.result import Result
from game.data.schemas import TownUpgradeDefinition

IN_SURGERY_LABEL = "In Surgery"


def ledger(
    company: CompanyState,
    definitions: GameDefinitions | None = None,
) -> dict[str, object]:
    objective = build_campaign_objective(company)
    installed_upgrades: list[str]
    if definitions is None:
        installed_upgrades = sorted(company.purchased_upgrade_ids)
    else:
        installed_upgrades = [
            definitions.town.upgrades[upgrade_id].name
            for upgrade_id in sorted(company.purchased_upgrade_ids)
            if upgrade_id in definitions.town.upgrades
        ]
    return {
        "name": company.name,
        "reputation": company.reputation,
        "coin": company.coin,
        "roster": len(company.roster),
        "active_party": len(active_roster(company)),
        "reserves": len(reserve_roster(company)),
        "deceased": len(company.deceased_heroes),
        "current_objective": objective.title,
        "objective_status": objective.status,
        "next_objective": objective.next_step,
        "chapter_status": objective.chapter_status,
        "installed_upgrades": installed_upgrades,
        "gear_inventory": dict(sorted(company.gear_inventory.items())),
        "equipped_gear": {
            hero.name: hero.equipped_gear_id
            for hero in living_roster(company)
            if hero.equipped_gear_id is not None
        },
        "known_breaches": sorted(company.known_breaches),
        "known_lore_entries": sorted(company.known_lore_entries),
        "active_contracts": sorted(company.active_contract_ids),
        "completed_contracts": sorted(company.completed_contract_ids),
        "report_count": len(company.expedition_reports),
        "company_timeline_count": len(company.company_timeline),
        "recent_timeline": [
            entry.summary for entry in company.company_timeline[-5:]
        ],
        "flags": sorted(key for key, value in company.flags.items() if value),
        "location": company.town_state.get("location", "Haven Town"),
    }


def town_summary(company: CompanyState, definitions: GameDefinitions) -> dict[str, object]:
    return {
        "ledger": ledger(company, definitions),
        "roster_cap": effective_roster_cap(company, definitions),
        "recruit_cost": definitions.town.recruit_cost,
        "recovery_cost": effective_recovery_cost(company, definitions),
        "services": definitions.town.services,
    }


def memorial(company: CompanyState) -> list[HeroState]:
    return list(company.deceased_heroes)


def hire_recruit(
    company: CompanyState,
    definitions: GameDefinitions,
    recruit: RecruitChoice,
) -> Result[HeroState]:
    roster_cap = effective_roster_cap(company, definitions)
    cost = definitions.town.recruit_cost
    if len(company.roster) >= roster_cap:
        return Result.fail(f"Roster cap reached ({roster_cap}).")
    if company.coin < cost:
        return Result.fail("Not enough Coin to hire that recruit.")
    if recruit.class_id not in definitions.hero_classes:
        return Result.fail(f"Unknown recruit class: {recruit.class_id}")

    hero_class = definitions.hero_classes[recruit.class_id]
    hero = HeroState(
        hero_id=_next_recruit_id(company, recruit),
        name=recruit.name,
        class_id=recruit.class_id,
        background=recruit.background,
        motive=recruit.motive,
        max_hp=hero_class.max_hp,
        hp=hero_class.max_hp,
        speed=hero_class.speed,
        accuracy=hero_class.accuracy,
        defense=hero_class.defense,
        damage=hero_class.damage,
        max_effort=hero_class.max_effort,
        effort=hero_class.max_effort,
        skills=list(hero_class.skills),
        formation_slot=_preferred_slot(definitions, recruit.class_id),
        personal_quirk=hero_class.personal_quirk,
    )
    spend_coin(company, cost)
    company.roster.append(hero)
    return Result.ok(
        hero,
        [
            HeroRecruitedEvent(
                message=f"{hero.name} joins the company.",
                hero_id=hero.hero_id,
                name=hero.name,
                class_id=hero.class_id,
                cost=cost,
            )
        ],
    )


def effective_surgery_cost(
    company: CompanyState,
    definitions: GameDefinitions,
) -> int:
    return max(0, definitions.town.surgery_cost)


def deep_surgery_candidates(company: CompanyState) -> list[HeroState]:
    return [
        hero
        for hero in living_roster(company)
        if hero.mortal_wounds >= 1 and not hero.in_surgery
    ]


def clear_surgery_recovery(company: CompanyState) -> list[str]:
    moments: list[str] = []
    for hero in company.roster:
        if hero.in_surgery:
            hero.in_surgery = False
            moments.append(f"{hero.name} is cleared from the surgery ward.")
    return moments


def deep_surgery(
    company: CompanyState,
    definitions: GameDefinitions,
    hero_id: str,
) -> Result[HeroState]:
    cost = effective_surgery_cost(company, definitions)
    living = {hero.hero_id: hero for hero in living_roster(company)}
    hero = living.get(hero_id)
    if hero is None:
        return Result.fail("Choose a living roster member.")
    if hero.in_surgery:
        return Result.fail(f"{hero.name} is already {IN_SURGERY_LABEL}.")
    if hero.mortal_wounds < 1:
        return Result.fail(f"{hero.name} has no Mortal Wounds to treat.")
    if company.coin < cost:
        return Result.fail("Not enough Coin to fund deep surgery.")

    hero.mortal_wounds -= 1
    hero.in_surgery = True
    events: list[GameEvent] = [
        DeepSurgeryEvent(
            message=(
                f"{hero.name} undergoes deep surgery. "
                f"{IN_SURGERY_LABEL} until the next expedition returns."
            ),
            hero_id=hero.hero_id,
            name=hero.name,
            cost=cost,
            remaining_mortal_wounds=hero.mortal_wounds,
        )
    ]
    events.extend(_remove_hero_from_active_party(company, hero.hero_id))
    spend_coin(company, cost)
    return Result.ok(hero, events)


def recover_company(company: CompanyState, definitions: GameDefinitions) -> Result[CompanyState]:
    cost = effective_recovery_cost(company, definitions)
    if company.coin < cost:
        return Result.fail("Not enough Coin to fund recovery.")
    recovered_ids: list[str] = []
    for hero in living_roster(company):
        stats = effective_hero_stats(hero, definitions)
        hero.hp = stats.max_hp
        hero.effort = stats.max_effort
        if hero.life_state == LifeState.DOWNED:
            hero.life_state = LifeState.ALIVE
        hero.strain = StrainTier.STEADY
        hero.strain_marks.clear()
        recovered_ids.append(hero.hero_id)
    spend_coin(company, cost)
    return Result.ok(
        company,
        [
            RecoveryEvent(
                message="The recovery ward restores the living roster.",
                hero_ids=recovered_ids,
                cost=cost,
            )
        ],
    )


def buy_supply(
    company: CompanyState,
    definitions: GameDefinitions,
    supply_id: str,
    quantity: int,
) -> Result[CompanyState]:
    if quantity <= 0:
        return Result.fail("Choose a positive quantity.")
    supply = definitions.supplies.catalog.get(supply_id)
    if supply is None:
        return Result.fail(f"Unknown supply: {supply_id}")
    unit_cost = effective_supply_cost(company, definitions, supply_id)
    cost = unit_cost * quantity
    if company.coin < cost:
        return Result.fail("Not enough Coin for those supplies.")
    spend_coin(company, cost)
    company.supplies[supply_id] = company.supplies.get(supply_id, 0) + quantity
    return Result.ok(
        company,
        [
            SuppliesPurchasedEvent(
                message=f"Purchased {quantity} {supply.name}.",
                supply_id=supply_id,
                quantity=quantity,
                cost=cost,
            )
        ],
    )


def purchase_upgrade(
    company: CompanyState,
    definitions: GameDefinitions,
    upgrade_id: str,
) -> Result[CompanyState]:
    upgrade = definitions.town.upgrades.get(upgrade_id)
    if upgrade is None:
        return Result.fail(f"Unknown upgrade: {upgrade_id}")
    unavailable = upgrade_unavailable_reason(company, definitions, upgrade_id)
    if unavailable:
        return Result.fail(unavailable)
    spend_coin(company, upgrade.cost)
    company.purchased_upgrade_ids.add(upgrade_id)
    return Result.ok(
        company,
        [
            CompanyEvent(
                message=f"Company upgrade installed - {upgrade.name}.",
            )
        ],
    )


def effective_roster_cap(
    company: CompanyState,
    definitions: GameDefinitions,
) -> int:
    return definitions.town.roster_cap + sum(
        upgrade.effects.roster_cap_bonus
        for upgrade in _purchased_upgrades(company, definitions)
    )


def effective_recovery_cost(
    company: CompanyState,
    definitions: GameDefinitions,
) -> int:
    return max(
        0,
        definitions.town.recovery_cost
        + sum(
            upgrade.effects.recovery_cost_delta
            for upgrade in _purchased_upgrades(company, definitions)
        ),
    )


def effective_supply_cost(
    company: CompanyState,
    definitions: GameDefinitions,
    supply_id: str,
) -> int:
    supply = definitions.supplies.catalog[supply_id]
    delta = sum(
        upgrade.effects.supply_cost_deltas.get(supply_id, 0)
        for upgrade in _purchased_upgrades(company, definitions)
    )
    return max(1, supply.cost + delta)


def upgrade_unavailable_reason(
    company: CompanyState,
    definitions: GameDefinitions,
    upgrade_id: str,
) -> str:
    upgrade = definitions.town.upgrades[upgrade_id]
    if upgrade_id in company.purchased_upgrade_ids:
        return "Upgrade already installed."
    for contract_id in upgrade.requires_completed_contracts:
        if contract_id not in company.completed_contract_ids:
            contract = definitions.contracts.get(contract_id)
            if contract is None:
                return "Complete the required contract first."
            return f"Complete {contract.name} first."
    for breach_id in upgrade.requires_known_breaches:
        if breach_id not in company.known_breaches:
            location = definitions.locations.get(breach_id)
            if location is None:
                return "Find the required breach first."
            return f"Find {location.name} first."
    if company.coin < upgrade.cost:
        return f"Need {upgrade.cost - company.coin} more Coin."
    return ""


def _purchased_upgrades(
    company: CompanyState,
    definitions: GameDefinitions,
) -> Iterator[TownUpgradeDefinition]:
    return (
        definitions.town.upgrades[upgrade_id]
        for upgrade_id in sorted(company.purchased_upgrade_ids)
        if upgrade_id in definitions.town.upgrades
    )


def assign_active_hero(
    company: CompanyState,
    hero_id: str,
    slot: FormationSlot,
) -> Result[CompanyState]:
    if hero_id == "__empty__":
        company.active_party_slots[slot] = None
        return Result.ok(
            company,
            [
                ActivePartyChangedEvent(
                    message=f"{slot.value} is now empty.",
                    active_party_slots={
                        active_slot.value: active_hero_id
                        for active_slot, active_hero_id in company.active_party_slots.items()
                    },
                )
            ],
        )

    living = {hero.hero_id: hero for hero in living_roster(company)}
    if hero_id not in living:
        return Result.fail("Choose a living roster member.")
    if living[hero_id].in_surgery:
        return Result.fail(f"That hero is {IN_SURGERY_LABEL} and cannot join the active party.")

    old_slot: FormationSlot | None = None
    for current_slot, current_hero_id in list(company.active_party_slots.items()):
        if current_hero_id == hero_id:
            old_slot = current_slot
            company.active_party_slots[current_slot] = None
            break
    displaced_id = company.active_party_slots.get(slot)
    company.active_party_slots[slot] = hero_id
    living[hero_id].formation_slot = slot
    if displaced_id is not None and old_slot is not None:
        company.active_party_slots[old_slot] = displaced_id
        living[displaced_id].formation_slot = old_slot
    return Result.ok(
        company,
        [
            ActivePartyChangedEvent(
                message=f"{living[hero_id].name} takes {slot.value}.",
                active_party_slots={
                    active_slot.value: active_hero_id
                    for active_slot, active_hero_id in company.active_party_slots.items()
                },
            )
        ],
    )


def town_service_event(service_id: str, message: str, *, cost: int = 0) -> TownServiceEvent:
    return TownServiceEvent(message=message, service_id=service_id, cost=cost)


def _remove_hero_from_active_party(
    company: CompanyState,
    hero_id: str,
) -> list[GameEvent]:
    changed = False
    for slot, active_id in list(company.active_party_slots.items()):
        if active_id == hero_id:
            company.active_party_slots[slot] = None
            changed = True
    if not changed:
        return []
    hero = next((entry for entry in living_roster(company) if entry.hero_id == hero_id), None)
    hero_name = hero.name if hero is not None else hero_id
    return [
        ActivePartyChangedEvent(
            message=f"{hero_name} is removed from the active party.",
            active_party_slots={
                active_slot.value: active_hero_id
                for active_slot, active_hero_id in company.active_party_slots.items()
            },
        )
    ]


def _next_recruit_id(company: CompanyState, recruit: RecruitChoice) -> str:
    stem = recruit.name.lower().replace(" ", "_")
    used_ids = {hero.hero_id for hero in company.roster}
    used_ids.update(hero.hero_id for hero in company.deceased_heroes)
    index = 1
    while f"recruit_{stem}_{index}" in used_ids:
        index += 1
    return f"recruit_{stem}_{index}"


def _preferred_slot(definitions: GameDefinitions, class_id: str) -> FormationSlot:
    for recruit in definitions.recruits.starting_roster:
        if recruit.class_id == class_id:
            return recruit.formation_slot
    return FormationSlot.FRONT_LEFT
