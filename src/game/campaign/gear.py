"""Company gear inventory and one-slot kit helpers."""

from __future__ import annotations

from dataclasses import dataclass

from game.campaign.company import CompanyState, HeroState
from game.campaign.economy import spend_coin
from game.combat.combat_state import LifeState
from game.content.definitions import GameDefinitions
from game.core.events import CompanyEvent
from game.core.result import Result
from game.data.schemas import GearDefinition


@dataclass(frozen=True)
class EffectiveHeroStats:
    max_hp: int
    max_effort: int
    accuracy: int
    damage: int


def effective_hero_stats(
    hero: HeroState,
    definitions: GameDefinitions | None = None,
) -> EffectiveHeroStats:
    gear = _equipped_gear(hero, definitions)
    effects = gear.effects if gear is not None else None
    return EffectiveHeroStats(
        max_hp=hero.max_hp + (effects.max_hp_bonus if effects is not None else 0),
        max_effort=hero.max_effort
        + (effects.max_effort_bonus if effects is not None else 0),
        accuracy=hero.accuracy + (effects.accuracy_bonus if effects is not None else 0),
        damage=hero.damage + (effects.damage_bonus if effects is not None else 0),
    )


def gear_effect_summary(gear: GearDefinition) -> str:
    pieces: list[str] = []
    if gear.effects.max_hp_bonus:
        pieces.append(f"Max HP +{gear.effects.max_hp_bonus}")
    if gear.effects.max_effort_bonus:
        pieces.append(f"Max Effort +{gear.effects.max_effort_bonus}")
    if gear.effects.accuracy_bonus:
        pieces.append(f"Accuracy +{gear.effects.accuracy_bonus}")
    if gear.effects.damage_bonus:
        pieces.append(f"Damage +{gear.effects.damage_bonus}")
    return ", ".join(pieces) or "No stat effect"


def available_gear_count(
    company: CompanyState,
    gear_id: str,
) -> int:
    owned = company.gear_inventory.get(gear_id, 0)
    equipped = sum(
        1
        for hero in company.roster
        if hero.life_state != LifeState.DEAD and hero.equipped_gear_id == gear_id
    )
    return max(0, owned - equipped)


def gear_unavailable_reason(
    company: CompanyState,
    definitions: GameDefinitions,
    gear_id: str,
) -> str:
    gear = definitions.gear[gear_id]
    if gear.cost is None:
        return "Not sold by the armory."
    for contract_id in gear.requires_completed_contracts:
        if contract_id not in company.completed_contract_ids:
            contract = definitions.contracts.get(contract_id)
            if contract is None:
                return "Complete the required contract first."
            return f"Complete {contract.name} first."
    for breach_id in gear.requires_known_breaches:
        if breach_id not in company.known_breaches:
            location = definitions.locations.get(breach_id)
            if location is None:
                return "Find the required breach first."
            return f"Find {location.name} first."
    if company.coin < gear.cost:
        return f"Need {gear.cost - company.coin} more Coin."
    return ""


def purchase_gear(
    company: CompanyState,
    definitions: GameDefinitions,
    gear_id: str,
) -> Result[CompanyState]:
    gear = definitions.gear.get(gear_id)
    if gear is None:
        return Result.fail(f"Unknown gear: {gear_id}")
    unavailable = gear_unavailable_reason(company, definitions, gear_id)
    if unavailable:
        return Result.fail(unavailable)
    cost = gear.cost or 0
    spend_coin(company, cost)
    company.gear_inventory[gear_id] = company.gear_inventory.get(gear_id, 0) + 1
    return Result.ok(
        company,
        [
            CompanyEvent(
                message=f"Gear purchased - {gear.name}.",
            )
        ],
    )


def equip_gear(
    company: CompanyState,
    definitions: GameDefinitions,
    hero_id: str,
    gear_id: str,
) -> Result[CompanyState]:
    hero = _living_hero(company, hero_id)
    if hero is None:
        return Result.fail("Choose a living hero.")
    if gear_id not in definitions.gear:
        return Result.fail(f"Unknown gear: {gear_id}")
    if hero.equipped_gear_id != gear_id and available_gear_count(company, gear_id) <= 0:
        return Result.fail("No available copy of that gear.")
    old_name = _gear_name(definitions, hero.equipped_gear_id)
    hero.equipped_gear_id = gear_id
    _clamp_hero_to_effective_stats(hero, definitions)
    new_name = definitions.gear[gear_id].name
    return Result.ok(
        company,
        [
            CompanyEvent(
                message=(
                    f"{hero.name} equips {new_name}."
                    if not old_name
                    else f"{hero.name} swaps {old_name} for {new_name}."
                ),
            )
        ],
    )


def unequip_gear(
    company: CompanyState,
    definitions: GameDefinitions,
    hero_id: str,
) -> Result[CompanyState]:
    hero = _living_hero(company, hero_id)
    if hero is None:
        return Result.fail("Choose a living hero.")
    if hero.equipped_gear_id is None:
        return Result.fail(f"{hero.name} has no gear equipped.")
    gear_name = _gear_name(definitions, hero.equipped_gear_id) or "gear"
    hero.equipped_gear_id = None
    _clamp_hero_to_effective_stats(hero, definitions)
    return Result.ok(
        company,
        [CompanyEvent(message=f"{hero.name} removes {gear_name}.")],
    )


def grant_gear(
    company: CompanyState,
    gear_rewards: dict[str, int],
) -> None:
    for gear_id, quantity in gear_rewards.items():
        if quantity <= 0:
            continue
        company.gear_inventory[gear_id] = company.gear_inventory.get(gear_id, 0) + quantity


def clear_dead_hero_gear(hero: HeroState) -> None:
    if hero.life_state == LifeState.DEAD:
        hero.equipped_gear_id = None


def _equipped_gear(
    hero: HeroState,
    definitions: GameDefinitions | None,
) -> GearDefinition | None:
    if definitions is None or hero.equipped_gear_id is None:
        return None
    return definitions.gear.get(hero.equipped_gear_id)


def _clamp_hero_to_effective_stats(
    hero: HeroState,
    definitions: GameDefinitions,
) -> None:
    stats = effective_hero_stats(hero, definitions)
    hero.hp = min(hero.hp, stats.max_hp)
    hero.effort = min(hero.effort, stats.max_effort)


def _living_hero(company: CompanyState, hero_id: str) -> HeroState | None:
    return next(
        (
            hero
            for hero in company.roster
            if hero.hero_id == hero_id and hero.life_state != LifeState.DEAD
        ),
        None,
    )


def _gear_name(
    definitions: GameDefinitions,
    gear_id: str | None,
) -> str:
    if gear_id is None:
        return ""
    gear = definitions.gear.get(gear_id)
    return gear.name if gear is not None else gear_id
