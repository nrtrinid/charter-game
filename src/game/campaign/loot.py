"""Town relic broker helpers."""

from __future__ import annotations

from game.campaign.company import CompanyState
from game.campaign.economy import add_coin
from game.content.definitions import GameDefinitions
from game.core.events import LootSoldEvent, LootTurnedInEvent
from game.core.result import Result


def sell_loot(
    company: CompanyState,
    definitions: GameDefinitions,
    item_id: str,
) -> Result[CompanyState]:
    loot = definitions.loot.get(item_id)
    if loot is None:
        return Result.fail(f"Unknown loot item: {item_id}")
    if loot.sell_price is None or loot.sell_price <= 0:
        return Result.fail(f"{loot.name} cannot be sold here.")
    owned = company.inventory.get(item_id, 0)
    if owned < 1:
        return Result.fail(f"No {loot.name.lower()} in company inventory.")

    company.inventory[item_id] = owned - 1
    if company.inventory[item_id] <= 0:
        company.inventory.pop(item_id, None)
    add_coin(company, loot.sell_price)
    return Result.ok(
        company,
        [
            LootSoldEvent(
                message=f"Sold {loot.name} for {loot.sell_price} Coin.",
                item_id=item_id,
                quantity=1,
                coin=loot.sell_price,
            )
        ],
    )


def turn_in_loot(
    company: CompanyState,
    definitions: GameDefinitions,
    item_id: str,
) -> Result[CompanyState]:
    loot = definitions.loot.get(item_id)
    if loot is None:
        return Result.fail(f"Unknown loot item: {item_id}")
    if not loot.turn_in_flag:
        return Result.fail(f"{loot.name} cannot be filed here.")
    if company.flags.get(loot.turn_in_flag):
        return Result.fail("That proof has already been filed.")
    owned = company.inventory.get(item_id, 0)
    if owned < 1:
        return Result.fail(f"No {loot.name.lower()} in company inventory.")

    company.inventory[item_id] = owned - 1
    if company.inventory[item_id] <= 0:
        company.inventory.pop(item_id, None)
    company.flags[loot.turn_in_flag] = True

    message = f"Filed {loot.name} with Haven's relic clerk."
    unlocked_contract_id = loot.turn_in_unlocks_contract
    if unlocked_contract_id:
        contract = definitions.contracts.get(unlocked_contract_id)
        if contract is not None:
            message = (
                f"{message} {contract.name} is now posted on the charter board."
            )
    return Result.ok(
        company,
        [
            LootTurnedInEvent(
                message=message,
                item_id=item_id,
                flag_id=loot.turn_in_flag,
                unlocked_contract_id=unlocked_contract_id,
            )
        ],
    )
