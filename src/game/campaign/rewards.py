"""Shared reward grants for campaign resources."""

from __future__ import annotations

from game.campaign.company import CompanyState
from game.campaign.economy import add_coin
from game.campaign.gear import grant_gear
from game.campaign.reputation import add_reputation
from game.core.events import GameEvent, LootGainedEvent
from game.data.schemas import ContractDefinition


def grant_contract_rewards(
    company: CompanyState,
    contract: ContractDefinition,
    node_id: str,
) -> list[GameEvent]:
    gear_rewards = dict(contract.reward_gear)
    if contract.reward_reputation:
        add_reputation(company, contract.reward_reputation)
    if contract.coin_reward:
        add_coin(company, contract.coin_reward)
    if gear_rewards:
        grant_gear(company, gear_rewards)
    if not (contract.reward_reputation or contract.coin_reward or gear_rewards):
        return []
    return [
        LootGainedEvent(
            message=contract_reward_message(
                contract.reward_reputation,
                contract.coin_reward,
                gear_rewards,
            ),
            node_id=node_id,
            inventory={},
            supplies={},
            reputation=contract.reward_reputation,
            coin=contract.coin_reward,
            gear=gear_rewards,
        )
    ]


def contract_reward_message(
    reputation: int,
    coin: int,
    gear_rewards: dict[str, int],
) -> str:
    pieces: list[str] = []
    if coin:
        pieces.append(f"{coin} Coin")
    if reputation:
        pieces.append(f"{reputation} reputation")
    pieces.extend(
        f"{quantity} {gear_id.replace('_', ' ')}"
        for gear_id, quantity in sorted(gear_rewards.items())
    )
    return "Reward gained: " + ", ".join(pieces) + "."
