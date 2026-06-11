"""Coin economy helpers."""

from __future__ import annotations

from game.campaign.company import CompanyState


def add_coin(company: CompanyState, amount: int) -> None:
    if amount <= 0:
        return
    company.coin += amount


def can_spend_coin(company: CompanyState, cost: int) -> bool:
    return cost <= 0 or company.coin >= cost


def spend_coin(company: CompanyState, cost: int) -> bool:
    if cost <= 0:
        return True
    if company.coin < cost:
        return False
    company.coin -= cost
    return True
