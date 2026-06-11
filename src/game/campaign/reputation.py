"""Reputation rules."""

from __future__ import annotations

from game.campaign.company import CompanyState


def add_reputation(company: CompanyState, amount: int) -> None:
    company.reputation = max(0, company.reputation + amount)
