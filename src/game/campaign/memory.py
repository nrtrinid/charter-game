"""Persistent consequence memory for expedition reports (facade)."""

from __future__ import annotations

from game.campaign.company import CompanyState, ExpeditionReportState
from game.campaign.memory_archive import (
    _record_company_timeline,
    _record_hero_memories,
)
from game.campaign.memory_capture import (
    _end_snapshots,
    capture_report_start,
)
from game.campaign.memory_event_signals import record_report_event_signals
from game.campaign.memory_post_expedition import (
    _apply_post_expedition_traits,
    _hero_outcomes,
)
from game.campaign.memory_util import _dedupe
from game.campaign.town import clear_surgery_recovery

__all__ = [
    "capture_report_start",
    "finalize_report_memory",
    "record_report_event_signals",
]


def finalize_report_memory(
    company: CompanyState,
    report: ExpeditionReportState,
    outcome: str,
) -> None:
    report.outcome = outcome
    report.end_reputation = company.reputation
    report.end_coin = company.coin
    report.end_supplies = dict(company.supplies)
    report.end_inventory = dict(company.inventory)
    report.end_gear_inventory = dict(company.gear_inventory)
    report.end_hero_states = _end_snapshots(company, report.participant_ids)
    report.hero_outcomes = _hero_outcomes(report)
    surgery_moments = clear_surgery_recovery(company)
    trait_moments = _apply_post_expedition_traits(company, report)
    report.end_hero_states = _end_snapshots(company, report.participant_ids)

    created_company_entries = _record_company_timeline(company, report)
    created_hero_entries = _record_hero_memories(company, report)
    notable_moments: list[str] = []
    notable_moments.extend(entry.summary for entry in created_company_entries)
    notable_moments.extend(entry.summary for entry in created_hero_entries)
    notable_moments.extend(surgery_moments)
    notable_moments.extend(trait_moments)
    notable_moments.extend(
        signal.message for signal in report.event_signals if signal.kind == "notable_beat"
    )
    report.notable_moments = _dedupe(notable_moments)
