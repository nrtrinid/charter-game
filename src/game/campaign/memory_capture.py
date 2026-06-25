"""Expedition report baseline capture and hero snapshots."""

from __future__ import annotations

from collections.abc import Sequence

from game.campaign.company import (
    CompanyState,
    ExpeditionReportState,
    HeroReportSnapshot,
    HeroState,
)


def capture_report_start(company: CompanyState, report: ExpeditionReportState) -> None:
    """Capture the operational baseline before expedition costs or rewards apply."""

    participants = _active_participant_ids(company)
    roster = {hero.hero_id: hero for hero in company.roster}
    report.participant_ids = [hero_id for hero_id in participants if hero_id in roster]
    report.start_reputation = company.reputation
    report.end_reputation = company.reputation
    report.start_coin = company.coin
    report.end_coin = company.coin
    report.start_supplies = dict(company.supplies)
    report.end_supplies = dict(company.supplies)
    report.start_inventory = dict(company.inventory)
    report.end_inventory = dict(company.inventory)
    report.start_gear_inventory = dict(company.gear_inventory)
    report.end_gear_inventory = dict(company.gear_inventory)
    report.start_hero_states = {
        hero_id: _snapshot_hero(roster[hero_id]) for hero_id in report.participant_ids
    }
    report.end_hero_states = dict(report.start_hero_states)
def _active_participant_ids(company: CompanyState) -> list[str]:
    ids: list[str] = []
    for hero_id in company.active_party_slots.values():
        if hero_id is not None and hero_id not in ids:
            ids.append(hero_id)
    return ids


def _end_snapshots(
    company: CompanyState,
    participant_ids: Sequence[str],
) -> dict[str, HeroReportSnapshot]:
    heroes = {hero.hero_id: hero for hero in (*company.roster, *company.deceased_heroes)}
    return {
        hero_id: _snapshot_hero(heroes[hero_id]) for hero_id in participant_ids if hero_id in heroes
    }


def _snapshot_hero(hero: HeroState) -> HeroReportSnapshot:
    return HeroReportSnapshot(
        hero_id=hero.hero_id,
        name=hero.name,
        class_id=hero.class_id,
        hp=hero.hp,
        max_hp=hero.max_hp,
        effort=hero.effort,
        max_effort=hero.max_effort,
        mortal_wounds=hero.mortal_wounds,
        morale=hero.morale.name,
        strain=hero.strain.name,
        life_state=hero.life_state.value,
        personal_quirk=hero.personal_quirk,
        quirks=list(hero.quirks),
        strain_marks=sorted(mark.value for mark in hero.strain_marks),
    )
def _hero_name(report: ExpeditionReportState, hero_id: str) -> str:
    snapshot = report.start_hero_states.get(hero_id) or report.end_hero_states.get(hero_id)
    return snapshot.name if snapshot is not None else hero_id
