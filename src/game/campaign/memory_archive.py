"""Persistent company timeline and hero memory archival."""

from __future__ import annotations

from game.campaign.company import (
    CompanyState,
    CompanyTimelineEntry,
    ExpeditionReportState,
    HeroMemoryEntry,
    ReportEventSignal,
)
from game.campaign.memory_capture import _hero_name
from game.campaign.memory_util import _label
from game.combat.combat_state import LifeState


def _record_company_timeline(
    company: CompanyState,
    report: ExpeditionReportState,
) -> list[CompanyTimelineEntry]:
    entries: list[CompanyTimelineEntry] = []
    entries.append(
        _add_timeline(
            company,
            kind="expedition_started",
            summary=(
                f"{_label(report.expedition_id)} expedition began"
                f" in {_label(report.dungeon_id or 'unknown')}."
            ),
            report=report,
        )
    )
    entries.append(
        _add_timeline(
            company,
            kind=_outcome_kind(report.outcome),
            summary=f"{_label(report.expedition_id)} expedition {_outcome_text(report.outcome)}.",
            report=report,
        )
    )

    for signal in report.event_signals:
        if signal.kind == "breach_discovered":
            entries.append(
                _add_timeline(
                    company,
                    kind="breach_discovered",
                    summary=signal.message,
                    report=report,
                    node_id=signal.node_id,
                )
            )
        elif signal.kind == "contract_completed":
            entries.append(
                _add_timeline(
                    company,
                    kind="contract_completed",
                    summary=signal.message,
                    report=report,
                    node_id=signal.node_id,
                )
            )
        elif signal.kind == "known_route_unlocked":
            entries.append(
                _add_timeline(
                    company,
                    kind="known_route_unlocked",
                    summary=signal.message,
                    report=report,
                    node_id=signal.node_id,
                )
            )
        elif signal.kind == "encounter_resolved":
            entries.append(
                _add_timeline(
                    company,
                    kind="encounter_resolved",
                    summary=signal.message,
                    report=report,
                    encounter_id=signal.encounter_id,
                )
            )
        elif signal.kind == "combat_retreat":
            entries.append(
                _add_timeline(
                    company,
                    kind="combat_retreat",
                    summary=signal.message,
                    report=report,
                    node_id=signal.node_id,
                    encounter_id=signal.encounter_id,
                )
            )
        elif signal.kind == "maze_route_collapsed":
            entries.append(
                _add_timeline(
                    company,
                    kind="maze_route_collapsed",
                    summary=signal.message,
                    report=report,
                    node_id=signal.node_id,
                )
            )
    return entries
def _record_hero_memories(
    company: CompanyState,
    report: ExpeditionReportState,
) -> list[HeroMemoryEntry]:
    entries: list[HeroMemoryEntry] = []
    for hero_id in report.participant_ids:
        start = report.start_hero_states.get(hero_id)
        end = report.end_hero_states.get(hero_id)
        if start is None or end is None:
            continue
        if not _has_hero_memory(company, hero_id, "first_expedition"):
            entries.append(
                _add_hero_memory(
                    company,
                    hero_id=hero_id,
                    hero_name=start.name,
                    kind="first_expedition",
                    summary=f"{start.name} first marched with the company.",
                    report=report,
                )
            )
        if report.breaches_discovered and end.life_state != LifeState.DEAD.value:
            entries.append(
                _add_hero_memory(
                    company,
                    hero_id=hero_id,
                    hero_name=start.name,
                    kind="breach_discovered",
                    summary=(
                        f"{start.name} stood with the company at "
                        f"{', '.join(report.breaches_discovered)}."
                    ),
                    report=report,
                )
            )

    seen_signals: set[tuple[str, str | None, str | None]] = set()
    for signal in report.event_signals:
        if signal.hero_id is None or signal.kind not in {"downed", "mortal_wound", "death"}:
            continue
        key = (signal.kind, signal.hero_id, signal.encounter_id)
        if key in seen_signals:
            continue
        seen_signals.add(key)
        entries.append(
            _add_hero_memory(
                company,
                hero_id=signal.hero_id,
                hero_name=signal.hero_name or _hero_name(report, signal.hero_id),
                kind=signal.kind,
                summary=_hero_signal_summary(signal),
                report=report,
                node_id=signal.node_id,
                encounter_id=signal.encounter_id,
            )
        )

    for outcome in report.hero_outcomes:
        if outcome.died:
            continue
        if outcome.wounded or outcome.downed or outcome.mortal_wounds_delta > 0:
            entries.append(
                _add_hero_memory(
                    company,
                    hero_id=outcome.hero_id,
                    hero_name=outcome.hero_name,
                    kind="survived_wounded",
                    summary=(
                        f"{outcome.hero_name} survived {_label(report.expedition_id)} "
                        "with wounds recorded."
                    ),
                    report=report,
                )
            )
    return entries
def _add_hero_memory(
    company: CompanyState,
    *,
    hero_id: str,
    hero_name: str,
    kind: str,
    summary: str,
    report: ExpeditionReportState,
    node_id: str | None = None,
    encounter_id: str | None = None,
) -> HeroMemoryEntry:
    entry = HeroMemoryEntry(
        entry_id=f"hero_memory_{len(company.hero_memories) + 1:04d}",
        hero_id=hero_id,
        hero_name=hero_name,
        kind=kind,
        summary=summary,
        expedition_id=report.expedition_id,
        dungeon_id=report.dungeon_id or "",
        node_id=node_id,
        encounter_id=encounter_id,
    )
    company.hero_memories.append(entry)
    return entry


def _add_timeline(
    company: CompanyState,
    *,
    kind: str,
    summary: str,
    report: ExpeditionReportState,
    node_id: str | None = None,
    encounter_id: str | None = None,
) -> CompanyTimelineEntry:
    entry = CompanyTimelineEntry(
        entry_id=f"company_timeline_{len(company.company_timeline) + 1:04d}",
        kind=kind,
        summary=summary,
        expedition_id=report.expedition_id,
        dungeon_id=report.dungeon_id or "",
        node_id=node_id,
        encounter_id=encounter_id,
    )
    company.company_timeline.append(entry)
    return entry


def _has_hero_memory(company: CompanyState, hero_id: str, kind: str) -> bool:
    return any(
        memory.hero_id == hero_id and memory.kind == kind for memory in company.hero_memories
    )


def _hero_signal_summary(signal: ReportEventSignal) -> str:
    name = signal.hero_name or "A hero"
    if signal.kind == "death":
        if signal.encounter_id is None:
            return f"{name} fell during the expedition."
        return f"{name} fell during {_label(signal.encounter_id)}."
    if signal.kind == "mortal_wound":
        return signal.message
    return signal.message
def _outcome_kind(outcome: str) -> str:
    if outcome == "defeat":
        return "expedition_defeated"
    if outcome == "descended_maze_depth_1":
        return "maze_descent"
    return "expedition_returned"


def _outcome_text(outcome: str) -> str:
    if outcome == "defeat":
        return "ended in defeat"
    if outcome == "descended_maze_depth_1":
        return "returned after a Maze descent"
    return "returned to Haven"
