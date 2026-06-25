"""Signal policy, deduplication, and legacy/synthetic memory sources."""

from __future__ import annotations

from collections.abc import Sequence

from game.campaign.company import ExpeditionReportState
from game.campaign.hero_memory import RecentSignal
from game.combat.combat_state import MoraleState

ROUTINE_FRESH_MEMORY_FAMILIES = frozenset(
    {
        "breach_witness",
        "maze_thread",
        "relic_greed",
    }
)
PERSONAL_STACKING_MEMORY_FAMILIES = frozenset(
    {
        "killing_blow",
        "marked_execution",
        "field_treatment",
        "morale_rally",
        "shaken_survival",
        "downed_survival",
        "broken_survival",
        "ally_downed_witnessed",
    }
)
ROUTINE_FRESH_MEMORY_TAGS = frozenset({"loot", "route"})
def _memory_signals_by_hero(
    report: ExpeditionReportState,
) -> dict[str, tuple[RecentSignal, ...]]:
    grouped: dict[str, list[RecentSignal]] = {}
    signals = [*_report_memory_signals(report), *_post_report_memory_signals(report)]
    for signal in signals:
        grouped.setdefault(signal.hero_id, []).append(signal)
    return {
        hero_id: tuple(sorted(signals, key=lambda signal: signal.order))
        for hero_id, signals in grouped.items()
    }


def _post_report_memory_signals(report: ExpeditionReportState) -> list[RecentSignal]:
    signals: list[RecentSignal] = []
    order = len(report.memory_signals) + 1
    for outcome in report.hero_outcomes:
        if outcome.died:
            continue
        end = report.end_hero_states.get(outcome.hero_id)
        if end is None:
            continue
        if outcome.downed:
            signals.append(
                RecentSignal(
                    hero_id=outcome.hero_id,
                    family_id="downed_survival",
                    tags=("downed", "survival"),
                    source_summary=f"{outcome.hero_name} survived being Downed.",
                    order=order,
                )
            )
            order += 1
        if end.morale == MoraleState.BROKEN.name:
            signals.append(
                RecentSignal(
                    hero_id=outcome.hero_id,
                    family_id="broken_survival",
                    tags=("morale", "survival"),
                    source_summary=f"{outcome.hero_name} came back Broken.",
                    order=order,
                )
            )
            order += 1
        elif end.morale == MoraleState.SHAKEN.name:
            signals.append(
                RecentSignal(
                    hero_id=outcome.hero_id,
                    family_id="shaken_survival",
                    tags=("morale", "survival"),
                    source_summary=f"{outcome.hero_name} came back Shaken.",
                    order=order,
                )
            )
            order += 1
    return signals


def _fresh_signal_allowlist(
    report: ExpeditionReportState,
) -> set[tuple[str, str, str | None, str | None, str, int]]:
    allowed: set[tuple[str, str, str | None, str | None, str, int]] = set()
    grouped: dict[
        tuple[str, str | None, str | None, str],
        list[RecentSignal],
    ] = {}
    for signal in _report_memory_signals(report):
        if not _is_routine_fresh_signal(signal) or _is_major_fresh_signal(signal):
            allowed.add(_signal_identity(signal))
            continue
        grouped.setdefault(_signal_group_key(signal), []).append(signal)

    participant_order = {
        hero_id: index for index, hero_id in enumerate(report.participant_ids)
    }
    for group_key, signals in grouped.items():
        ordered = sorted(
            signals,
            key=lambda signal: (participant_order.get(signal.hero_id, 999), signal.hero_id),
        )
        if not ordered:
            continue
        selected_index = _stable_index(
            "|".join(
                (
                    report.expedition_id,
                    report.dungeon_id or "",
                    report.outcome,
                    group_key[0],
                    group_key[1] or "",
                    group_key[2] or "",
                    group_key[3],
                    ",".join(report.participant_ids),
                )
            ),
            len(ordered),
        )
        allowed.add(_signal_identity(ordered[selected_index]))
    return allowed


def _is_routine_fresh_signal(signal: RecentSignal) -> bool:
    if signal.family_id in PERSONAL_STACKING_MEMORY_FAMILIES:
        return False
    return (
        signal.family_id in ROUTINE_FRESH_MEMORY_FAMILIES
        or bool(ROUTINE_FRESH_MEMORY_TAGS.intersection(signal.tags))
    )


def _is_major_fresh_signal(signal: RecentSignal) -> bool:
    return signal.family_id == "breach_witness"


def _signal_group_key(
    signal: RecentSignal,
) -> tuple[str, str | None, str | None, str]:
    return (
        signal.family_id,
        signal.node_id,
        signal.encounter_id,
        signal.source_summary,
    )


def _signal_identity(
    signal: RecentSignal,
) -> tuple[str, str, str | None, str | None, str, int]:
    return (
        signal.hero_id,
        signal.family_id,
        signal.node_id,
        signal.encounter_id,
        signal.source_summary,
        signal.order,
    )


def _stable_index(seed_text: str, count: int) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(seed_text)) % count


def _report_memory_signals(report: ExpeditionReportState) -> tuple[RecentSignal, ...]:
    if report.memory_signals:
        return tuple(sorted(report.memory_signals, key=lambda signal: signal.order))
    return tuple(_legacy_report_memory_signals(report))


def _legacy_report_memory_signals(report: ExpeditionReportState) -> list[RecentSignal]:
    signals: list[RecentSignal] = []
    for event_signal in report.event_signals:
        if event_signal.kind == "killing_blow" and event_signal.hero_id is not None:
            signals.append(
                _recent_signal(
                    report,
                    hero_id=event_signal.hero_id,
                    family_id="killing_blow",
                    tags=("kill", "combat"),
                    source_summary=event_signal.message,
                    node_id=event_signal.node_id,
                    encounter_id=event_signal.encounter_id,
                )
            )
        elif event_signal.kind in {"tag_frozen", "tag_shocked"} and event_signal.hero_id:
            signals.append(
                _recent_signal(
                    report,
                    hero_id=event_signal.hero_id,
                    family_id="frost_shock",
                    tags=("frozen", "shock", "combat"),
                    source_summary=event_signal.message,
                    node_id=event_signal.node_id,
                    encounter_id=event_signal.encounter_id,
                )
            )

    if report.breaches_discovered:
        for hero_id in report.participant_ids:
            signals.append(
                _recent_signal(
                    report,
                    hero_id=hero_id,
                    family_id="breach_witness",
                    tags=("breach", "maze"),
                    source_summary="Breach discovered: "
                    + ", ".join(report.breaches_discovered)
                    + ".",
                )
            )
    if report.outcome == "descended_maze_depth_1" or any(
        signal.kind == "maze_route_collapsed" for signal in report.event_signals
    ):
        for hero_id in report.participant_ids:
            signals.append(
                _recent_signal(
                    report,
                    hero_id=hero_id,
                    family_id="maze_thread",
                    tags=("maze", "route"),
                    source_summary="The company carried a Maze route home.",
                )
            )
    if sum(report.loot.values()) > 0:
        for hero_id in report.participant_ids:
            signals.append(
                _recent_signal(
                    report,
                    hero_id=hero_id,
                    family_id="relic_greed",
                    tags=("loot", "greed"),
                    source_summary="The expedition returned with loot.",
                )
            )
    return _dedupe_memory_signals(signals)


def _recent_signal(
    report: ExpeditionReportState,
    *,
    hero_id: str,
    family_id: str,
    tags: tuple[str, ...],
    source_summary: str,
    node_id: str | None = None,
    encounter_id: str | None = None,
    score: int = 1,
) -> RecentSignal:
    return RecentSignal(
        hero_id=hero_id,
        family_id=family_id,
        score=score,
        tags=tags,
        source_summary=source_summary,
        node_id=node_id,
        encounter_id=encounter_id,
        order=len(report.memory_signals) + 1,
    )


def _dedupe_memory_signals(signals: Sequence[RecentSignal]) -> list[RecentSignal]:
    deduped: list[RecentSignal] = []
    seen: set[tuple[str, str, str | None, str | None, str]] = set()
    for index, signal in enumerate(signals, start=1):
        key = (
            signal.hero_id,
            signal.family_id,
            signal.node_id,
            signal.encounter_id,
            signal.source_summary,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            RecentSignal(
                hero_id=signal.hero_id,
                family_id=signal.family_id,
                score=signal.score,
                tags=signal.tags,
                source_summary=signal.source_summary,
                node_id=signal.node_id,
                encounter_id=signal.encounter_id,
                order=signal.order or index,
            )
        )
    return deduped
