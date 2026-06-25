"""Ingest combat and expedition events into report memory signals."""

from __future__ import annotations

from collections.abc import Sequence

from game.campaign.company import ExpeditionReportState, ReportEventSignal
from game.campaign.hero_memory import RecentSignal
from game.campaign.memory_capture import _hero_name
from game.core.events import (
    BreachDiscoveredEvent,
    CombatRetreatedEvent,
    ContractCompletedEvent,
    DamageEvent,
    DeathEvent,
    DownedEvent,
    DungeonActionEvent,
    EncounterEndedEvent,
    EncounterStartedEvent,
    ExpeditionEvent,
    ExpeditionReturnedEvent,
    GameEvent,
    LootGainedEvent,
    MazeFrontierOpenedEvent,
    MazeRouteCollapsedEvent,
    MemorySignalEvent,
    StatusChangedEvent,
)
from game.core.hci import EventImportance, event_importance


def record_report_event_signals(
    report: ExpeditionReportState,
    events: Sequence[GameEvent],
) -> None:
    """Persist only the event facts needed for report finalization."""

    current_node_id: str | None = None
    current_encounter_id: str | None = None
    last_damage_source_by_target: dict[str, str] = {}
    participant_ids = set(report.participant_ids)
    for event in events:
        if isinstance(event, ExpeditionEvent):
            current_node_id = event.node_id
            if event.message.startswith(("Known route charted:", "Charted approach mapped:")):
                _append_signal(
                    report,
                    ReportEventSignal(
                        kind="known_route_unlocked",
                        message=event.message,
                        node_id=event.node_id,
                    ),
                )
            if event.node_id == "maze_depth_1" or (
                event.node_id.startswith("maze_run_") and event.node_id.endswith("_entry")
            ):
                _append_party_memory_signal(
                    report,
                    family_id="maze_thread",
                    message=event.message,
                    tags=("maze", "route"),
                    node_id=event.node_id,
                    encounter_id=current_encounter_id,
                )
        elif isinstance(event, EncounterStartedEvent):
            current_encounter_id = event.encounter_id
        elif isinstance(event, EncounterEndedEvent):
            _append_signal(
                report,
                ReportEventSignal(
                    kind="encounter_resolved",
                    message=event.message,
                    encounter_id=event.encounter_id,
                ),
            )
            current_encounter_id = None
        elif isinstance(event, BreachDiscoveredEvent):
            current_node_id = event.node_id
            _append_signal(
                report,
                ReportEventSignal(
                    kind="breach_discovered",
                    message=event.message,
                    node_id=event.node_id,
                ),
            )
            _append_party_memory_signal(
                report,
                family_id="breach_witness",
                message=event.message,
                tags=("breach", "maze"),
                node_id=event.node_id,
                encounter_id=current_encounter_id,
            )
        elif isinstance(event, ContractCompletedEvent):
            current_node_id = event.node_id
            _append_signal(
                report,
                ReportEventSignal(
                    kind="contract_completed",
                    message=event.message,
                    node_id=event.node_id,
                ),
            )
        elif isinstance(event, ExpeditionReturnedEvent):
            _append_signal(
                report,
                ReportEventSignal(
                    kind="expedition_returned",
                    message=event.message,
                ),
            )
        elif isinstance(event, MazeFrontierOpenedEvent):
            _append_signal(
                report,
                ReportEventSignal(
                    kind="maze_frontier_opened",
                    message=event.message,
                    node_id=event.node_id,
                ),
            )
            _append_party_memory_signal(
                report,
                family_id="maze_thread",
                message=event.message,
                tags=("maze", "frontier"),
                node_id=event.node_id,
                encounter_id=current_encounter_id,
            )
        elif isinstance(event, MazeRouteCollapsedEvent):
            _append_signal(
                report,
                ReportEventSignal(
                    kind="maze_route_collapsed",
                    message=event.message,
                    node_id=event.source_node_id,
                ),
            )
            _append_party_memory_signal(
                report,
                family_id="maze_thread",
                message=event.message,
                tags=("maze", "route"),
                node_id=event.source_node_id,
                encounter_id=current_encounter_id,
            )
        elif isinstance(event, CombatRetreatedEvent):
            _append_signal(
                report,
                ReportEventSignal(
                    kind="combat_retreat",
                    message=event.message,
                    node_id=event.to_node_id,
                    encounter_id=event.encounter_id,
                ),
            )
            if event.actor_id in participant_ids:
                _append_memory_signal(
                    report,
                    RecentSignal(
                        hero_id=event.actor_id,
                        family_id="combat_retreat",
                        tags=("retreat", "morale"),
                        source_summary=event.message,
                        node_id=event.to_node_id,
                        encounter_id=event.encounter_id,
                    ),
                )
        elif isinstance(event, MemorySignalEvent):
            if event.hero_id in participant_ids:
                _append_memory_signal(
                    report,
                    RecentSignal(
                        hero_id=event.hero_id,
                        family_id=event.family_id,
                        score=event.score,
                        tags=event.tags,
                        source_summary=event.source_summary or event.message,
                        node_id=event.node_id or current_node_id,
                        encounter_id=event.encounter_id or current_encounter_id,
                    ),
                )
        elif isinstance(event, LootGainedEvent):
            if event.inventory or event.gear:
                _append_party_memory_signal(
                    report,
                    family_id="relic_greed",
                    message=event.message,
                    tags=("loot", "greed"),
                    node_id=event.node_id,
                    encounter_id=current_encounter_id,
                )
        elif isinstance(event, DungeonActionEvent):
            if event.loot:
                _append_party_memory_signal(
                    report,
                    family_id="relic_greed",
                    message=event.message,
                    tags=("loot", "greed"),
                    node_id=event.node_id,
                    encounter_id=current_encounter_id,
                )
        elif isinstance(event, DamageEvent):
            last_damage_source_by_target[event.target_id] = event.source_id
        elif isinstance(event, DeathEvent) and event.actor_id not in participant_ids:
            source_id = last_damage_source_by_target.get(event.actor_id)
            if source_id in participant_ids:
                _append_signal(
                    report,
                    ReportEventSignal(
                        kind="killing_blow",
                        message=event.message,
                        hero_id=source_id,
                        hero_name=_hero_name(report, source_id),
                        node_id=current_node_id,
                        encounter_id=current_encounter_id,
                    ),
                )
        elif isinstance(event, DownedEvent) and event.actor_id in participant_ids:
            _append_signal(
                report,
                ReportEventSignal(
                    kind="downed",
                    message=event.message,
                    hero_id=event.actor_id,
                    hero_name=_hero_name(report, event.actor_id),
                    node_id=current_node_id,
                    encounter_id=current_encounter_id,
                ),
            )
        elif isinstance(event, DeathEvent) and event.actor_id in participant_ids:
            _append_signal(
                report,
                ReportEventSignal(
                    kind="death",
                    message=event.message,
                    hero_id=event.actor_id,
                    hero_name=_hero_name(report, event.actor_id),
                    node_id=current_node_id,
                    encounter_id=current_encounter_id,
                ),
            )
        elif (
            isinstance(event, StatusChangedEvent)
            and event.status == "mortal_wound"
            and event.added
            and event.actor_id in participant_ids
        ):
            _append_signal(
                report,
                ReportEventSignal(
                    kind="mortal_wound",
                    message=event.message,
                    hero_id=event.actor_id,
                    hero_name=_hero_name(report, event.actor_id),
                    node_id=current_node_id,
                    encounter_id=current_encounter_id,
                ),
            )
        elif (
            isinstance(event, StatusChangedEvent)
            and event.status in {"frozen", "shocked"}
            and event.added
            and event.actor_id in participant_ids
        ):
            _append_signal(
                report,
                ReportEventSignal(
                    kind=f"tag_{event.status}",
                    message=event.message,
                    hero_id=event.actor_id,
                    hero_name=_hero_name(report, event.actor_id),
                    node_id=current_node_id,
                    encounter_id=current_encounter_id,
                ),
            )
        if event_importance(event) in {EventImportance.IMPORTANT, EventImportance.CRITICAL}:
            _append_signal(
                report,
                ReportEventSignal(
                    kind="notable_beat",
                    message=event.message,
                    node_id=current_node_id,
                    encounter_id=current_encounter_id,
                ),
            )

def _append_signal(report: ExpeditionReportState, signal: ReportEventSignal) -> None:
    key = (
        signal.kind,
        signal.message,
        signal.hero_id,
        signal.node_id,
        signal.encounter_id,
    )
    if key not in {
        (
            existing.kind,
            existing.message,
            existing.hero_id,
            existing.node_id,
            existing.encounter_id,
        )
        for existing in report.event_signals
    }:
        report.event_signals.append(signal)


def _append_party_memory_signal(
    report: ExpeditionReportState,
    *,
    family_id: str,
    message: str,
    tags: tuple[str, ...],
    node_id: str | None = None,
    encounter_id: str | None = None,
) -> None:
    for hero_id in report.participant_ids:
        _append_memory_signal(
            report,
            RecentSignal(
                hero_id=hero_id,
                family_id=family_id,
                tags=tags,
                source_summary=message,
                node_id=node_id,
                encounter_id=encounter_id,
            ),
        )


def _append_memory_signal(report: ExpeditionReportState, signal: RecentSignal) -> None:
    order = signal.order or len(report.memory_signals) + 1
    normalized = RecentSignal(
        hero_id=signal.hero_id,
        family_id=signal.family_id,
        score=signal.score,
        tags=signal.tags,
        source_summary=signal.source_summary,
        node_id=signal.node_id,
        encounter_id=signal.encounter_id,
        order=order,
    )
    key = (
        normalized.hero_id,
        normalized.family_id,
        normalized.node_id,
        normalized.encounter_id,
        normalized.source_summary,
    )
    if key not in {
        (
            existing.hero_id,
            existing.family_id,
            existing.node_id,
            existing.encounter_id,
            existing.source_summary,
        )
        for existing in report.memory_signals
    }:
        report.memory_signals.append(normalized)
