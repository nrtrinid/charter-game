"""Persistent consequence memory for expedition reports."""

from __future__ import annotations

from collections.abc import Sequence

from game.campaign.company import (
    CompanyState,
    CompanyTimelineEntry,
    ExpeditionReportState,
    HeroMemoryEntry,
    HeroReportOutcome,
    HeroReportSnapshot,
    HeroState,
    ReportEventSignal,
)
from game.campaign.hero_memory import (
    MEMORY_FAMILIES,
    ManifestationResult,
    RecentSignal,
    apply_signal_to_career,
    apply_signal_to_fresh_memories,
    flat_quirks_from_slots,
    manifest_pending_memories,
    memory_family,
)
from game.campaign.town import clear_surgery_recovery
from game.combat.combat_state import LifeState, MoraleState, StrainMark, StrainTier
from game.combat.traits import QUIRK_GOLD_FEVER
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
from game.core.rng import GameRng

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


def _apply_post_expedition_traits(
    company: CompanyState,
    report: ExpeditionReportState,
) -> list[str]:
    moments: list[str] = []
    participant_ids = set(report.participant_ids)
    heroes = {hero.hero_id: hero for hero in company.roster}
    memory_signals = _memory_signals_by_hero(report)
    fresh_signal_allowlist = _fresh_signal_allowlist(report)
    manifestation_rng = GameRng(_manifestation_seed(report))

    for hero in company.roster:
        if hero.hero_id not in participant_ids:
            _rest_hero_strain(hero)

    for outcome in report.hero_outcomes:
        outcome_hero = heroes.get(outcome.hero_id)
        if outcome_hero is None or outcome.died:
            continue
        strain_moments = _apply_expedition_strain(outcome_hero, outcome, report)
        moments.extend(strain_moments)

        moments.extend(
            _apply_hero_memory_signals(
                company,
                outcome_hero,
                report,
                memory_signals.get(outcome.hero_id, ()),
                fresh_signal_allowlist,
                manifestation_rng,
            )
        )

        if QUIRK_GOLD_FEVER in outcome_hero.quirks:
            loot_total = sum(report.loot.values())
            if loot_total > 0 and _raise_hero_morale(outcome_hero):
                moments.append(f"{outcome_hero.name}'s Coin Fever brightened after the haul.")
            elif (
                loot_total == 0 and report.outcome != "defeat" and _lower_hero_morale(outcome_hero)
            ):
                moments.append(f"{outcome_hero.name}'s Coin Fever soured without loot.")
    return moments


def _apply_hero_memory_signals(
    company: CompanyState,
    hero: HeroState,
    report: ExpeditionReportState,
    signals: Sequence[RecentSignal],
    fresh_signal_allowlist: set[tuple[str, str, str | None, str | None, str, int]],
    rng: GameRng,
) -> list[str]:
    moments: list[str] = []
    routine_fresh_applied: set[str] = set()
    for signal in signals:
        hero.career_signals = apply_signal_to_career(hero.career_signals, signal)
        if signal.family_id not in MEMORY_FAMILIES:
            continue
        fresh_signal = signal
        if _is_routine_fresh_signal(signal):
            if signal.family_id in routine_fresh_applied:
                continue
            if _signal_identity(signal) not in fresh_signal_allowlist:
                continue
            routine_fresh_applied.add(signal.family_id)
            fresh_signal = RecentSignal(
                hero_id=signal.hero_id,
                family_id=signal.family_id,
                score=1,
                tags=signal.tags,
                source_summary=signal.source_summary,
                node_id=signal.node_id,
                encounter_id=signal.encounter_id,
                order=signal.order,
            )
        update = apply_signal_to_fresh_memories(hero.fresh_memories, fresh_signal)
        hero.fresh_memories = list(update.memories)
        family = memory_family(signal.family_id)
        if update.strengthened:
            moments.append(f"{hero.name}'s {family.display_name} deepens.")
        elif update.replaced_family_id:
            moments.append(f"{hero.name}'s {family.display_name} takes hold.")
        else:
            moments.append(f"{hero.name}'s {family.display_name} takes hold.")
        for summary in update.permanent_memory_summaries:
            hero_summary = _hero_permanent_memory_summary(hero, summary)
            _add_hero_memory(
                company,
                hero_id=hero.hero_id,
                hero_name=hero.name,
                kind="fresh_memory_settled",
                summary=hero_summary,
                report=report,
                node_id=signal.node_id,
                encounter_id=signal.encounter_id,
            )
            moments.append(hero_summary)

    fresh_memories, earned_slots, results = manifest_pending_memories(
        hero_id=hero.hero_id,
        hero_name=hero.name,
        fresh_memories=hero.fresh_memories,
        earned_slots=hero.earned_quirk_slots,
        career_signals=hero.career_signals,
        rng=rng,
    )
    hero.fresh_memories = fresh_memories
    hero.earned_quirk_slots = earned_slots
    hero.quirks = flat_quirks_from_slots(earned_slots)
    for result in results:
        record_summaries = _manifestation_record_summaries(hero, result)
        moments.extend(record_summaries)
        summaries = record_summaries or (
            (result.permanent_memory_summary,) if result.permanent_memory_summary else ()
        )
        for summary in summaries:
            _add_hero_memory(
                company,
                hero_id=hero.hero_id,
                hero_name=hero.name,
                kind=_manifestation_memory_kind(result.outcome),
                summary=summary,
                report=report,
            )
    return moments


def _strain_mark_candidate(
    outcome: HeroReportOutcome,
    report: ExpeditionReportState,
) -> StrainMark | None:
    if outcome.downed or outcome.mortal_wounds_delta > 0:
        return StrainMark.BATTERED
    if report.outcome == "defeat" or any(
        signal.kind in {"combat_retreat", "maze_route_collapsed"} for signal in report.event_signals
    ):
        return StrainMark.FRAYED
    start = report.start_hero_states.get(outcome.hero_id)
    end = report.end_hero_states.get(outcome.hero_id)
    if start is not None and end is not None:
        effort_spent = start.effort - end.effort
        if effort_spent >= 2 or end.effort == 0:
            return StrainMark.DRAINED
    if any(signal.kind == "encounter_resolved" for signal in report.event_signals):
        return StrainMark.WINDED
    return None


def _apply_expedition_strain(
    hero: HeroState,
    outcome: HeroReportOutcome,
    report: ExpeditionReportState,
) -> list[str]:
    moments: list[str] = []
    had_encounter = any(signal.kind == "encounter_resolved" for signal in report.event_signals)
    if had_encounter:
        previous = hero.strain
        hero.strain = StrainTier(min(hero.strain.value + 1, StrainTier.SPENT.value))
        if hero.strain != previous and hero.strain.value >= StrainTier.WORN.value:
            moments.append(f"{hero.name}'s Strain rose to {hero.strain.name.title()}.")

    mark = _strain_mark_candidate(outcome, report)
    if mark is not None and mark not in hero.strain_marks:
        hero.strain_marks.add(mark)
        moments.append(f"{hero.name} came back {_trait_label(mark.value)}.")
    return moments


def _rest_hero_strain(hero: HeroState) -> None:
    if hero.strain.value > StrainTier.STEADY.value:
        hero.strain = StrainTier(hero.strain.value - 1)
    if hero.strain.value <= StrainTier.STEADY.value:
        hero.strain_marks.clear()


def _raise_hero_morale(hero: HeroState) -> bool:
    new_value = min(hero.morale.value + 1, MoraleState.INSPIRED.value)
    if new_value == hero.morale.value:
        return False
    hero.morale = MoraleState(new_value)
    return True


def _lower_hero_morale(hero: HeroState) -> bool:
    minimum = (
        MoraleState.STEADY.value
        if hero.personal_quirk == "grave_calm"
        else MoraleState.BROKEN.value
    )
    new_value = max(hero.morale.value - 1, minimum)
    if new_value == hero.morale.value:
        return False
    hero.morale = MoraleState(new_value)
    return True


def _trait_label(trait_id: str) -> str:
    return trait_id.replace("_", " ").title()


def _hero_permanent_memory_summary(hero: HeroState, summary: str) -> str:
    if "permanent memory" in summary:
        return summary.replace("permanent memory", f"{hero.name}'s permanent memories")
    return f"{summary} ({hero.name})"


def _manifestation_memory_kind(outcome: str) -> str:
    if outcome == "reinforced":
        return "quirk_reinforced"
    if outcome in {"added", "replaced"}:
        return "quirk_manifested"
    return "fresh_memory_settled"


def _manifestation_record_summaries(
    hero: HeroState,
    result: ManifestationResult,
) -> tuple[str, ...]:
    outcome = getattr(result, "outcome", "")
    memory_name = getattr(result, "memory_display_name", "") or _label(
        getattr(result, "memory_family_id", "")
    )
    quirk_name = getattr(result, "quirk_display_name", "") or _label(
        getattr(result, "quirk_id", "")
    )
    if outcome == "added":
        return (f"{hero.name} developed {quirk_name} from repeated {memory_name}.",)
    if outcome == "reinforced":
        lines: tuple[str, ...] = (f"{quirk_name} deepened from another {memory_name}.",)
        if getattr(result, "locked", False):
            lines += (f"{quirk_name} became locked.",)
        return lines
    if outcome == "replaced":
        replaced_name = getattr(result, "replaced_quirk_display_name", "") or _label(
            getattr(result, "replaced_quirk_id", "")
        )
        return (f"{quirk_name} replaced {replaced_name} from {memory_name}.",)
    if outcome == "no_eligible_quirk":
        return (f"{memory_name} settled without a new quirk.",)
    if outcome == "all_locked":
        return (f"{memory_name} settled without a new quirk; all earned slots were locked.",)
    if outcome == "already_locked":
        return (f"{quirk_name} was already locked from another {memory_name}.",)
    return tuple(str(message) for message in getattr(result, "messages", ()) if message)


def _manifestation_seed(report: ExpeditionReportState) -> int:
    seed_text = "|".join(
        (
            report.expedition_id,
            report.dungeon_id or "",
            report.outcome,
            ",".join(report.participant_ids),
            ",".join(report.rooms_entered),
            str(len(report.event_signals)),
            str(len(report.memory_signals)),
        )
    )
    return sum((index + 1) * ord(character) for index, character in enumerate(seed_text))


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


def _hero_outcomes(report: ExpeditionReportState) -> list[HeroReportOutcome]:
    outcomes: list[HeroReportOutcome] = []
    for hero_id in report.participant_ids:
        start = report.start_hero_states.get(hero_id)
        end = report.end_hero_states.get(hero_id)
        if start is None or end is None:
            continue
        died = end.life_state == LifeState.DEAD.value
        downed = end.life_state == LifeState.DOWNED.value
        wound_delta = end.mortal_wounds - start.mortal_wounds
        wounded = end.hp < end.max_hp or wound_delta > 0
        if died:
            status = "died"
        elif downed:
            status = "downed"
        elif wounded:
            status = "wounded"
        else:
            status = "survived"
        outcomes.append(
            HeroReportOutcome(
                hero_id=hero_id,
                hero_name=start.name,
                class_id=start.class_id,
                status=status,
                start_hp=start.hp,
                end_hp=end.hp,
                max_hp=end.max_hp,
                start_mortal_wounds=start.mortal_wounds,
                end_mortal_wounds=end.mortal_wounds,
                mortal_wounds_delta=wound_delta,
                died=died,
                downed=downed,
                wounded=wounded,
            )
        )
    return outcomes


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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


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


def _hero_name(report: ExpeditionReportState, hero_id: str) -> str:
    snapshot = report.start_hero_states.get(hero_id) or report.end_hero_states.get(hero_id)
    return snapshot.name if snapshot is not None else hero_id


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


def _label(value: str) -> str:
    return value.replace("_", " ").title()
