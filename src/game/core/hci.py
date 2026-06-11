"""Shared HCI analysis models for player-facing clarity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from game.core.events import (
    ActivePartyChangedEvent,
    BreachDiscoveredEvent,
    CombatEffectEvent,
    CombatEndedEvent,
    CombatRetreatDeclaredEvent,
    CombatRetreatedEvent,
    ContractCompletedEvent,
    DamageEvent,
    DeathEvent,
    DownedEvent,
    DungeonActionEvent,
    EncounterEndedEvent,
    EncounterStartedEvent,
    EventType,
    ExpeditionEvent,
    ExpeditionReturnedEvent,
    GameEvent,
    HealingEvent,
    HeroRecruitedEvent,
    LootGainedEvent,
    LoreDiscoveredEvent,
    MazeFrontierOpenedEvent,
    MazeRouteCollapsedEvent,
    MoveEvent,
    RecoveryEvent,
    RecruitmentOfferedEvent,
    RoundEndedEvent,
    RoundStartedEvent,
    SkillUsedEvent,
    SuppliesPurchasedEvent,
    TownServiceEvent,
    TurnDelayedEvent,
    TurnPassedEvent,
)


class EventImportance(StrEnum):
    NOISE = "noise"
    NORMAL = "normal"
    IMPORTANT = "important"
    CRITICAL = "critical"


_IMPORTANCE_RANK = {
    EventImportance.NOISE: 0,
    EventImportance.NORMAL: 1,
    EventImportance.IMPORTANT: 2,
    EventImportance.CRITICAL: 3,
}


@dataclass(frozen=True)
class HeroStateSnapshot:
    hero_id: str
    name: str
    hp: int
    max_hp: int
    effort: int
    max_effort: int
    mortal_wounds: int
    statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateSnapshot:
    has_company: bool = False
    company_name: str = ""
    reputation: int | None = None
    coin: int | None = None
    supplies: tuple[tuple[str, int], ...] = ()
    inventory: tuple[tuple[str, int], ...] = ()
    location_id: str = ""
    location_name: str = ""
    active_expedition_id: str = ""
    dungeon_node_id: str = ""
    dungeon_pending_combat_node_id: str = ""
    combat_encounter_id: str = ""
    combat_round: int | None = None
    combat_actor_id: str = ""
    combat_selected_skill_id: str = ""
    heroes: tuple[HeroStateSnapshot, ...] = ()


@dataclass(frozen=True)
class StateDelta:
    category: str
    key: str
    label: str
    before: str
    after: str
    summary: str
    importance: EventImportance = EventImportance.NORMAL
    delta: int | None = None


@dataclass(frozen=True)
class EventBeat:
    title: str
    events: list[GameEvent]
    style: str = "cyan"
    combat: bool = False
    importance: EventImportance = EventImportance.NORMAL


@dataclass
class HciResultAnalysis:
    before: StateSnapshot
    after: StateSnapshot
    deltas: tuple[StateDelta, ...] = ()
    beats: tuple[EventBeat, ...] = ()
    summary: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class TacticalBrief:
    action: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    danger: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()


def build_hci_result_analysis(
    before: StateSnapshot,
    after: StateSnapshot,
    events: list[GameEvent],
    *,
    error: str | None = None,
) -> HciResultAnalysis:
    deltas = tuple(compare_state_snapshots(before, after))
    beats = tuple(build_event_beats(events))
    summary = tuple(_analysis_summary(before, after, deltas, beats, error))
    return HciResultAnalysis(
        before=before,
        after=after,
        deltas=deltas,
        beats=beats,
        summary=summary,
        error=error,
    )


def compare_state_snapshots(
    before: StateSnapshot,
    after: StateSnapshot,
) -> list[StateDelta]:
    deltas: list[StateDelta] = []
    if not before.has_company and after.has_company:
        deltas.append(
            StateDelta(
                category="company",
                key="company",
                label="Company",
                before="none",
                after=after.company_name,
                summary=f"Company active: {after.company_name}.",
                importance=EventImportance.IMPORTANT,
            )
        )
    elif before.has_company and not after.has_company:
        deltas.append(
            StateDelta(
                category="company",
                key="company",
                label="Company",
                before=before.company_name,
                after="none",
                summary=f"Company cleared: {before.company_name}.",
                importance=EventImportance.IMPORTANT,
            )
        )
    elif before.company_name != after.company_name and after.has_company:
        deltas.append(
            StateDelta(
                category="company",
                key="company",
                label="Company",
                before=before.company_name,
                after=after.company_name,
                summary=f"Company changed to {after.company_name}.",
                importance=EventImportance.IMPORTANT,
            )
        )

    if before.reputation is not None and after.reputation is not None:
        _append_int_delta(
            deltas,
            category="resource",
            key="reputation",
            label="Reputation",
            before=before.reputation,
            after=after.reputation,
        )
    if before.coin is not None and after.coin is not None:
        _append_int_delta(
            deltas,
            category="resource",
            key="coin",
            label="Coin",
            before=before.coin,
            after=after.coin,
        )

    _append_item_deltas(deltas, "supply", before.supplies, after.supplies, "Supply")
    _append_item_deltas(deltas, "inventory", before.inventory, after.inventory, "Inventory")

    if before.location_id != after.location_id or before.location_name != after.location_name:
        if after.location_name:
            deltas.append(
                StateDelta(
                    category="location",
                    key="location",
                    label="Location",
                    before=before.location_name or before.location_id or "none",
                    after=after.location_name or after.location_id,
                    summary=f"Location: {after.location_name or after.location_id}.",
                    importance=EventImportance.IMPORTANT,
                )
            )

    if before.dungeon_node_id != after.dungeon_node_id:
        if before.dungeon_node_id or after.dungeon_node_id:
            deltas.append(
                StateDelta(
                    category="dungeon",
                    key="node",
                    label="Dungeon Room",
                    before=before.dungeon_node_id or "none",
                    after=after.dungeon_node_id or "none",
                    summary=f"Dungeon room: {after.dungeon_node_id or 'none'}.",
                    importance=EventImportance.IMPORTANT,
                )
            )

    if before.dungeon_pending_combat_node_id != after.dungeon_pending_combat_node_id:
        pending = after.dungeon_pending_combat_node_id
        deltas.append(
            StateDelta(
                category="dungeon",
                key="pending_combat",
                label="Pending Combat",
                before=before.dungeon_pending_combat_node_id or "none",
                after=pending or "none",
                summary=(
                    f"Combat pending in {pending}."
                    if pending
                    else "Pending dungeon combat cleared."
                ),
                importance=EventImportance.IMPORTANT,
            )
        )

    if before.combat_encounter_id != after.combat_encounter_id:
        if before.combat_encounter_id or after.combat_encounter_id:
            deltas.append(
                StateDelta(
                    category="combat",
                    key="encounter",
                    label="Combat",
                    before=before.combat_encounter_id or "none",
                    after=after.combat_encounter_id or "none",
                    summary=(
                        f"Combat active: {after.combat_encounter_id}."
                        if after.combat_encounter_id
                        else "Combat ended."
                    ),
                    importance=EventImportance.CRITICAL
                    if before.combat_encounter_id and not after.combat_encounter_id
                    else EventImportance.IMPORTANT,
                )
            )

    if before.combat_actor_id != after.combat_actor_id and after.combat_actor_id:
        deltas.append(
            StateDelta(
                category="combat",
                key="actor",
                label="Active Actor",
                before=before.combat_actor_id or "none",
                after=after.combat_actor_id,
                summary=f"Next actor: {after.combat_actor_id}.",
                importance=EventImportance.NORMAL,
            )
        )

    deltas.extend(_hero_deltas(before.heroes, after.heroes))
    return deltas


def build_event_beats(events: list[GameEvent]) -> list[EventBeat]:
    beats: list[EventBeat] = []
    combat_events: list[GameEvent] = []

    def flush_combat() -> None:
        nonlocal combat_events
        if not combat_events:
            return
        beats.append(
            EventBeat(
                title="Combat Encounter",
                events=combat_events,
                style="red",
                combat=True,
                importance=max_event_importance(combat_events),
            )
        )
        combat_events = []

    for event in events:
        if is_combat_event(event):
            combat_events.append(event)
            if isinstance(event, EncounterEndedEvent):
                flush_combat()
            continue
        flush_combat()
        beats.append(
            EventBeat(
                title=event_title(event),
                events=[event],
                style=event_style(event),
                importance=event_importance(event),
            )
        )
    flush_combat()
    return beats


def event_importance(event: GameEvent) -> EventImportance:
    if isinstance(event, DeathEvent | BreachDiscoveredEvent | CombatRetreatedEvent):
        return EventImportance.CRITICAL
    if isinstance(event, CombatRetreatDeclaredEvent):
        return EventImportance.IMPORTANT
    if isinstance(event, DownedEvent):
        return EventImportance.CRITICAL
    if isinstance(event, CombatEndedEvent):
        return (
            EventImportance.CRITICAL
            if event.victor != "heroes"
            else EventImportance.IMPORTANT
        )
    if isinstance(event, EncounterEndedEvent):
        return (
            EventImportance.CRITICAL
            if event.victor != "heroes"
            else EventImportance.IMPORTANT
        )
    if isinstance(
        event,
        (
            ContractCompletedEvent,
            DungeonActionEvent,
            EncounterStartedEvent,
            ExpeditionReturnedEvent,
            HeroRecruitedEvent,
            LootGainedEvent,
            MazeFrontierOpenedEvent,
            MazeRouteCollapsedEvent,
            RecoveryEvent,
            SuppliesPurchasedEvent,
        ),
    ):
        return EventImportance.IMPORTANT
    if isinstance(event, ExpeditionEvent) and (event.major_beat or event.first_visit):
        return EventImportance.IMPORTANT
    if isinstance(
        event,
        DamageEvent
        | CombatEffectEvent
        | HealingEvent
        | MoveEvent
        | SkillUsedEvent
        | TurnDelayedEvent
        | TurnPassedEvent,
    ):
        return EventImportance.NORMAL
    if isinstance(event, RoundStartedEvent | RoundEndedEvent):
        return EventImportance.NOISE
    if isinstance(event, TownServiceEvent) and event.service_id in {"town", "ledger"}:
        return EventImportance.NOISE
    if isinstance(event, RecruitmentOfferedEvent | ActivePartyChangedEvent):
        return EventImportance.NORMAL
    if event.event_type in {EventType.SAVE, EventType.LOAD}:
        return EventImportance.IMPORTANT
    return EventImportance.NORMAL


def max_event_importance(events: list[GameEvent]) -> EventImportance:
    importance = EventImportance.NOISE
    for event in events:
        candidate = event_importance(event)
        if _IMPORTANCE_RANK[candidate] > _IMPORTANCE_RANK[importance]:
            importance = candidate
    return importance


def tactical_brief_lines(hci: HciResultAnalysis) -> list[str]:
    return _format_tactical_brief(build_tactical_brief(hci))


def build_tactical_brief(hci: HciResultAnalysis) -> TacticalBrief:
    if hci.error:
        return TacticalBrief(
            action=("Command blocked.",),
            changed=("No state changed.",),
            danger=(hci.error,),
            next_steps=("Choose another enabled action.",),
        )
    return _brief_from_parts(hci.before, hci.after, hci.deltas, hci.beats)


def event_title(event: GameEvent) -> str:
    if isinstance(event, ExpeditionEvent):
        return event.node_id.replace("_", " ").title()
    if isinstance(event, LootGainedEvent):
        return "Loot"
    if isinstance(event, DungeonActionEvent):
        return "Dungeon Action"
    if isinstance(event, BreachDiscoveredEvent):
        return "Breach"
    if isinstance(event, LoreDiscoveredEvent):
        return "Rumor"
    if isinstance(event, ContractCompletedEvent):
        return "Contract"
    if isinstance(event, ExpeditionReturnedEvent):
        return "Return"
    if isinstance(event, MazeFrontierOpenedEvent):
        return "Maze Frontier"
    if isinstance(event, MazeRouteCollapsedEvent):
        return "Maze Route"
    if isinstance(
        event,
        (
            ActivePartyChangedEvent,
            HeroRecruitedEvent,
            RecoveryEvent,
            RecruitmentOfferedEvent,
            SuppliesPurchasedEvent,
            TownServiceEvent,
        ),
    ):
        return "Town"
    if event.event_type in {EventType.SAVE, EventType.LOAD}:
        return "Save"
    return "Company"


def event_style(event: GameEvent) -> str:
    importance = event_importance(event)
    if importance == EventImportance.CRITICAL:
        return "red"
    if isinstance(event, LootGainedEvent | ContractCompletedEvent | DungeonActionEvent):
        return "green"
    if isinstance(event, LoreDiscoveredEvent):
        return "magenta"
    if isinstance(
        event,
        ExpeditionEvent | BreachDiscoveredEvent | ExpeditionReturnedEvent | MazeRouteCollapsedEvent,
    ):
        return "cyan"
    if event.event_type in {EventType.SAVE, EventType.LOAD}:
        return "green"
    return "blue"


def is_combat_event(event: GameEvent) -> bool:
    return event.event_type in {
        EventType.ENCOUNTER_STARTED,
        EventType.ROUND_STARTED,
        EventType.ROUND_ENDED,
        EventType.ENCOUNTER_ENDED,
        EventType.DAMAGE,
        EventType.HEALING,
        EventType.MOVE,
        EventType.TURN_DELAYED,
        EventType.TURN_PASSED,
        EventType.COMBAT_RETREAT_DECLARED,
        EventType.COMBAT_RETREATED,
        EventType.ENEMY_INTENT,
        EventType.REACTION_USED,
        EventType.REACTION_SKIPPED,
        EventType.DOWNED,
        EventType.DEATH,
        EventType.COMBAT_ENDED,
        EventType.SKILL_USED,
        EventType.MISS,
        EventType.COMBAT_EFFECT,
        EventType.STATUS_CHANGED,
    }


def _append_int_delta(
    deltas: list[StateDelta],
    *,
    category: str,
    key: str,
    label: str,
    before: int,
    after: int,
) -> None:
    if before == after:
        return
    delta = after - before
    deltas.append(
        StateDelta(
            category=category,
            key=key,
            label=label,
            before=str(before),
            after=str(after),
            delta=delta,
            summary=f"{label}: {before}->{after} ({_signed(delta)}).",
            importance=EventImportance.IMPORTANT,
        )
    )


def _append_item_deltas(
    deltas: list[StateDelta],
    category: str,
    before_items: tuple[tuple[str, int], ...],
    after_items: tuple[tuple[str, int], ...],
    label_prefix: str,
) -> None:
    before = dict(before_items)
    after = dict(after_items)
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key, 0)
        after_value = after.get(key, 0)
        if before_value == after_value:
            continue
        delta = after_value - before_value
        label = f"{label_prefix}: {key}"
        deltas.append(
            StateDelta(
                category=category,
                key=key,
                label=label,
                before=str(before_value),
                after=str(after_value),
                delta=delta,
                summary=f"{label}: {before_value}->{after_value} ({_signed(delta)}).",
                importance=EventImportance.IMPORTANT,
            )
        )


def _hero_deltas(
    before_heroes: tuple[HeroStateSnapshot, ...],
    after_heroes: tuple[HeroStateSnapshot, ...],
) -> list[StateDelta]:
    deltas: list[StateDelta] = []
    before = {hero.hero_id: hero for hero in before_heroes}
    after = {hero.hero_id: hero for hero in after_heroes}
    for hero_id in sorted(set(before) | set(after)):
        old = before.get(hero_id)
        new = after.get(hero_id)
        if old is None and new is not None:
            deltas.append(
                StateDelta(
                    category="hero",
                    key=f"{hero_id}:roster",
                    label=new.name,
                    before="absent",
                    after="present",
                    summary=f"{new.name} joined the roster.",
                    importance=EventImportance.IMPORTANT,
                )
            )
            continue
        if old is not None and new is None:
            deltas.append(
                StateDelta(
                    category="hero",
                    key=f"{hero_id}:roster",
                    label=old.name,
                    before="present",
                    after="absent",
                    summary=f"{old.name} left the active roster.",
                    importance=EventImportance.CRITICAL,
                )
            )
            continue
        if old is None or new is None:
            continue
        if old.hp != new.hp:
            delta = new.hp - old.hp
            deltas.append(
                StateDelta(
                    category="hero",
                    key=f"{hero_id}:hp",
                    label=f"{new.name} HP",
                    before=str(old.hp),
                    after=str(new.hp),
                    delta=delta,
                    summary=f"{new.name} HP: {old.hp}->{new.hp} ({_signed(delta)}).",
                    importance=EventImportance.CRITICAL
                    if new.hp <= 0
                    else EventImportance.IMPORTANT,
                )
            )
        if old.effort != new.effort:
            delta = new.effort - old.effort
            deltas.append(
                StateDelta(
                    category="hero",
                    key=f"{hero_id}:effort",
                    label=f"{new.name} Effort",
                    before=str(old.effort),
                    after=str(new.effort),
                    delta=delta,
                    summary=(
                        f"{new.name} Effort: {old.effort}->{new.effort}"
                        f" ({_signed(delta)})."
                    ),
                    importance=EventImportance.NORMAL,
                )
            )
        if old.mortal_wounds != new.mortal_wounds:
            delta = new.mortal_wounds - old.mortal_wounds
            deltas.append(
                StateDelta(
                    category="hero",
                    key=f"{hero_id}:mortal_wounds",
                    label=f"{new.name} Mortal Wounds",
                    before=str(old.mortal_wounds),
                    after=str(new.mortal_wounds),
                    delta=delta,
                    summary=(
                        f"{new.name} Mortal Wounds: {old.mortal_wounds}->{new.mortal_wounds}"
                        f" ({_signed(delta)})."
                    ),
                    importance=EventImportance.CRITICAL if delta > 0 else EventImportance.IMPORTANT,
                )
            )
        if old.statuses != new.statuses:
            added = sorted(set(new.statuses) - set(old.statuses))
            removed = sorted(set(old.statuses) - set(new.statuses))
            pieces: list[str] = []
            if added:
                pieces.append(f"gained {', '.join(added)}")
            if removed:
                pieces.append(f"cleared {', '.join(removed)}")
            deltas.append(
                StateDelta(
                    category="hero",
                    key=f"{hero_id}:statuses",
                    label=f"{new.name} Status",
                    before=", ".join(old.statuses) or "none",
                    after=", ".join(new.statuses) or "none",
                    summary=f"{new.name} status: {'; '.join(pieces)}.",
                    importance=EventImportance.CRITICAL
                    if any(status in {"dead", "downed"} for status in added)
                    else EventImportance.IMPORTANT,
                )
            )
    return deltas


def _analysis_summary(
    before: StateSnapshot,
    after: StateSnapshot,
    deltas: tuple[StateDelta, ...],
    beats: tuple[EventBeat, ...],
    error: str | None,
) -> list[str]:
    return _format_tactical_brief(
        TacticalBrief(
            action=("Command blocked.",),
            changed=("No state changed.",),
            danger=(error,),
            next_steps=("Choose another enabled action.",),
        )
        if error
        else _brief_from_parts(before, after, deltas, beats)
    )


def _brief_from_parts(
    before: StateSnapshot,
    after: StateSnapshot,
    deltas: tuple[StateDelta, ...],
    beats: tuple[EventBeat, ...],
) -> TacticalBrief:
    action_lines = _action_lines(beats)
    changed_lines = _changed_lines(deltas)
    danger_lines = _danger_lines(after, deltas, beats)
    return TacticalBrief(
        action=tuple(action_lines or ["Command resolved."]),
        changed=tuple(changed_lines or ["No tracked state change."]),
        danger=tuple(danger_lines or ["No critical condition change."]),
        next_steps=tuple(_next_lines(before, after)),
    )


def _format_tactical_brief(brief: TacticalBrief) -> list[str]:
    lines: list[str] = []
    sections = (
        ("Action", brief.action),
        ("Changed", brief.changed),
        ("Danger / Condition", brief.danger),
        ("Next", brief.next_steps),
    )
    for title, section_lines in sections:
        if not section_lines:
            continue
        if lines:
            lines.append("")
        lines.append(title)
        lines.extend(f"- {line}" for line in section_lines if line)
    return lines


def _action_lines(beats: tuple[EventBeat, ...]) -> list[str]:
    lines: list[str] = []
    for beat in _prioritized_beats(beats):
        for event in beat.events:
            if event_importance(event) == EventImportance.NOISE:
                continue
            lines.append(event.message)
            if len(lines) >= 4:
                return _dedupe(lines)
    return _dedupe(lines)


def _changed_lines(deltas: tuple[StateDelta, ...]) -> list[str]:
    return _dedupe([delta.summary for delta in sorted(deltas, key=_delta_sort_key)])[:6]


def _danger_lines(
    after: StateSnapshot,
    deltas: tuple[StateDelta, ...],
    beats: tuple[EventBeat, ...],
) -> list[str]:
    lines: list[str] = []
    for beat in _prioritized_beats(beats):
        if _IMPORTANCE_RANK[beat.importance] < _IMPORTANCE_RANK[EventImportance.CRITICAL]:
            continue
        for event in beat.events:
            if _is_danger_event(event):
                lines.append(event.message)
    lines.extend(_hero_condition_lines(after))
    if after.combat_encounter_id:
        lines.append("Combat remains active.")
    if after.dungeon_pending_combat_node_id:
        lines.append(f"Room combat pending: {after.dungeon_pending_combat_node_id}.")
    return _dedupe(lines)[:5]


def _next_lines(before: StateSnapshot, after: StateSnapshot) -> list[str]:
    if after.combat_encounter_id:
        return ["Resolve the active combat turn."]
    if after.dungeon_pending_combat_node_id:
        return ["Clear the pending room combat before moving."]
    if after.dungeon_node_id:
        return ["Choose a room action, route, or safe return."]
    if after.active_expedition_id:
        return ["Continue the active expedition."]
    if before.location_id != after.location_id and after.location_name:
        return [f"Continue from {after.location_name}."]
    if after.location_name:
        return [f"Continue in {after.location_name}."]
    if after.has_company:
        return ["Choose the next company command."]
    return ["Choose the next command."]


def _prioritized_beats(beats: tuple[EventBeat, ...]) -> list[EventBeat]:
    prioritized = sorted(
        enumerate(beats),
        key=lambda entry: (
            -_IMPORTANCE_RANK[entry[1].importance],
            0 if entry[1].combat else 1,
            entry[0],
        ),
    )
    return [beat for _index, beat in prioritized]


def _delta_sort_key(delta: StateDelta) -> tuple[int, int, int, str]:
    category_rank = {
        "hero": 0,
        "combat": 1,
        "dungeon": 2,
        "resource": 3,
        "supply": 4,
        "inventory": 5,
        "location": 6,
        "company": 7,
    }.get(delta.category, 8)
    loss_rank = 0 if delta.delta is not None and delta.delta < 0 else 1
    return (-_IMPORTANCE_RANK[delta.importance], category_rank, loss_rank, delta.label)


def _is_danger_event(event: GameEvent) -> bool:
    return isinstance(
        event,
        (
            BreachDiscoveredEvent,
            CombatRetreatDeclaredEvent,
            CombatRetreatedEvent,
            DeathEvent,
            DownedEvent,
        ),
    ) or (
        isinstance(event, CombatEndedEvent | EncounterEndedEvent)
        and event.victor != "heroes"
    )


def _hero_condition_lines(after: StateSnapshot) -> list[str]:
    lines: list[str] = []
    for hero in after.heroes:
        statuses = set(hero.statuses)
        if "dead" in statuses:
            lines.append(f"{hero.name} is dead.")
        elif "downed" in statuses or hero.hp <= 0:
            lines.append(f"{hero.name} is downed.")
        elif hero.mortal_wounds > 0:
            lines.append(f"{hero.name} carries {hero.mortal_wounds} mortal wound(s).")
    return lines


def _dedupe(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique


def _signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def hci_value(value: Any) -> str:
    return str(value)
