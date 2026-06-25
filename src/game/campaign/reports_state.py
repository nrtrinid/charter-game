"""Expedition report state for campaign saves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.campaign.hero_memory import RecentSignal
from game.campaign.serde_helpers import (
    life_state_from_raw,
    strain_mark_values_from_raw,
    strain_name_from_raw,
)
from game.combat.combat_state import LifeState, MoraleState, StrainTier


def _life_state_value_from_raw(data: dict[str, Any]) -> str:
    return life_state_from_raw(data, legacy=False).value


def _strain_name_from_raw(data: dict[str, Any]) -> str:
    return strain_name_from_raw(data, legacy=False)


def _strain_marks_from_raw(data: dict[str, Any]) -> list[str]:
    return strain_mark_values_from_raw(data, legacy=False)


@dataclass
class HeroReportSnapshot:
    hero_id: str
    name: str
    class_id: str
    hp: int
    max_hp: int
    effort: int
    max_effort: int
    mortal_wounds: int
    morale: str = MoraleState.STEADY.name
    strain: str = StrainTier.STEADY.name
    life_state: str = LifeState.ALIVE.value
    personal_quirk: str | None = None
    quirks: list[str] = field(default_factory=list)
    strain_marks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "name": self.name,
            "class_id": self.class_id,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "effort": self.effort,
            "max_effort": self.max_effort,
            "mortal_wounds": self.mortal_wounds,
            "morale": self.morale,
            "strain": self.strain,
            "life_state": self.life_state,
            "personal_quirk": self.personal_quirk,
            "quirks": list(self.quirks),
            "strain_marks": list(self.strain_marks),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeroReportSnapshot:
        return cls(
            hero_id=str(data["hero_id"]),
            name=str(data["name"]),
            class_id=str(data["class_id"]),
            hp=int(data["hp"]),
            max_hp=int(data["max_hp"]),
            effort=int(data["effort"]),
            max_effort=int(data["max_effort"]),
            mortal_wounds=int(data.get("mortal_wounds", 0)),
            morale=str(data.get("morale", MoraleState.STEADY.name)),
            strain=_strain_name_from_raw(data),
            life_state=_life_state_value_from_raw(data),
            personal_quirk=(
                None
                if data.get("personal_quirk") is None
                else str(data["personal_quirk"])
            ),
            quirks=[str(quirk) for quirk in data.get("quirks", [])],
            strain_marks=_strain_marks_from_raw(data),
        )


@dataclass
class ReportEventSignal:
    kind: str
    message: str
    hero_id: str | None = None
    hero_name: str = ""
    node_id: str | None = None
    encounter_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "hero_id": self.hero_id,
            "hero_name": self.hero_name,
            "node_id": self.node_id,
            "encounter_id": self.encounter_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportEventSignal:
        return cls(
            kind=str(data["kind"]),
            message=str(data["message"]),
            hero_id=None if data.get("hero_id") is None else str(data["hero_id"]),
            hero_name=str(data.get("hero_name", "")),
            node_id=None if data.get("node_id") is None else str(data["node_id"]),
            encounter_id=None if data.get("encounter_id") is None else str(data["encounter_id"]),
        )


@dataclass
class HeroReportOutcome:
    hero_id: str
    hero_name: str
    class_id: str
    status: str
    start_hp: int
    end_hp: int
    max_hp: int
    start_mortal_wounds: int
    end_mortal_wounds: int
    mortal_wounds_delta: int
    died: bool = False
    downed: bool = False
    wounded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "hero_name": self.hero_name,
            "class_id": self.class_id,
            "status": self.status,
            "start_hp": self.start_hp,
            "end_hp": self.end_hp,
            "max_hp": self.max_hp,
            "start_mortal_wounds": self.start_mortal_wounds,
            "end_mortal_wounds": self.end_mortal_wounds,
            "mortal_wounds_delta": self.mortal_wounds_delta,
            "died": self.died,
            "downed": self.downed,
            "wounded": self.wounded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeroReportOutcome:
        return cls(
            hero_id=str(data["hero_id"]),
            hero_name=str(data["hero_name"]),
            class_id=str(data["class_id"]),
            status=str(data["status"]),
            start_hp=int(data["start_hp"]),
            end_hp=int(data["end_hp"]),
            max_hp=int(data["max_hp"]),
            start_mortal_wounds=int(data.get("start_mortal_wounds", 0)),
            end_mortal_wounds=int(data.get("end_mortal_wounds", 0)),
            mortal_wounds_delta=int(data.get("mortal_wounds_delta", 0)),
            died=bool(data.get("died", False)),
            downed=bool(data.get("downed", False)),
            wounded=bool(data.get("wounded", False)),
        )


@dataclass
class ExpeditionReportState:
    expedition_id: str
    dungeon_id: str | None
    outcome: str = "in_progress"
    rooms_entered: list[str] = field(default_factory=list)
    room_names: dict[str, str] = field(default_factory=dict)
    encounters_resolved: list[str] = field(default_factory=list)
    loot: dict[str, int] = field(default_factory=dict)
    supplies: dict[str, int] = field(default_factory=dict)
    gear: dict[str, int] = field(default_factory=dict)
    reputation_gained: int = 0
    coin_gained: int = 0
    breaches_discovered: list[str] = field(default_factory=list)
    room_actions: list[str] = field(default_factory=list)
    supplies_spent: dict[str, int] = field(default_factory=dict)
    participant_ids: list[str] = field(default_factory=list)
    start_reputation: int = 0
    end_reputation: int = 0
    start_coin: int = 0
    end_coin: int = 0
    start_supplies: dict[str, int] = field(default_factory=dict)
    end_supplies: dict[str, int] = field(default_factory=dict)
    start_inventory: dict[str, int] = field(default_factory=dict)
    end_inventory: dict[str, int] = field(default_factory=dict)
    start_gear_inventory: dict[str, int] = field(default_factory=dict)
    end_gear_inventory: dict[str, int] = field(default_factory=dict)
    start_hero_states: dict[str, HeroReportSnapshot] = field(default_factory=dict)
    end_hero_states: dict[str, HeroReportSnapshot] = field(default_factory=dict)
    hero_outcomes: list[HeroReportOutcome] = field(default_factory=list)
    event_signals: list[ReportEventSignal] = field(default_factory=list)
    memory_signals: list[RecentSignal] = field(default_factory=list)
    notable_moments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expedition_id": self.expedition_id,
            "dungeon_id": self.dungeon_id,
            "outcome": self.outcome,
            "rooms_entered": list(self.rooms_entered),
            "room_names": dict(self.room_names),
            "encounters_resolved": list(self.encounters_resolved),
            "loot": dict(self.loot),
            "supplies": dict(self.supplies),
            "gear": dict(self.gear),
            "reputation_gained": self.reputation_gained,
            "coin_gained": self.coin_gained,
            "breaches_discovered": list(self.breaches_discovered),
            "room_actions": list(self.room_actions),
            "supplies_spent": dict(self.supplies_spent),
            "participant_ids": list(self.participant_ids),
            "start_reputation": self.start_reputation,
            "end_reputation": self.end_reputation,
            "start_coin": self.start_coin,
            "end_coin": self.end_coin,
            "start_supplies": dict(self.start_supplies),
            "end_supplies": dict(self.end_supplies),
            "start_inventory": dict(self.start_inventory),
            "end_inventory": dict(self.end_inventory),
            "start_gear_inventory": dict(self.start_gear_inventory),
            "end_gear_inventory": dict(self.end_gear_inventory),
            "start_hero_states": {
                hero_id: snapshot.to_dict()
                for hero_id, snapshot in self.start_hero_states.items()
            },
            "end_hero_states": {
                hero_id: snapshot.to_dict()
                for hero_id, snapshot in self.end_hero_states.items()
            },
            "hero_outcomes": [outcome.to_dict() for outcome in self.hero_outcomes],
            "event_signals": [signal.to_dict() for signal in self.event_signals],
            "memory_signals": [signal.to_dict() for signal in self.memory_signals],
            "notable_moments": list(self.notable_moments),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpeditionReportState:
        dungeon_id = data.get("dungeon_id")
        return cls(
            expedition_id=str(data["expedition_id"]),
            dungeon_id=None if dungeon_id is None else str(dungeon_id),
            outcome=str(data.get("outcome", "in_progress")),
            rooms_entered=[str(node_id) for node_id in data.get("rooms_entered", [])],
            room_names={
                str(node_id): str(name)
                for node_id, name in data.get("room_names", {}).items()
            },
            encounters_resolved=[
                str(encounter_id) for encounter_id in data.get("encounters_resolved", [])
            ],
            loot={str(key): int(value) for key, value in data.get("loot", {}).items()},
            supplies={str(key): int(value) for key, value in data.get("supplies", {}).items()},
            gear={str(key): int(value) for key, value in data.get("gear", {}).items()},
            reputation_gained=int(data.get("reputation_gained", 0)),
            coin_gained=int(data.get("coin_gained", 0)),
            breaches_discovered=[
                str(breach_id) for breach_id in data.get("breaches_discovered", [])
            ],
            room_actions=[str(action_id) for action_id in data.get("room_actions", [])],
            supplies_spent={
                str(key): int(value)
                for key, value in data.get("supplies_spent", {}).items()
            },
            participant_ids=[str(hero_id) for hero_id in data.get("participant_ids", [])],
            start_reputation=int(data.get("start_reputation", 0)),
            end_reputation=int(data.get("end_reputation", 0)),
            start_coin=int(data.get("start_coin", 0)),
            end_coin=int(data.get("end_coin", 0)),
            start_supplies={
                str(key): int(value)
                for key, value in data.get("start_supplies", {}).items()
            },
            end_supplies={
                str(key): int(value)
                for key, value in data.get("end_supplies", {}).items()
            },
            start_inventory={
                str(key): int(value)
                for key, value in data.get("start_inventory", {}).items()
            },
            end_inventory={
                str(key): int(value)
                for key, value in data.get("end_inventory", {}).items()
            },
            start_gear_inventory={
                str(key): int(value)
                for key, value in data.get("start_gear_inventory", {}).items()
            },
            end_gear_inventory={
                str(key): int(value)
                for key, value in data.get("end_gear_inventory", {}).items()
            },
            start_hero_states={
                str(hero_id): HeroReportSnapshot.from_dict(snapshot)
                for hero_id, snapshot in data.get("start_hero_states", {}).items()
                if isinstance(snapshot, dict)
            },
            end_hero_states={
                str(hero_id): HeroReportSnapshot.from_dict(snapshot)
                for hero_id, snapshot in data.get("end_hero_states", {}).items()
                if isinstance(snapshot, dict)
            },
            hero_outcomes=[
                HeroReportOutcome.from_dict(outcome)
                for outcome in data.get("hero_outcomes", [])
                if isinstance(outcome, dict)
            ],
            event_signals=[
                ReportEventSignal.from_dict(signal)
                for signal in data.get("event_signals", [])
                if isinstance(signal, dict)
            ],
            memory_signals=[
                RecentSignal.from_dict(signal)
                for signal in data.get("memory_signals", [])
                if isinstance(signal, dict)
            ],
            notable_moments=[str(moment) for moment in data.get("notable_moments", [])],
        )

