"""Campaign memory and timeline state for saves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HeroMemoryEntry:
    entry_id: str
    hero_id: str
    hero_name: str
    kind: str
    summary: str
    expedition_id: str
    dungeon_id: str
    node_id: str | None = None
    encounter_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "hero_id": self.hero_id,
            "hero_name": self.hero_name,
            "kind": self.kind,
            "summary": self.summary,
            "expedition_id": self.expedition_id,
            "dungeon_id": self.dungeon_id,
            "node_id": self.node_id,
            "encounter_id": self.encounter_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeroMemoryEntry:
        return cls(
            entry_id=str(data["entry_id"]),
            hero_id=str(data["hero_id"]),
            hero_name=str(data["hero_name"]),
            kind=str(data["kind"]),
            summary=str(data["summary"]),
            expedition_id=str(data["expedition_id"]),
            dungeon_id=str(data["dungeon_id"]),
            node_id=None if data.get("node_id") is None else str(data["node_id"]),
            encounter_id=None if data.get("encounter_id") is None else str(data["encounter_id"]),
        )


@dataclass
class CompanyTimelineEntry:
    entry_id: str
    kind: str
    summary: str
    expedition_id: str
    dungeon_id: str
    node_id: str | None = None
    encounter_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "summary": self.summary,
            "expedition_id": self.expedition_id,
            "dungeon_id": self.dungeon_id,
            "node_id": self.node_id,
            "encounter_id": self.encounter_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompanyTimelineEntry:
        return cls(
            entry_id=str(data["entry_id"]),
            kind=str(data["kind"]),
            summary=str(data["summary"]),
            expedition_id=str(data["expedition_id"]),
            dungeon_id=str(data["dungeon_id"]),
            node_id=None if data.get("node_id") is None else str(data["node_id"]),
            encounter_id=None if data.get("encounter_id") is None else str(data["encounter_id"]),
        )


@dataclass
class DungeonMemoryState:
    dungeon_id: str
    visited_node_ids: list[str] = field(default_factory=list)
    cleared_node_ids: list[str] = field(default_factory=list)
    completed_action_ids: list[str] = field(default_factory=list)
    revealed_exit_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dungeon_id": self.dungeon_id,
            "visited_node_ids": list(self.visited_node_ids),
            "cleared_node_ids": list(self.cleared_node_ids),
            "completed_action_ids": list(self.completed_action_ids),
            "revealed_exit_ids": list(self.revealed_exit_ids),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        dungeon_id: str | None = None,
    ) -> DungeonMemoryState:
        return cls(
            dungeon_id=str(data.get("dungeon_id") or dungeon_id or ""),
            visited_node_ids=[str(node_id) for node_id in data.get("visited_node_ids", [])],
            cleared_node_ids=[str(node_id) for node_id in data.get("cleared_node_ids", [])],
            completed_action_ids=[
                str(action_id) for action_id in data.get("completed_action_ids", [])
            ],
            revealed_exit_ids=[str(exit_id) for exit_id in data.get("revealed_exit_ids", [])],
        )


@dataclass
class WorldLocationMemoryState:
    location_id: str
    visited: bool = False
    visit_count: int = 0
    discovered_node_ids: list[str] = field(default_factory=list)
    cleared_threat_node_ids: list[str] = field(default_factory=list)
    consumed_rumor_ids: list[str] = field(default_factory=list)
    unlocked_shortcut_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_id": self.location_id,
            "visited": self.visited,
            "visit_count": self.visit_count,
            "discovered_node_ids": list(self.discovered_node_ids),
            "cleared_threat_node_ids": list(self.cleared_threat_node_ids),
            "consumed_rumor_ids": list(self.consumed_rumor_ids),
            "unlocked_shortcut_ids": list(self.unlocked_shortcut_ids),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        location_id: str | None = None,
    ) -> WorldLocationMemoryState:
        return cls(
            location_id=str(data.get("location_id") or location_id or ""),
            visited=bool(data.get("visited", False)),
            visit_count=int(data.get("visit_count", 0)),
            discovered_node_ids=[
                str(node_id) for node_id in data.get("discovered_node_ids", [])
            ],
            cleared_threat_node_ids=[
                str(node_id) for node_id in data.get("cleared_threat_node_ids", [])
            ],
            consumed_rumor_ids=[
                str(rumor_id) for rumor_id in data.get("consumed_rumor_ids", [])
            ],
            unlocked_shortcut_ids=[
                str(shortcut_id) for shortcut_id in data.get("unlocked_shortcut_ids", [])
            ],
        )


@dataclass
class BreachMemoryState:
    source_node_id: str
    run_count: int = 0
    collapsed_run_ids: list[str] = field(default_factory=list)
    scouted_run_ids: list[str] = field(default_factory=list)
    hunt_run_ids: list[str] = field(default_factory=list)
    last_pressure_id: str = ""
    pressure_counts: dict[str, int] = field(default_factory=dict)
    last_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "run_count": self.run_count,
            "collapsed_run_ids": list(self.collapsed_run_ids),
            "scouted_run_ids": list(self.scouted_run_ids),
            "hunt_run_ids": list(self.hunt_run_ids),
            "last_pressure_id": self.last_pressure_id,
            "pressure_counts": dict(self.pressure_counts),
            "last_seed": self.last_seed,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_node_id: str | None = None,
    ) -> BreachMemoryState:
        return cls(
            source_node_id=str(data.get("source_node_id") or source_node_id or ""),
            run_count=int(data.get("run_count", 0)),
            collapsed_run_ids=[str(run_id) for run_id in data.get("collapsed_run_ids", [])],
            scouted_run_ids=[str(run_id) for run_id in data.get("scouted_run_ids", [])],
            hunt_run_ids=[str(run_id) for run_id in data.get("hunt_run_ids", [])],
            last_pressure_id=str(data.get("last_pressure_id", "")),
            pressure_counts={
                str(pressure_id): int(count)
                for pressure_id, count in data.get("pressure_counts", {}).items()
            },
            last_seed=None if data.get("last_seed") is None else int(data["last_seed"]),
        )

