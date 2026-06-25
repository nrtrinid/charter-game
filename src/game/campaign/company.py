"""Runtime campaign and company state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.campaign.contract_state import ContractRecordState
from game.campaign.expedition_state import (  # noqa: F401
    ExpeditionSessionState,
    GeneratedDungeonState,
    MazeRecipe,
)
from game.campaign.hero_state import HeroState  # noqa: F401
from game.campaign.memory_state import (
    BreachMemoryState,
    CompanyTimelineEntry,
    DungeonMemoryState,
    HeroMemoryEntry,
    WorldLocationMemoryState,
)
from game.campaign.recruitment_state import (  # noqa: F401
    RecruitmentOfferState,
    RecruitmentState,
)
from game.campaign.reports_state import (  # noqa: F401
    ExpeditionReportState,
    HeroReportOutcome,
    HeroReportSnapshot,
    ReportEventSignal,
)
from game.combat.formation import FormationSlot
from game.content.definitions import GameDefinitions

SAVE_VERSION = 13
STARTING_COIN = 8


@dataclass
class CompanyState:
    company_id: str
    name: str
    roster: list[HeroState]
    supplies: dict[str, int]
    inventory: dict[str, int] = field(default_factory=dict)
    gear_inventory: dict[str, int] = field(default_factory=dict)
    reputation: int = 0
    coin: int = STARTING_COIN
    known_breaches: set[str] = field(default_factory=set)
    known_route_ids: set[str] = field(default_factory=set)
    known_lore_entries: set[str] = field(default_factory=set)
    active_contract_ids: set[str] = field(default_factory=set)
    completed_contract_ids: set[str] = field(default_factory=set)
    expedition_history: list[str] = field(default_factory=list)
    deceased_heroes: list[HeroState] = field(default_factory=list)
    hero_memories: list[HeroMemoryEntry] = field(default_factory=list)
    company_timeline: list[CompanyTimelineEntry] = field(default_factory=list)
    town_state: dict[str, Any] = field(
        default_factory=lambda: {"location": "Haven Town", "location_id": "haven"}
    )
    flags: dict[str, bool] = field(default_factory=dict)
    dungeon_memory: dict[str, DungeonMemoryState] = field(default_factory=dict)
    world_memory: dict[str, WorldLocationMemoryState] = field(default_factory=dict)
    recruitment_state: RecruitmentState = field(default_factory=RecruitmentState)
    contract_records: dict[str, ContractRecordState] = field(default_factory=dict)
    breach_memory: dict[str, BreachMemoryState] = field(default_factory=dict)
    purchased_upgrade_ids: set[str] = field(default_factory=set)
    active_party_slots: dict[FormationSlot, str | None] = field(default_factory=dict)
    active_expedition: ExpeditionSessionState | None = None
    last_expedition_report: ExpeditionReportState | None = None
    expedition_reports: list[ExpeditionReportState] = field(default_factory=list)
    save_version: int = SAVE_VERSION

    def __post_init__(self) -> None:
        self.save_version = SAVE_VERSION
        if not self.active_party_slots:
            self.active_party_slots = _active_party_from_roster(self.roster)
        else:
            self.active_party_slots = _normalize_active_party_slots(self.active_party_slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "save_version": self.save_version,
            "company_id": self.company_id,
            "name": self.name,
            "roster": [hero.to_dict() for hero in self.roster],
            "supplies": dict(self.supplies),
            "inventory": dict(self.inventory),
            "gear_inventory": dict(self.gear_inventory),
            "reputation": self.reputation,
            "coin": self.coin,
            "known_breaches": sorted(self.known_breaches),
            "known_route_ids": sorted(self.known_route_ids),
            "known_lore_entries": sorted(self.known_lore_entries),
            "active_contract_ids": sorted(self.active_contract_ids),
            "completed_contract_ids": sorted(self.completed_contract_ids),
            "expedition_history": list(self.expedition_history),
            "deceased_heroes": [hero.to_dict() for hero in self.deceased_heroes],
            "hero_memories": [memory.to_dict() for memory in self.hero_memories],
            "company_timeline": [entry.to_dict() for entry in self.company_timeline],
            "town_state": dict(self.town_state),
            "flags": dict(self.flags),
            "dungeon_memory": {
                dungeon_id: memory.to_dict()
                for dungeon_id, memory in sorted(self.dungeon_memory.items())
            },
            "world_memory": {
                location_id: memory.to_dict()
                for location_id, memory in sorted(self.world_memory.items())
            },
            "recruitment_state": self.recruitment_state.to_dict(),
            "contract_records": {
                contract_id: record.to_dict()
                for contract_id, record in sorted(self.contract_records.items())
            },
            "breach_memory": {
                source_node_id: memory.to_dict()
                for source_node_id, memory in sorted(self.breach_memory.items())
            },
            "purchased_upgrade_ids": sorted(self.purchased_upgrade_ids),
            "active_party_slots": {
                slot.value: actor_id for slot, actor_id in self.active_party_slots.items()
            },
            "active_expedition": (
                self.active_expedition.to_dict()
                if self.active_expedition is not None
                else None
            ),
            "last_expedition_report": (
                self.last_expedition_report.to_dict()
                if self.last_expedition_report is not None
                else None
            ),
            "expedition_reports": [
                report.to_dict() for report in self.expedition_reports
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompanyState:
        from game.campaign.migrations import upgrade_raw_save

        data = upgrade_raw_save(data)
        roster = _parse_roster(data.get("roster", []))
        active_party_slots = _parse_active_party_slots(data.get("active_party_slots"), roster)
        dungeon_memory, world_memory, breach_memory = _parse_memory_maps(data)
        last_expedition_report, expedition_reports = _parse_reports(data)
        supplies_raw = data.get("supplies")
        supplies = (
            {str(key): int(value) for key, value in supplies_raw.items()}
            if isinstance(supplies_raw, dict)
            else {}
        )
        recruitment_raw = data.get("recruitment_state")
        contract_records_raw = data.get("contract_records")
        active_expedition_raw = data.get("active_expedition")
        return cls(
            company_id=str(data.get("company_id", "")),
            name=str(data.get("name", "")),
            roster=roster,
            supplies=supplies,
            inventory={str(key): int(value) for key, value in data.get("inventory", {}).items()},
            gear_inventory={
                str(key): int(value)
                for key, value in data.get("gear_inventory", {}).items()
            },
            reputation=int(data.get("reputation", 0)),
            coin=int(data.get("coin", STARTING_COIN)),
            known_breaches={str(breach) for breach in data.get("known_breaches", [])},
            known_route_ids={str(route_id) for route_id in data.get("known_route_ids", [])},
            known_lore_entries={
                str(lore_id) for lore_id in data.get("known_lore_entries", [])
            },
            active_contract_ids={
                str(contract_id) for contract_id in data.get("active_contract_ids", [])
            },
            completed_contract_ids={
                str(contract_id) for contract_id in data.get("completed_contract_ids", [])
            },
            expedition_history=[str(entry) for entry in data.get("expedition_history", [])],
            deceased_heroes=_parse_roster(data.get("deceased_heroes", [])),
            hero_memories=[
                HeroMemoryEntry.from_dict(memory)
                for memory in _as_dict_list(data.get("hero_memories"))
                if isinstance(memory, dict)
            ],
            company_timeline=[
                CompanyTimelineEntry.from_dict(entry)
                for entry in _as_dict_list(data.get("company_timeline"))
                if isinstance(entry, dict)
            ],
            town_state=_normalize_town_state(data.get("town_state")),
            flags={str(key): bool(value) for key, value in data.get("flags", {}).items()},
            dungeon_memory=dungeon_memory,
            world_memory=world_memory,
            recruitment_state=RecruitmentState.from_dict(recruitment_raw)
            if isinstance(recruitment_raw, dict)
            else RecruitmentState(),
            contract_records={
                str(contract_id): ContractRecordState.from_dict(
                    record,
                    contract_id=str(contract_id),
                )
                for contract_id, record in (
                    contract_records_raw.items()
                    if isinstance(contract_records_raw, dict)
                    else ()
                )
                if isinstance(record, dict)
            },
            breach_memory=breach_memory,
            purchased_upgrade_ids={
                str(upgrade_id) for upgrade_id in data.get("purchased_upgrade_ids", [])
            },
            active_party_slots=active_party_slots,
            active_expedition=ExpeditionSessionState.from_dict(active_expedition_raw)
            if isinstance(active_expedition_raw, dict)
            else None,
            last_expedition_report=last_expedition_report,
            expedition_reports=expedition_reports,
            save_version=int(data.get("save_version", SAVE_VERSION)),
        )


def create_new_company(
    definitions: GameDefinitions,
    name: str = "Haven Charter",
    company_id: str = "company_001",
) -> CompanyState:
    roster: list[HeroState] = []
    for recruit in definitions.recruits.starting_roster:
        hero_class = definitions.hero_classes[recruit.class_id]
        roster.append(
            HeroState(
                hero_id=recruit.id,
                name=recruit.name,
                class_id=recruit.class_id,
                background=recruit.background,
                motive=recruit.motive,
                max_hp=hero_class.max_hp,
                hp=hero_class.max_hp,
                speed=hero_class.speed,
                accuracy=hero_class.accuracy,
                defense=hero_class.defense,
                damage=hero_class.damage,
                max_effort=hero_class.max_effort,
                effort=hero_class.max_effort,
                skills=list(hero_class.skills),
                formation_slot=recruit.formation_slot,
                personal_quirk=hero_class.personal_quirk,
            )
        )
    starting_location = definitions.locations[definitions.world.starting_settlement].name
    starting_location_id = definitions.world.starting_settlement
    active_contract_ids = {
        contract.id
        for contract in definitions.contracts.values()
        if contract.available_at_start
    }
    return CompanyState(
        company_id=company_id,
        name=name,
        roster=roster,
        supplies=dict(definitions.supplies.starting),
        active_contract_ids=active_contract_ids,
        contract_records={
            contract_id: ContractRecordState(
                contract_id=contract_id,
                state="active",
                accepted_count=1,
            )
            for contract_id in active_contract_ids
        },
        town_state={
            "location": starting_location,
            "location_id": starting_location_id,
            "regional_node_id": "town_gate",
            "services": "charter_house",
        },
        world_memory={
            starting_location_id: WorldLocationMemoryState(
                location_id=starting_location_id,
                visited=True,
                visit_count=1,
            )
        },
        active_party_slots=_active_party_from_roster(roster),
    )


def world_location_memory(
    company: CompanyState,
    location_id: str,
) -> WorldLocationMemoryState:
    memory = company.world_memory.get(location_id)
    if memory is None:
        memory = WorldLocationMemoryState(location_id=location_id)
        company.world_memory[location_id] = memory
    return memory


def record_world_visit(company: CompanyState, location_id: str) -> None:
    memory = world_location_memory(company, location_id)
    memory.visited = True
    memory.visit_count += 1


def record_world_node_discovered(
    company: CompanyState,
    location_id: str,
    node_id: str,
) -> None:
    _append_once(world_location_memory(company, location_id).discovered_node_ids, node_id)


def record_world_node_cleared(
    company: CompanyState,
    location_id: str,
    node_id: str,
) -> None:
    _append_once(world_location_memory(company, location_id).cleared_threat_node_ids, node_id)


def record_world_shortcut(
    company: CompanyState,
    location_id: str,
    shortcut_id: str,
) -> None:
    _append_once(world_location_memory(company, location_id).unlocked_shortcut_ids, shortcut_id)


def record_world_rumor_consumed(
    company: CompanyState,
    location_id: str,
    rumor_id: str,
) -> None:
    _append_once(world_location_memory(company, location_id).consumed_rumor_ids, rumor_id)


def contract_record(
    company: CompanyState,
    contract_id: str,
) -> ContractRecordState:
    record = company.contract_records.get(contract_id)
    if record is None:
        record = ContractRecordState(contract_id=contract_id)
        company.contract_records[contract_id] = record
    return record


def breach_memory(
    company: CompanyState,
    source_node_id: str,
) -> BreachMemoryState:
    memory = company.breach_memory.get(source_node_id)
    if memory is None:
        memory = BreachMemoryState(source_node_id=source_node_id)
        company.breach_memory[source_node_id] = memory
    return memory


def _as_dict_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_roster(raw: object) -> list[HeroState]:
    return [
        HeroState.from_dict(hero)
        for hero in _as_dict_list(raw)
        if isinstance(hero, dict)
    ]


def _parse_active_party_slots(
    raw: object,
    roster: list[HeroState],
) -> dict[FormationSlot, str | None]:
    if isinstance(raw, dict):
        return _active_party_from_raw(raw)
    return _active_party_from_roster(roster)


def _parse_memory_maps(
    data: dict[str, Any],
) -> tuple[
    dict[str, DungeonMemoryState],
    dict[str, WorldLocationMemoryState],
    dict[str, BreachMemoryState],
]:
    dungeon_memory_raw = data.get("dungeon_memory")
    world_memory_raw = data.get("world_memory")
    breach_memory_raw = data.get("breach_memory")
    dungeon_memory = {
        str(dungeon_id): DungeonMemoryState.from_dict(
            memory,
            dungeon_id=str(dungeon_id),
        )
        for dungeon_id, memory in (
            dungeon_memory_raw.items() if isinstance(dungeon_memory_raw, dict) else ()
        )
        if isinstance(memory, dict)
    }
    world_memory = {
        str(location_id): WorldLocationMemoryState.from_dict(
            memory,
            location_id=str(location_id),
        )
        for location_id, memory in (
            world_memory_raw.items() if isinstance(world_memory_raw, dict) else ()
        )
        if isinstance(memory, dict)
    }
    breach_memory = {
        str(source_node_id): BreachMemoryState.from_dict(
            memory,
            source_node_id=str(source_node_id),
        )
        for source_node_id, memory in (
            breach_memory_raw.items() if isinstance(breach_memory_raw, dict) else ()
        )
        if isinstance(memory, dict)
    }
    return dungeon_memory, world_memory, breach_memory


def _parse_reports(
    data: dict[str, Any],
) -> tuple[ExpeditionReportState | None, list[ExpeditionReportState]]:
    last_report_raw = data.get("last_expedition_report")
    last_expedition_report = (
        ExpeditionReportState.from_dict(last_report_raw)
        if isinstance(last_report_raw, dict)
        else None
    )
    expedition_reports = [
        ExpeditionReportState.from_dict(report)
        for report in _as_dict_list(data.get("expedition_reports"))
        if isinstance(report, dict)
    ]
    return last_expedition_report, expedition_reports


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


_LOCATION_ID_BY_DISPLAY_NAME: dict[str, str] = {
    "Haven Town": "haven",
    "Haven East Gate": "old_road",
    "Old Road": "old_road",
    "Abandoned Toll Post": "old_road",
    "Bandit Camp": "old_road",
    "Hunter's Trail": "blackwood_forest",
    "Wolf Hollow": "blackwood_forest",
    "Dry Creek Bed": "blackwood_forest",
    "Bramble Shrine": "blackwood_forest",
    "Hidden Deer Path": "blackwood_forest",
    "Black Stone Sinkhole": "shallow_cave",
    "Shallow Cave Entrance": "shallow_cave",
    "Cave Mouth": "shallow_cave",
    "Forked Descent": "shallow_cave",
    "Fungus-Lit Gallery": "shallow_cave",
    "Old Works Cache": "shallow_cave",
    "Narrow Crawl": "shallow_cave",
    "Black Stone Gate": "shallow_cave",
    "Maze-Touched Lair": "shallow_cave",
    "Cave Maw Brute Defeated": "shallow_cave",
    "Shallow Cave Breach": "shallow_cave_breach",
    "Pandora's Maze Depth 1": "pandoras_maze_depth_1",
}


def _normalize_town_state(raw: object) -> dict[str, Any]:
    state = dict(raw) if isinstance(raw, dict) else {"location": "Haven Town"}
    location = str(state.get("location") or "Haven Town")
    state["location"] = location
    if not state.get("location_id"):
        state["location_id"] = _LOCATION_ID_BY_DISPLAY_NAME.get(location, "haven")
    if not state.get("regional_node_id"):
        location_id = str(state.get("location_id") or "haven")
        if location_id == "shallow_cave":
            state["regional_node_id"] = "shallow_cave_entrance"
        else:
            state["regional_node_id"] = "town_gate"
    return state


def _empty_active_party_slots() -> dict[FormationSlot, str | None]:
    return dict.fromkeys(FormationSlot)


def _active_party_from_roster(roster: list[HeroState]) -> dict[FormationSlot, str | None]:
    active_party_slots = _empty_active_party_slots()
    for hero in roster:
        if active_party_slots[hero.formation_slot] is None:
            active_party_slots[hero.formation_slot] = hero.hero_id
    return active_party_slots


def _active_party_from_raw(data: dict[object, object]) -> dict[FormationSlot, str | None]:
    active_party_slots = _empty_active_party_slots()
    for raw_slot, raw_actor_id in data.items():
        slot = FormationSlot(str(raw_slot))
        active_party_slots[slot] = None if raw_actor_id is None else str(raw_actor_id)
    return active_party_slots


def _normalize_active_party_slots(
    active_party_slots: dict[FormationSlot, str | None],
) -> dict[FormationSlot, str | None]:
    normalized = _empty_active_party_slots()
    for slot, actor_id in active_party_slots.items():
        normalized[FormationSlot(str(slot))] = actor_id
    return normalized
