"""Runtime campaign and company state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.campaign.hero_memory import (
    EarnedQuirkSlotState,
    FreshMemoryState,
    RecentSignal,
    flat_quirks_from_slots,
    synthesize_earned_slots,
)
from game.combat.combat_state import (
    ActorStatus,
    FatigueState,
    LifeState,
    MoraleState,
    StatusSetProxy,
    StrainMark,
    StrainTier,
    Tag,
    fatigue_from_strain,
    life_state_from_statuses,
    strain_from_fatigue,
    tags_from_legacy_statuses,
)
from game.combat.formation import FormationSlot
from game.content.definitions import GameDefinitions
from game.data.schemas import ExpeditionNodeDefinition

SAVE_VERSION = 13
STARTING_COIN = 8


_CONDITION_TO_STRAIN_MARK: dict[str, StrainMark] = {
    "winded": StrainMark.WINDED,
    "drained": StrainMark.DRAINED,
    "battered": StrainMark.BATTERED,
    "frayed": StrainMark.FRAYED,
}


def _life_state_from_raw(data: dict[str, Any]) -> LifeState:
    if data.get("life_state") is not None:
        return LifeState(str(data["life_state"]))
    statuses = _legacy_statuses_from_raw(data)
    return life_state_from_statuses(statuses)


def _legacy_statuses_from_raw(data: dict[str, Any]) -> set[ActorStatus]:
    statuses: set[ActorStatus] = set()
    for raw in data.get("statuses", []):
        value = str(raw)
        try:
            statuses.add(ActorStatus(value))
            continue
        except ValueError:
            pass
        try:
            statuses.add(ActorStatus[value])
        except KeyError:
            continue
    return statuses


def _tags_from_raw(data: dict[str, Any]) -> set[Tag]:
    tags: set[Tag] = set()
    for raw in data.get("tags", []):
        value = str(raw)
        try:
            tags.add(Tag[value])
            continue
        except KeyError:
            pass
        normalized = value.upper()
        try:
            tags.add(Tag[normalized])
        except KeyError:
            continue
    tags.update(tags_from_legacy_statuses(_legacy_statuses_from_raw(data)))
    return tags


def _strain_name_from_raw(data: dict[str, Any]) -> str:
    if data.get("strain") is not None:
        return str(data["strain"])
    fatigue_raw = data.get("fatigue", FatigueState.STEADY.name)
    try:
        fatigue = FatigueState[str(fatigue_raw)]
    except KeyError:
        fatigue = FatigueState.STEADY
    strain = strain_from_fatigue(fatigue)
    if any(condition in {"spent", "exhausted"} for condition in data.get("conditions", [])):
        strain = StrainTier.SPENT
    return strain.name


def _strain_marks_from_raw(data: dict[str, Any]) -> set[StrainMark]:
    marks = _strain_marks_from_conditions(data.get("strain_marks", ()))
    marks.update(_strain_marks_from_conditions(data.get("conditions", ())))
    return marks


def _strain_marks_from_conditions(raw_values: object) -> set[StrainMark]:
    if not isinstance(raw_values, list | tuple | set):
        return set()
    marks: set[StrainMark] = set()
    for raw in raw_values:
        value = str(raw)
        if value in _CONDITION_TO_STRAIN_MARK:
            marks.add(_CONDITION_TO_STRAIN_MARK[value])
        elif value in {"spent", "exhausted"}:
            marks.update({StrainMark.WINDED, StrainMark.DRAINED, StrainMark.FRAYED})
    return marks


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
            encounter_id=(
                None if data.get("encounter_id") is None else str(data["encounter_id"])
            ),
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
            encounter_id=(
                None if data.get("encounter_id") is None else str(data["encounter_id"])
            ),
        )


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
            life_state=_life_state_from_raw(data).value,
            personal_quirk=(
                None
                if data.get("personal_quirk") is None
                else str(data["personal_quirk"])
            ),
            quirks=[str(quirk) for quirk in data.get("quirks", [])],
            strain_marks=sorted(mark.value for mark in _strain_marks_from_raw(data)),
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
            encounter_id=(
                None if data.get("encounter_id") is None else str(data["encounter_id"])
            ),
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
                str(key): int(value) for key, value in data.get("supplies_spent", {}).items()
            },
            participant_ids=[str(hero_id) for hero_id in data.get("participant_ids", [])],
            start_reputation=int(data.get("start_reputation", 0)),
            end_reputation=int(data.get("end_reputation", 0)),
            start_coin=int(data.get("start_coin", 0)),
            end_coin=int(data.get("end_coin", 0)),
            start_supplies={
                str(key): int(value) for key, value in data.get("start_supplies", {}).items()
            },
            end_supplies={
                str(key): int(value) for key, value in data.get("end_supplies", {}).items()
            },
            start_inventory={
                str(key): int(value) for key, value in data.get("start_inventory", {}).items()
            },
            end_inventory={
                str(key): int(value) for key, value in data.get("end_inventory", {}).items()
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
            notable_moments=[
                str(moment) for moment in data.get("notable_moments", [])
            ],
        )


@dataclass
class MazeRecipe:
    pressure_id: str
    route_length: int
    combat_budget: int
    hazard_budget: int
    reward_lure: bool
    include_hunt: bool
    enemy_policy_id: str
    pressure_tags: tuple[str, ...] = ()
    layout_style: str = "winding"
    branch_budget: int = 2
    room_palette: str = "stone"
    encounter_style: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pressure_id": self.pressure_id,
            "route_length": self.route_length,
            "combat_budget": self.combat_budget,
            "hazard_budget": self.hazard_budget,
            "reward_lure": self.reward_lure,
            "include_hunt": self.include_hunt,
            "enemy_policy_id": self.enemy_policy_id,
            "pressure_tags": list(self.pressure_tags),
            "layout_style": self.layout_style,
            "branch_budget": self.branch_budget,
            "room_palette": self.room_palette,
            "encounter_style": self.encounter_style,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MazeRecipe:
        return cls(
            pressure_id=str(data["pressure_id"]),
            route_length=int(data["route_length"]),
            combat_budget=int(data["combat_budget"]),
            hazard_budget=int(data["hazard_budget"]),
            reward_lure=bool(data["reward_lure"]),
            include_hunt=bool(data["include_hunt"]),
            enemy_policy_id=str(data["enemy_policy_id"]),
            pressure_tags=tuple(str(tag) for tag in data.get("pressure_tags", ())),
            layout_style=str(data.get("layout_style", "winding")),
            branch_budget=int(data.get("branch_budget", 2)),
            room_palette=str(data.get("room_palette", "stone")),
            encounter_style=str(data.get("encounter_style", "standard")),
        )


def _infer_main_spine_length_from_nodes(
    nodes: list[ExpeditionNodeDefinition],
    *,
    run_id: str,
    recipe: MazeRecipe | None,
) -> int:
    prefix = f"{run_id}_room_"
    indices = [
        int(node.id.removeprefix(prefix))
        for node in nodes
        if node.id.startswith(prefix) and node.id.removeprefix(prefix).isdigit()
    ]
    if indices:
        return max(indices)
    if recipe is not None:
        return max(3, recipe.route_length)
    return GENERATED_MAZE_DEFAULT_SPINE_LENGTH


GENERATED_MAZE_DEFAULT_SPINE_LENGTH = 3


@dataclass
class GeneratedDungeonState:
    run_id: str
    seed: int
    source_node_id: str
    return_node_id: str
    dungeon_id: str
    entry_node_id: str
    nodes: list[ExpeditionNodeDefinition]
    recipe: MazeRecipe | None = None
    visited_node_ids: list[str] = field(default_factory=list)
    cleared_node_ids: list[str] = field(default_factory=list)
    collapsed: bool = False
    main_spine_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "source_node_id": self.source_node_id,
            "return_node_id": self.return_node_id,
            "dungeon_id": self.dungeon_id,
            "entry_node_id": self.entry_node_id,
            "nodes": [node.model_dump(mode="json") for node in self.nodes],
            "recipe": self.recipe.to_dict() if self.recipe is not None else None,
            "visited_node_ids": list(self.visited_node_ids),
            "cleared_node_ids": list(self.cleared_node_ids),
            "collapsed": self.collapsed,
            "main_spine_length": self.main_spine_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedDungeonState:
        recipe_raw = data.get("recipe")
        run_id = str(data["run_id"])
        nodes = [
            ExpeditionNodeDefinition.model_validate(node)
            for node in data.get("nodes", [])
            if isinstance(node, dict)
        ]
        recipe = (
            MazeRecipe.from_dict(recipe_raw) if isinstance(recipe_raw, dict) else None
        )
        main_spine_length_raw = data.get("main_spine_length")
        if main_spine_length_raw is not None:
            main_spine_length = int(main_spine_length_raw)
        else:
            main_spine_length = _infer_main_spine_length_from_nodes(
                nodes,
                run_id=run_id,
                recipe=recipe,
            )
        return cls(
            run_id=run_id,
            seed=int(data["seed"]),
            source_node_id=str(data["source_node_id"]),
            return_node_id=str(data["return_node_id"]),
            dungeon_id=str(data["dungeon_id"]),
            entry_node_id=str(data["entry_node_id"]),
            nodes=nodes,
            recipe=recipe,
            visited_node_ids=[
                str(node_id) for node_id in data.get("visited_node_ids", [])
            ],
            cleared_node_ids=[
                str(node_id) for node_id in data.get("cleared_node_ids", [])
            ],
            collapsed=bool(data.get("collapsed", False)),
            main_spine_length=main_spine_length,
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
            visited_node_ids=[
                str(node_id) for node_id in data.get("visited_node_ids", [])
            ],
            cleared_node_ids=[
                str(node_id) for node_id in data.get("cleared_node_ids", [])
            ],
            completed_action_ids=[
                str(action_id) for action_id in data.get("completed_action_ids", [])
            ],
            revealed_exit_ids=[
                str(exit_id) for exit_id in data.get("revealed_exit_ids", [])
            ],
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
class RecruitmentOfferState:
    name: str
    class_id: str
    background: str = ""
    motive: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_id": self.class_id,
            "background": self.background,
            "motive": self.motive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecruitmentOfferState:
        return cls(
            name=str(data["name"]),
            class_id=str(data["class_id"]),
            background=str(data.get("background", "")),
            motive=str(data.get("motive", "")),
        )


@dataclass
class RecruitmentState:
    current_offers: list[RecruitmentOfferState] = field(default_factory=list)
    refresh_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_offers": [offer.to_dict() for offer in self.current_offers],
            "refresh_count": self.refresh_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecruitmentState:
        return cls(
            current_offers=[
                RecruitmentOfferState.from_dict(offer)
                for offer in data.get("current_offers", [])
                if isinstance(offer, dict)
            ],
            refresh_count=int(data.get("refresh_count", 0)),
        )


@dataclass
class ContractRecordState:
    contract_id: str
    state: str = "available"
    accepted_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    last_run_id: str = ""
    rooms_scouted: int = 0
    hunt_cleared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "state": self.state,
            "accepted_count": self.accepted_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "last_run_id": self.last_run_id,
            "rooms_scouted": self.rooms_scouted,
            "hunt_cleared": self.hunt_cleared,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        contract_id: str | None = None,
    ) -> ContractRecordState:
        return cls(
            contract_id=str(data.get("contract_id") or contract_id or ""),
            state=str(data.get("state", "available")),
            accepted_count=int(data.get("accepted_count", 0)),
            completed_count=int(data.get("completed_count", 0)),
            failed_count=int(data.get("failed_count", 0)),
            last_run_id=str(data.get("last_run_id", "")),
            rooms_scouted=int(data.get("rooms_scouted", 0)),
            hunt_cleared=bool(data.get("hunt_cleared", False)),
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
            collapsed_run_ids=[
                str(run_id) for run_id in data.get("collapsed_run_ids", [])
            ],
            scouted_run_ids=[
                str(run_id) for run_id in data.get("scouted_run_ids", [])
            ],
            hunt_run_ids=[str(run_id) for run_id in data.get("hunt_run_ids", [])],
            last_pressure_id=str(data.get("last_pressure_id", "")),
            pressure_counts={
                str(pressure_id): int(count)
                for pressure_id, count in data.get("pressure_counts", {}).items()
            },
            last_seed=(
                None if data.get("last_seed") is None else int(data["last_seed"])
            ),
        )


@dataclass
class ExpeditionSessionState:
    expedition_id: str
    dungeon_id: str
    current_node_id: str
    previous_node_id: str = ""
    visited_node_ids: list[str] = field(default_factory=list)
    cleared_node_ids: list[str] = field(default_factory=list)
    completed_action_ids: list[str] = field(default_factory=list)
    revealed_exit_ids: list[str] = field(default_factory=list)
    pending_combat_node_id: str | None = None
    generated_dungeon: GeneratedDungeonState | None = None
    report: ExpeditionReportState | None = None

    def __post_init__(self) -> None:
        if self.current_node_id not in self.visited_node_ids:
            self.visited_node_ids.append(self.current_node_id)
        if self.report is None:
            self.report = ExpeditionReportState(
                expedition_id=self.expedition_id,
                dungeon_id=self.dungeon_id,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expedition_id": self.expedition_id,
            "dungeon_id": self.dungeon_id,
            "current_node_id": self.current_node_id,
            "previous_node_id": self.previous_node_id,
            "visited_node_ids": list(self.visited_node_ids),
            "cleared_node_ids": list(self.cleared_node_ids),
            "completed_action_ids": list(self.completed_action_ids),
            "revealed_exit_ids": list(self.revealed_exit_ids),
            "pending_combat_node_id": self.pending_combat_node_id,
            "generated_dungeon": (
                self.generated_dungeon.to_dict()
                if self.generated_dungeon is not None
                else None
            ),
            "report": self.report.to_dict() if self.report is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpeditionSessionState:
        report_raw = data.get("report")
        generated_raw = data.get("generated_dungeon")
        return cls(
            expedition_id=str(data["expedition_id"]),
            dungeon_id=str(data["dungeon_id"]),
            current_node_id=str(data["current_node_id"]),
            previous_node_id=str(data.get("previous_node_id", "")),
            visited_node_ids=[str(node_id) for node_id in data.get("visited_node_ids", [])],
            cleared_node_ids=[str(node_id) for node_id in data.get("cleared_node_ids", [])],
            completed_action_ids=[
                str(action_id) for action_id in data.get("completed_action_ids", [])
            ],
            revealed_exit_ids=[str(exit_id) for exit_id in data.get("revealed_exit_ids", [])],
            pending_combat_node_id=(
                None
                if data.get("pending_combat_node_id") is None
                else str(data["pending_combat_node_id"])
            ),
            generated_dungeon=GeneratedDungeonState.from_dict(generated_raw)
            if isinstance(generated_raw, dict)
            else None,
            report=ExpeditionReportState.from_dict(report_raw)
            if isinstance(report_raw, dict)
            else None,
        )


@dataclass
class HeroState:
    hero_id: str
    name: str
    class_id: str
    max_hp: int
    hp: int
    speed: int
    accuracy: int
    defense: int
    damage: int
    max_effort: int
    effort: int
    skills: list[str]
    formation_slot: FormationSlot
    background: str = ""
    motive: str = ""
    life_state: LifeState = LifeState.ALIVE
    morale: MoraleState = MoraleState.STEADY
    strain: StrainTier = StrainTier.STEADY
    tags: set[Tag] = field(default_factory=set)
    quirks: list[str] = field(default_factory=list)
    strain_marks: set[StrainMark] = field(default_factory=set)
    personal_quirk: str | None = None
    mortal_wounds: int = 0
    in_surgery: bool = False
    equipped_gear_id: str | None = None
    career_signals: dict[str, int] = field(default_factory=dict)
    fresh_memories: list[FreshMemoryState] = field(default_factory=list)
    earned_quirk_slots: list[EarnedQuirkSlotState] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.earned_quirk_slots = synthesize_earned_slots(
            self.quirks,
            self.earned_quirk_slots,
        )
        self.quirks = flat_quirks_from_slots(self.earned_quirk_slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "name": self.name,
            "class_id": self.class_id,
            "background": self.background,
            "motive": self.motive,
            "max_hp": self.max_hp,
            "hp": self.hp,
            "speed": self.speed,
            "accuracy": self.accuracy,
            "defense": self.defense,
            "damage": self.damage,
            "max_effort": self.max_effort,
            "effort": self.effort,
            "skills": list(self.skills),
            "formation_slot": self.formation_slot.value,
            "life_state": self.life_state.value,
            "morale": self.morale.name,
            "strain": self.strain.name,
            "tags": sorted(tag.name for tag in self.tags),
            "quirks": flat_quirks_from_slots(self.earned_quirk_slots)
            if self.earned_quirk_slots
            else list(self.quirks),
            "strain_marks": sorted(mark.value for mark in self.strain_marks),
            "personal_quirk": self.personal_quirk,
            "mortal_wounds": self.mortal_wounds,
            "in_surgery": self.in_surgery,
            "equipped_gear_id": self.equipped_gear_id,
            "career_signals": dict(self.career_signals),
            "fresh_memories": [memory.to_dict() for memory in self.fresh_memories],
            "earned_quirk_slots": [
                slot.to_dict() for slot in self.earned_quirk_slots
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeroState:
        return cls(
            hero_id=str(data["hero_id"]),
            name=str(data["name"]),
            class_id=str(data["class_id"]),
            background=str(data.get("background", "")),
            motive=str(data.get("motive", "")),
            max_hp=int(data["max_hp"]),
            hp=int(data["hp"]),
            speed=int(data["speed"]),
            accuracy=int(data["accuracy"]),
            defense=int(data["defense"]),
            damage=int(data["damage"]),
            max_effort=int(data["max_effort"]),
            effort=int(data["effort"]),
            skills=[str(skill) for skill in data["skills"]],
            formation_slot=FormationSlot(str(data["formation_slot"])),
            life_state=_life_state_from_raw(data),
            morale=MoraleState[str(data.get("morale", MoraleState.STEADY.name))],
            strain=StrainTier[_strain_name_from_raw(data)],
            tags=_tags_from_raw(data),
            quirks=[str(quirk) for quirk in data.get("quirks", [])],
            strain_marks=_strain_marks_from_raw(data),
            personal_quirk=(
                None
                if data.get("personal_quirk") is None
                else str(data["personal_quirk"])
            ),
            mortal_wounds=int(data.get("mortal_wounds", 0)),
            in_surgery=bool(data.get("in_surgery", False)),
            equipped_gear_id=(
                None
                if data.get("equipped_gear_id") is None
                else str(data["equipped_gear_id"])
            ),
            career_signals={
                str(key): int(value)
                for key, value in data.get("career_signals", {}).items()
            },
            fresh_memories=[
                FreshMemoryState.from_dict(memory)
                for memory in data.get("fresh_memories", [])
                if isinstance(memory, dict)
            ],
            earned_quirk_slots=[
                EarnedQuirkSlotState.from_dict(slot)
                for slot in data.get("earned_quirk_slots", [])
                if isinstance(slot, dict)
            ],
        )

    @property
    def statuses(self) -> set[ActorStatus]:
        return StatusSetProxy(self)

    @statuses.setter
    def statuses(self, value: set[ActorStatus]) -> None:
        self.life_state = life_state_from_statuses(value)
        self.tags.update(tags_from_legacy_statuses(value))

    @property
    def fatigue(self) -> FatigueState:
        return fatigue_from_strain(self.strain)

    @fatigue.setter
    def fatigue(self, value: FatigueState) -> None:
        self.strain = strain_from_fatigue(value)

    @property
    def conditions(self) -> list[str]:
        if self.strain == StrainTier.SPENT:
            return ["spent"]
        return sorted(mark.value for mark in self.strain_marks)

    @conditions.setter
    def conditions(self, value: list[str]) -> None:
        strain = self.strain
        marks = _strain_marks_from_conditions(value)
        if any(condition in {"spent", "exhausted"} for condition in value):
            strain = StrainTier.SPENT
        self.strain = strain
        self.strain_marks = marks


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
        roster = [HeroState.from_dict(hero) for hero in data["roster"]]
        active_party_raw = data.get("active_party_slots")
        active_party_slots = (
            _active_party_from_raw(active_party_raw)
            if isinstance(active_party_raw, dict)
            else _active_party_from_roster(roster)
        )
        active_expedition_raw = data.get("active_expedition")
        last_report_raw = data.get("last_expedition_report")
        dungeon_memory_raw = data.get("dungeon_memory")
        world_memory_raw = data.get("world_memory")
        recruitment_raw = data.get("recruitment_state")
        contract_records_raw = data.get("contract_records")
        breach_memory_raw = data.get("breach_memory")
        return cls(
            company_id=str(data["company_id"]),
            name=str(data["name"]),
            roster=roster,
            supplies={str(key): int(value) for key, value in data["supplies"].items()},
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
            deceased_heroes=[
                HeroState.from_dict(hero) for hero in data.get("deceased_heroes", [])
            ],
            hero_memories=[
                HeroMemoryEntry.from_dict(memory)
                for memory in data.get("hero_memories", [])
                if isinstance(memory, dict)
            ],
            company_timeline=[
                CompanyTimelineEntry.from_dict(entry)
                for entry in data.get("company_timeline", [])
                if isinstance(entry, dict)
            ],
            town_state=_normalize_town_state(data.get("town_state")),
            flags={str(key): bool(value) for key, value in data.get("flags", {}).items()},
            dungeon_memory={
                str(dungeon_id): DungeonMemoryState.from_dict(
                    memory,
                    dungeon_id=str(dungeon_id),
                )
                for dungeon_id, memory in (
                    dungeon_memory_raw.items()
                    if isinstance(dungeon_memory_raw, dict)
                    else ()
                )
                if isinstance(memory, dict)
            },
            world_memory={
                str(location_id): WorldLocationMemoryState.from_dict(
                    memory,
                    location_id=str(location_id),
                )
                for location_id, memory in (
                    world_memory_raw.items()
                    if isinstance(world_memory_raw, dict)
                    else ()
                )
                if isinstance(memory, dict)
            },
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
            breach_memory={
                str(source_node_id): BreachMemoryState.from_dict(
                    memory,
                    source_node_id=str(source_node_id),
                )
                for source_node_id, memory in (
                    breach_memory_raw.items()
                    if isinstance(breach_memory_raw, dict)
                    else ()
                )
                if isinstance(memory, dict)
            },
            purchased_upgrade_ids={
                str(upgrade_id) for upgrade_id in data.get("purchased_upgrade_ids", [])
            },
            active_party_slots=active_party_slots,
            active_expedition=ExpeditionSessionState.from_dict(active_expedition_raw)
            if isinstance(active_expedition_raw, dict)
            else None,
            last_expedition_report=ExpeditionReportState.from_dict(last_report_raw)
            if isinstance(last_report_raw, dict)
            else None,
            expedition_reports=[
                ExpeditionReportState.from_dict(report)
                for report in data.get("expedition_reports", [])
                if isinstance(report, dict)
            ],
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
