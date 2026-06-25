"""Expedition session and generated dungeon state for campaign saves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.campaign.reports_state import ExpeditionReportState
from game.data.schemas import ExpeditionNodeDefinition


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


GENERATED_MAZE_DEFAULT_SPINE_LENGTH = 3


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
        recipe = MazeRecipe.from_dict(recipe_raw) if isinstance(recipe_raw, dict) else None
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
            visited_node_ids=[str(node_id) for node_id in data.get("visited_node_ids", [])],
            cleared_node_ids=[str(node_id) for node_id in data.get("cleared_node_ids", [])],
            collapsed=bool(data.get("collapsed", False)),
            main_spine_length=main_spine_length,
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
                self.generated_dungeon.to_dict() if self.generated_dungeon is not None else None
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
            visited_node_ids=[
                str(node_id) for node_id in data.get("visited_node_ids", [])
            ],
            cleared_node_ids=[
                str(node_id) for node_id in data.get("cleared_node_ids", [])
            ],
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
            report=(
                ExpeditionReportState.from_dict(report_raw)
                if isinstance(report_raw, dict)
                else None
            ),
        )

