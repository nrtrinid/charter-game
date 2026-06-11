"""Deterministic generated Maze breach routes."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from game.campaign.company import GeneratedDungeonState, MazeRecipe
from game.core.rng import GameRng
from game.data.schemas import ExpeditionNodeDefinition, ExpeditionRoomActionDefinition
from game.expedition.node import ExpeditionNodeType

GENERATED_MAZE_DUNGEON_ID = "generated_maze_breach"
GENERATED_MAZE_CONTRACT_ID = "shallow_cave_breach_scout"
GENERATED_MAZE_HUNT_CONTRACT_ID = "shallow_cave_breach_hunt"
GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID = "shallow_cave_breach_scout_posting"
GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID = "shallow_cave_breach_hunt_posting"
GENERATED_MAZE_HUNT_ENCOUNTER_ID = "generated_maze_hunt"
GENERATED_MAZE_REQUIRED_ROOMS = 3
MAX_MAZE_PLACEMENT_RADIUS = 16
Position = tuple[int, int]
DIRECTION_STEPS: tuple[Position, ...] = ((0, 1), (1, 0), (-1, 0), (0, -1))
PreviewNodeType = Literal["maze", "curio", "combat"]

NORMAL_ROOM_TEMPLATES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "Black Stone Room",
        "Black stone holds a room together just long enough for the company to name its exits.",
        "The walls agree with each other for now.",
        "The route keeps its shape, but only while watched.",
        "Boots slow whenever the room seems to breathe.",
    ),
    (
        "Auction-Chalk Niche",
        "Chalk prices cover the wall beside old hammer marks and newer fingerprints.",
        "Someone has tried to value the impossible and failed in neat columns.",
        "The route bends past a chalked bid no one should answer.",
        "The company keeps their hands away from the marked stone.",
    ),
    (
        "Folded Corridor",
        "The corridor folds twice before admitting it is a room.",
        "Lantern light repeats at the corners without matching the flame.",
        "The path is visible, but distance is clearly lying.",
        "Nobody agrees how many steps it took to cross.",
    ),
    (
        "Upside-Down Landing",
        "A stair landing clings to the wall with no stairs above or below it.",
        "Dust drifts upward whenever anyone speaks.",
        "The exit waits where the ceiling would be in a kinder place.",
        "The party keeps low, as if height has become risky.",
    ),
)

CURIO_ROOM_TEMPLATES: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    (
        "Numbered Hall",
        "Numbers are carved into the stones in a hand that changes size between each mark.",
        "The count goes higher than the visible rooms should allow.",
        "One side passage glints; the main route continues darker.",
        "The Scribe refuses to say which number is missing.",
        "mark_route",
        "Mark Route",
    ),
    (
        "Listening Arch",
        "An archway hums with voices from rooms the company has not entered yet.",
        "The arch repeats each footstep once before it happens.",
        "The route beyond the arch sounds nearer than it looks.",
        "Everyone lowers their voice without being told.",
        "listen_at_arch",
        "Listen At Arch",
    ),
    (
        "Lantern Court",
        "Dead lanterns hang in a square around one warm patch of floor.",
        "One lantern swings whenever someone thinks about leaving.",
        "A dimmer route opens beside the warm floor.",
        "The company walks around the lit patch.",
        "map_lanterns",
        "Map Lanterns",
    ),
)

COMBAT_ROOM_TEMPLATES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "Leech Alcove",
        "The chamber waits with the stillness of a place that has heard breathing before.",
        "A wet scrape answers the lantern.",
        "The route narrows through the chamber.",
        "The line tightens before the first shape moves.",
    ),
    (
        "Black-Pulse Chamber",
        "Every heartbeat comes back from the chamber wall a half-beat late.",
        "The air bruises around the lantern flame.",
        "The route continues through the pulse if the company can hold formation.",
        "Someone counts breaths and loses count at three.",
    ),
    (
        "Teeth-In-Stone Crossing",
        "Stone teeth jut from the floor in rows too even to be natural.",
        "The gaps between the teeth are full of old movement.",
        "The only clean exit lies past the crossing.",
        "The front line raises weapons before anything shows itself.",
    ),
)


def generate_maze_breach_route(
    *,
    run_number: int,
    source_node_id: str,
    return_node_id: str,
    rng: GameRng,
    seed: int | None = None,
    include_hunt: bool = False,
    recipe: MazeRecipe | None = None,
) -> GeneratedDungeonState:
    route_seed = seed if seed is not None else rng.randint(1, 999_999)
    route_rng = GameRng(route_seed)
    run_id = f"maze_run_{run_number:04d}"
    prefix = run_id
    if recipe is None:
        recipe = _legacy_recipe(
            route_length=3 + route_rng.randint(0, 2),
            include_hunt=include_hunt,
        )
    route_length = max(3, recipe.route_length)
    combat_budget = max(0, recipe.combat_budget)

    entry_id = f"{prefix}_entry"
    occupied_positions: set[Position] = {(0, 0)}
    main_positions = _main_path_positions(
        route_length,
        route_rng,
        occupied_positions,
        layout_style=recipe.layout_style,
    )
    rooms: list[ExpeditionNodeDefinition] = [
        ExpeditionNodeDefinition(
            id=entry_id,
            name="Breach Threshold",
            node_type=ExpeditionNodeType.MAZE,
            text=(
                "The breach folds behind the company into black stone, chalk dust, "
                "and a corridor no map has earned yet."
            ),
            scene_state="The route is new enough to feel offended by footsteps.",
            route_hint="The way back is still visible through the breach-light.",
            party_hint="Everyone looks back once.",
            major_beat=True,
            map_id=GENERATED_MAZE_DUNGEON_ID,
            position=(0, 0),
            exits=[f"{prefix}_room_1"],
        ),
    ]

    linear_ids = [f"{prefix}_room_{index}" for index in range(1, route_length + 1)]
    curio_index = min(2, route_length)
    combat_index = min(max(3, curio_index + 1), route_length)
    used_names: set[str] = {"Breach Threshold"}
    for index, node_id in enumerate(linear_ids, start=1):
        exits = []
        if index > 1:
            exits.append(linear_ids[index - 2])
        else:
            exits.append(entry_id)
        if index < len(linear_ids):
            exits.append(linear_ids[index])
        rooms.append(
            _linear_room(
                prefix,
                node_id,
                index,
                exits,
                route_rng,
                position=main_positions[index - 1],
                curio_index=curio_index,
                combat_index=combat_index,
                combat_budget=combat_budget,
                room_palette=recipe.room_palette,
                encounter_style=recipe.encounter_style,
                used_names=used_names,
            )
        )

    terminal_forward_position = _planned_spine_extension_position(
        route_seed=route_seed,
        run_id=run_id,
        nodes_by_id={node.id: node for node in rooms},
        frontier_id=linear_ids[-1],
        next_depth=route_length + 1,
        recipe=recipe,
        occupied_positions=occupied_positions,
    )
    reserved_branch_positions = (
        {terminal_forward_position} if terminal_forward_position is not None else set()
    )

    if _echo_branch_enabled(recipe):
        echo_position = _try_branch_position(
            main_positions[0],
            occupied_positions,
            route_rng,
            reserved_positions=reserved_branch_positions,
        )
        if echo_position is not None:
            echo_id = f"{prefix}_echo"
            _append_room_exit(rooms, linear_ids[0], echo_id)
            echo_name = _choose_unique_option(
                route_rng,
                ("Echo Pocket", "Chalk-Quiet Spur"),
                used_names,
            )
            used_names.add(echo_name)
            rooms.append(
                ExpeditionNodeDefinition(
                    id=echo_id,
                    name=echo_name,
                    node_type=ExpeditionNodeType.CURIO,
                    text=(
                        "A short spur holds the company's last few words, arranged as if "
                        "they were clues."
                    ),
                    scene_state="The walls listen better than they should.",
                    route_hint="The main route waits one room back.",
                    party_hint="The company stops talking until the echo fades.",
                    map_id=GENERATED_MAZE_DUNGEON_ID,
                    position=echo_position,
                    exits=[linear_ids[0]],
                    actions=[
                        ExpeditionRoomActionDefinition(
                            id="study_echo",
                            label="Study Echo",
                            description="Compare the echo to the route just walked.",
                            result_text="The echo reveals one false turn before it closes.",
                        )
                    ],
                )
            )

    if _reward_branch_enabled(recipe):
        reward_position = _try_branch_position(
            main_positions[curio_index - 1],
            occupied_positions,
            route_rng,
            reserved_positions=reserved_branch_positions,
        )
        if reward_position is not None:
            reward_id = f"{prefix}_reward"
            _append_room_exit(rooms, linear_ids[curio_index - 1], reward_id)
            reward_name = _choose_unique_option(
                route_rng,
                ("Maze-Glass Shelf", "Coin-Slit Reliquary"),
                used_names,
            )
            used_names.add(reward_name)
            rooms.append(
                ExpeditionNodeDefinition(
                    id=reward_id,
                    name=reward_name,
                    node_type=ExpeditionNodeType.CURIO,
                    text=(
                        "A shelf of black stone offers a coin-thin shard of Maze-glass, "
                        "left as if the room expects payment later."
                    ),
                    scene_state="The shard reflects the party from the wrong side.",
                    route_hint="The main route waits beyond the numbered hall.",
                    map_id=GENERATED_MAZE_DUNGEON_ID,
                    position=reward_position,
                    exits=[linear_ids[curio_index - 1]],
                    actions=[
                        ExpeditionRoomActionDefinition(
                            id="take_maze_glass",
                            label="Take Maze-Glass",
                            description="Wrap the shard before the room changes its mind.",
                            result_text="The company wraps a shard of Maze-glass for the ledger.",
                            loot={"maze_glass": 1},
                        )
                    ],
                )
            )

    if recipe.include_hunt:
        hunt_position = _try_branch_position(
            main_positions[-1],
            occupied_positions,
            route_rng,
            reserved_positions=reserved_branch_positions,
        )
        if hunt_position is not None:
            hunt_id = f"{prefix}_hunt_lair"
            _append_room_exit(rooms, linear_ids[-1], hunt_id)
            hunt_name = _choose_unique_option(
                route_rng,
                ("Marked Lair", "Maw-Marked Den"),
                used_names,
            )
            used_names.add(hunt_name)
            rooms.append(
                ExpeditionNodeDefinition(
                    id=hunt_id,
                    name=hunt_name,
                    node_type=ExpeditionNodeType.BOSS,
                    text=(
                        "The contract mark burns into the wall here. Something deeper "
                        "has learned to wait where hunters expect prey."
                    ),
                    scene_state="The room holds still in the manner of a jaw.",
                    route_hint="The main route remains behind; the marked thing waits ahead.",
                    party_hint="The company spreads out before anyone volunteers to step first.",
                    major_beat=True,
                    map_id=GENERATED_MAZE_DUNGEON_ID,
                    position=hunt_position,
                    encounter=GENERATED_MAZE_HUNT_ENCOUNTER_ID,
                    exits=[linear_ids[-1]],
                )
            )
    if _hard_branch_enabled(recipe, combat_budget):
        hard_room_position = _try_branch_position(
            main_positions[-1],
            occupied_positions,
            route_rng,
            reserved_positions=reserved_branch_positions,
        )
        if hard_room_position is not None:
            hard_room_id = f"{prefix}_hard_room"
            _append_room_exit(rooms, linear_ids[-1], hard_room_id)
            hard_room_name = _choose_unique_option(
                route_rng,
                ("Wrong-Turn Cell", "Debt-Bell Dead End"),
                used_names,
            )
            used_names.add(hard_room_name)
            rooms.append(
                ExpeditionNodeDefinition(
                    id=hard_room_id,
                    name=hard_room_name,
                    node_type=ExpeditionNodeType.COMBAT,
                    text=(
                        "The route pinches into a cell where earlier bootprints arrive from "
                        "several directions and none leave cleanly."
                    ),
                    scene_state="The cell is optional. The sound inside is not inviting.",
                    route_hint="The marked path remains behind; profit gathers ahead.",
                    party_hint="Someone says the sensible thing and nobody moves yet.",
                    major_beat=True,
                    map_id=GENERATED_MAZE_DUNGEON_ID,
                    position=hard_room_position,
                    encounter=_hard_room_encounter(recipe.encounter_style, route_rng),
                    exits=[linear_ids[-1]],
                )
            )

    return GeneratedDungeonState(
        run_id=run_id,
        seed=route_seed,
        source_node_id=source_node_id,
        return_node_id=return_node_id,
        dungeon_id=GENERATED_MAZE_DUNGEON_ID,
        entry_node_id=entry_id,
        nodes=rooms,
        recipe=recipe,
        main_spine_length=route_length,
    )


@dataclass(frozen=True)
class MazeSpineExtensionResult:
    node: ExpeditionNodeDefinition
    new_depth: int
    previous_frontier_id: str


@dataclass(frozen=True)
class ExtensionPreviewContent:
    name: str
    node_type: PreviewNodeType


@dataclass(frozen=True)
class FrontierExitPreview:
    exit_id: str
    label: str
    kind: Literal["forward", "spur"]
    node_type: PreviewNodeType


def derive_maze_extension_seed(route_seed: int, run_id: str, seed_offset: int) -> int:
    """Stable per-extension seed mix. Does not use Python hash()."""
    value = route_seed & 0x7FFFFFFF
    for char in run_id:
        value = (value * 131 + ord(char)) & 0x7FFFFFFF
    value = (value * 131 + seed_offset) & 0x7FFFFFFF
    return value or 1


def _extension_rng_offsets(
    next_depth: int,
    kind: Literal["forward", "spur"],
) -> tuple[int, int]:
    """Return separate RNG offsets for layout position and room content."""
    if kind == "forward":
        return next_depth, next_depth * 10_000 + 2
    return next_depth * 10_000 + 1, next_depth * 10_000 + 3


def frontier_node_id(generated: GeneratedDungeonState) -> str:
    return f"{generated.run_id}_room_{generated.main_spine_length}"


def is_main_spine_room(node_id: str, run_id: str) -> bool:
    match = re.fullmatch(rf"{re.escape(run_id)}_room_(\d+)", node_id)
    return match is not None


def is_generated_spur_room(node_id: str, run_id: str) -> bool:
    return re.fullmatch(rf"{re.escape(run_id)}_spur_\d+", node_id) is not None


def _route_used_names(nodes: Sequence[ExpeditionNodeDefinition]) -> set[str]:
    return {node.name for node in nodes}


def _append_room_exit(
    rooms: list[ExpeditionNodeDefinition],
    node_id: str,
    exit_id: str,
) -> None:
    for index, node in enumerate(rooms):
        if node.id != node_id:
            continue
        if exit_id in node.exits:
            return
        rooms[index] = node.model_copy(update={"exits": [*node.exits, exit_id]})
        return
    raise ValueError(f"Generated Maze anchor room is missing: {node_id}")


def infer_main_spine_length(
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
    return GENERATED_MAZE_REQUIRED_ROOMS


def extend_maze_main_spine(generated: GeneratedDungeonState) -> MazeSpineExtensionResult:
    if generated.collapsed:
        raise ValueError("Cannot extend a collapsed Maze route.")
    recipe = generated.recipe or _legacy_recipe(route_length=3, include_hunt=False)
    previous_frontier_id = frontier_node_id(generated)
    nodes_by_id = {node.id: node for node in generated.nodes}
    frontier = nodes_by_id.get(previous_frontier_id)
    if frontier is None:
        raise ValueError("Maze frontier room is missing from the active route.")
    next_depth = generated.main_spine_length + 1
    next_node_id = f"{generated.run_id}_room_{next_depth}"
    if next_node_id in nodes_by_id:
        raise ValueError("Maze frontier already has a deeper room.")

    _position_offset, content_offset = _extension_rng_offsets(next_depth, "forward")
    content_rng = GameRng(
        derive_maze_extension_seed(generated.seed, generated.run_id, content_offset)
    )
    occupied_positions: set[Position] = {
        node.position
        for node in generated.nodes
        if node.position is not None
    }
    new_position = _planned_spine_extension_position(
        route_seed=generated.seed,
        run_id=generated.run_id,
        nodes_by_id=nodes_by_id,
        frontier_id=previous_frontier_id,
        next_depth=next_depth,
        recipe=recipe,
        occupied_positions=occupied_positions,
    )
    if new_position is None:
        raise ValueError("Maze frontier has no adjacent space for a deeper room.")
    back_exit_id = previous_frontier_id
    used_names = _route_used_names(generated.nodes)
    new_room = _extended_linear_room(
        generated.run_id,
        next_node_id,
        next_depth,
        [back_exit_id],
        content_rng,
        position=new_position,
        recipe=recipe,
        used_names=used_names,
    )
    frontier_index = next(
        index for index, node in enumerate(generated.nodes) if node.id == previous_frontier_id
    )
    updated_frontier = frontier.model_copy(update={"exits": [*frontier.exits, next_node_id]})
    generated.nodes[frontier_index] = updated_frontier
    generated.nodes.append(new_room)
    generated.main_spine_length = next_depth
    return MazeSpineExtensionResult(
        node=new_room,
        new_depth=next_depth,
        previous_frontier_id=previous_frontier_id,
    )


def frontier_exit_previews(
    generated: GeneratedDungeonState,
    current_node_id: str,
) -> tuple[FrontierExitPreview, ...]:
    if generated.collapsed or current_node_id != frontier_node_id(generated):
        return ()
    recipe = generated.recipe or _legacy_recipe(route_length=3, include_hunt=False)
    next_depth = generated.main_spine_length + 1
    nodes_by_id = {node.id: node for node in generated.nodes}
    used_names = _route_used_names(generated.nodes)
    occupied_positions: set[Position] = {
        node.position for node in generated.nodes if node.position is not None
    }
    previews: list[FrontierExitPreview] = []
    forward_id = f"{generated.run_id}_room_{next_depth}"
    forward_position = _planned_spine_extension_position(
        route_seed=generated.seed,
        run_id=generated.run_id,
        nodes_by_id=nodes_by_id,
        frontier_id=current_node_id,
        next_depth=next_depth,
        recipe=recipe,
        occupied_positions=occupied_positions,
    )
    if forward_id not in nodes_by_id and forward_position is not None:
        forward_content = _preview_extension_content(
            generated.seed,
            generated.run_id,
            next_depth,
            recipe,
            kind="forward",
            used_names=used_names,
        )
        previews.append(
            FrontierExitPreview(
                exit_id=forward_id,
                label=forward_content.name,
                kind="forward",
                node_type=forward_content.node_type,
            )
        )
    spur_id = f"{generated.run_id}_spur_{next_depth}"
    reserved_positions = {forward_position} if forward_position is not None else set()
    spur_position = (
        _planned_spur_position(
            route_seed=generated.seed,
            run_id=generated.run_id,
            nodes_by_id=nodes_by_id,
            frontier_id=current_node_id,
            next_depth=next_depth,
            recipe=recipe,
            occupied_positions=occupied_positions,
            reserved_positions=reserved_positions,
        )
        if forward_position is not None and _spur_branch_enabled(recipe, next_depth)
        else None
    )
    if (
        spur_id not in nodes_by_id
        and _spur_branch_enabled(recipe, next_depth)
        and spur_position is not None
    ):
        spur_content = _preview_extension_content(
            generated.seed,
            generated.run_id,
            next_depth,
            recipe,
            kind="spur",
            used_names=used_names,
        )
        previews.append(
            FrontierExitPreview(
                exit_id=spur_id,
                label=spur_content.name,
                kind="spur",
                node_type=spur_content.node_type,
            )
        )
    return tuple(previews)


def frontier_preview_map_positions(
    generated: GeneratedDungeonState,
    previews: Sequence[FrontierExitPreview],
) -> dict[str, Position]:
    """Read-only map coordinates for unstable frontier previews."""
    if not previews:
        return {}
    recipe = generated.recipe or _legacy_recipe(route_length=3, include_hunt=False)
    nodes_by_id = {node.id: node for node in generated.nodes}
    frontier_id = frontier_node_id(generated)
    next_depth = generated.main_spine_length + 1
    occupied_positions: set[Position] = {
        node.position for node in generated.nodes if node.position is not None
    }
    forward_position = _planned_spine_extension_position(
        route_seed=generated.seed,
        run_id=generated.run_id,
        nodes_by_id=nodes_by_id,
        frontier_id=frontier_id,
        next_depth=next_depth,
        recipe=recipe,
        occupied_positions=occupied_positions,
    )
    positions: dict[str, Position] = {}
    for preview in previews:
        if preview.kind != "forward":
            continue
        if forward_position is not None:
            positions[preview.exit_id] = forward_position
    reserved_positions = {forward_position} if forward_position is not None else set()
    for preview in previews:
        if preview.kind != "spur":
            continue
        position = _planned_spur_position(
            route_seed=generated.seed,
            run_id=generated.run_id,
            nodes_by_id=nodes_by_id,
            frontier_id=frontier_id,
            next_depth=next_depth,
            recipe=recipe,
            occupied_positions=occupied_positions,
            reserved_positions=reserved_positions,
        )
        if position is not None:
            positions[preview.exit_id] = position
    return positions


def frontier_preview_map_position(
    generated: GeneratedDungeonState,
    preview: FrontierExitPreview,
) -> Position:
    """Read-only map coordinates for a single unstable frontier exit preview."""
    positions = frontier_preview_map_positions(generated, (preview,))
    position = positions.get(preview.exit_id)
    if position is None:
        raise ValueError(f"No preview position available for {preview.exit_id}.")
    return position


def extend_maze_spur(generated: GeneratedDungeonState) -> MazeSpineExtensionResult:
    if generated.collapsed:
        raise ValueError("Cannot extend a collapsed Maze route.")
    recipe = generated.recipe or _legacy_recipe(route_length=3, include_hunt=False)
    previous_frontier_id = frontier_node_id(generated)
    nodes_by_id = {node.id: node for node in generated.nodes}
    frontier = nodes_by_id.get(previous_frontier_id)
    if frontier is None:
        raise ValueError("Maze frontier room is missing from the active route.")
    next_depth = generated.main_spine_length + 1
    spur_id = f"{generated.run_id}_spur_{next_depth}"
    if spur_id in nodes_by_id:
        raise ValueError("Maze frontier spur already exists.")
    if not _spur_branch_enabled(recipe, next_depth):
        raise ValueError("No unstable spur is available at this frontier.")

    _position_offset, content_offset = _extension_rng_offsets(next_depth, "spur")
    content_rng = GameRng(
        derive_maze_extension_seed(generated.seed, generated.run_id, content_offset)
    )
    occupied_positions: set[Position] = {
        node.position for node in generated.nodes if node.position is not None
    }
    forward_position = _planned_spine_extension_position(
        route_seed=generated.seed,
        run_id=generated.run_id,
        nodes_by_id=nodes_by_id,
        frontier_id=previous_frontier_id,
        next_depth=next_depth,
        recipe=recipe,
        occupied_positions=occupied_positions,
    )
    reserved_positions = {forward_position} if forward_position is not None else set()
    spur_position = _planned_spur_position(
        route_seed=generated.seed,
        run_id=generated.run_id,
        nodes_by_id=nodes_by_id,
        frontier_id=previous_frontier_id,
        next_depth=next_depth,
        recipe=recipe,
        occupied_positions=occupied_positions,
        reserved_positions=reserved_positions,
    )
    if spur_position is None:
        raise ValueError("No adjacent unstable spur is available at this frontier.")
    used_names = _route_used_names(generated.nodes)
    spur_room = _extended_spur_room(
        generated.run_id,
        spur_id,
        previous_frontier_id,
        content_rng,
        position=spur_position,
        recipe=recipe,
        used_names=used_names,
    )
    frontier_index = next(
        index for index, node in enumerate(generated.nodes) if node.id == previous_frontier_id
    )
    updated_frontier = frontier.model_copy(update={"exits": [*frontier.exits, spur_id]})
    generated.nodes[frontier_index] = updated_frontier
    generated.nodes.append(spur_room)
    return MazeSpineExtensionResult(
        node=spur_room,
        new_depth=generated.main_spine_length,
        previous_frontier_id=previous_frontier_id,
    )


def resolve_generated_maze_travel(
    generated: GeneratedDungeonState,
    *,
    current_node_id: str,
    target_node_id: str,
) -> MazeSpineExtensionResult | None:
    if generated.collapsed or target_node_id in generated_nodes_by_id(generated):
        return None
    if current_node_id != frontier_node_id(generated):
        return None
    preview_by_id = {
        preview.exit_id: preview for preview in frontier_exit_previews(generated, current_node_id)
    }
    preview = preview_by_id.get(target_node_id)
    if preview is None:
        return None
    if preview.kind == "forward":
        return extend_maze_main_spine(generated)
    return extend_maze_spur(generated)


def generated_nodes_by_id(
    generated: GeneratedDungeonState | None,
) -> dict[str, ExpeditionNodeDefinition]:
    if generated is None or generated.collapsed:
        return {}
    return {node.id: node for node in generated.nodes}


def _choose_unique_template(
    rng: GameRng,
    templates: tuple[Any, ...],
    used_names: set[str],
    *,
    name_index: int = 0,
) -> tuple[Any, ...]:
    if not templates:
        raise ValueError("No templates available.")
    remaining = list(templates)
    ordered: list[tuple[Any, ...]] = []
    while remaining:
        template = rng.choice(tuple(remaining))
        remaining.remove(template)
        ordered.append(template)
    for template in ordered:
        if template[name_index] not in used_names:
            return template
    return ordered[0]


def _choose_unique_option(
    rng: GameRng,
    options: tuple[str, ...],
    used_names: set[str],
) -> str:
    remaining = list(options)
    ordered: list[str] = []
    while remaining:
        option = rng.choice(tuple(remaining))
        remaining.remove(option)
        ordered.append(option)
    for option in ordered:
        if option not in used_names:
            return option
    return ordered[0]


def _preview_extension_content(
    route_seed: int,
    run_id: str,
    next_depth: int,
    recipe: MazeRecipe,
    *,
    kind: Literal["forward", "spur"],
    used_names: set[str],
) -> ExtensionPreviewContent:
    _position_offset, content_offset = _extension_rng_offsets(next_depth, kind)
    content_rng = GameRng(derive_maze_extension_seed(route_seed, run_id, content_offset))
    if kind == "spur":
        return _extended_spur_preview_content(content_rng, recipe, used_names)
    return _extended_linear_preview_content(next_depth, content_rng, recipe, used_names)


def _extended_linear_preview_content(
    depth: int,
    content_rng: GameRng,
    recipe: MazeRecipe,
    used_names: set[str],
) -> ExtensionPreviewContent:
    if _extension_depth_is_curio(depth):
        name = _choose_unique_template(
            content_rng,
            _curio_templates_for_palette(recipe.room_palette),
            used_names,
        )[0]
        return ExtensionPreviewContent(name=name, node_type="curio")
    if _extension_depth_is_combat(depth):
        name = _choose_unique_template(
            content_rng,
            _combat_templates_for_palette(recipe.room_palette),
            used_names,
        )[0]
        return ExtensionPreviewContent(name=name, node_type="combat")
    name = _choose_unique_template(
        content_rng,
        _normal_templates_for_palette(recipe.room_palette),
        used_names,
    )[0]
    return ExtensionPreviewContent(name=name, node_type="maze")


def _extended_spur_preview_content(
    content_rng: GameRng,
    recipe: MazeRecipe,
    used_names: set[str],
) -> ExtensionPreviewContent:
    if recipe.encounter_style == "brute" and content_rng.chance(40):
        name = _choose_unique_template(
            content_rng,
            _combat_templates_for_palette(recipe.room_palette),
            used_names,
        )[0]
        return ExtensionPreviewContent(name=name, node_type="combat")
    name = _choose_unique_template(
        content_rng,
        _curio_templates_for_palette(recipe.room_palette),
        used_names,
    )[0]
    return ExtensionPreviewContent(name=name, node_type="curio")


def _extended_linear_room(
    prefix: str,
    node_id: str,
    depth: int,
    exits: list[str],
    rng: GameRng,
    *,
    position: Position,
    recipe: MazeRecipe,
    used_names: set[str],
) -> ExpeditionNodeDefinition:
    if _extension_depth_is_curio(depth):
        name, text, scene_state, route_hint, party_hint, action_id, label = _choose_unique_template(
            rng,
            _curio_templates_for_palette(recipe.room_palette),
            used_names,
        )
        used_names.add(name)
        action_description, action_result = _curio_action_text(action_id)
        return ExpeditionNodeDefinition(
            id=node_id,
            name=name,
            node_type=ExpeditionNodeType.CURIO,
            text=text,
            scene_state=scene_state,
            route_hint=route_hint,
            party_hint=party_hint,
            map_id=GENERATED_MAZE_DUNGEON_ID,
            position=position,
            exits=exits,
            actions=[
                ExpeditionRoomActionDefinition(
                    id=action_id,
                    label=label,
                    description=action_description,
                    result_text=action_result,
                    loot={},
                    coin_reward=0,
                )
            ],
        )
    if _extension_depth_is_combat(depth):
        name, text, scene_state, route_hint, party_hint = _choose_unique_template(
            rng,
            _combat_templates_for_palette(recipe.room_palette),
            used_names,
        )
        used_names.add(name)
        return ExpeditionNodeDefinition(
            id=node_id,
            name=name,
            node_type=ExpeditionNodeType.COMBAT,
            text=text,
            scene_state=scene_state,
            route_hint=route_hint,
            party_hint=party_hint,
            major_beat=True,
            map_id=GENERATED_MAZE_DUNGEON_ID,
            position=position,
            encounter=_main_combat_encounter(recipe.encounter_style, rng),
            exits=exits,
        )
    name, text, scene_state, route_hint, party_hint = _choose_unique_template(
        rng,
        _normal_templates_for_palette(recipe.room_palette),
        used_names,
    )
    used_names.add(name)
    return ExpeditionNodeDefinition(
        id=node_id,
        name=name,
        node_type=ExpeditionNodeType.MAZE,
        text=text,
        scene_state=scene_state,
        route_hint=route_hint,
        party_hint=party_hint,
        map_id=GENERATED_MAZE_DUNGEON_ID,
        position=position,
        exits=exits,
    )


def _extension_depth_is_curio(depth: int) -> bool:
    return depth % 4 == 2


def _extension_depth_is_combat(depth: int) -> bool:
    return depth >= 3 and depth % 3 == 0


def _spur_branch_enabled(recipe: MazeRecipe, next_depth: int) -> bool:
    if recipe.branch_budget < 1:
        return False
    if recipe.layout_style == "winding" and recipe.branch_budget < 2 and not recipe.reward_lure:
        return False
    return recipe.reward_lure and next_depth % 2 == 0


def _extended_spur_room(
    prefix: str,
    node_id: str,
    frontier_id: str,
    rng: GameRng,
    *,
    position: Position,
    recipe: MazeRecipe,
    used_names: set[str],
) -> ExpeditionNodeDefinition:
    if recipe.encounter_style == "brute" and rng.chance(40):
        name, text, scene_state, route_hint, party_hint = _choose_unique_template(
            rng,
            _combat_templates_for_palette(recipe.room_palette),
            used_names,
        )
        used_names.add(name)
        return ExpeditionNodeDefinition(
            id=node_id,
            name=name,
            node_type=ExpeditionNodeType.COMBAT,
            text=text,
            scene_state=scene_state,
            route_hint=route_hint,
            party_hint=party_hint,
            major_beat=True,
            map_id=GENERATED_MAZE_DUNGEON_ID,
            position=position,
            encounter=_main_combat_encounter(recipe.encounter_style, rng),
            exits=[frontier_id],
        )
    name, text, scene_state, route_hint, party_hint, action_id, label = _choose_unique_template(
        rng,
        _curio_templates_for_palette(recipe.room_palette),
        used_names,
    )
    used_names.add(name)
    action_description, action_result = _curio_action_text(action_id)
    return ExpeditionNodeDefinition(
        id=node_id,
        name=name,
        node_type=ExpeditionNodeType.CURIO,
        text=text,
        scene_state=scene_state,
        route_hint=route_hint,
        party_hint=party_hint,
        map_id=GENERATED_MAZE_DUNGEON_ID,
        position=position,
        exits=[frontier_id],
        actions=[
            ExpeditionRoomActionDefinition(
                id=action_id,
                label=label,
                description=action_description,
                result_text=action_result,
                loot={},
                coin_reward=0,
            )
        ],
    )


def _linear_room(
    prefix: str,
    node_id: str,
    index: int,
    exits: list[str],
    rng: GameRng,
    *,
    position: Position,
    curio_index: int,
    combat_index: int,
    combat_budget: int,
    room_palette: str,
    encounter_style: str,
    used_names: set[str],
) -> ExpeditionNodeDefinition:
    if index == curio_index:
        name, text, scene_state, route_hint, party_hint, action_id, label = _choose_unique_template(
            rng,
            _curio_templates_for_palette(room_palette),
            used_names,
        )
        used_names.add(name)
        action_description, action_result = _curio_action_text(action_id)
        return ExpeditionNodeDefinition(
            id=node_id,
            name=name,
            node_type=ExpeditionNodeType.CURIO,
            text=text,
            scene_state=scene_state,
            route_hint=route_hint,
            party_hint=party_hint,
            map_id=GENERATED_MAZE_DUNGEON_ID,
            position=position,
            exits=exits,
            actions=[
                ExpeditionRoomActionDefinition(
                    id=action_id,
                    label=label,
                    description=action_description,
                    result_text=action_result,
                )
            ],
        )
    if index == combat_index and combat_budget >= 1:
        name, text, scene_state, route_hint, party_hint = _choose_unique_template(
            rng,
            _combat_templates_for_palette(room_palette),
            used_names,
        )
        used_names.add(name)
        return ExpeditionNodeDefinition(
            id=node_id,
            name=name,
            node_type=ExpeditionNodeType.COMBAT,
            text=text,
            scene_state=scene_state,
            route_hint=route_hint,
            party_hint=party_hint,
            major_beat=True,
            map_id=GENERATED_MAZE_DUNGEON_ID,
            position=position,
            encounter=_main_combat_encounter(encounter_style, rng),
            exits=exits,
        )
    name, text, scene_state, route_hint, party_hint = _choose_unique_template(
        rng,
        _normal_templates_for_palette(room_palette),
        used_names,
    )
    used_names.add(name)
    return ExpeditionNodeDefinition(
        id=node_id,
        name=name,
        node_type=ExpeditionNodeType.MAZE,
        text=text,
        scene_state=scene_state,
        route_hint=route_hint,
        party_hint=party_hint,
        map_id=GENERATED_MAZE_DUNGEON_ID,
        position=position,
        exits=exits,
    )


def _legacy_recipe(route_length: int, include_hunt: bool) -> MazeRecipe:
    pressure_tags = ("legacy", "hunt") if include_hunt else ("legacy",)
    return MazeRecipe(
        pressure_id="legacy_breach_route",
        route_length=route_length,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=include_hunt,
        enemy_policy_id="basic",
        pressure_tags=pressure_tags,
    )


def _reward_branch_enabled(recipe: MazeRecipe) -> bool:
    return recipe.reward_lure and recipe.branch_budget >= 1


def _hard_branch_enabled(recipe: MazeRecipe, combat_budget: int) -> bool:
    return combat_budget >= 2 and recipe.branch_budget >= 2


def _echo_branch_enabled(recipe: MazeRecipe) -> bool:
    return recipe.branch_budget >= 3 and "fork" in recipe.layout_style


def _main_combat_encounter(encounter_style: str, rng: GameRng) -> str:
    if encounter_style == "light":
        return "generated_maze_probe"
    if encounter_style == "brute":
        return "generated_maze_stalker"
    return rng.choice(
        (
            "generated_maze_probe",
            "generated_maze_pattern_cell",
            "maze_depth_1",
        )
    )


def _hard_room_encounter(encounter_style: str, rng: GameRng) -> str:
    if encounter_style == "light":
        return "generated_maze_pattern_cell"
    if encounter_style == "brute":
        return rng.choice(("shallow_cave", "generated_maze_stalker"))
    return "generated_maze_stalker"


def _normal_templates_for_palette(
    room_palette: str,
) -> tuple[tuple[str, str, str, str, str], ...]:
    if room_palette == "market":
        return (
            NORMAL_ROOM_TEMPLATES[1],
            NORMAL_ROOM_TEMPLATES[2],
        )
    if room_palette == "maw":
        return (
            NORMAL_ROOM_TEMPLATES[0],
            NORMAL_ROOM_TEMPLATES[3],
        )
    if room_palette == "glass":
        return (
            NORMAL_ROOM_TEMPLATES[2],
            NORMAL_ROOM_TEMPLATES[3],
        )
    return NORMAL_ROOM_TEMPLATES


def _curio_templates_for_palette(
    room_palette: str,
) -> tuple[tuple[str, str, str, str, str, str, str], ...]:
    if room_palette == "market":
        return (
            CURIO_ROOM_TEMPLATES[0],
            CURIO_ROOM_TEMPLATES[2],
        )
    if room_palette == "maw":
        return (
            CURIO_ROOM_TEMPLATES[1],
            CURIO_ROOM_TEMPLATES[2],
        )
    return CURIO_ROOM_TEMPLATES


def _combat_templates_for_palette(
    room_palette: str,
) -> tuple[tuple[str, str, str, str, str], ...]:
    if room_palette == "maw":
        return (
            COMBAT_ROOM_TEMPLATES[1],
            COMBAT_ROOM_TEMPLATES[2],
        )
    if room_palette == "glass":
        return (
            COMBAT_ROOM_TEMPLATES[0],
            COMBAT_ROOM_TEMPLATES[1],
        )
    return COMBAT_ROOM_TEMPLATES


def _curio_action_text(action_id: str) -> tuple[str, str]:
    if action_id == "listen_at_arch":
        return (
            "Listen long enough to learn which echo belongs to the next room.",
            "The company catches one useful echo before the arch falls quiet.",
        )
    if action_id == "map_lanterns":
        return (
            "Sketch the dead lanterns and the one warm square of floor.",
            "The lantern pattern gives the route one more point of agreement.",
        )
    return (
        "Mark the wall while the route still agrees it has one.",
        "A chalk mark holds long enough to be useful.",
    )


def _main_path_positions(
    route_length: int,
    rng: GameRng,
    occupied: set[Position],
    *,
    layout_style: str,
) -> list[Position]:
    positions: list[Position] = []
    current = (0, 0)
    for index in range(route_length):
        preferred = ((1, 0), (-1, 0)) if index == 1 and positions and positions[0][0] == 0 else None
        current = _open_spine_extension_position(
            current,
            occupied,
            rng,
            preferred=preferred or _layout_preference(layout_style, index),
        )
        occupied.add(current)
        positions.append(current)
    return positions


def _layout_preference(layout_style: str, index: int) -> tuple[Position, ...] | None:
    if layout_style == "forked" and index % 2 == 0:
        return ((1, 0), (-1, 0), (0, 1))
    if layout_style == "dead_end_heavy":
        return ((0, 1), (1, 0), (-1, 0))
    return None


def _branch_position(
    anchor: Position,
    occupied: set[Position],
    rng: GameRng,
    *,
    forbidden_directions: tuple[Position, ...] = (),
    reserved_positions: set[Position] | None = None,
) -> Position:
    position = _open_cardinal_adjacent_position(
        anchor,
        occupied,
        rng,
        forbidden_directions=forbidden_directions,
        reserved_positions=reserved_positions,
    )
    occupied.add(position)
    return position


def _try_branch_position(
    anchor: Position,
    occupied: set[Position],
    rng: GameRng,
    *,
    forbidden_directions: tuple[Position, ...] = (),
    reserved_positions: set[Position] | None = None,
) -> Position | None:
    try:
        return _branch_position(
            anchor,
            occupied,
            rng,
            forbidden_directions=forbidden_directions,
            reserved_positions=reserved_positions,
        )
    except ValueError:
        return None


def _open_cardinal_adjacent_position(
    anchor: Position,
    occupied: set[Position],
    rng: GameRng,
    *,
    preferred: tuple[Position, ...] | None = None,
    forbidden_directions: tuple[Position, ...] = (),
    reserved_positions: set[Position] | None = None,
) -> Position:
    return _open_adjacent_position(
        anchor,
        occupied,
        rng,
        preferred=preferred,
        forbidden_directions=forbidden_directions,
        reserved_positions=reserved_positions,
        allow_distant=False,
    )


def _open_spine_extension_position(
    anchor: Position,
    occupied: set[Position],
    rng: GameRng,
    *,
    preferred: tuple[Position, ...] | None = None,
) -> Position:
    """Place a forward spine room on a one-step cardinal cell."""
    candidates = _open_adjacent_position_candidates(
        anchor,
        occupied,
        rng,
        preferred=preferred,
    )
    if not candidates:
        raise ValueError("Generated Maze route ran out of adjacent map positions.")
    for candidate in candidates:
        next_occupied = {*occupied, candidate}
        if _has_open_cardinal_neighbor(candidate, next_occupied):
            return candidate
    return candidates[0]


def _planned_spine_extension_position(
    *,
    route_seed: int,
    run_id: str,
    nodes_by_id: Mapping[str, ExpeditionNodeDefinition],
    frontier_id: str,
    next_depth: int,
    recipe: MazeRecipe,
    occupied_positions: set[Position],
) -> Position | None:
    frontier = nodes_by_id.get(frontier_id)
    if frontier is None:
        return None
    anchor = frontier.position if frontier.position is not None else (0, 0)
    position_offset, _content_offset = _extension_rng_offsets(next_depth, "forward")
    position_rng = GameRng(derive_maze_extension_seed(route_seed, run_id, position_offset))
    spine_preferences = _spine_extension_preferences(
        nodes_by_id,
        run_id,
        frontier_id,
        next_depth,
        recipe,
    )
    try:
        return _open_spine_extension_position(
            anchor,
            set(occupied_positions),
            position_rng,
            preferred=spine_preferences,
        )
    except ValueError:
        return None


def _planned_spur_position(
    *,
    route_seed: int,
    run_id: str,
    nodes_by_id: Mapping[str, ExpeditionNodeDefinition],
    frontier_id: str,
    next_depth: int,
    recipe: MazeRecipe,
    occupied_positions: set[Position],
    reserved_positions: set[Position],
) -> Position | None:
    frontier = nodes_by_id.get(frontier_id)
    if frontier is None:
        return None
    anchor = frontier.position if frontier.position is not None else (0, 0)
    position_offset, _content_offset = _extension_rng_offsets(next_depth, "spur")
    position_rng = GameRng(derive_maze_extension_seed(route_seed, run_id, position_offset))
    return _try_branch_position(
        anchor,
        set(occupied_positions),
        position_rng,
        reserved_positions=reserved_positions,
    )


def _spine_step(from_position: Position, to_position: Position) -> Position | None:
    dx = to_position[0] - from_position[0]
    dy = to_position[1] - from_position[1]
    if abs(dx) + abs(dy) != 1:
        return None
    return (dx, dy)


def _terminal_spine_forbidden_direction(
    main_positions: Sequence[Position],
    *,
    entry_position: Position,
) -> tuple[Position, ...]:
    if not main_positions:
        return ()
    if len(main_positions) == 1:
        step = _spine_step(entry_position, main_positions[0])
    else:
        step = _spine_step(main_positions[-2], main_positions[-1])
    return (step,) if step is not None else ()


def _forward_spine_forbidden_directions(
    nodes_by_id: Mapping[str, ExpeditionNodeDefinition],
    run_id: str,
    frontier_id: str,
    next_depth: int,
    recipe: MazeRecipe,
) -> tuple[Position, ...]:
    """Directions spurs must not use so the forward spine can continue."""
    preferences = _spine_extension_preferences(
        nodes_by_id,
        run_id,
        frontier_id,
        next_depth,
        recipe,
    )
    if not preferences:
        return ()
    return (preferences[0],)


def _forward_spine_cell(
    nodes_by_id: Mapping[str, ExpeditionNodeDefinition],
    run_id: str,
    frontier_id: str,
    next_depth: int,
    recipe: MazeRecipe,
) -> Position | None:
    frontier = nodes_by_id.get(frontier_id)
    if frontier is None or frontier.position is None:
        return None
    forbidden = _forward_spine_forbidden_directions(
        nodes_by_id,
        run_id,
        frontier_id,
        next_depth,
        recipe,
    )
    if not forbidden:
        return None
    dx, dy = forbidden[0]
    return (frontier.position[0] + dx, frontier.position[1] + dy)


def _spine_extension_preferences(
    nodes_by_id: Mapping[str, ExpeditionNodeDefinition],
    run_id: str,
    frontier_id: str,
    next_depth: int,
    recipe: MazeRecipe,
) -> tuple[Position, ...]:
    frontier = nodes_by_id[frontier_id]
    if frontier.position is None:
        return _layout_preference(recipe.layout_style, next_depth - 1) or ()
    frontier_depth = next_depth - 1
    previous_id = (
        f"{run_id}_entry"
        if frontier_depth <= 1
        else f"{run_id}_room_{frontier_depth - 1}"
    )
    previous = nodes_by_id.get(previous_id)
    preferences: list[Position] = []
    if previous is not None and previous.position is not None:
        step = _spine_step(previous.position, frontier.position)
        if step is not None:
            preferences.append(step)
    layout = _layout_preference(recipe.layout_style, next_depth - 1)
    if layout is not None:
        for direction in layout:
            if direction not in preferences:
                preferences.append(direction)
    return tuple(preferences)


def _open_adjacent_position(
    anchor: Position,
    occupied: set[Position],
    rng: GameRng,
    *,
    preferred: tuple[Position, ...] | None = None,
    forbidden_directions: tuple[Position, ...] = (),
    reserved_positions: set[Position] | None = None,
    allow_distant: bool = True,
) -> Position:
    candidates = _open_adjacent_position_candidates(
        anchor,
        occupied,
        rng,
        preferred=preferred,
        forbidden_directions=forbidden_directions,
        reserved_positions=reserved_positions,
    )
    if candidates:
        return candidates[0]
    if not allow_distant:
        raise ValueError("Generated Maze route ran out of adjacent map positions.")
    reserved = reserved_positions or set()
    for radius in range(2, MAX_MAZE_PLACEMENT_RADIUS + 1):
        distant_candidates: list[Position] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) != radius:
                    continue
                candidate = (anchor[0] + dx, anchor[1] + dy)
                if candidate not in occupied and candidate not in reserved:
                    distant_candidates.append(candidate)
        if distant_candidates:
            return rng.choice(tuple(distant_candidates))
    raise ValueError(
        f"Generated Maze route ran out of map positions at {anchor} "
        f"with {len(occupied)} occupied cells."
    )


def _open_adjacent_position_candidates(
    anchor: Position,
    occupied: set[Position],
    rng: GameRng,
    *,
    preferred: tuple[Position, ...] | None = None,
    forbidden_directions: tuple[Position, ...] = (),
    reserved_positions: set[Position] | None = None,
) -> tuple[Position, ...]:
    forbidden = set(forbidden_directions)
    reserved = reserved_positions or set()
    candidates: list[Position] = []
    seen_directions: set[Position] = set()

    def add_candidate(dx: int, dy: int) -> None:
        direction = (dx, dy)
        if direction in seen_directions:
            return
        seen_directions.add(direction)
        if direction in forbidden:
            return
        candidate = (anchor[0] + dx, anchor[1] + dy)
        if candidate in occupied or candidate in reserved:
            return
        candidates.append(candidate)

    if preferred:
        for dx, dy in _shuffled_directions_from(preferred, rng):
            add_candidate(dx, dy)
    for dx, dy in _shuffled_directions(rng):
        add_candidate(dx, dy)
    return tuple(candidates)


def _has_open_cardinal_neighbor(position: Position, occupied: set[Position]) -> bool:
    return any(
        (position[0] + dx, position[1] + dy) not in occupied
        for dx, dy in DIRECTION_STEPS
    )


def _shuffled_directions(rng: GameRng) -> tuple[Position, ...]:
    return _shuffled_directions_from(DIRECTION_STEPS, rng)


def _shuffled_directions_from(
    directions: tuple[Position, ...],
    rng: GameRng,
) -> tuple[Position, ...]:
    remaining = list(directions)
    shuffled: list[Position] = []
    while remaining:
        direction = rng.choice(tuple(remaining))
        remaining.remove(direction)
        shuffled.append(direction)
    return tuple(shuffled)
