"""App-facing view models for terminal rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from game.combat.combat_state import Combatant, Team
from game.content.definitions import GameDefinitions
from game.data.schemas import ExpeditionNodeDefinition
from game.expedition.node import ExpeditionNodeType


def _combatant_art_lines(
    definitions: GameDefinitions,
    combatant: Combatant,
) -> tuple[str, ...]:
    return _art_lines(_combatant_art_asset(definitions, combatant))

def _combatant_art_frames(
    definitions: GameDefinitions,
    combatant: Combatant,
) -> Mapping[str, tuple[tuple[str, ...], ...]]:
    return _art_frames(_combatant_art_asset(definitions, combatant))

def _combatant_art_frame_impacts(
    definitions: GameDefinitions,
    combatant: Combatant,
) -> Mapping[str, int]:
    return _art_frame_impacts(_combatant_art_asset(definitions, combatant))

def _art_frame_impacts(asset: Any | None) -> Mapping[str, int]:
    if asset is None:
        return {}
    return {
        name: metadata.impact_frame
        for name, metadata in getattr(asset, "frame_metadata", {}).items()
        if metadata.impact_frame is not None
    }

def _art_frame_holds(asset: Any | None) -> Mapping[str, tuple[int, ...]]:
    if asset is None:
        return {}
    return {
        name: tuple(int(getattr(frame, "hold", 2)) for frame in frames)
        for name, frames in getattr(asset, "frames", {}).items()
    }

def _combatant_art_asset(
    definitions: GameDefinitions,
    combatant: Combatant,
) -> Any | None:
    if combatant.team == Team.HERO:
        return _hero_art_asset(definitions, combatant.actor_id, combatant.class_id)
    return definitions.art.enemies.get(combatant.class_id)

def _hero_art_asset(
    definitions: GameDefinitions,
    hero_id: str,
    class_id: str,
) -> Any | None:
    hero_art = definitions.art.heroes.get(hero_id)
    if hero_art is not None:
        return hero_art
    return definitions.art.hero_classes.get(class_id)

def _dungeon_node_art_asset(
    definitions: GameDefinitions,
    session: Any,
    node: ExpeditionNodeDefinition,
) -> Any | None:
    exact = definitions.art.dungeon_nodes.get(node.id)
    if exact is not None:
        return exact
    if not node.id.startswith("maze_run_"):
        return None
    art_key = _generated_maze_art_key(session, node)
    return definitions.art.dungeon_nodes.get(art_key)

def _generated_maze_art_key(session: Any, node: ExpeditionNodeDefinition) -> str:
    if node.id.endswith("_entry"):
        return "generated_maze_entry"
    if node.id.endswith("_hunt_lair"):
        return "generated_maze_hunt"
    if node.id.endswith("_hard_room"):
        return "generated_maze_hard_room"
    if node.id.endswith("_reward"):
        return "generated_maze_reward"
    if node.id.endswith("_echo"):
        return "generated_maze_curio"
    if node.node_type == ExpeditionNodeType.COMBAT:
        return "generated_maze_combat"
    if node.node_type == ExpeditionNodeType.CURIO:
        return "generated_maze_curio"
    generated = getattr(session, "generated_dungeon", None)
    recipe = getattr(generated, "recipe", None)
    palette = getattr(recipe, "room_palette", "stone")
    return f"generated_maze_{palette}_room"

def _art_lines(asset: Any | None) -> tuple[str, ...]:
    if asset is None:
        return ()
    return tuple(asset.lines)

def _art_display_name(asset: Any | None) -> str:
    if asset is None:
        return ""
    return str(getattr(asset, "display_name", "") or "")

def _art_glyph(asset: Any | None) -> str:
    if asset is None:
        return ""
    return str(getattr(asset, "glyph", "") or "")

def _art_mini_lines(asset: Any | None) -> tuple[str, ...]:
    if asset is None:
        return ()
    mini = getattr(asset, "mini", None)
    return tuple(getattr(mini, "lines", ()) or ())

def _art_mini_frames(asset: Any | None) -> Mapping[str, tuple[tuple[str, ...], ...]]:
    if asset is None:
        return {}
    mini = getattr(asset, "mini", None)
    if mini is None:
        return {}
    return {
        name: tuple(tuple(frame.lines) for frame in frames) for name, frames in mini.frames.items()
    }

def _derive_mini_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    if not lines:
        return ()
    compact = [line.strip() for line in lines if line.strip()]
    if not compact:
        return ()
    if len(compact) >= 3:
        return (compact[0], compact[len(compact) // 2], compact[-1])
    return (*compact, *("" for _ in range(3 - len(compact))))

def _art_frames(asset: Any | None) -> Mapping[str, tuple[tuple[str, ...], ...]]:
    if asset is None:
        return {}
    return {
        name: tuple(tuple(frame.lines) for frame in frames) for name, frames in asset.frames.items()
    }
