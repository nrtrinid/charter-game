"""YAML loading and Pydantic validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from game.content.definitions import GameDefinitions
from game.data.schemas import (
    ArtFile,
    EnemiesFile,
    ExpeditionsFile,
    GearFile,
    HeroesFile,
    LootFile,
    RecruitsFile,
    SkillsFile,
    SuppliesFile,
    TownFile,
    TraitsFile,
    TraitType,
    WorldFile,
)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def load_game_definitions(data_dir: Path | None = None) -> GameDefinitions:
    root = data_dir or default_data_dir()
    definitions = GameDefinitions(
        heroes_file=HeroesFile.model_validate(load_yaml(root / "heroes.yaml")),
        enemies_file=EnemiesFile.model_validate(load_yaml(root / "enemies.yaml")),
        skills_file=SkillsFile.model_validate(load_yaml(root / "skills.yaml")),
        traits_file=TraitsFile.model_validate(load_yaml(root / "traits.yaml")),
        recruits_file=RecruitsFile.model_validate(load_yaml(root / "recruits.yaml")),
        expeditions_file=ExpeditionsFile.model_validate(load_yaml(root / "expeditions.yaml")),
        gear_file=GearFile.model_validate(load_yaml(root / "gear.yaml")),
        loot_file=LootFile.model_validate(load_yaml(root / "loot.yaml")),
        supplies_file=SuppliesFile.model_validate(load_yaml(root / "supplies.yaml")),
        town_file=TownFile.model_validate(load_yaml(root / "town.yaml")),
        world_file=_load_world_file(root),
        art_file=_load_art_file(root),
    )
    validate_references(definitions)
    return definitions


def validate_references(definitions: GameDefinitions) -> None:
    _assert_keys_match_ids("hero class", definitions.hero_classes)
    _assert_keys_match_ids("skill", definitions.skills)
    _assert_keys_match_ids("trait", definitions.traits)
    _assert_keys_match_ids("enemy", definitions.enemies)
    _assert_keys_match_ids("expedition", definitions.expeditions)
    _assert_keys_match_ids("encounter", definitions.encounters)
    _assert_keys_match_ids("gear", definitions.gear)
    _assert_keys_match_ids("loot", definitions.loot)
    _assert_keys_match_ids("supply", definitions.supplies.catalog)
    _assert_keys_match_ids("town service", definitions.town.services)
    _assert_keys_match_ids("town upgrade", definitions.town.upgrades)
    _assert_keys_match_ids("location", definitions.locations)
    _assert_keys_match_ids("contract", definitions.contracts)
    _assert_keys_match_ids("rumor", definitions.rumors)
    _validate_art_references(definitions)

    _require(
        definitions.world.starting_settlement in definitions.locations,
        f"Unknown starting settlement: {definitions.world.starting_settlement}",
    )

    for hero_class in definitions.hero_classes.values():
        for skill_id in hero_class.skills:
            _require(skill_id in definitions.skills, f"Unknown hero skill: {skill_id}")
        if hero_class.personal_quirk is not None:
            trait = definitions.traits.get(hero_class.personal_quirk)
            if trait is None:
                raise ValueError(f"Unknown personal quirk: {hero_class.personal_quirk}")
            _require(
                trait.type == TraitType.PERSONAL,
                f"Hero class personal quirk must be personal: {hero_class.personal_quirk}",
            )

    for enemy in definitions.enemies.values():
        for skill_id in enemy.skills:
            _require(skill_id in definitions.skills, f"Unknown enemy skill: {skill_id}")

    for starting_recruit in definitions.recruits.starting_roster:
        _require(
            starting_recruit.class_id in definitions.hero_classes,
            f"Unknown recruit class: {starting_recruit.class_id}",
        )

    for pool_entry in definitions.recruits.recruitment_pool:
        _require(
            pool_entry.class_id in definitions.hero_classes,
            f"Unknown recruit class: {pool_entry.class_id}",
        )

    for supply_id in definitions.supplies.starting:
        _require(supply_id in definitions.supplies.catalog, f"Unknown starting supply: {supply_id}")

    for upgrade in definitions.town.upgrades.values():
        for required_contract_id in upgrade.requires_completed_contracts:
            _require(
                required_contract_id in definitions.contracts,
                f"Unknown upgrade required contract: {required_contract_id}",
            )
        for breach_id in upgrade.requires_known_breaches:
            _require(
                breach_id in definitions.locations,
                f"Unknown upgrade required breach: {breach_id}",
            )
        for supply_id in upgrade.effects.supply_cost_deltas:
            _require(
                supply_id in definitions.supplies.catalog,
                f"Unknown upgrade supply cost effect: {supply_id}",
            )

    for gear in definitions.gear.values():
        _require(
            gear.slot == "kit",
            f"Unsupported gear slot for {gear.id}: {gear.slot}",
        )
        for required_contract_id in gear.requires_completed_contracts:
            _require(
                required_contract_id in definitions.contracts,
                f"Unknown gear required contract: {required_contract_id}",
            )
        for breach_id in gear.requires_known_breaches:
            _require(
                breach_id in definitions.locations,
                f"Unknown gear required breach: {breach_id}",
            )

    for contract in definitions.contracts.values():
        _require(
            contract.location_id in definitions.locations,
            f"Unknown contract location: {contract.location_id}",
        )
        if contract.expedition_id is not None:
            _require(
                contract.expedition_id in definitions.expeditions,
                f"Unknown contract expedition: {contract.expedition_id}",
            )
        for required_contract_id in contract.posted_after_completed_contracts:
            _require(
                required_contract_id in definitions.contracts,
                f"Unknown contract posting prerequisite: {required_contract_id}",
            )
        for breach_id in contract.posted_after_known_breaches:
            _require(
                breach_id in definitions.locations,
                f"Unknown contract posting breach: {breach_id}",
            )
        for required_contract_id in contract.requires_completed_contracts:
            _require(
                required_contract_id in definitions.contracts,
                f"Unknown required contract: {required_contract_id}",
            )
        for breach_id in contract.requires_known_breaches:
            _require(
                breach_id in definitions.locations,
                f"Unknown required breach: {breach_id}",
            )
        for gear_id, quantity in contract.reward_gear.items():
            _require(gear_id in definitions.gear, f"Unknown contract gear reward: {gear_id}")
            _require(quantity > 0, f"Contract gear reward must be positive: {contract.id}")
        for flag_id in contract.posted_after_flags:
            _require(bool(flag_id), f"Contract posting flag must not be empty: {contract.id}")

    for loot in definitions.loot.values():
        _require(
            bool(loot.sell_price is not None or loot.turn_in_flag or loot.description),
            f"Loot item must be catalogued with a purpose: {loot.id}",
        )
        if loot.turn_in_unlocks_contract:
            _require(
                loot.turn_in_unlocks_contract in definitions.contracts,
                f"Unknown turn-in contract unlock: {loot.turn_in_unlocks_contract}",
            )

    for encounter in definitions.encounters.values():
        for encounter_enemy in encounter.enemies:
            _require(
                encounter_enemy.enemy_id in definitions.enemies,
                f"Unknown encounter enemy: {encounter_enemy.enemy_id}",
            )

    for expedition in definitions.expeditions.values():
        node_ids = {node.id for node in expedition.nodes}
        _validate_spatial_nodes(expedition.nodes)
        for node in expedition.nodes:
            if node.next_node is not None:
                _require(node.next_node in node_ids, f"Unknown next node: {node.next_node}")
            for exit_node in node.exits:
                _require(exit_node in node_ids, f"Unknown exit node: {exit_node}")
            action_ids = {action.id for action in node.actions}
            _require(
                len(action_ids) == len(node.actions),
                f"Duplicate action id in node: {node.id}",
            )
            for action in node.actions:
                for item_id in action.inventory_requirements:
                    _require(
                        bool(item_id),
                        "Action item requirements must not be empty.",
                    )
                for supply_id in action.supply_costs:
                    _require(
                        supply_id in definitions.supplies.catalog,
                        f"Unknown action supply cost: {supply_id}",
                    )
                for supply_id in action.supply_rewards:
                    _require(
                        supply_id in definitions.supplies.catalog,
                        f"Unknown action supply reward: {supply_id}",
                    )
                for exit_node in action.reveal_exits:
                    _require(exit_node in node_ids, f"Unknown revealed exit node: {exit_node}")
                for contract_id in action.requires_active_contracts:
                    _require(
                        contract_id in definitions.contracts,
                        f"Unknown required active contract: {contract_id}",
                    )
                if action.complete_contract is not None:
                    _require(
                        action.complete_contract in definitions.contracts,
                        f"Unknown completed contract: {action.complete_contract}",
                    )
                for item_id in action.loot:
                    _require(
                        item_id in definitions.loot,
                        f"Unknown action loot item: {item_id}",
                    )
            if node.encounter is not None:
                _require(
                    node.encounter in definitions.encounters,
                    f"Unknown encounter: {node.encounter}",
                )
            if node.complete_contract is not None:
                _require(
                    node.complete_contract in definitions.contracts,
                    f"Unknown completed contract: {node.complete_contract}",
                )
            for lore_id in node.lore_entries:
                _require(lore_id in definitions.rumors, f"Unknown lore entry: {lore_id}")
            for supply_id in node.supply_rewards:
                _require(
                    supply_id in definitions.supplies.catalog,
                    f"Unknown supply reward: {supply_id}",
                )


def _validate_spatial_nodes(nodes: Sequence[Any]) -> None:
    positions: dict[tuple[str, int, int], str] = {}
    nodes_by_id = {node.id: node for node in nodes}
    for node in nodes:
        if node.position is None:
            continue
        position_key = (node.map_id, node.position[0], node.position[1])
        previous = positions.get(position_key)
        _require(
            previous is None,
            f"Duplicate map position {node.map_id} {node.position}: "
            f"{previous} and {node.id}",
        )
        positions[position_key] = node.id

    for node in nodes:
        if node.position is None:
            continue
        linked_node_ids = [*node.exits]
        for action in node.actions:
            linked_node_ids.extend(action.reveal_exits)
        for linked_node_id in linked_node_ids:
            linked_node = nodes_by_id.get(linked_node_id)
            if linked_node is None or linked_node.position is None:
                continue
            if linked_node.map_id != node.map_id:
                continue
            dx = abs(linked_node.position[0] - node.position[0])
            dy = abs(linked_node.position[1] - node.position[1])
            _require(
                dx == 0 or dy == 0,
                f"Map link must be cardinal: {node.id} -> {linked_node_id}",
            )


def _assert_keys_match_ids(label: str, values: Mapping[str, object]) -> None:
    for key, value in values.items():
        actual_id = getattr(value, "id", None)
        _require(key == actual_id, f"{label} key {key!r} does not match id {actual_id!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_art_references(definitions: GameDefinitions) -> None:
    for class_id in definitions.art.hero_classes:
        _require(class_id in definitions.hero_classes, f"Unknown hero class art: {class_id}")
    starting_hero_ids = {hero.id for hero in definitions.recruits.starting_roster}
    for hero_id in definitions.art.heroes:
        _require(hero_id in starting_hero_ids, f"Unknown hero art: {hero_id}")
    for enemy_id in definitions.art.enemies:
        _require(enemy_id in definitions.enemies, f"Unknown enemy art: {enemy_id}")
    node_ids = {
        node.id
        for expedition in definitions.expeditions.values()
        for node in expedition.nodes
    }
    for node_id in definitions.art.dungeon_nodes:
        _require(
            node_id in node_ids or node_id.startswith("generated_maze_"),
            f"Unknown dungeon node art: {node_id}",
        )
    for group_name, assets in (
        ("hero class", definitions.art.hero_classes),
        ("hero", definitions.art.heroes),
        ("enemy", definitions.art.enemies),
        ("dungeon node", definitions.art.dungeon_nodes),
    ):
        for asset_id, asset in assets.items():
            _validate_art_asset(asset, f"{group_name} art {asset_id}")


def _validate_art_asset(asset: Any, label: str) -> None:
    base_line_count = len(getattr(asset, "lines", ()) or ())
    for frame_name, frames in (getattr(asset, "frames", {}) or {}).items():
        for index, frame in enumerate(frames):
            line_count = len(getattr(frame, "lines", ()) or ())
            if base_line_count:
                _require(
                    line_count == base_line_count,
                    f"{label} frame {frame_name}[{index}] must have "
                    f"{base_line_count} lines.",
                )
            _require(line_count > 0, f"{label} frame {frame_name}[{index}] is empty.")
        metadata = (getattr(asset, "frame_metadata", {}) or {}).get(frame_name)
        impact_frame = getattr(metadata, "impact_frame", None)
        if impact_frame is not None:
            _require(
                int(impact_frame) < len(frames),
                f"{label} frame {frame_name} impact_frame is outside its frame list.",
            )

    mini = getattr(asset, "mini", None)
    if mini is None:
        return
    mini_lines = tuple(getattr(mini, "lines", ()) or ())
    if mini_lines:
        _require(len(mini_lines) == 3, f"{label} mini.lines must have exactly 3 lines.")
        _require(
            max(len(line) for line in mini_lines) <= 12,
            f"{label} mini.lines should be 12 columns or narrower.",
        )
    for frame_name, frames in (getattr(mini, "frames", {}) or {}).items():
        for index, frame in enumerate(frames):
            frame_lines = tuple(getattr(frame, "lines", ()) or ())
            _require(
                len(frame_lines) == 3,
                f"{label} mini frame {frame_name}[{index}] must have exactly 3 lines.",
            )
            _require(
                max(len(line) for line in frame_lines) <= 12,
                f"{label} mini frame {frame_name}[{index}] should be 12 columns or narrower.",
            )


def _load_art_file(root: Path) -> ArtFile:
    path = root / "art.yaml"
    if path.exists():
        return ArtFile.model_validate(load_yaml(path))
    return ArtFile()


def _load_world_file(root: Path) -> WorldFile:
    path = root / "world.yaml"
    if path.exists():
        return WorldFile.model_validate(load_yaml(path))
    return WorldFile.model_validate(
        {
            "starting_settlement": "haven_town",
            "difficulty": {
                "id": "standard",
                "name": "Standard",
                "summary": "Default campaign assumptions.",
            },
            "locations": {
                "haven_town": {
                    "id": "haven_town",
                    "name": "Haven Town",
                    "kind": "settlement",
                    "act": 1,
                    "description": "The charter company's starting settlement.",
                }
            },
        }
    )
