"""Pydantic schemas for authored YAML content."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game.combat.formation import FormationSlot
from game.combat.targeting import AttackType, SkillUsableFrom
from game.expedition.node import ExpeditionNodeType


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HeroClassDefinition(StrictModel):
    id: str
    name: str
    max_hp: int = Field(gt=0)
    speed: int
    accuracy: int
    defense: int
    damage: int
    max_effort: int = Field(ge=0)
    skills: list[str]
    personal_quirk: str | None = None


class HeroesFile(StrictModel):
    classes: dict[str, HeroClassDefinition]


class SkillDefinition(StrictModel):
    id: str
    name: str
    category: str
    effort_cost: int = Field(ge=0)
    attack_type: AttackType
    usable_from: SkillUsableFrom = SkillUsableFrom.ANY_POSITION
    accuracy: int
    damage: int
    damage_min: int | None = Field(default=None, ge=0)
    damage_max: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    effect_text: str = ""
    reaction_window: bool = False
    intent_label: str | None = None
    threat_level: str = "normal"
    obvious_effect: str = ""

    @model_validator(mode="after")
    def _validate_damage_range(self) -> SkillDefinition:
        if (self.damage_min is None) != (self.damage_max is None):
            raise ValueError("damage_min and damage_max must be authored together.")
        if (
            self.damage_min is not None
            and self.damage_max is not None
            and self.damage_min > self.damage_max
        ):
            raise ValueError("damage_min must be less than or equal to damage_max.")
        return self


class SkillsFile(StrictModel):
    skills: dict[str, SkillDefinition]


class TraitType(StrEnum):
    PERSONAL = "personal"
    EARNED = "earned"
    CONDITION = "condition"
    STRAIN_MARK = "strain_mark"


class TraitDefinition(StrictModel):
    id: str
    name: str
    type: TraitType
    description: str = ""
    positive_text: str = ""
    negative_text: str = ""
    tags: list[str] = Field(default_factory=list)


class TraitsFile(StrictModel):
    traits: dict[str, TraitDefinition]


class EnemyDefinition(StrictModel):
    id: str
    name: str
    max_hp: int = Field(gt=0)
    speed: int
    accuracy: int
    defense: int
    damage: int
    max_effort: int = Field(ge=0)
    skills: list[str]
    formation_slot: FormationSlot
    tags: list[str] = Field(default_factory=list)


class EnemiesFile(StrictModel):
    enemies: dict[str, EnemyDefinition]


class ArtFrameDefinition(StrictModel):
    lines: list[str] = Field(default_factory=list)
    hold: int = Field(default=2, ge=1)


class ArtFrameGroupMetadata(StrictModel):
    impact_frame: int | None = Field(default=None, ge=0)


class ArtMiniDefinition(StrictModel):
    lines: list[str] = Field(default_factory=list)
    frames: dict[str, list[ArtFrameDefinition]] = Field(default_factory=dict)


class ArtAssetDefinition(StrictModel):
    display_name: str = ""
    glyph: str = ""
    mini: ArtMiniDefinition | None = None
    lines: list[str] = Field(default_factory=list)
    frames: dict[str, list[ArtFrameDefinition]] = Field(default_factory=dict)
    frame_metadata: dict[str, ArtFrameGroupMetadata] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_frame_groups(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw_frames = data.get("frames")
        if not isinstance(raw_frames, dict):
            return data

        frames: dict[str, object] = {}
        metadata: dict[str, dict[str, int | None]] = {}
        for name, group in raw_frames.items():
            if isinstance(group, dict):
                frames[name] = group.get("frames", [])
                metadata[name] = {"impact_frame": group.get("impact_frame")}
            else:
                frames[name] = group
        normalized = dict(data)
        normalized["frames"] = frames
        if metadata:
            normalized["frame_metadata"] = {
                **dict(normalized.get("frame_metadata") or {}),
                **metadata,
            }
        return normalized


class ArtFile(StrictModel):
    hero_classes: dict[str, ArtAssetDefinition] = Field(default_factory=dict)
    heroes: dict[str, ArtAssetDefinition] = Field(default_factory=dict)
    enemies: dict[str, ArtAssetDefinition] = Field(default_factory=dict)
    dungeon_nodes: dict[str, ArtAssetDefinition] = Field(default_factory=dict)


class RecruitDefinition(StrictModel):
    id: str
    name: str
    class_id: str
    formation_slot: FormationSlot
    background: str = ""
    motive: str = ""


class RecruitTableEntry(StrictModel):
    name: str
    class_id: str
    background: str = ""
    motive: str = ""


class RecruitsFile(StrictModel):
    starting_roster: list[RecruitDefinition]
    recruitment_pool: list[RecruitTableEntry] = Field(default_factory=list)


class ExpeditionRoomActionDefinition(StrictModel):
    id: str
    label: str
    description: str = ""
    result_text: str
    once: bool = True
    requires_cleared: bool = False
    inventory_requirements: dict[str, int] = Field(default_factory=dict)
    supply_costs: dict[str, int] = Field(default_factory=dict)
    supply_rewards: dict[str, int] = Field(default_factory=dict)
    loot: dict[str, int] = Field(default_factory=dict)
    reputation_reward: int = 0
    coin_reward: int = Field(default=0, ge=0)
    reveal_exits: list[str] = Field(default_factory=list)
    requires_active_contracts: list[str] = Field(default_factory=list)
    complete_contract: str | None = None
    flags_set: dict[str, bool] = Field(default_factory=dict)
    history: str | None = None


class ExpeditionNodeDefinition(StrictModel):
    id: str
    name: str
    node_type: ExpeditionNodeType
    text: str
    scene_state: str = ""
    revisit_text: str = ""
    route_hint: str = ""
    party_hint: str = ""
    major_beat: bool = False
    map_id: str = "default"
    position: tuple[int, int] | None = None
    encounter: str | None = None
    next_node: str | None = None
    exits: list[str] = Field(default_factory=list)
    safe_return: bool = False
    reputation_reward: int = 0
    coin_reward: int = Field(default=0, ge=0)
    breach_id: str | None = None
    known_route_unlock: str | None = None
    loot: dict[str, int] = Field(default_factory=dict)
    supply_rewards: dict[str, int] = Field(default_factory=dict)
    history: str | None = None
    horror_morale_loss: int = 0
    actions: list[ExpeditionRoomActionDefinition] = Field(default_factory=list)
    lore_entries: list[str] = Field(default_factory=list)
    flags_set: dict[str, bool] = Field(default_factory=dict)
    complete_contract: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_horror_order_loss(cls, data: object) -> object:
        if not isinstance(data, dict) or "horror_order_loss" not in data:
            return data
        normalized = dict(data)
        normalized.setdefault("horror_morale_loss", normalized.pop("horror_order_loss"))
        return normalized


class ExpeditionDefinition(StrictModel):
    id: str
    name: str
    nodes: list[ExpeditionNodeDefinition]


class EncounterEnemyDefinition(StrictModel):
    enemy_id: str
    actor_id: str
    formation_slot: FormationSlot


class EncounterDefinition(StrictModel):
    id: str
    enemies: list[EncounterEnemyDefinition]


class ExpeditionsFile(StrictModel):
    expeditions: dict[str, ExpeditionDefinition]
    encounters: dict[str, EncounterDefinition] = Field(default_factory=dict)


class SupplyDefinition(StrictModel):
    id: str
    name: str
    cost: int = Field(ge=0)
    description: str = ""


class SuppliesFile(StrictModel):
    starting: dict[str, int]
    catalog: dict[str, SupplyDefinition]


class GearEffectDefinition(StrictModel):
    max_hp_bonus: int = 0
    max_effort_bonus: int = 0
    accuracy_bonus: int = 0
    damage_bonus: int = 0


class GearDefinition(StrictModel):
    id: str
    name: str
    description: str = ""
    slot: str = "kit"
    cost: int | None = Field(default=None, ge=0)
    requires_completed_contracts: list[str] = Field(default_factory=list)
    requires_known_breaches: list[str] = Field(default_factory=list)
    effects: GearEffectDefinition = Field(default_factory=GearEffectDefinition)
    tags: list[str] = Field(default_factory=list)


class GearFile(StrictModel):
    gear: dict[str, GearDefinition] = Field(default_factory=dict)


class LootDefinition(StrictModel):
    id: str
    name: str
    description: str = ""
    sell_price: int | None = Field(default=None, ge=0)
    turn_in_flag: str = ""
    turn_in_unlocks_contract: str = ""


class LootFile(StrictModel):
    loot: dict[str, LootDefinition] = Field(default_factory=dict)


class TownServiceDefinition(StrictModel):
    id: str
    name: str
    description: str = ""


class TownUpgradeEffectDefinition(StrictModel):
    roster_cap_bonus: int = Field(default=0, ge=0)
    recovery_cost_delta: int = 0
    supply_cost_deltas: dict[str, int] = Field(default_factory=dict)


class TownUpgradeDefinition(StrictModel):
    id: str
    name: str
    description: str = ""
    cost: int = Field(ge=0)
    requires_completed_contracts: list[str] = Field(default_factory=list)
    requires_known_breaches: list[str] = Field(default_factory=list)
    effects: TownUpgradeEffectDefinition = Field(
        default_factory=TownUpgradeEffectDefinition
    )


class TownFile(StrictModel):
    roster_cap: int = Field(gt=0)
    recruit_offer_count: int = Field(gt=0)
    recruit_cost: int = Field(ge=0)
    recovery_cost: int = Field(ge=0)
    surgery_cost: int = Field(ge=0)
    services: dict[str, TownServiceDefinition]
    upgrades: dict[str, TownUpgradeDefinition] = Field(default_factory=dict)


class DifficultyProfileDefinition(StrictModel):
    id: str
    name: str
    summary: str
    principles: list[str] = Field(default_factory=list)


class LocationDefinition(StrictModel):
    id: str
    name: str
    kind: str
    act: int = Field(ge=1)
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class ContractDefinition(StrictModel):
    id: str
    name: str
    act: int = Field(ge=1)
    location_id: str
    expedition_id: str | None = None
    difficulty: int = Field(ge=1, le=5)
    reward_reputation: int = Field(ge=0)
    coin_reward: int = Field(default=0, ge=0)
    reward_gear: dict[str, int] = Field(default_factory=dict)
    summary: str = ""
    available_at_start: bool = False
    board_order: int | None = Field(default=None, ge=0)
    posted_after_completed_contracts: list[str] = Field(default_factory=list)
    posted_after_known_breaches: list[str] = Field(default_factory=list)
    posted_after_flags: list[str] = Field(default_factory=list)
    requires_completed_contracts: list[str] = Field(default_factory=list)
    requires_known_breaches: list[str] = Field(default_factory=list)
    locked_reason: str = ""
    tags: list[str] = Field(default_factory=list)
    generated_maze_pressure_id: str = ""
    generated_maze_required_rooms: int = Field(default=0, ge=0)
    generated_maze_required_action_count: int = Field(default=0, ge=0)
    generated_maze_required_loot: dict[str, int] = Field(default_factory=dict)
    generated_maze_required_combat_clears: int = Field(default=0, ge=0)
    generated_maze_requires_hunt: bool = False


class RumorDefinition(StrictModel):
    id: str
    title: str
    text: str
    source: str = ""
    act: int = Field(ge=1)
    tags: list[str] = Field(default_factory=list)


class WorldFile(StrictModel):
    maze_name: str = "Pandora's Maze"
    starting_settlement: str
    difficulty: DifficultyProfileDefinition
    locations: dict[str, LocationDefinition]
    contracts: dict[str, ContractDefinition] = Field(default_factory=dict)
    rumors: dict[str, RumorDefinition] = Field(default_factory=dict)
