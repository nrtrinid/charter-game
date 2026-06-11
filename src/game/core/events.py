"""Structured events returned by engine systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EventType(StrEnum):
    ENCOUNTER_STARTED = "encounter_started"
    ROUND_STARTED = "round_started"
    ROUND_ENDED = "round_ended"
    ENCOUNTER_ENDED = "encounter_ended"
    DAMAGE = "damage"
    HEALING = "healing"
    MOVE = "move"
    TURN_PASSED = "turn_passed"
    TURN_DELAYED = "turn_delayed"
    COMBAT_RETREAT_DECLARED = "combat_retreat_declared"
    COMBAT_RETREATED = "combat_retreated"
    ENEMY_INTENT = "enemy_intent"
    REACTION_USED = "reaction_used"
    REACTION_SKIPPED = "reaction_skipped"
    DOWNED = "downed"
    DEATH = "death"
    COMBAT_ENDED = "combat_ended"
    SKILL_USED = "skill_used"
    MISS = "miss"
    COMBAT_EFFECT = "combat_effect"
    STATUS_CHANGED = "status_changed"
    MEMORY_SIGNAL = "memory_signal"
    EXPEDITION = "expedition"
    DUNGEON_ACTION = "dungeon_action"
    LOOT_GAINED = "loot_gained"
    BREACH_DISCOVERED = "breach_discovered"
    LORE_DISCOVERED = "lore_discovered"
    CONTRACT_COMPLETED = "contract_completed"
    EXPEDITION_RETURNED = "expedition_returned"
    MAZE_ROUTE_COLLAPSED = "maze_route_collapsed"
    MAZE_FRONTIER_OPENED = "maze_frontier_opened"
    COMPANY = "company"
    TOWN_SERVICE = "town_service"
    RECRUITMENT_OFFERED = "recruitment_offered"
    HERO_RECRUITED = "hero_recruited"
    RECOVERY = "recovery"
    SUPPLIES_PURCHASED = "supplies_purchased"
    ACTIVE_PARTY_CHANGED = "active_party_changed"
    SAVE = "save"
    LOAD = "load"


@dataclass(frozen=True)
class EncounterStartedEvent:
    message: str
    encounter_id: str
    encounter_name: str
    actor_ids: list[str]
    event_type: EventType = field(init=False, default=EventType.ENCOUNTER_STARTED)


@dataclass(frozen=True)
class RoundStartedEvent:
    message: str
    encounter_id: str
    round_number: int
    actor_ids: list[str]
    event_type: EventType = field(init=False, default=EventType.ROUND_STARTED)


@dataclass(frozen=True)
class RoundEndedEvent:
    message: str
    encounter_id: str
    round_number: int
    event_type: EventType = field(init=False, default=EventType.ROUND_ENDED)


@dataclass(frozen=True)
class EncounterEndedEvent:
    message: str
    encounter_id: str
    victor: str
    event_type: EventType = field(init=False, default=EventType.ENCOUNTER_ENDED)


@dataclass(frozen=True)
class DamageEvent:
    message: str
    source_id: str
    target_id: str
    amount: int
    hp_before: int | None = None
    event_type: EventType = field(init=False, default=EventType.DAMAGE)


@dataclass(frozen=True)
class HealingEvent:
    message: str
    source_id: str
    target_id: str
    amount: int
    event_type: EventType = field(init=False, default=EventType.HEALING)


@dataclass(frozen=True)
class MoveEvent:
    message: str
    actor_id: str
    from_slot: str
    to_slot: str
    event_type: EventType = field(init=False, default=EventType.MOVE)


@dataclass(frozen=True)
class TurnPassedEvent:
    message: str
    actor_id: str
    encounter_id: str
    event_type: EventType = field(init=False, default=EventType.TURN_PASSED)


@dataclass(frozen=True)
class TurnDelayedEvent:
    message: str
    actor_id: str
    encounter_id: str
    event_type: EventType = field(init=False, default=EventType.TURN_DELAYED)


@dataclass(frozen=True)
class CombatRetreatDeclaredEvent:
    message: str
    actor_id: str
    encounter_id: str
    event_type: EventType = field(init=False, default=EventType.COMBAT_RETREAT_DECLARED)


@dataclass(frozen=True)
class CombatRetreatedEvent:
    message: str
    actor_id: str
    encounter_id: str
    from_node_id: str | None = None
    to_node_id: str | None = None
    event_type: EventType = field(init=False, default=EventType.COMBAT_RETREATED)


@dataclass(frozen=True)
class EnemyIntentEvent:
    message: str
    enemy_id: str
    enemy_name: str
    skill_id: str
    skill_name: str
    label: str
    target_id: str
    target_name: str
    threat_level: str
    obvious_effect: str
    hit_chance: int | None = None
    damage_estimate: int | None = None
    damage_label: str = ""
    event_type: EventType = field(init=False, default=EventType.ENEMY_INTENT)


@dataclass(frozen=True)
class ReactionUsedEvent:
    message: str
    reaction_id: str
    reaction_kind: str
    actor_id: str
    actor_name: str
    enemy_id: str
    skill_id: str
    target_id: str
    event_type: EventType = field(init=False, default=EventType.REACTION_USED)


@dataclass(frozen=True)
class ReactionSkippedEvent:
    message: str
    enemy_id: str
    skill_id: str
    target_id: str
    event_type: EventType = field(init=False, default=EventType.REACTION_SKIPPED)


@dataclass(frozen=True)
class DownedEvent:
    message: str
    actor_id: str
    event_type: EventType = field(init=False, default=EventType.DOWNED)


@dataclass(frozen=True)
class DeathEvent:
    message: str
    actor_id: str
    event_type: EventType = field(init=False, default=EventType.DEATH)


@dataclass(frozen=True)
class CombatEndedEvent:
    message: str
    victor: str
    event_type: EventType = field(init=False, default=EventType.COMBAT_ENDED)


@dataclass(frozen=True)
class SkillUsedEvent:
    message: str
    actor_id: str
    skill_id: str
    target_id: str | None
    event_type: EventType = field(init=False, default=EventType.SKILL_USED)


@dataclass(frozen=True)
class MissEvent:
    message: str
    actor_id: str
    target_id: str
    event_type: EventType = field(init=False, default=EventType.MISS)


@dataclass(frozen=True)
class CombatEffectEvent:
    message: str
    actor_id: str
    effect_type: str
    label: str
    delta: int = 0
    before: int | None = None
    after: int | None = None
    source_kind: str = ""
    source_id: str = ""
    target_id: str | None = None
    resource: str = ""
    tag: str = ""
    emphasis: str = "normal"
    event_type: EventType = field(init=False, default=EventType.COMBAT_EFFECT)


@dataclass(frozen=True)
class StatusChangedEvent:
    message: str
    actor_id: str
    status: str
    added: bool
    event_type: EventType = field(init=False, default=EventType.STATUS_CHANGED)


@dataclass(frozen=True)
class MemorySignalEvent:
    message: str
    hero_id: str
    family_id: str
    score: int = 1
    tags: tuple[str, ...] = ()
    source_summary: str = ""
    node_id: str | None = None
    encounter_id: str | None = None
    event_type: EventType = field(init=False, default=EventType.MEMORY_SIGNAL)


@dataclass(frozen=True)
class ExpeditionEvent:
    message: str
    node_id: str
    first_visit: bool = True
    major_beat: bool = False
    event_type: EventType = field(init=False, default=EventType.EXPEDITION)


@dataclass(frozen=True)
class DungeonActionEvent:
    message: str
    node_id: str
    action_id: str
    label: str
    supply_costs: dict[str, int] = field(default_factory=dict)
    supply_rewards: dict[str, int] = field(default_factory=dict)
    loot: dict[str, int] = field(default_factory=dict)
    reputation: int = 0
    coin: int = 0
    event_type: EventType = field(init=False, default=EventType.DUNGEON_ACTION)


@dataclass(frozen=True)
class LootGainedEvent:
    message: str
    node_id: str
    inventory: dict[str, int]
    supplies: dict[str, int]
    reputation: int = 0
    coin: int = 0
    gear: dict[str, int] = field(default_factory=dict)
    event_type: EventType = field(init=False, default=EventType.LOOT_GAINED)


@dataclass(frozen=True)
class BreachDiscoveredEvent:
    message: str
    node_id: str
    breach_id: str
    event_type: EventType = field(init=False, default=EventType.BREACH_DISCOVERED)


@dataclass(frozen=True)
class LoreDiscoveredEvent:
    message: str
    node_id: str
    lore_id: str
    title: str
    event_type: EventType = field(init=False, default=EventType.LORE_DISCOVERED)


@dataclass(frozen=True)
class ContractCompletedEvent:
    message: str
    node_id: str
    contract_id: str
    name: str
    event_type: EventType = field(init=False, default=EventType.CONTRACT_COMPLETED)


@dataclass(frozen=True)
class ExpeditionReturnedEvent:
    message: str
    expedition_id: str
    location: str
    event_type: EventType = field(init=False, default=EventType.EXPEDITION_RETURNED)


@dataclass(frozen=True)
class MazeRouteCollapsedEvent:
    message: str
    run_id: str
    source_node_id: str
    rooms_visited: int
    main_depth_reached: int = 0
    event_type: EventType = field(init=False, default=EventType.MAZE_ROUTE_COLLAPSED)


@dataclass(frozen=True)
class MazeFrontierOpenedEvent:
    message: str
    run_id: str
    node_id: str
    depth: int
    event_type: EventType = field(init=False, default=EventType.MAZE_FRONTIER_OPENED)


@dataclass(frozen=True)
class CompanyEvent:
    message: str
    event_type: EventType = field(init=False, default=EventType.COMPANY)


@dataclass(frozen=True)
class TownServiceEvent:
    message: str
    service_id: str
    cost: int = 0
    event_type: EventType = field(init=False, default=EventType.TOWN_SERVICE)


@dataclass(frozen=True)
class LootSoldEvent:
    message: str
    item_id: str
    quantity: int
    coin: int
    event_type: EventType = field(init=False, default=EventType.TOWN_SERVICE)


@dataclass(frozen=True)
class LootTurnedInEvent:
    message: str
    item_id: str
    flag_id: str
    unlocked_contract_id: str = ""
    event_type: EventType = field(init=False, default=EventType.TOWN_SERVICE)


@dataclass(frozen=True)
class RecruitmentOfferedEvent:
    message: str
    offers: list[dict[str, str]]
    event_type: EventType = field(init=False, default=EventType.RECRUITMENT_OFFERED)


@dataclass(frozen=True)
class HeroRecruitedEvent:
    message: str
    hero_id: str
    name: str
    class_id: str
    cost: int
    event_type: EventType = field(init=False, default=EventType.HERO_RECRUITED)


@dataclass(frozen=True)
class RecoveryEvent:
    message: str
    hero_ids: list[str]
    cost: int
    event_type: EventType = field(init=False, default=EventType.RECOVERY)


@dataclass(frozen=True)
class DeepSurgeryEvent:
    message: str
    hero_id: str
    name: str
    cost: int
    remaining_mortal_wounds: int
    event_type: EventType = field(init=False, default=EventType.TOWN_SERVICE)


@dataclass(frozen=True)
class SuppliesPurchasedEvent:
    message: str
    supply_id: str
    quantity: int
    cost: int
    event_type: EventType = field(init=False, default=EventType.SUPPLIES_PURCHASED)


@dataclass(frozen=True)
class ActivePartyChangedEvent:
    message: str
    active_party_slots: dict[str, str | None]
    event_type: EventType = field(init=False, default=EventType.ACTIVE_PARTY_CHANGED)


@dataclass(frozen=True)
class SaveEvent:
    message: str
    path: str
    event_type: EventType = field(init=False, default=EventType.SAVE)


@dataclass(frozen=True)
class LoadEvent:
    message: str
    path: str
    event_type: EventType = field(init=False, default=EventType.LOAD)


GameEvent = (
    EncounterStartedEvent
    | RoundStartedEvent
    | RoundEndedEvent
    | EncounterEndedEvent
    | DamageEvent
    | HealingEvent
    | MoveEvent
    | TurnPassedEvent
    | TurnDelayedEvent
    | CombatRetreatDeclaredEvent
    | CombatRetreatedEvent
    | EnemyIntentEvent
    | ReactionUsedEvent
    | ReactionSkippedEvent
    | DownedEvent
    | DeathEvent
    | CombatEndedEvent
    | SkillUsedEvent
    | MissEvent
    | CombatEffectEvent
    | StatusChangedEvent
    | MemorySignalEvent
    | ExpeditionEvent
    | DungeonActionEvent
    | LootGainedEvent
    | BreachDiscoveredEvent
    | LoreDiscoveredEvent
    | ContractCompletedEvent
    | ExpeditionReturnedEvent
    | MazeRouteCollapsedEvent
    | MazeFrontierOpenedEvent
    | CompanyEvent
    | TownServiceEvent
    | LootSoldEvent
    | LootTurnedInEvent
    | RecruitmentOfferedEvent
    | HeroRecruitedEvent
    | RecoveryEvent
    | DeepSurgeryEvent
    | SuppliesPurchasedEvent
    | ActivePartyChangedEvent
    | SaveEvent
    | LoadEvent
)
