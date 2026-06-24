"""App-facing view models for terminal rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from game.app.actions import (
    ActionProvider,
    ScreenAction,
    ScreenActionKind,
    ScreenActionRisk,
    join_detail,
)
from game.app.contracts import contract_board_ids, contract_board_state
from game.app.manual_combat import (
    ManualCombatSession,
    can_delay_hero,
    heal_amount_range_for_skill,
    legal_move_slots,
    legal_reaction_options,
    legal_skill_ids,
    legal_target_ids,
    skill_target_ids,
    skill_unavailable_reason,
    visible_skill_ids,
)
from game.campaign.company import (
    CompanyState,
    ExpeditionReportState,
    HeroMemoryEntry,
    HeroReportOutcome,
    HeroState,
)
from game.campaign.gear import (
    available_gear_count,
    effective_hero_stats,
    gear_effect_summary,
    gear_unavailable_reason,
)
from game.campaign.objectives import (
    BLACKWOOD_ROAD_CHARTER_ID,
    BREACH_STALKER_HUNT_ID,
    SHALLOW_CAVE_BREACH_SCOUT_ID,
    CampaignObjectiveView,
    build_campaign_objective,
)
from game.campaign.recruitment import RecruitChoice
from game.campaign.roster import active_roster, living_roster, reserve_roster
from game.campaign.town import (
    IN_SURGERY_LABEL,
    deep_surgery_candidates,
    effective_roster_cap,
    effective_surgery_cost,
    upgrade_unavailable_reason,
)
from game.combat.combat_state import Combatant, LifeState, Team
from game.combat.damage_range import format_damage_label
from game.combat.formation import (
    Formation,
    FormationSlot,
    back_slot_for,
    is_back,
)
from game.combat.preview import preview_attack
from game.combat.targeting import skill_position_label
from game.content.definitions import GameDefinitions
from game.core.events import GameEvent
from game.data.schemas import ExpeditionNodeDefinition, ExpeditionRoomActionDefinition
from game.expedition.dungeon import (
    SHALLOW_CAVE_BREACH_NODE_ID,
    active_dungeon_nodes,
    revealed_exit_node_ids,
    room_action_key,
)
from game.expedition.expedition import OPENING_BREACH_PENDING_FLAG
from game.expedition.generated_maze import (
    GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
    GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
    FrontierExitPreview,
    frontier_exit_previews,
    frontier_node_id,
    frontier_preview_map_positions,
    generated_nodes_by_id,
    is_generated_spur_room,
    is_main_spine_room,
)
from game.expedition.node import ExpeditionNodeType
from game.expedition.travel import (
    CAVE_REGIONAL_ID,
    HAVEN_REGIONAL_ID,
    REGIONAL_CAVE_ANCHOR_NODE_ID,
    REGIONAL_EAST_GATE_NODE_ID,
    REGIONAL_OVERWORLD_MAP_ID,
    REGIONAL_OVERWORLD_NODE_IDS,
    get_regional_node_id,
    regional_available_exit_ids,
    regional_known_exit_ids_by_node,
    regional_overworld_nodes,
    world_location_id_for_node,
)
from game.ui.wounds import mortal_wound_badge

EMPTY_SLOT_VALUE = "__empty__"
QUEST_PICKUP_NODE_ID = "town_gate"
BLACKWOOD_CAVE_TARGET_NODE_ID = "shallow_cave_entrance"
BLACKWOOD_BOSS_TARGET_NODE_ID = "maze_touched_lair"
BREACH_TARGET_NODE_ID = "maze_breach"


@dataclass(frozen=True)
class CombatActorView:
    actor_id: str
    name: str
    team: str
    slot: str
    hp: int
    max_hp: int
    effort: int
    max_effort: int
    mortal_wounds: int
    morale: str
    strain: str
    tags: tuple[str, ...]
    life_state: str
    statuses: tuple[str, ...] = ()
    personal_quirk: str = ""
    quirks: tuple[str, ...] = ()
    strain_marks: tuple[str, ...] = ()
    acting: bool = False
    class_id: str = ""
    display_name: str = ""
    glyph: str = ""
    mini_lines: tuple[str, ...] = ()
    mini_frames: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    art_lines: tuple[str, ...] = ()
    art_frames: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    art_frame_holds: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    art_frame_impacts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.statuses:
            object.__setattr__(self, "statuses", _life_state_labels(self.life_state))

    @property
    def fatigue(self) -> str:
        return self.strain

    @property
    def conditions(self) -> tuple[str, ...]:
        return self.strain_marks


@dataclass(frozen=True)
class CombatSkillOption:
    action: ScreenAction
    skill_id: str
    name: str
    effort_cost: int
    attack_type: str
    usable_from: str
    usable_from_label: str
    flavor_text: str
    effect_text: str
    unavailable_reason: str
    intent: str
    damage_estimate: int
    damage_label: str
    target_count: int


@dataclass(frozen=True)
class CombatTargetOption:
    action: ScreenAction
    target_id: str
    name: str
    slot: str
    hp: int
    max_hp: int
    life_state: str
    hit_chance: int
    damage_estimate: int
    damage_label: str
    legality_reason: str
    intent: str

    @property
    def statuses(self) -> tuple[str, ...]:
        return _life_state_labels(self.life_state)


@dataclass(frozen=True)
class CombatMoveOption:
    action: ScreenAction
    from_slot: str
    to_slot: str
    actor_name: str
    occupant_name: str
    description: str
    before_formation: tuple[tuple[str, str], ...] = ()
    after_formation: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CombatEnemyIntentView:
    enemy_id: str
    enemy_name: str
    skill_id: str
    skill_name: str
    label: str
    target_id: str
    target_name: str
    threat_level: str
    obvious_effect: str
    debug_hit_chance: int | None = None
    debug_damage_estimate: int | None = None
    debug_damage_label: str = ""


@dataclass(frozen=True)
class CombatReactionOption:
    action: ScreenAction
    reaction_id: str | None
    kind: str
    actor_id: str | None
    actor_name: str
    cost: int
    summary: str


@dataclass(frozen=True)
class CombatTurnOrderEntry:
    actor_id: str
    name: str
    team: str
    life_state: str
    active: bool = False
    acted: bool = False

    @property
    def statuses(self) -> tuple[str, ...]:
        return _life_state_labels(self.life_state)


@dataclass(frozen=True)
class CombatView:
    encounter_id: str
    encounter_name: str
    round_number: int
    cohesion: str
    current_actor: CombatActorView | None
    selected_skill_id: str | None
    party: tuple[CombatActorView, ...]
    enemies: tuple[CombatActorView, ...]
    commands: tuple[ScreenAction, ...] = ()
    skills: tuple[CombatSkillOption, ...] = ()
    targets: tuple[CombatTargetOption, ...] = ()
    moves: tuple[CombatMoveOption, ...] = ()
    pending_enemy_intent: CombatEnemyIntentView | None = None
    reaction_options: tuple[CombatReactionOption, ...] = ()
    recent_events: tuple[GameEvent, ...] = ()
    turn_order: tuple[CombatTurnOrderEntry, ...] = ()


@dataclass(frozen=True)
class HeroListEntry:
    hero_id: str
    name: str
    class_id: str
    slot: str
    hp: int
    max_hp: int
    effort: int
    max_effort: int
    mortal_wounds: int
    morale: str
    strain: str
    life_state: str
    personal_quirk: str = ""
    quirks: tuple[str, ...] = ()
    strain_marks: tuple[str, ...] = ()
    memory_count: int = 0
    latest_memory: str = ""
    equipped_gear: str = ""
    stat_bonus: str = ""

    @property
    def statuses(self) -> tuple[str, ...]:
        return _life_state_labels(self.life_state)

    @property
    def fatigue(self) -> str:
        return self.strain

    @property
    def conditions(self) -> tuple[str, ...]:
        return self.strain_marks


@dataclass(frozen=True)
class HeroSheetTraitView:
    trait_id: str
    name: str
    kind: str
    description: str = ""
    positive_text: str = ""
    negative_text: str = ""
    stability: str = ""


@dataclass(frozen=True)
class HeroSheetFreshMemoryView:
    family_id: str
    name: str
    intensity: int
    tags: tuple[str, ...] = ()
    source_summary: str = ""
    pending_manifestation: bool = False


@dataclass(frozen=True)
class HeroSheetMemoryEntryView:
    summary: str
    kind: str
    expedition_id: str
    dungeon_id: str
    node_id: str = ""
    encounter_id: str = ""


@dataclass(frozen=True)
class HeroSheetSignalView:
    label: str
    score: int


@dataclass(frozen=True)
class HeroSheetView:
    hero_id: str
    name: str
    class_id: str
    class_name: str
    roster_state: str
    slot: str
    background: str = ""
    motive: str = ""
    hp: int = 0
    max_hp: int = 0
    effort: int = 0
    max_effort: int = 0
    speed: int = 0
    accuracy: int = 0
    defense: int = 0
    damage: int = 0
    morale: str = ""
    strain: str = ""
    life_state: str = ""
    statuses: tuple[str, ...] = ()
    mortal_wounds: int = 0
    equipped_gear: str = ""
    equipped_gear_description: str = ""
    stat_bonus: str = ""
    personal_quirk: HeroSheetTraitView | None = None
    earned_quirks: tuple[HeroSheetTraitView, ...] = ()
    strain_marks: tuple[HeroSheetTraitView, ...] = ()
    fresh_memories: tuple[HeroSheetFreshMemoryView, ...] = ()
    permanent_memories: tuple[HeroSheetMemoryEntryView, ...] = ()
    career_signals: tuple[HeroSheetSignalView, ...] = ()
    available_kits: tuple[GearItemView, ...] = ()
    can_manage_gear: bool = True
    gear_manage_reason: str = ""

    @property
    def latest_memory(self) -> str:
        return self.permanent_memories[0].summary if self.permanent_memories else ""


@dataclass(frozen=True)
class RosterSectionView:
    title: str
    heroes: tuple[HeroListEntry, ...]


@dataclass(frozen=True)
class MemorialEntryView:
    hero_id: str
    name: str
    class_id: str
    mortal_wounds: int
    final_memory: str = ""


@dataclass(frozen=True)
class FormationSlotView:
    slot: FormationSlot
    slot_label: str
    hero_id: str | None
    hero_name: str
    condition: str
    class_name: str = ""
    vitals_line: str = ""
    protection_line: str = ""
    abnormal_status: str = ""
    mortal_wounds: int = 0


@dataclass(frozen=True)
class ContractBoardEntryView:
    contract_id: str
    name: str
    summary: str
    reward_reputation: int
    coin_reward: int
    difficulty: int
    state: str
    unavailable_reason: str = ""


@dataclass(frozen=True)
class TownUpgradeView:
    upgrade_id: str
    name: str
    description: str
    cost: int
    state: str
    effect_summary: str
    unavailable_reason: str = ""


@dataclass(frozen=True)
class TownDashboardView:
    company_name: str
    location: str
    reputation: int
    coin: int
    roster_cap: int
    active_count: int
    reserve_count: int
    wounded_count: int
    downed_count: int
    deceased_count: int
    objective: CampaignObjectiveView
    active_party: tuple[HeroListEntry, ...]
    reserves: tuple[HeroListEntry, ...]
    services: tuple[ScreenAction, ...]
    contract_board: tuple[ContractBoardEntryView, ...] = ()
    upgrades: tuple[TownUpgradeView, ...] = ()
    upgrade_actions: tuple[ScreenAction, ...] = ()


@dataclass(frozen=True)
class WorldLocationView:
    location_id: str
    name: str
    kind: str
    description: str
    known: bool
    visit_count: int = 0
    discovered_count: int = 0
    cleared_count: int = 0
    shortcut_count: int = 0
    consumed_rumor_count: int = 0
    related_contracts: tuple[str, ...] = ()
    memory_summary: str = ""


@dataclass(frozen=True)
class WorldDestinationView:
    destination_id: str
    location_id: str
    label: str
    description: str
    available: bool = True
    route_kind: str = "travel"


@dataclass(frozen=True)
class WorldView:
    company_name: str
    current_location_id: str
    current_location_name: str
    current_location_kind: str
    current_location_description: str
    breadcrumb: str
    objective: CampaignObjectiveView
    known_locations: tuple[WorldLocationView, ...]
    travel_destinations: tuple[WorldDestinationView, ...]
    active_contracts: tuple[tuple[str, str, str], ...]
    known_routes: tuple[str, ...]


@dataclass(frozen=True)
class ArrivalBriefView:
    title: str
    location_id: str
    location_name: str
    origin_name: str
    path: tuple[str, ...]
    flavor_lines: tuple[str, ...]
    what_changed: tuple[str, ...]
    next_objective: str
    record_label: str = "Latest filed company record"


@dataclass(frozen=True)
class RegionalMapView:
    current_node_id: str
    current_node_name: str
    other_node_id: str
    other_node_name: str
    route_charted: bool
    travel_available: bool
    travel_cost: str
    travel_flavor: str | None
    objective: CampaignObjectiveView
    arrival_context: ArrivalBriefView | None
    active_contracts: tuple[tuple[str, str, str], ...]
    breadcrumb: str = ""
    location_description: str = ""
    place_text: str = ""
    anchor_kind: str = "overworld"
    current_map_id: str = REGIONAL_OVERWORLD_MAP_ID
    map_nodes: tuple[DungeonMapNodeView, ...] = ()
    exits: tuple[DungeonMapNodeView, ...] = ()
    inventory: tuple[tuple[str, int], ...] = ()
    supplies: tuple[tuple[str, int], ...] = ()
    place_title: str = ""
    opening_route_available: bool = False
    opening_route_label: str = "Take the Old Road"
    room_actions: tuple[DungeonActionView, ...] = ()


@dataclass(frozen=True)
class ShellStatusView:
    company_name: str | None
    location: str = "No company loaded"
    reputation: int = 0
    coin: int = 0
    active_count: int = 0
    reserve_count: int = 0
    wounded_count: int = 0
    downed_count: int = 0
    breaches: str = "none"
    save_status: str = "empty"
    save_path: str = ""


@dataclass(frozen=True)
class FormationView:
    slots: tuple[FormationSlotView, ...]
    assignable_heroes: tuple[HeroListEntry, ...]
    actions: tuple[ScreenAction, ...]


@dataclass(frozen=True)
class SupplyShopView:
    reputation: int
    coin: int
    actions: tuple[ScreenAction, ...]


@dataclass(frozen=True)
class RelicBrokerView:
    reputation: int
    coin: int
    inventory: tuple[tuple[str, int], ...]
    actions: tuple[ScreenAction, ...]


@dataclass(frozen=True)
class DeepSurgeryCandidateView:
    hero_id: str
    name: str
    class_name: str
    mortal_wounds: int


@dataclass(frozen=True)
class DeepSurgeryView:
    coin: int
    surgery_cost: int
    candidates: tuple[DeepSurgeryCandidateView, ...]
    actions: tuple[ScreenAction, ...]


@dataclass(frozen=True)
class GearItemView:
    gear_id: str
    name: str
    description: str
    owned_count: int
    equipped_count: int
    available_count: int
    cost: int | None
    state: str
    effect_summary: str
    unavailable_reason: str = ""


@dataclass(frozen=True)
class GearHeroView:
    hero_id: str
    name: str
    class_id: str
    equipped_gear_id: str = ""
    equipped_gear_name: str = ""
    condition: str = ""


@dataclass(frozen=True)
class GearInventoryView:
    reputation: int
    coin: int
    can_manage: bool
    can_purchase: bool
    manage_reason: str = ""
    purchase_reason: str = ""
    items: tuple[GearItemView, ...] = ()
    heroes: tuple[GearHeroView, ...] = ()
    actions: tuple[ScreenAction, ...] = ()


@dataclass(frozen=True)
class RecruitOfferView:
    name: str
    class_id: str
    class_name: str
    background: str
    motive: str
    cost: int


@dataclass(frozen=True)
class RecruitOffersView:
    reputation: int
    coin: int
    roster_count: int
    roster_cap: int
    offers: tuple[RecruitOfferView, ...]
    actions: tuple[ScreenAction, ...]


@dataclass(frozen=True)
class DungeonRoomView:
    node_id: str
    name: str
    node_type: str
    text: str
    safe_return: bool
    cleared: bool
    scene_state: str = ""
    route_hint: str = ""
    party_hint: str = ""
    first_visit: bool = False
    major_beat: bool = False
    art_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class DungeonMapNodeView:
    node_id: str
    name: str
    node_type: str
    status: str
    current: bool
    known: bool
    visited: bool
    cleared: bool
    safe_return: bool
    map_id: str = ""
    exit_node_ids: tuple[str, ...] = ()
    map_x: int | None = None
    map_y: int | None = None
    direction: str = ""
    location_id: str = ""
    memory_summary: str = ""
    memory_notes: tuple[str, ...] = ()
    inventory_rewards: tuple[tuple[str, int], ...] = ()
    supply_rewards: tuple[tuple[str, int], ...] = ()
    reputation_reward: int = 0
    coin_reward: int = 0
    inventory_requirements: tuple[tuple[str, int], ...] = ()
    supply_costs: tuple[tuple[str, int], ...] = ()
    action_summaries: tuple[str, ...] = ()
    quest_marker: bool = False


@dataclass(frozen=True)
class DungeonActionView:
    action_id: str
    label: str
    description: str
    state: str
    requirements: tuple[tuple[str, int], ...]
    cost: tuple[tuple[str, int], ...]
    reward: tuple[str, ...]


@dataclass(frozen=True)
class DungeonView:
    expedition_id: str
    dungeon_id: str
    current_room: DungeonRoomView
    map_nodes: tuple[DungeonMapNodeView, ...]
    exits: tuple[DungeonMapNodeView, ...]
    room_actions: tuple[DungeonActionView, ...]
    actions: tuple[ScreenAction, ...]
    recent_events: tuple[GameEvent, ...] = ()
    current_map_id: str = ""
    previous_node_id: str = ""
    inventory: tuple[tuple[str, int], ...] = ()
    supplies: tuple[tuple[str, int], ...] = ()
    maze_main_depth: int = 0
    maze_frontier_depth: int = 0
    maze_off_spine_hint: str = ""


@dataclass(frozen=True)
class ExpeditionReportView:
    expedition_id: str
    dungeon_id: str | None
    outcome: str
    rooms_entered: tuple[str, ...]
    encounters_resolved: tuple[str, ...]
    loot: tuple[tuple[str, int], ...]
    supplies: tuple[tuple[str, int], ...]
    gear: tuple[tuple[str, int], ...]
    reputation_gained: int
    coin_gained: int
    breaches_discovered: tuple[str, ...]
    room_actions: tuple[str, ...]
    supplies_spent: tuple[tuple[str, int], ...]
    reputation_start: int
    reputation_end: int
    coin_start: int
    coin_end: int
    hero_outcomes: tuple[str, ...]
    notable_moments: tuple[str, ...]
    what_changed: tuple[str, ...]
    objective: CampaignObjectiveView
    next_objective: str
    wounded_count: int
    downed_count: int
    deceased_count: int
    actions: tuple[ScreenAction, ...]
    reputation_delta: int = 0
    coin_delta: int = 0
    supply_deltas: tuple[tuple[str, int, int, int], ...] = ()
    inventory_deltas: tuple[tuple[str, int, int, int], ...] = ()
    gear_deltas: tuple[tuple[str, int, int, int], ...] = ()


def build_shell_status(
    company: CompanyState | None,
    save_path: str,
    *,
    save_exists: bool,
    definitions: GameDefinitions | None = None,
) -> ShellStatusView:
    if company is None:
        return ShellStatusView(
            company_name=None,
            save_status="present" if save_exists else "empty",
            save_path=save_path,
        )
    active = active_roster(company)
    reserves = reserve_roster(company)
    living = living_roster(company)
    wounded_count = sum(
        1
        for hero in living
        if hero.hp
        < (
            effective_hero_stats(hero, definitions).max_hp
            if definitions is not None
            else hero.max_hp
        )
        or hero.mortal_wounds > 0
    )
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    return ShellStatusView(
        company_name=company.name,
        location=str(company.town_state.get("location", "Haven Town")),
        reputation=company.reputation,
        coin=company.coin,
        active_count=len(active),
        reserve_count=len(reserves),
        wounded_count=wounded_count,
        downed_count=downed_count,
        breaches=", ".join(sorted(company.known_breaches)) or "none",
        save_status="present" if save_exists else "empty",
        save_path=save_path,
    )


def build_world_view(company: CompanyState, definitions: GameDefinitions) -> WorldView:
    current_location_id = _current_world_location_id(company, definitions)
    current_location = definitions.locations[current_location_id]
    known_location_ids = _known_world_location_ids(company, definitions, current_location_id)
    objective = build_campaign_objective(company)
    active_contracts = tuple(
        (
            contract.id,
            contract.name,
            contract.summary,
        )
        for contract in sorted(
            (
                contract
                for contract in definitions.contracts.values()
                if contract.id in company.active_contract_ids
            ),
            key=lambda contract: contract.name,
        )
    )
    return WorldView(
        company_name=company.name,
        current_location_id=current_location.id,
        current_location_name=current_location.name,
        current_location_kind=current_location.kind,
        current_location_description=current_location.description,
        breadcrumb=_world_breadcrumb(current_location),
        objective=objective,
        known_locations=tuple(
            _world_location_view(
                company,
                definitions,
                location.id,
                known=location.id in known_location_ids,
            )
            for location in definitions.locations.values()
            if location.id in known_location_ids
        ),
        travel_destinations=_travel_destinations(
            company,
            definitions,
            current_location.id,
        ),
        active_contracts=active_contracts,
        known_routes=tuple(sorted(company.known_route_ids)),
    )


def build_regional_map_view(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    travel_flavor: str | None = None,
    arrival_context: ArrivalBriefView | None = None,
) -> RegionalMapView:
    nodes = regional_overworld_nodes(definitions)
    current_node_id = _current_regional_node_id(company, definitions)
    current_node = nodes[current_node_id]
    route_charted = "shallow_cave" in company.known_route_ids
    charted_hop_id = (
        CAVE_REGIONAL_ID
        if current_node_id == REGIONAL_EAST_GATE_NODE_ID
        else HAVEN_REGIONAL_ID
    )
    other_node_id = charted_hop_id
    other_location = definitions.locations[other_node_id]
    active_contracts = tuple(
        (
            contract.id,
            contract.name,
            contract.summary,
        )
        for contract in sorted(
            (
                contract
                for contract in definitions.contracts.values()
                if contract.id in company.active_contract_ids
            ),
            key=lambda contract: contract.name,
        )
    )
    memory = company.dungeon_memory.get("shallow_cave")
    visited_ids = set(memory.visited_node_ids if memory else [])
    cleared_ids = set(memory.cleared_node_ids if memory else [])
    visited_ids.add(current_node_id)
    known_ids = set(visited_ids)
    for world_memory in company.world_memory.values():
        known_ids.update(
            node_id for node_id in world_memory.discovered_node_ids if node_id in nodes
        )
    for node_id in list(known_ids):
        if node_id not in nodes:
            continue
        known_ids.update(nodes[node_id].exits)
        if memory is not None:
            prefix = f"{node_id}->"
            known_ids.update(
                revealed_exit_id.removeprefix(prefix)
                for revealed_exit_id in memory.revealed_exit_ids
                if revealed_exit_id.startswith(prefix)
            )
    known_ids = {node_id for node_id in known_ids if node_id in REGIONAL_OVERWORLD_NODE_IDS}
    known_ids.add(current_node_id)
    known_exit_ids_by_node = regional_known_exit_ids_by_node(
        definitions,
        known_ids,
        memory,
    )
    quest_target_node_ids = _quest_target_node_ids(company, None)
    map_nodes = tuple(
        _dungeon_node_view(
            node_id,
            nodes,
            company=company,
            session=None,
            current_id=current_node_id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=set(memory.completed_action_ids if memory else []),
            exit_node_ids=known_exit_ids_by_node.get(node_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for node_id in nodes
        if node_id in known_ids
    )
    walk_exit_ids = regional_available_exit_ids(company, definitions, current_node_id)
    exits = tuple(
        _dungeon_node_view(
            exit_id,
            nodes,
            company=company,
            session=None,
            current_id=current_node_id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=set(memory.completed_action_ids if memory else []),
            exit_node_ids=known_exit_ids_by_node.get(exit_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for exit_id in walk_exit_ids
    )
    if current_node_id == REGIONAL_EAST_GATE_NODE_ID:
        breadcrumb = "World > Haven > East Gate"
        place_title = "East Gate"
        anchor_kind = "east_gate"
    elif current_node_id == REGIONAL_CAVE_ANCHOR_NODE_ID:
        breadcrumb = "World > Shallow Cave"
        place_title = current_node.name
        anchor_kind = "shallow_cave"
    else:
        breadcrumb = "World > Old Road Wilds"
        place_title = current_node.name
        anchor_kind = "overworld"
    first_visit = memory is None or current_node_id not in memory.visited_node_ids
    place_text = (
        current_node.text
        if first_visit or not current_node.revisit_text
        else current_node.revisit_text
    )
    pending_regional_combat = bool(company.town_state.get("pending_regional_combat_node_id"))
    current_cleared = (
        current_node_id in cleared_ids or current_node.encounter is None
    )
    room_actions: tuple[DungeonActionView, ...] = ()
    if not pending_regional_combat:
        room_actions = tuple(
            _dungeon_action_view(
                action,
                company,
                current_node_id=current_node_id,
                current_cleared=current_cleared,
                completed_action_ids=set(memory.completed_action_ids if memory else []),
            )
            for action in current_node.actions
            if _room_action_visible(company, action)
        )
    return RegionalMapView(
        current_node_id=current_node_id,
        current_node_name=current_node.name,
        other_node_id=other_node_id,
        other_node_name=other_location.name,
        route_charted=route_charted,
        travel_available=route_charted,
        travel_cost="1 ration when available",
        travel_flavor=travel_flavor,
        objective=build_campaign_objective(company),
        arrival_context=arrival_context,
        active_contracts=active_contracts,
        breadcrumb=breadcrumb,
        location_description=current_node.text,
        place_text=place_text,
        anchor_kind=anchor_kind,
        current_map_id=REGIONAL_OVERWORLD_MAP_ID,
        map_nodes=map_nodes,
        exits=exits,
        inventory=tuple(sorted(company.inventory.items())),
        supplies=tuple(sorted(company.supplies.items())),
        place_title=place_title,
        opening_route_available=(
            current_node_id == REGIONAL_EAST_GATE_NODE_ID and not route_charted
        ),
        opening_route_label="Take the Old Road",
        room_actions=room_actions,
    )


def build_regional_render_view(regional: RegionalMapView) -> DungeonView:
    """Adapt a regional map view for dungeon-style map/minimap rendering."""
    current_node = next(
        node for node in regional.map_nodes if node.node_id == regional.current_node_id
    )
    return DungeonView(
        expedition_id="",
        dungeon_id="",
        current_map_id=regional.current_map_id,
        current_room=DungeonRoomView(
            node_id=current_node.node_id,
            name=current_node.name,
            node_type=current_node.node_type,
            text=regional.place_text or regional.location_description,
            safe_return=True,
            cleared=True,
        ),
        map_nodes=regional.map_nodes,
        exits=regional.exits,
        room_actions=(),
        actions=ActionProvider.regional_map_travel_actions(regional),
        inventory=regional.inventory,
        supplies=regional.supplies,
    )


def build_regional_arrival_context(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    origin_name: str = "",
    flavor_line: str = "",
) -> ArrivalBriefView | None:
    report = company.last_expedition_report
    if report is None:
        return None
    regional_node_id = _current_regional_node_id(company, definitions)
    nodes = regional_overworld_nodes(definitions)
    if regional_node_id in nodes:
        location_name = nodes[regional_node_id].name
        location_id = world_location_id_for_node(nodes[regional_node_id])
    else:
        location_id = _current_world_location_id(company, definitions)
        location_name = definitions.locations[location_id].name
    objective = build_campaign_objective(company)
    flavor_lines = (flavor_line,) if flavor_line else ()
    path = (origin_name, location_name) if origin_name else (location_name,)
    return ArrivalBriefView(
        title=f"Returned to {location_name}",
        location_id=location_id,
        location_name=location_name,
        origin_name=origin_name,
        path=path,
        flavor_lines=flavor_lines,
        what_changed=_arrival_change_lines(company, report),
        next_objective=objective.next_step,
    )


def _world_location_view(
    company: CompanyState,
    definitions: GameDefinitions,
    location_id: str,
    *,
    known: bool,
) -> WorldLocationView:
    location = definitions.locations[location_id]
    memory = company.world_memory.get(location_id)
    visit_count = memory.visit_count if memory is not None else 0
    discovered_count = len(memory.discovered_node_ids) if memory is not None else 0
    cleared_count = len(memory.cleared_threat_node_ids) if memory is not None else 0
    shortcut_count = len(memory.unlocked_shortcut_ids) if memory is not None else 0
    consumed_rumor_count = len(memory.consumed_rumor_ids) if memory is not None else 0
    memory_summary = _join_detail(
        f"visited {visit_count}x" if visit_count else "",
        f"{discovered_count} rooms discovered" if discovered_count else "",
        f"{cleared_count} threats cleared" if cleared_count else "",
        f"{shortcut_count} shortcuts" if shortcut_count else "",
        f"{consumed_rumor_count} rumors filed" if consumed_rumor_count else "",
    )
    return WorldLocationView(
        location_id=location.id,
        name=location.name,
        kind=location.kind,
        description=location.description,
        known=known,
        visit_count=visit_count,
        discovered_count=discovered_count,
        cleared_count=cleared_count,
        shortcut_count=shortcut_count,
        consumed_rumor_count=consumed_rumor_count,
        related_contracts=_location_contract_lines(company, definitions, location_id),
        memory_summary=memory_summary,
    )


def build_combat_view(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    *,
    retreat_available: bool = False,
    debug_combat_preview: bool = False,
) -> CombatView:
    current_actor = session.pending_hero()
    selected_skill_id = session.selected_skill_id
    skill_ids = visible_skill_ids(session, definitions)
    legal_skill_id_set = set(legal_skill_ids(session, definitions))
    move_slots = legal_move_slots(session)
    commands = ActionProvider.combat_commands(
        has_skills=bool(skill_ids),
        has_usable_skills=bool(legal_skill_id_set),
        has_moves=bool(move_slots),
        can_delay=can_delay_hero(session),
        can_act=current_actor is not None,
        retreat_available=retreat_available,
        actor_name=current_actor.name if current_actor is not None else "",
    )
    skills: list[CombatSkillOption] = []
    for index, skill_id in enumerate(skill_ids, start=1):
        skill = definitions.skills[skill_id]
        enabled = skill_id in legal_skill_id_set
        unavailable_reason = (
            ""
            if enabled
            else skill_unavailable_reason(
                session,
                definitions,
                skill_id,
            )
        )
        target_ids = (
            legal_target_ids(session, definitions, skill_id)
            if enabled
            else skill_target_ids(session, definitions, skill_id)
        )
        intent = _skill_intent(skill.tags)
        damage_estimate = 0
        damage_label = "0"
        if current_actor is not None and target_ids and enabled:
            if intent == "heal":
                heal_min, heal_max = heal_amount_range_for_skill(
                    session,
                    definitions,
                    skill_id,
                )
                damage_estimate = heal_max
                damage_label = f"{heal_min}-{heal_max}" if heal_min != heal_max else str(heal_max)
            else:
                preview = preview_attack(
                    session.state,
                    current_actor.actor_id,
                    skill,
                    target_ids[0],
                )
                damage_estimate = preview.damage
                damage_label = preview.damage_label
        skill_effect_description = _skill_description(
            intent,
            skill.attack_type.value,
            damage_label,
            len(target_ids),
        )
        skills.append(
            CombatSkillOption(
                action=ScreenAction(
                    number=str(index),
                    label=skill.name,
                    value=skill.id,
                    aliases=(skill.id,),
                    enabled=enabled,
                    default=enabled and not any(option.action.enabled for option in skills),
                    description=skill_effect_description,
                    kind=ScreenActionKind.COMBAT,
                    risk=ScreenActionRisk.COSTLY if skill.effort_cost else ScreenActionRisk.LOW,
                    cost=f"{skill.effort_cost} Effort" if skill.effort_cost else "",
                    unavailable_reason=unavailable_reason,
                    preview=join_detail(
                        (
                            f"{current_actor.name if current_actor is not None else 'Actor'} "
                            f"prepares {skill.name}."
                        ),
                        f"Usable: {skill_position_label(skill)}.",
                        skill.description,
                        skill_effect_description,
                    ),
                    result_hint=join_detail(
                        "Choose a legal target next." if enabled else unavailable_reason,
                        skill_effect_description,
                    ),
                ),
                skill_id=skill.id,
                name=skill.name,
                effort_cost=skill.effort_cost,
                attack_type=skill.attack_type.value,
                usable_from=skill.usable_from.value,
                usable_from_label=skill_position_label(skill, compact=True),
                flavor_text=skill.description,
                effect_text=skill.effect_text or skill_effect_description,
                unavailable_reason=unavailable_reason,
                intent=intent,
                damage_estimate=damage_estimate,
                damage_label=damage_label,
                target_count=len(target_ids),
            )
        )

    targets: list[CombatTargetOption] = []
    if current_actor is not None and selected_skill_id is not None:
        target_ids = sorted(
            legal_target_ids(session, definitions, selected_skill_id),
            key=lambda target_id: _target_sort_key(session, target_id),
        )
        skill = definitions.skills[selected_skill_id]
        intent = _skill_intent(skill.tags)
        single_target = len(target_ids) == 1
        for index, target_id in enumerate(target_ids, start=1):
            target = session.state.actor(target_id)
            if intent == "heal":
                hit_chance = 100
                heal_min, heal_max = heal_amount_range_for_skill(
                    session,
                    definitions,
                    selected_skill_id,
                    target_id,
                )
                amount = heal_max
                amount_label = f"{heal_min}-{heal_max}" if heal_min != heal_max else str(heal_max)
                legality_reason = "living ally"
                effect_label = format_damage_label(heal_min, heal_max, "heal")
            else:
                preview = preview_attack(session.state, current_actor.actor_id, skill, target_id)
                hit_chance = preview.hit_chance
                amount = preview.damage
                amount_label = preview.damage_label
                legality_reason = preview.legality_reason
                effect_label = format_damage_label(preview.damage_min, preview.damage_max)
            slot = session.state.formation_for(target.team).slot_of(target_id)
            targets.append(
                CombatTargetOption(
                    action=ScreenAction(
                        number=str(index),
                        label=target.name,
                        value=target_id,
                        aliases=(target_id,),
                        default=index == 1 or single_target,
                        description=f"{hit_chance}% hit, {effect_label}",
                        kind=ScreenActionKind.COMBAT,
                        risk=ScreenActionRisk.LOW,
                        preview=join_detail(
                            f"{target.name}: HP {target.hp}/{target.max_hp}",
                            f"Legal: {legality_reason}",
                            f"Projected: {hit_chance}% hit, {effect_label}",
                        ),
                        result_hint=(
                            f"Enter commits {skill.name} on {target.name}: "
                            f"{hit_chance}% hit, {effect_label}."
                        ),
                    ),
                    target_id=target_id,
                    name=target.name,
                    slot=slot.value if slot is not None else "-",
                    hp=target.hp,
                    max_hp=target.max_hp,
                    life_state=target.life_state.value,
                    hit_chance=hit_chance,
                    damage_estimate=amount,
                    damage_label=amount_label,
                    legality_reason=legality_reason,
                    intent=intent,
                )
            )

    moves: list[CombatMoveOption] = []
    from_slot = "-"
    if current_actor is not None:
        actor_slot = session.state.party_formation.slot_of(current_actor.actor_id)
        if actor_slot is not None:
            from_slot = actor_slot.value
    for index, to_slot in enumerate(move_slots, start=1):
        occupant_id = session.state.party_formation.actor_at(to_slot)
        occupant = session.state.actor(occupant_id) if occupant_id is not None else None
        actor_name = current_actor.name if current_actor is not None else "Actor"
        occupant_name = occupant.name if occupant is not None else "empty"
        moving_to = _slot_display(to_slot.value)
        moving_from = _slot_display(from_slot)
        if occupant is None:
            label = f"{moving_to} — open slot"
            description = f"{actor_name} shifts from {moving_from} to {moving_to}."
            result_hint = "Turn ends and protection changes immediately."
        else:
            label = f"{moving_to} — swap with {occupant_name}"
            description = (
                f"{actor_name} at {moving_from} swaps with {occupant_name} at {moving_to}."
            )
            result_hint = "Turn ends and protection changes immediately."
        before_formation = _formation_preview_slots(session.state.party_formation, session)
        after_formation = _move_preview_slots(
            before_formation,
            from_slot=from_slot,
            to_slot=to_slot.value,
        )
        moves.append(
            CombatMoveOption(
                action=ScreenAction(
                    number=str(index),
                    label=label,
                    value=to_slot.value,
                    aliases=(to_slot.value.lower(), to_slot.value.replace("_", " ").lower()),
                    default=index == 1,
                    description=f"{from_slot} -> {to_slot.value}",
                    kind=ScreenActionKind.COMBAT,
                    risk=ScreenActionRisk.LOW,
                    preview=join_detail(
                        description,
                        "Before -> after preview available in focus detail.",
                    ),
                    result_hint=result_hint,
                ),
                from_slot=from_slot,
                to_slot=to_slot.value,
                actor_name=actor_name,
                occupant_name=occupant_name,
                description=description,
                before_formation=before_formation,
                after_formation=after_formation,
            )
        )

    return CombatView(
        encounter_id=session.encounter_id,
        encounter_name=session.encounter_name,
        round_number=session.state.round_number,
        cohesion=session.state.derive_cohesion().name.title(),
        current_actor=_combatant_view(session, definitions, current_actor, acting=True)
        if current_actor is not None
        else None,
        selected_skill_id=selected_skill_id,
        party=tuple(
            _combatant_view(session, definitions, combatant, acting=current_actor == combatant)
            for combatant in _slot_ordered_combatants(session, Team.HERO)
        ),
        enemies=tuple(
            _combatant_view(session, definitions, combatant, acting=current_actor == combatant)
            for combatant in _slot_ordered_combatants(session, Team.ENEMY)
        ),
        commands=commands,
        skills=tuple(skills),
        targets=tuple(targets),
        moves=tuple(moves),
        pending_enemy_intent=_enemy_intent_view(
            session,
            debug_combat_preview=debug_combat_preview,
        ),
        reaction_options=_reaction_options(session),
        recent_events=tuple(session.recent_events[-8:]),
        turn_order=_turn_order_entries(session),
    )


def build_town_dashboard(
    company: CompanyState,
    definitions: GameDefinitions,
) -> TownDashboardView:
    active = active_roster(company)
    reserves = reserve_roster(company)
    living = living_roster(company)
    wounded_count = sum(1 for hero in living if hero.hp < hero.max_hp or hero.mortal_wounds > 0)
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    services = ActionProvider.town_services(company, definitions)
    objective = build_campaign_objective(company)
    return TownDashboardView(
        company_name=company.name,
        location=str(company.town_state.get("location", "Haven Town")),
        reputation=company.reputation,
        coin=company.coin,
        roster_cap=effective_roster_cap(company, definitions),
        active_count=len(active),
        reserve_count=len(reserves),
        wounded_count=wounded_count,
        downed_count=downed_count,
        deceased_count=len(company.deceased_heroes),
        objective=objective,
        active_party=tuple(
            _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions) for hero in active
        ),
        reserves=tuple(
            _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
            for hero in reserves
        ),
        services=services,
        contract_board=_contract_board_entries(company, definitions),
        upgrades=_upgrade_entries(company, definitions),
        upgrade_actions=ActionProvider.upgrade_actions(company, definitions),
    )


def _contract_board_entries(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[ContractBoardEntryView, ...]:
    return tuple(
        _contract_board_entry(company, definitions, contract_id)
        for contract_id in contract_board_ids(company, definitions)
        if _contract_board_is_visible(company, definitions, contract_id)
    )


def _contract_board_is_visible(
    company: CompanyState,
    definitions: GameDefinitions,
    contract_id: str,
) -> bool:
    state, _reason = contract_board_state(company, definitions, contract_id)
    return state in {"available", "active"}


def _contract_board_entry(
    company: CompanyState,
    definitions: GameDefinitions,
    contract_id: str,
) -> ContractBoardEntryView:
    contract = definitions.contracts[contract_id]
    state, unavailable_reason = contract_board_state(company, definitions, contract_id)
    return ContractBoardEntryView(
        contract_id=contract.id,
        name=contract.name,
        summary=contract.summary,
        reward_reputation=contract.reward_reputation,
        coin_reward=contract.coin_reward,
        difficulty=contract.difficulty,
        state=state,
        unavailable_reason=unavailable_reason,
    )


def _upgrade_entries(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[TownUpgradeView, ...]:
    entries: list[TownUpgradeView] = []
    for upgrade in definitions.town.upgrades.values():
        unavailable = upgrade_unavailable_reason(company, definitions, upgrade.id)
        if upgrade.id in company.purchased_upgrade_ids:
            state = "installed"
        elif not unavailable:
            state = "available"
        elif unavailable.startswith(("Complete ", "Find ")):
            state = "locked"
        else:
            state = "unavailable"
        entries.append(
            TownUpgradeView(
                upgrade_id=upgrade.id,
                name=upgrade.name,
                description=upgrade.description,
                cost=upgrade.cost,
                state=state,
                effect_summary=_upgrade_effect_summary(upgrade.effects),
                unavailable_reason=unavailable,
            )
        )
    return tuple(entries)


def build_roster_sections(
    company: CompanyState,
    definitions: GameDefinitions | None = None,
) -> tuple[RosterSectionView, ...]:
    return (
        RosterSectionView(
            "Active Party",
            tuple(
                _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
                for hero in active_roster(company)
            ),
        ),
        RosterSectionView(
            "Reserves",
            tuple(
                _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
                for hero in reserve_roster(company)
            ),
        ),
        RosterSectionView(
            "Memorial",
            tuple(
                _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
                for hero in company.deceased_heroes
            ),
        ),
    )


def build_memorial_entries(company: CompanyState) -> tuple[MemorialEntryView, ...]:
    entries: list[MemorialEntryView] = []
    for hero in company.deceased_heroes:
        memories = _hero_memories(company, hero.hero_id)
        final_memory = memories[-1].summary if memories else ""
        entries.append(
            MemorialEntryView(
                hero_id=hero.hero_id,
                name=hero.name,
                class_id=hero.class_id,
                mortal_wounds=hero.mortal_wounds,
                final_memory=final_memory,
            )
        )
    return tuple(entries)


def build_formation_view(
    company: CompanyState,
    definitions: GameDefinitions | None = None,
) -> FormationView:
    roster_by_id = {hero.hero_id: hero for hero in company.roster}
    formation = Formation.from_mapping(company.active_party_slots)
    protectors = _town_protectors(roster_by_id)
    slots: list[FormationSlotView] = []
    for slot in FormationSlot:
        hero = roster_by_id.get(company.active_party_slots.get(slot) or "")
        if hero is not None:
            stats = effective_hero_stats(hero, definitions)
            hero_class = (
                definitions.hero_classes.get(hero.class_id) if definitions is not None else None
            )
            class_name = (
                hero_class.name if hero_class is not None else _trait_label(hero.class_id)
            )
            vitals_line = (
                f"{hero.hp}/{stats.max_hp} HP, {hero.effort}/{stats.max_effort} Effort"
            )
            slots.append(
                FormationSlotView(
                    slot=slot,
                    slot_label=slot.value,
                    hero_id=hero.hero_id,
                    hero_name=hero.name,
                    condition=_hero_condition(hero, definitions),
                    class_name=class_name,
                    vitals_line=vitals_line,
                    protection_line=_formation_protection_line(
                        slot,
                        formation,
                        protectors,
                        roster_by_id,
                    ),
                    abnormal_status=_hero_abnormal_status(hero, definitions),
                    mortal_wounds=hero.mortal_wounds,
                )
            )
        else:
            slots.append(
                FormationSlotView(
                    slot=slot,
                    slot_label=slot.value,
                    hero_id=None,
                    hero_name="empty",
                    condition="",
                    mortal_wounds=0,
                )
            )
    actions: list[ScreenAction] = []
    for index, slot in enumerate(FormationSlot, start=1):
        slot_actor_id = company.active_party_slots.get(slot)
        slot_name = _slot_display(slot.value)
        label = (
            f"{slot_name}: {roster_by_id[slot_actor_id].name}"
            if slot_actor_id is not None and slot_actor_id in roster_by_id
            else f"{slot_name}: [empty]"
        )
        actions.append(
            ScreenAction(
                str(index),
                label,
                slot.value,
                (slot.value.lower(), slot.value.replace("_", " ").lower()),
                kind=ScreenActionKind.TOWN,
                preview=f"Choose who holds {slot_name}.",
                result_hint="Formation changes protection lanes before the next fight.",
            )
        )
    actions.append(
        ScreenAction("5", "Back", "back", ("back", "b"), kind=ScreenActionKind.NAVIGATE)
    )
    return FormationView(
        slots=tuple(slots),
        assignable_heroes=tuple(
            _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
            for hero in living_roster(company)
            if not hero.in_surgery
        ),
        actions=tuple(actions),
    )


def build_deep_surgery_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> DeepSurgeryView:
    candidates = deep_surgery_candidates(company)
    return DeepSurgeryView(
        coin=company.coin,
        surgery_cost=effective_surgery_cost(company, definitions),
        candidates=tuple(
            DeepSurgeryCandidateView(
                hero_id=hero.hero_id,
                name=hero.name,
                class_name=_trait_label(hero.class_id),
                mortal_wounds=hero.mortal_wounds,
            )
            for hero in candidates
        ),
        actions=ActionProvider.deep_surgery_hero_actions(company, definitions),
    )


def build_supply_shop_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> SupplyShopView:
    return SupplyShopView(
        reputation=company.reputation,
        coin=company.coin,
        actions=ActionProvider.supply_shop_actions(company, definitions),
    )


def build_relic_broker_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> RelicBrokerView:
    return RelicBrokerView(
        reputation=company.reputation,
        coin=company.coin,
        inventory=tuple(sorted(company.inventory.items())),
        actions=ActionProvider.relic_broker_actions(company, definitions),
    )


def build_gear_inventory_view(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    can_manage: bool,
    can_purchase: bool,
    manage_reason: str = "",
    purchase_reason: str = "",
) -> GearInventoryView:
    return GearInventoryView(
        reputation=company.reputation,
        coin=company.coin,
        can_manage=can_manage,
        can_purchase=can_purchase,
        manage_reason=manage_reason,
        purchase_reason=purchase_reason,
        items=_gear_item_views(company, definitions),
        heroes=_gear_hero_views(company, definitions),
        actions=ActionProvider.gear_actions(
            company,
            definitions,
            can_manage=can_manage,
            can_purchase=can_purchase,
            manage_reason=manage_reason,
            purchase_reason=purchase_reason,
        ),
    )


def build_hero_sheet_view(
    company: CompanyState,
    definitions: GameDefinitions,
    hero_id: str,
    *,
    can_manage_gear: bool = True,
    gear_manage_reason: str = "",
) -> HeroSheetView | None:
    hero = next(
        (candidate for candidate in (*company.roster, *company.deceased_heroes)
         if candidate.hero_id == hero_id),
        None,
    )
    if hero is None:
        return None
    stats = effective_hero_stats(hero, definitions)
    hero_class = definitions.hero_classes.get(hero.class_id)
    gear = (
        definitions.gear.get(hero.equipped_gear_id)
        if hero.equipped_gear_id is not None
        else None
    )
    active_ids = {active.hero_id for active in active_roster(company)}
    reserve_ids = {reserve.hero_id for reserve in reserve_roster(company)}
    memories = tuple(reversed(_hero_memories(company, hero.hero_id)))
    if hero.in_surgery:
        roster_state = IN_SURGERY_LABEL
    else:
        roster_state = _hero_roster_state(hero.hero_id, active_ids, reserve_ids)
    return HeroSheetView(
        hero_id=hero.hero_id,
        name=hero.name,
        class_id=hero.class_id,
        class_name=hero_class.name if hero_class is not None else _trait_label(hero.class_id),
        roster_state=roster_state,
        slot=hero.formation_slot.value,
        background=hero.background,
        motive=hero.motive,
        hp=hero.hp,
        max_hp=stats.max_hp,
        effort=hero.effort,
        max_effort=stats.max_effort,
        speed=hero.speed,
        accuracy=stats.accuracy,
        defense=hero.defense,
        damage=stats.damage,
        morale=hero.morale.name.title(),
        strain=hero.strain.name.title(),
        life_state=hero.life_state.value,
        statuses=_life_state_labels(hero.life_state.value),
        mortal_wounds=hero.mortal_wounds,
        equipped_gear=gear.name if gear is not None else "",
        equipped_gear_description=gear.description if gear is not None else "",
        stat_bonus=_stat_bonus_summary(hero, stats),
        personal_quirk=_sheet_trait(hero.personal_quirk, definitions, kind="personal"),
        earned_quirks=tuple(
            trait
            for slot in hero.earned_quirk_slots
            if (
                trait := _sheet_trait(
                    slot.quirk_id,
                    definitions,
                    kind="earned",
                    stability=slot.stability,
                )
            )
            is not None
        ),
        strain_marks=tuple(
            trait
            for mark in sorted(hero.strain_marks, key=lambda mark: mark.value)
            if (
                trait := _sheet_trait(mark.value, definitions, kind="strain", stability="")
            )
            is not None
        ),
        fresh_memories=tuple(
            HeroSheetFreshMemoryView(
                family_id=memory.family_id,
                name=memory.display_name,
                intensity=memory.intensity,
                tags=tuple(memory.tags),
                source_summary=_player_memory_summary(memory.source_summary),
                pending_manifestation=memory.pending_manifestation,
            )
            for memory in sorted(
                hero.fresh_memories,
                key=lambda memory: (memory.refreshed_order, memory.created_order),
                reverse=True,
            )
        ),
        permanent_memories=tuple(
            HeroSheetMemoryEntryView(
                summary=memory.summary,
                kind=memory.kind,
                expedition_id=memory.expedition_id,
                dungeon_id=memory.dungeon_id,
                node_id=memory.node_id or "",
                encounter_id=memory.encounter_id or "",
            )
            for memory in memories
        ),
        career_signals=tuple(
            HeroSheetSignalView(_signal_label(signal_id), score)
            for signal_id, score in sorted(
                hero.career_signals.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if score
        ),
        available_kits=tuple(
            item for item in _gear_item_views(company, definitions) if item.owned_count > 0
        ),
        can_manage_gear=can_manage_gear,
        gear_manage_reason=gear_manage_reason,
    )


def _gear_item_views(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[GearItemView, ...]:
    entries: list[GearItemView] = []
    for gear in definitions.gear.values():
        owned = company.gear_inventory.get(gear.id, 0)
        available = available_gear_count(company, gear.id)
        equipped = max(0, owned - available)
        unavailable = gear_unavailable_reason(company, definitions, gear.id)
        if owned:
            state = "owned"
        elif gear.cost is None:
            state = "reward"
        elif not unavailable:
            state = "available"
        elif unavailable.startswith(("Complete ", "Find ")):
            state = "locked"
        else:
            state = "unavailable"
        entries.append(
            GearItemView(
                gear_id=gear.id,
                name=gear.name,
                description=gear.description,
                owned_count=owned,
                equipped_count=equipped,
                available_count=available,
                cost=gear.cost,
                state=state,
                effect_summary=gear_effect_summary(gear),
                unavailable_reason=unavailable,
            )
        )
    return tuple(entries)


def _gear_hero_views(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[GearHeroView, ...]:
    entries: list[GearHeroView] = []
    for hero in living_roster(company):
        gear_id = hero.equipped_gear_id or ""
        gear = definitions.gear.get(gear_id) if gear_id else None
        entries.append(
            GearHeroView(
                hero_id=hero.hero_id,
                name=hero.name,
                class_id=hero.class_id,
                equipped_gear_id=gear_id,
                equipped_gear_name=gear.name if gear is not None else "",
                condition=_hero_condition(hero, definitions),
            )
        )
    return tuple(entries)


def build_recruit_offers_view(
    company: CompanyState,
    definitions: GameDefinitions,
    offers: Sequence[RecruitChoice],
) -> RecruitOffersView:
    cost = definitions.town.recruit_cost
    offer_views: list[RecruitOfferView] = []
    for offer in offers:
        name = offer.name
        class_id = offer.class_id
        hero_class = definitions.hero_classes.get(class_id)
        offer_views.append(
            RecruitOfferView(
                name=name,
                class_id=class_id,
                class_name=hero_class.name if hero_class is not None else _trait_label(class_id),
                background=offer.background,
                motive=offer.motive,
                cost=cost,
            )
        )
    return RecruitOffersView(
        reputation=company.reputation,
        coin=company.coin,
        roster_count=len(company.roster),
        roster_cap=effective_roster_cap(company, definitions),
        offers=tuple(offer_views),
        actions=ActionProvider.recruit_offer_actions(company, definitions, offers),
    )


def build_dungeon_view(
    company: CompanyState,
    definitions: GameDefinitions,
    recent_events: Sequence[GameEvent] = (),
    first_visit_node_id: str | None = None,
) -> DungeonView:
    session = company.active_expedition
    if session is None:
        raise ValueError("No active dungeon expedition.")
    nodes = active_dungeon_nodes(definitions, session)
    current = nodes[session.current_node_id]
    current_map_id = current.map_id
    current_cleared = current.id in session.cleared_node_ids
    current_first_visit = current.id == first_visit_node_id
    current_text = (
        current.text if current_first_visit or not current.revisit_text else current.revisit_text
    )
    current_exit_ids = list(
        dict.fromkeys([*current.exits, *revealed_exit_node_ids(session, current.id)])
    )
    current_exit_ids = _ordered_route_ids(
        current_exit_ids,
        nodes=nodes,
        current_node_id=current.id,
    )
    anchor_ids = set(session.visited_node_ids)
    anchor_ids.add(current.id)
    known_ids = set(anchor_ids)
    for node_id in anchor_ids:
        if node_id in nodes:
            known_ids.update(nodes[node_id].exits)
            known_ids.update(revealed_exit_node_ids(session, node_id))
    movement_enabled = current_cleared and session.pending_combat_node_id is None
    generated_nodes = generated_nodes_by_id(session.generated_dungeon)
    inside_generated_maze = current.id in generated_nodes
    at_generated_threshold = (
        session.generated_dungeon is not None
        and not session.generated_dungeon.collapsed
        and current.id == session.generated_dungeon.entry_node_id
    )
    opening_breach_at_breach = bool(
        company.flags.get(OPENING_BREACH_PENDING_FLAG, False)
        and session.current_node_id == SHALLOW_CAVE_BREACH_NODE_ID
    )
    can_enter_generated_maze = (
        current.node_type.value == "breach"
        and (session.generated_dungeon is None or session.generated_dungeon.collapsed)
        and not opening_breach_at_breach
    )
    generated_active = (
        session.generated_dungeon is not None and not session.generated_dungeon.collapsed
    )
    generated_dungeon = (
        session.generated_dungeon
        if session.generated_dungeon is not None and not session.generated_dungeon.collapsed
        else None
    )
    maze_frontier_depth = generated_dungeon.main_spine_length if generated_dungeon else 0
    at_maze_frontier = (
        generated_dungeon is not None
        and inside_generated_maze
        and session.current_node_id == frontier_node_id(generated_dungeon)
    )
    maze_main_depth = 0
    if generated_dungeon is not None and is_main_spine_room(
        session.current_node_id,
        generated_dungeon.run_id,
    ):
        room_suffix = session.current_node_id.rsplit("_room_", 1)[-1]
        if room_suffix.isdigit():
            maze_main_depth = int(room_suffix)
    frontier_previews = (
        frontier_exit_previews(generated_dungeon, current.id)
        if generated_dungeon is not None
        and current_cleared
        and session.pending_combat_node_id is None
        else ()
    )
    preview_by_id = {preview.exit_id: preview for preview in frontier_previews}
    if frontier_previews:
        current_exit_ids = list(
            dict.fromkeys([*current_exit_ids, *(preview.exit_id for preview in frontier_previews)])
        )

    maze_off_spine_hint = ""
    if (
        generated_active
        and session.generated_dungeon is not None
        and not at_maze_frontier
        and is_generated_spur_room(
            session.current_node_id,
            session.generated_dungeon.run_id,
        )
    ):
        frontier_id = frontier_node_id(session.generated_dungeon)
        if frontier_id in current.exits:
            maze_off_spine_hint = (
                "Return to the main route to push deeper or take unstable exits."
            )

    preview_exit_ids = {
        preview.exit_id for preview in frontier_previews if preview.exit_id not in nodes
    }
    if preview_exit_ids:
        known_ids.update(preview_exit_ids)

    known_exit_ids_by_node: dict[str, tuple[str, ...]] = {}
    for node_id in known_ids:
        if node_id not in nodes:
            continue
        exit_ids = tuple(
            dict.fromkeys(
                [*nodes[node_id].exits, *revealed_exit_node_ids(session, node_id)]
            )
        )
        known_exit_ids_by_node[node_id] = tuple(
            exit_id
            for exit_id in exit_ids
            if exit_id in preview_exit_ids
            or (
                exit_id in known_ids
                and exit_id in nodes
                and nodes[exit_id].map_id == nodes[node_id].map_id
            )
        )
    if preview_exit_ids and generated_active and session.generated_dungeon is not None:
        frontier_id = frontier_node_id(session.generated_dungeon)
        if frontier_id in known_exit_ids_by_node:
            preview_ordered = tuple(
                preview.exit_id
                for preview in frontier_previews
                if preview.exit_id in preview_exit_ids
            )
            existing = known_exit_ids_by_node[frontier_id]
            known_exit_ids_by_node[frontier_id] = tuple(
                dict.fromkeys([*existing, *preview_ordered])
            )
    visited_ids = set(session.visited_node_ids)
    cleared_ids = set(session.cleared_node_ids)
    completed_action_ids = set(session.completed_action_ids)
    quest_target_node_ids = _quest_target_node_ids(company, session)
    map_nodes = tuple(
        _dungeon_node_view(
            node_id,
            nodes,
            company=company,
            session=session,
            current_id=current.id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=completed_action_ids,
            exit_node_ids=known_exit_ids_by_node.get(node_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for node_id in nodes
        if node_id in known_ids and nodes[node_id].map_id == current_map_id
    )
    if frontier_previews and session.generated_dungeon is not None:
        frontier_position = current.position
        preview_positions = frontier_preview_map_positions(
            session.generated_dungeon,
            frontier_previews,
        )
        preview_map_nodes: list[DungeonMapNodeView] = []
        for preview in frontier_previews:
            if preview.exit_id in nodes:
                continue
            position = preview_positions[preview.exit_id]
            preview_map_nodes.append(
                _frontier_preview_exit_view(
                    preview,
                    current_map_id=current_map_id,
                    map_x=position[0],
                    map_y=position[1],
                    direction=(
                        _direction_between_positions(frontier_position, position)
                        if frontier_position is not None
                        else ""
                    ),
                )
            )
        if preview_map_nodes:
            map_nodes = (*map_nodes, *preview_map_nodes)
    exits = tuple(
        _frontier_preview_exit_view(preview_by_id[node_id], current_map_id=current_map_id)
        if node_id in preview_by_id and node_id not in nodes
        else _dungeon_node_view(
            node_id,
            nodes,
            company=company,
            session=session,
            current_id=current.id,
            visited_ids=visited_ids,
            cleared_ids=cleared_ids,
            known_ids=known_ids,
            completed_action_ids=completed_action_ids,
            exit_node_ids=known_exit_ids_by_node.get(node_id, ()),
            quest_target_node_ids=quest_target_node_ids,
        )
        for node_id in current_exit_ids
    )
    room_actions = tuple(
        _dungeon_action_view(
            action,
            company,
            current_node_id=current.id,
            current_cleared=current_cleared,
            completed_action_ids=set(session.completed_action_ids),
        )
        for action in current.actions
        if _room_action_visible(company, action)
    )

    actions = ActionProvider.dungeon_actions(
        room_actions,
        exits,
        movement_enabled=movement_enabled,
        pending_combat=session.pending_combat_node_id is not None,
        current_map_id=current_map_id,
        current_safe_return=current.safe_return,
        can_enter_generated_maze=can_enter_generated_maze,
        can_retrace_generated_maze=inside_generated_maze and not at_generated_threshold,
        can_withdraw_generated_maze=at_generated_threshold,
        unstable_frontier_exit_ids=tuple(preview.exit_id for preview in frontier_previews),
        previous_node_id=session.previous_node_id,
        current_node_id=current.id,
        opening_breach_at_breach=opening_breach_at_breach,
    )
    return DungeonView(
        expedition_id=session.expedition_id,
        dungeon_id=session.dungeon_id,
        current_map_id=current_map_id,
        previous_node_id=session.previous_node_id,
        maze_main_depth=maze_main_depth,
        maze_frontier_depth=maze_frontier_depth,
        maze_off_spine_hint=maze_off_spine_hint,
        current_room=DungeonRoomView(
            node_id=current.id,
            name=current.name,
            node_type=current.node_type.value,
            text=current_text,
            safe_return=current.safe_return,
            cleared=current_cleared,
            scene_state=current.scene_state,
            route_hint=current.route_hint,
            party_hint=current.party_hint,
            first_visit=current_first_visit,
            major_beat=current.major_beat,
            art_lines=_art_lines(_dungeon_node_art_asset(definitions, session, current)),
        ),
        map_nodes=map_nodes,
        exits=exits,
        room_actions=room_actions,
        actions=tuple(actions),
        recent_events=tuple(recent_events[-8:]),
        inventory=tuple(sorted(company.inventory.items())),
        supplies=tuple(sorted(company.supplies.items())),
    )


def build_expedition_report_view(
    company: CompanyState,
    definitions: GameDefinitions,
) -> ExpeditionReportView:
    report = company.last_expedition_report
    if report is None:
        raise ValueError("No expedition report is available.")
    nodes = {
        node.id: node
        for expedition in definitions.expeditions.values()
        for node in expedition.nodes
    }
    living = living_roster(company)
    wounded_count = sum(1 for hero in living if hero.hp < hero.max_hp or hero.mortal_wounds > 0)
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    actions = ActionProvider.report_actions(company, definitions)
    objective = build_campaign_objective(company)
    return ExpeditionReportView(
        expedition_id=report.expedition_id,
        dungeon_id=report.dungeon_id,
        outcome=report.outcome,
        rooms_entered=tuple(
            _node_name(nodes, node_id, report.room_names) for node_id in report.rooms_entered
        ),
        encounters_resolved=tuple(
            encounter_id.replace("_", " ").title() for encounter_id in report.encounters_resolved
        ),
        loot=tuple(sorted(report.loot.items())),
        supplies=tuple(sorted(report.supplies.items())),
        gear=tuple(sorted(report.gear.items())),
        reputation_gained=report.reputation_gained,
        coin_gained=report.coin_gained,
        breaches_discovered=tuple(report.breaches_discovered),
        room_actions=tuple(
            _room_action_name(nodes, action_key) for action_key in report.room_actions
        ),
        supplies_spent=tuple(sorted(report.supplies_spent.items())),
        reputation_start=report.start_reputation,
        reputation_end=report.end_reputation,
        reputation_delta=report.end_reputation - report.start_reputation,
        coin_start=report.start_coin,
        coin_end=report.end_coin,
        coin_delta=report.end_coin - report.start_coin,
        supply_deltas=_item_deltas(report.start_supplies, report.end_supplies),
        inventory_deltas=_item_deltas(report.start_inventory, report.end_inventory),
        gear_deltas=_item_deltas(
            report.start_gear_inventory,
            report.end_gear_inventory,
        ),
        hero_outcomes=tuple(_hero_outcome_line(outcome) for outcome in report.hero_outcomes),
        notable_moments=tuple(report.notable_moments),
        what_changed=_report_change_lines(report),
        objective=objective,
        next_objective=objective.next_step,
        wounded_count=wounded_count,
        downed_count=downed_count,
        deceased_count=len(company.deceased_heroes),
        actions=actions,
    )


def _slot_ordered_combatants(
    session: ManualCombatSession,
    team: Team,
) -> tuple[Combatant, ...]:
    formation = session.state.formation_for(team)
    side = session.state.side_for(team)
    ordered: list[Combatant] = []
    for slot in FormationSlot:
        actor_id = formation.actor_at(slot)
        if actor_id is not None and actor_id in side:
            ordered.append(side[actor_id])
    for combatant in side.values():
        if combatant not in ordered:
            ordered.append(combatant)
    return tuple(ordered)


def _formation_preview_slots(
    formation: Any,
    session: ManualCombatSession,
) -> tuple[tuple[str, str], ...]:
    slots: list[tuple[str, str]] = []
    for slot in FormationSlot:
        actor_id = formation.actor_at(slot)
        actor_name = session.state.actor(actor_id).name if actor_id is not None else "empty"
        slots.append((slot.value, actor_name))
    return tuple(slots)


def _move_preview_slots(
    slots: tuple[tuple[str, str], ...],
    *,
    from_slot: str,
    to_slot: str,
) -> tuple[tuple[str, str], ...]:
    names_by_slot = dict(slots)
    from_name = names_by_slot.get(from_slot, "empty")
    to_name = names_by_slot.get(to_slot, "empty")
    names_by_slot[from_slot] = to_name
    names_by_slot[to_slot] = from_name
    return tuple((slot, names_by_slot.get(slot, "empty")) for slot, _name in slots)


def _formation_slot_summaries(
    slots: Mapping[FormationSlot, str | None],
    roster_by_id: Mapping[str, HeroState],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            slot.value,
            roster_by_id[hero_id].name
            if (hero_id := slots.get(slot)) is not None and hero_id in roster_by_id
            else "empty",
        )
        for slot in FormationSlot
    )


def preview_assign_hero(
    slots: Mapping[FormationSlot, str | None],
    roster_by_id: Mapping[str, HeroState],
    hero_id: str,
    target_slot: FormationSlot,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Simulate assign_active_hero swap/displace for before/after previews."""
    before = _formation_slot_summaries(slots, roster_by_id)
    after_slots = dict(slots)
    old_slot: FormationSlot | None = None
    for slot, occupant_id in after_slots.items():
        if occupant_id == hero_id:
            old_slot = slot
            after_slots[slot] = None
            break
    displaced_id = after_slots.get(target_slot)
    after_slots[target_slot] = hero_id
    if displaced_id is not None and old_slot is not None:
        after_slots[old_slot] = displaced_id
    elif displaced_id is not None:
        after_slots[target_slot] = hero_id
    after = _formation_slot_summaries(after_slots, roster_by_id)
    return before, after


@dataclass(frozen=True)
class _TownProtector:
    hero_id: str

    def can_protect(self) -> bool:
        return True


def _town_protectors(roster_by_id: Mapping[str, HeroState]) -> Mapping[str, _TownProtector]:
    protectors: dict[str, _TownProtector] = {}
    for hero_id, hero in roster_by_id.items():
        if hero.life_state == LifeState.ALIVE and hero.hp > 0:
            protectors[hero_id] = _TownProtector(hero_id=hero_id)
    return protectors


def _formation_protection_line(
    slot: FormationSlot,
    formation: Formation,
    protectors: Mapping[str, _TownProtector],
    roster_by_id: Mapping[str, HeroState],
) -> str:
    if is_back(slot):
        protector_id = formation.protector_for(slot, protectors)
        if protector_id is not None and protector_id in roster_by_id:
            return f"protected by {roster_by_id[protector_id].name}"
        return ""
    front_actor_id = formation.actor_at(slot)
    if front_actor_id is None:
        return ""
    back_slot = back_slot_for(slot)
    back_actor_id = formation.actor_at(back_slot)
    if back_actor_id is None:
        return ""
    if formation.protector_for(back_slot, protectors) != front_actor_id:
        return ""
    if back_actor_id in roster_by_id:
        return f"protects {roster_by_id[back_actor_id].name}"
    return ""


def _hero_abnormal_status(
    hero: HeroState,
    definitions: GameDefinitions | None = None,
) -> str:
    pieces: list[str] = []
    statuses = ", ".join(
        status
        for status in _life_state_labels(hero.life_state.value)
        if status != "ready"
    )
    if statuses:
        pieces.append(statuses)
    if hero.in_surgery:
        pieces.append(IN_SURGERY_LABEL)
    if hero.mortal_wounds:
        pieces.append(mortal_wound_badge(hero.mortal_wounds))
    if hero.strain != hero.strain.STEADY:
        pieces.append(f"Strain {hero.strain.name.title()}")
    if hero.strain_marks:
        mark_text = ", ".join(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda item: item.value)
        )
        pieces.append(f"Marks {mark_text}")
    return ", ".join(pieces)


def hero_protection_line(
    company: CompanyState,
    hero_id: str,
) -> str:
    roster_by_id = {hero.hero_id: hero for hero in company.roster}
    hero = roster_by_id.get(hero_id)
    if hero is None:
        return ""
    formation = Formation.from_mapping(company.active_party_slots)
    protectors = _town_protectors(roster_by_id)
    slot = hero.formation_slot
    return _formation_protection_line(slot, formation, protectors, roster_by_id)


def _slot_display(slot: str) -> str:
    return slot.replace("_", " ").title()


def _enemy_intent_view(
    session: ManualCombatSession,
    *,
    debug_combat_preview: bool,
) -> CombatEnemyIntentView | None:
    intent = session.pending_enemy_intent
    if intent is None:
        return None
    return CombatEnemyIntentView(
        enemy_id=intent.enemy_id,
        enemy_name=intent.enemy_name,
        skill_id=intent.skill_id,
        skill_name=intent.skill_name,
        label=intent.label,
        target_id=intent.target_id,
        target_name=intent.target_name,
        threat_level=intent.threat_level,
        obvious_effect=intent.obvious_effect,
        debug_hit_chance=intent.hit_chance if debug_combat_preview else None,
        debug_damage_estimate=intent.damage_estimate if debug_combat_preview else None,
        debug_damage_label=intent.damage_label if debug_combat_preview else "",
    )


def _turn_order_entries(session: ManualCombatSession) -> tuple[CombatTurnOrderEntry, ...]:
    entries: list[CombatTurnOrderEntry] = []
    for index, initiative_entry in enumerate(session.initiative):
        actor = session.state.actor(initiative_entry.actor_id)
        if actor is None:
            continue
        entries.append(
            CombatTurnOrderEntry(
                actor_id=actor.actor_id,
                name=actor.name,
                team=actor.team.value,
                life_state=actor.life_state.value,
                active=not session.ended and index == session.turn_index,
                acted=index < session.turn_index,
            )
        )
    return tuple(entries)


def _reaction_options(session: ManualCombatSession) -> tuple[CombatReactionOption, ...]:
    if session.pending_enemy_intent is None:
        return ()
    options: list[CombatReactionOption] = []
    for index, reaction in enumerate(legal_reaction_options(session), start=1):
        options.append(
            CombatReactionOption(
                action=ScreenAction(
                    str(index),
                    _reaction_label(reaction.kind, reaction.actor_name),
                    reaction.reaction_id,
                    (reaction.kind, reaction.actor_name.lower().replace(" ", "_")),
                    description=reaction.summary,
                    kind=ScreenActionKind.COMBAT,
                    risk=ScreenActionRisk.COSTLY if reaction.cost else ScreenActionRisk.LOW,
                    cost=f"{reaction.cost} Effort" if reaction.cost else "",
                    preview=join_detail(
                        f"{reaction.actor_name} reacts.",
                        reaction.summary,
                    ),
                    result_hint="Spend the listed Effort to interrupt or soften the enemy action.",
                ),
                reaction_id=reaction.reaction_id,
                kind=reaction.kind,
                actor_id=reaction.actor_id,
                actor_name=reaction.actor_name,
                cost=reaction.cost,
                summary=reaction.summary,
            )
        )
    options.append(
        CombatReactionOption(
            action=ScreenAction(
                str(len(options) + 1),
                "Skip Reaction",
                "skip",
                ("s", "none", "wait"),
                default=not options,
                description="Let the enemy action resolve normally.",
                kind=ScreenActionKind.COMBAT,
                risk=ScreenActionRisk.RISKY,
                preview="Skip protection and let the enemy action resolve normally.",
                result_hint="The threatened target takes the full pending action if it lands.",
            ),
            reaction_id=None,
            kind="skip",
            actor_id=None,
            actor_name="",
            cost=0,
            summary="Let the enemy action resolve normally.",
        )
    )
    return tuple(options)


def _reaction_label(kind: str, actor_name: str) -> str:
    labels = {
        "watchman_intercede": "Keep Watch",
        "cutpurse_evade": "Slip Away",
        "field_surgeon_stabilize": "Field Dress",
        "scribe_disrupt": "Annotate",
    }
    return f"{labels.get(kind, kind.replace('_', ' ').title())}: {actor_name}"


def _target_sort_key(
    session: ManualCombatSession,
    target_id: str,
) -> tuple[int, str]:
    target = session.state.actor(target_id)
    slot = session.state.formation_for(target.team).slot_of(target_id)
    slot_priority = {
        FormationSlot.FRONT_LEFT: 0,
        FormationSlot.FRONT_RIGHT: 1,
        FormationSlot.BACK_LEFT: 2,
        FormationSlot.BACK_RIGHT: 3,
    }
    if slot is None:
        return 99, target.name
    return slot_priority.get(slot, 99), target.name


def _combatant_view(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    combatant: Combatant,
    *,
    acting: bool = False,
) -> CombatActorView:
    slot = session.state.formation_for(combatant.team).slot_of(combatant.actor_id)
    art_asset = _combatant_art_asset(definitions, combatant)
    return CombatActorView(
        actor_id=combatant.actor_id,
        name=combatant.name,
        team=combatant.team.value,
        slot=slot.value if slot is not None else "-",
        hp=combatant.hp,
        max_hp=combatant.max_hp,
        effort=combatant.effort,
        max_effort=combatant.max_effort,
        mortal_wounds=combatant.mortal_wounds,
        morale=combatant.morale.name.title(),
        strain=combatant.strain.name.title(),
        tags=tuple(sorted(tag.name.title() for tag in combatant.tags)),
        life_state=combatant.life_state.value,
        personal_quirk=_trait_label(combatant.personal_quirk),
        quirks=tuple(combatant.quirks),
        strain_marks=tuple(
            _trait_label(mark.value)
            for mark in sorted(combatant.strain_marks, key=lambda mark: mark.value)
        ),
        acting=acting,
        class_id=combatant.class_id,
        display_name=_art_display_name(art_asset),
        glyph=_art_glyph(art_asset),
        mini_lines=_art_mini_lines(art_asset),
        mini_frames=_art_mini_frames(art_asset),
        art_lines=_art_lines(art_asset),
        art_frames=_art_frames(art_asset),
        art_frame_holds=_art_frame_holds(art_asset),
        art_frame_impacts=_art_frame_impacts(art_asset),
    )


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


def build_hero_portrait_view(
    hero: HeroState,
    definitions: GameDefinitions,
    *,
    slot: str = "",
) -> CombatActorView:
    stats = effective_hero_stats(hero, definitions)
    art_asset = _hero_art_asset(definitions, hero.hero_id, hero.class_id)
    slot_label = slot or hero.formation_slot.value
    return CombatActorView(
        actor_id=hero.hero_id,
        name=hero.name,
        team="hero",
        slot=slot_label,
        hp=hero.hp,
        max_hp=stats.max_hp,
        effort=hero.effort,
        max_effort=stats.max_effort,
        mortal_wounds=hero.mortal_wounds,
        morale=hero.morale.name.title(),
        strain=hero.strain.name.title(),
        tags=(),
        life_state=hero.life_state.value,
        class_id=hero.class_id,
        display_name=_art_display_name(art_asset),
        glyph=_art_glyph(art_asset),
        mini_lines=_art_mini_lines(art_asset),
        mini_frames=_art_mini_frames(art_asset),
        art_lines=_art_lines(art_asset) or _derive_mini_lines(_art_mini_lines(art_asset)),
        art_frames=_art_frames(art_asset),
        art_frame_holds=_art_frame_holds(art_asset),
        art_frame_impacts=_art_frame_impacts(art_asset),
        strain_marks=tuple(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda mark: mark.value)
        ),
    )


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


_WORLD_LOCATION_FALLBACKS: dict[str, str] = {
    "Haven Town": "haven",
    "Haven East Gate": "old_road",
    "Old Road": "old_road",
    "Shallow Cave Entrance": "shallow_cave",
    "Cave Mouth": "shallow_cave",
    "Shallow Cave": "shallow_cave",
    "Shallow Cave Breach": "shallow_cave_breach",
    "Pandora's Maze Depth 1": "pandoras_maze_depth_1",
}


def _current_world_location_id(
    company: CompanyState,
    definitions: GameDefinitions,
) -> str:
    raw_location_id = str(company.town_state.get("location_id") or "")
    if raw_location_id in definitions.locations:
        return raw_location_id
    location_name = str(company.town_state.get("location") or "")
    for location in definitions.locations.values():
        if location.name == location_name:
            return location.id
    fallback = _WORLD_LOCATION_FALLBACKS.get(location_name)
    if fallback in definitions.locations:
        return fallback
    return definitions.world.starting_settlement


def _current_regional_node_id(
    company: CompanyState,
    definitions: GameDefinitions,
) -> str:
    node_id = get_regional_node_id(company)
    nodes = regional_overworld_nodes(definitions)
    if node_id in nodes:
        return node_id
    location_id = _current_world_location_id(company, definitions)
    if location_id == CAVE_REGIONAL_ID:
        return REGIONAL_CAVE_ANCHOR_NODE_ID
    return REGIONAL_EAST_GATE_NODE_ID


def _known_world_location_ids(
    company: CompanyState,
    definitions: GameDefinitions,
    current_location_id: str,
) -> set[str]:
    known = {definitions.world.starting_settlement, current_location_id}
    known.update(
        location_id for location_id, memory in company.world_memory.items() if memory.visited
    )
    known.update(
        contract.location_id
        for contract in definitions.contracts.values()
        if contract.id in company.active_contract_ids
        or contract.id in company.completed_contract_ids
        or (
            contract.id in company.contract_records
            and company.contract_records[contract.id].state
            in {"active", "completed", "repeatable_completed"}
        )
    )
    if "shallow_cave" in company.known_route_ids or company.flags.get(
        "shallow_cave_discovered",
        False,
    ):
        known.add("shallow_cave")
    known.update(
        breach_id for breach_id in company.known_breaches if breach_id in definitions.locations
    )
    if company.flags.get("maze_leak_confirmed", False):
        known.add("shallow_cave_breach")
    if "maze_depth_1_scouted" in company.expedition_history:
        known.add("pandoras_maze_depth_1")
    return {location_id for location_id in known if location_id in definitions.locations}


def _travel_destinations(
    company: CompanyState,
    definitions: GameDefinitions,
    current_location_id: str,
) -> tuple[WorldDestinationView, ...]:
    if current_location_id == definitions.world.starting_settlement:
        destinations = [
            _destination(
                definitions,
                "old_road",
                "Old Road",
                _join_detail(
                    "Leave through Haven East Gate and travel the wilderness route.",
                    "Cost: 1 ration when available.",
                    _contract_context(company, definitions, "old_road"),
                ),
                route_kind="road",
            )
        ]
        if "shallow_cave" in company.known_route_ids:
            destinations.append(
                _destination(
                    definitions,
                    "shallow_cave",
                    "Charted Approach: Shallow Cave",
                    (
                        "Follow the charted road to the cave mouth. This summarizes "
                        "the solved approach and spends normal route supplies."
                    ),
                    route_kind="charted_approach",
                )
            )
        return tuple(destinations)
    if current_location_id == "shallow_cave":
        return (
            _destination(
                definitions,
                definitions.world.starting_settlement,
                "Haven Town",
                "Return along the charted road to Haven.",
                route_kind="return_road",
            ),
        )
    return (
        _destination(
            definitions,
            definitions.world.starting_settlement,
            "Haven Town",
            "Return to the charter town.",
        ),
    )


def _location_contract_lines(
    company: CompanyState,
    definitions: GameDefinitions,
    location_id: str,
) -> tuple[str, ...]:
    lines: list[str] = []
    for contract in sorted(
        (
            contract
            for contract in definitions.contracts.values()
            if contract.location_id == location_id
            and (
                contract.id in company.active_contract_ids
                or contract.id in company.completed_contract_ids
                or contract.id in company.contract_records
            )
        ),
        key=lambda contract: contract.name,
    ):
        lines.append(f"{_contract_state_label(company, contract.id)}: {contract.name}")
    return tuple(lines)


def _contract_state_label(company: CompanyState, contract_id: str) -> str:
    record = company.contract_records.get(contract_id)
    if contract_id in company.active_contract_ids or (
        record is not None and record.state == "active"
    ):
        return "Active"
    if contract_id in company.completed_contract_ids or (
        record is not None and record.state in {"completed", "repeatable_completed"}
    ):
        return "Completed"
    return "Known"


def _destination(
    definitions: GameDefinitions,
    location_id: str,
    label: str,
    description: str,
    *,
    route_kind: str = "travel",
) -> WorldDestinationView:
    location = definitions.locations[location_id]
    return WorldDestinationView(
        destination_id=location.id,
        location_id=location.id,
        label=label,
        description=description,
        route_kind=route_kind,
    )


def _ordered_route_ids(
    node_ids: Sequence[str],
    *,
    nodes: Mapping[str, ExpeditionNodeDefinition],
    current_node_id: str,
) -> list[str]:
    current = nodes[current_node_id]
    if current.position is None:
        return list(node_ids)
    indexed_ids = list(enumerate(node_ids))
    indexed_ids.sort(
        key=lambda item: _route_sort_key(
            nodes[item[1]],
            current=current,
            authored_index=item[0],
        )
    )
    return [node_id for _index, node_id in indexed_ids]


def _route_sort_key(
    node: ExpeditionNodeDefinition,
    *,
    current: ExpeditionNodeDefinition,
    authored_index: int,
) -> tuple[int, int, int, int]:
    if node.position is None or current.position is None:
        return (1, 99, 99, authored_index)
    dx = node.position[0] - current.position[0]
    dy = node.position[1] - current.position[1]
    return (0, _compass_rank(dx, dy), abs(dx) + abs(dy), authored_index)


def _compass_rank(dx: int, dy: int) -> int:
    if dx == 0 and dy < 0:
        return 0
    if dx > 0 and dy == 0:
        return 1
    if dx == 0 and dy > 0:
        return 2
    if dx < 0 and dy == 0:
        return 3
    if dy < 0:
        return 0
    if dx > 0:
        return 1
    if dy > 0:
        return 2
    if dx < 0:
        return 3
    return 4


def _contract_context(
    company: CompanyState,
    definitions: GameDefinitions,
    location_id: str,
) -> str:
    contracts = [
        contract
        for contract in definitions.contracts.values()
        if contract.id in company.active_contract_ids and contract.location_id == location_id
    ]
    if not contracts:
        return ""
    return "Contract: " + "; ".join(contract.name for contract in contracts)


def _world_breadcrumb(location: Any) -> str:
    if location.id == "haven":
        return "World > Haven"
    return f"World > {location.name}"


def _hero_entry(
    hero: HeroState,
    memories: Sequence[HeroMemoryEntry] = (),
    definitions: GameDefinitions | None = None,
) -> HeroListEntry:
    stats = effective_hero_stats(hero, definitions)
    gear = (
        definitions.gear.get(hero.equipped_gear_id)
        if (definitions is not None and hero.equipped_gear_id is not None)
        else None
    )
    return HeroListEntry(
        hero_id=hero.hero_id,
        name=hero.name,
        class_id=hero.class_id,
        slot=hero.formation_slot.value,
        hp=hero.hp,
        max_hp=stats.max_hp,
        effort=hero.effort,
        max_effort=stats.max_effort,
        mortal_wounds=hero.mortal_wounds,
        morale=hero.morale.name.title(),
        strain=hero.strain.name.title(),
        life_state=hero.life_state.value,
        personal_quirk=_trait_label(hero.personal_quirk),
        quirks=tuple(_trait_label(quirk) for quirk in hero.quirks),
        strain_marks=tuple(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda mark: mark.value)
        ),
        memory_count=len(memories),
        latest_memory=memories[-1].summary if memories else "",
        equipped_gear=gear.name if gear is not None else "",
        stat_bonus=_stat_bonus_summary(hero, stats),
    )


def _hero_memories(
    company: CompanyState,
    hero_id: str,
) -> tuple[HeroMemoryEntry, ...]:
    return tuple(memory for memory in company.hero_memories if memory.hero_id == hero_id)


def _sheet_trait(
    trait_id: str | None,
    definitions: GameDefinitions,
    *,
    kind: str,
    stability: str = "",
) -> HeroSheetTraitView | None:
    if not trait_id:
        return None
    trait = definitions.traits.get(trait_id)
    return HeroSheetTraitView(
        trait_id=trait_id,
        name=trait.name if trait is not None else _trait_label(trait_id),
        kind=kind,
        description=trait.description if trait is not None else "",
        positive_text=trait.positive_text if trait is not None else "",
        negative_text=trait.negative_text if trait is not None else "",
        stability=stability,
    )


def _hero_roster_state(
    hero_id: str,
    active_ids: set[str],
    reserve_ids: set[str],
) -> str:
    if hero_id in active_ids:
        return "Active"
    if hero_id in reserve_ids:
        return "Reserve"
    return "Memorial"


def _signal_label(signal_id: str) -> str:
    label = {
        "killing_blow": "Killing Blows",
        "marked_execution": "Marked Executions",
        "relic_greed": "Relic Greed",
        "maze_thread": "Maze Thread",
        "breach_witness": "Breach Witness",
        "field_treatment": "Field Treatment",
        "morale_rally": "Morale Rally",
        "shaken_survival": "Shaken Survival",
        "downed_survival": "Downed Survival",
        "broken_survival": "Broken Survival",
        "ally_downed_witnessed": "Ally Downed Witnesses",
        "frost_shock": "Frost Shock",
        "tag:combat": "combat",
        "tag:kill": "killing",
        "tag:maze": "Maze exposure",
        "tag:greed": "greed",
        "tag:loot": "loot",
        "tag:route": "route pressure",
        "tag:breach": "breach exposure",
        "tag:marked": "marked targets",
        "tag:support": "support",
        "tag:morale": "morale pressure",
        "tag:healing": "healing",
        "tag:downed": "downed allies",
        "tag:ally": "ally pressure",
        "tag:survival": "survival",
        "tag:frozen": "freezing shock",
        "tag:shock": "shock",
        "tag:final_kill": "fight-ending kills",
        "tag:basic": "basic finishes",
        "tag:steady": "calm killing",
        "tag:shaken": "desperate killing",
        "tag:fractured": "fractured-line kills",
        "tag:wounded": "wounded killing",
        "tag:low_hp": "finisher targets",
        "tag:effort_kill": "costly kills",
        "tag:boss": "boss kills",
    }.get(signal_id)
    if label is not None:
        return label
    if signal_id.startswith("tag:"):
        return _trait_label(signal_id.removeprefix("tag:")).lower()
    return _trait_label(signal_id)


def _player_memory_summary(summary: str) -> str:
    if not summary:
        return ""
    replacements = {
        "shallow_cave_breach": "Shallow Cave Breach",
    }
    for raw, label in replacements.items():
        summary = summary.replace(raw, label)
    return summary


def _item_deltas(
    start: Mapping[str, int],
    end: Mapping[str, int],
) -> tuple[tuple[str, int, int, int], ...]:
    item_ids = sorted(set(start) | set(end))
    return tuple(
        (
            item_id,
            start.get(item_id, 0),
            end.get(item_id, 0),
            end.get(item_id, 0) - start.get(item_id, 0),
        )
        for item_id in item_ids
        if start.get(item_id, 0) != end.get(item_id, 0)
    )


def _report_change_lines(report: ExpeditionReportState) -> tuple[str, ...]:
    lines: list[str] = []
    if report.end_reputation != report.start_reputation:
        lines.append(
            "Reputation "
            f"{report.start_reputation}->{report.end_reputation} "
            f"({_signed(report.end_reputation - report.start_reputation)})."
        )
    if report.end_coin != report.start_coin:
        lines.append(
            "Coin "
            f"{report.start_coin}->{report.end_coin} "
            f"({_signed(report.end_coin - report.start_coin)})."
        )
    for signal in report.event_signals:
        if signal.kind in {
            "breach_discovered",
            "contract_completed",
            "known_route_unlocked",
            "maze_route_collapsed",
            "maze_frontier_opened",
        }:
            lines.append(signal.message)
    if report.breaches_discovered:
        lines.append("Breach memory updated: " + ", ".join(report.breaches_discovered) + ".")
    if report.room_actions:
        lines.append(f"Ledger recorded {len(report.room_actions)} room action(s).")
    if report.gear:
        lines.append(
            "Armory updated: "
            + ", ".join(f"{gear_id} x{quantity}" for gear_id, quantity in report.gear.items())
            + "."
        )
    if report.hero_outcomes:
        lines.append(f"Roster report updated for {len(report.hero_outcomes)} participant(s).")
    return tuple(_dedupe(lines))


def _arrival_change_lines(
    company: CompanyState,
    report: ExpeditionReportState,
) -> tuple[str, ...]:
    lines: list[str] = []
    for signal in report.event_signals:
        if signal.kind in {
            "breach_discovered",
            "contract_completed",
            "known_route_unlocked",
            "maze_route_collapsed",
            "maze_frontier_opened",
        }:
            lines.append(signal.message)
    if report.breaches_discovered:
        lines.append("Breach recorded: " + ", ".join(report.breaches_discovered) + ".")
    if report.reputation_gained:
        lines.append(f"Reputation +{report.reputation_gained}.")
    if report.coin_gained:
        lines.append(f"Coin +{report.coin_gained}.")
    if report.gear:
        lines.append(
            "Gear added: "
            + ", ".join(f"{gear_id} x{quantity}" for gear_id, quantity in report.gear.items())
            + "."
        )
    living = living_roster(company)
    wounded_count = sum(1 for hero in living if hero.hp < hero.max_hp or hero.mortal_wounds > 0)
    downed_count = sum(1 for hero in living if hero.life_state == LifeState.DOWNED)
    if wounded_count:
        lines.append(f"Wounded: {wounded_count}.")
    if downed_count:
        lines.append(f"Downed: {downed_count}.")
    if company.deceased_heroes:
        lines.append(f"Memorial: {len(company.deceased_heroes)}.")
    if not lines:
        lines.append("Company record filed.")
    return tuple(_dedupe(lines))


def _upgrade_effect_summary(effects: Any) -> str:
    pieces: list[str] = []
    if effects.roster_cap_bonus:
        pieces.append(f"Roster cap +{effects.roster_cap_bonus}")
    if effects.recovery_cost_delta:
        pieces.append(f"Recovery cost {effects.recovery_cost_delta:+d}")
    for supply_id, delta in sorted(effects.supply_cost_deltas.items()):
        pieces.append(f"{supply_id.replace('_', ' ').title()} cost {delta:+d}")
    return _join_detail(*pieces)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _signed(value: int) -> str:
    return f"+{value}" if value >= 0 else str(value)


def _hero_outcome_line(outcome: HeroReportOutcome) -> str:
    pieces = [
        f"{outcome.hero_name} ({outcome.class_id})",
        outcome.status,
        f"HP {outcome.start_hp}->{outcome.end_hp}/{outcome.max_hp}",
    ]
    if outcome.mortal_wounds_delta:
        sign = "+" if outcome.mortal_wounds_delta > 0 else ""
        pieces.append(
            f"Mortal Wounds {sign}{outcome.mortal_wounds_delta} "
            f"({mortal_wound_badge(outcome.end_mortal_wounds)})"
        )
    else:
        pieces.append(mortal_wound_badge(outcome.end_mortal_wounds))
    return _join_detail(*pieces)


def _hero_condition(
    hero: HeroState,
    definitions: GameDefinitions | None = None,
) -> str:
    stats = effective_hero_stats(hero, definitions)
    statuses = ", ".join(_life_state_labels(hero.life_state.value))
    pieces = [
        f"{hero.hp}/{stats.max_hp} HP",
        f"{hero.effort}/{stats.max_effort} Effort",
        mortal_wound_badge(hero.mortal_wounds),
    ]
    if statuses != "ready":
        pieces.append(statuses)
    if hero.strain != hero.strain.STEADY:
        pieces.append(f"Strain {hero.strain.name.title()}")
    if hero.strain_marks:
        mark_text = ", ".join(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda item: item.value)
        )
        pieces.append(f"Marks {mark_text}")
    return ", ".join(pieces)


def _stat_bonus_summary(hero: HeroState, stats: Any) -> str:
    pieces: list[str] = []
    if stats.max_hp != hero.max_hp:
        pieces.append(f"Max HP +{stats.max_hp - hero.max_hp}")
    if stats.max_effort != hero.max_effort:
        pieces.append(f"Max Effort +{stats.max_effort - hero.max_effort}")
    if stats.accuracy != hero.accuracy:
        pieces.append(f"Accuracy +{stats.accuracy - hero.accuracy}")
    if stats.damage != hero.damage:
        pieces.append(f"Damage +{stats.damage - hero.damage}")
    return ", ".join(pieces)


def _trait_label(trait_id: str | None) -> str:
    if not trait_id:
        return ""
    return trait_id.replace("_", " ").title()


def _life_state_labels(life_state: str) -> tuple[str, ...]:
    if life_state == LifeState.ALIVE.value:
        return ("ready",)
    return (life_state,)


def _skill_description(
    intent: str,
    attack_type: str,
    amount_label: str,
    target_count: int,
) -> str:
    if intent == "heal":
        return f"heals up to {amount_label} HP, {target_count} allies"
    return f"{attack_type}, {target_count} targets"


def _skill_intent(tags: Sequence[str]) -> str:
    tag_set = set(tags)
    if "treatment" in tag_set or "heal" in tag_set or "support" in tag_set:
        return "heal"
    if tag_set & {"debuff", "control", "horror", "status"}:
        return "debuff"
    return "attack"


def _frontier_preview_exit_view(
    preview: FrontierExitPreview,
    *,
    current_map_id: str,
    map_x: int | None = None,
    map_y: int | None = None,
    direction: str = "",
) -> DungeonMapNodeView:
    return DungeonMapNodeView(
        node_id=preview.exit_id,
        name=preview.label,
        map_id=current_map_id,
        node_type=preview.node_type,
        status="unstable",
        exit_node_ids=(),
        map_x=map_x,
        map_y=map_y,
        direction=direction,
        location_id="",
        memory_summary="Unstable passage — room not yet mapped.",
        memory_notes=(),
        inventory_rewards=(),
        supply_rewards=(),
        reputation_reward=0,
        coin_reward=0,
        inventory_requirements=(),
        supply_costs=(),
        action_summaries=(),
        current=False,
        known=True,
        visited=False,
        cleared=False,
        safe_return=False,
    )


def _dungeon_quest_marker(
    node: ExpeditionNodeDefinition,
    *,
    quest_target_node_ids: set[str],
) -> bool:
    return node.id in quest_target_node_ids


def _quest_target_node_ids(
    company: CompanyState,
    session: Any,
) -> set[str]:
    active_contract_ids = set(company.active_contract_ids)
    if not active_contract_ids:
        return {QUEST_PICKUP_NODE_ID}
    if BLACKWOOD_ROAD_CHARTER_ID in active_contract_ids:
        cave_memory = company.dungeon_memory.get("shallow_cave")
        visited_ids = set(cave_memory.visited_node_ids if cave_memory else ())
        cleared_ids = set(cave_memory.cleared_node_ids if cave_memory else ())
        if BLACKWOOD_CAVE_TARGET_NODE_ID not in visited_ids:
            return {BLACKWOOD_CAVE_TARGET_NODE_ID}
        if BLACKWOOD_BOSS_TARGET_NODE_ID not in cleared_ids:
            return {BLACKWOOD_BOSS_TARGET_NODE_ID}
        return set()
    if _active_hunt_contract_ids(active_contract_ids):
        generated = getattr(session, "generated_dungeon", None)
        if generated is not None and not getattr(generated, "collapsed", False):
            hunt_node_id = next(
                (node.id for node in generated.nodes if node.id.endswith("_hunt_lair")),
                "",
            )
            if hunt_node_id and hunt_node_id not in generated.cleared_node_ids:
                return {hunt_node_id}
        return {BREACH_TARGET_NODE_ID}
    if _active_scout_contract_ids(active_contract_ids):
        return {BREACH_TARGET_NODE_ID}
    return set()


def _active_hunt_contract_ids(active_contract_ids: set[str]) -> set[str]:
    return active_contract_ids & {
        BREACH_STALKER_HUNT_ID,
        GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
    }


def _active_scout_contract_ids(active_contract_ids: set[str]) -> set[str]:
    return active_contract_ids & {
        SHALLOW_CAVE_BREACH_SCOUT_ID,
        GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
    }


def _dungeon_node_view(
    node_id: str,
    nodes: Mapping[str, ExpeditionNodeDefinition],
    *,
    company: CompanyState,
    session: Any,
    current_id: str,
    visited_ids: set[str],
    cleared_ids: set[str],
    known_ids: set[str],
    completed_action_ids: set[str],
    exit_node_ids: Sequence[str] = (),
    quest_target_node_ids: set[str] | None = None,
) -> DungeonMapNodeView:
    node = nodes[node_id]
    visited = node_id in visited_ids
    cleared = node_id in cleared_ids
    current = node_id == current_id
    known = node_id in known_ids
    if current:
        status = "current"
    elif cleared:
        status = "cleared"
    elif visited:
        status = "visited"
    else:
        status = "known"
    return DungeonMapNodeView(
        node_id=node_id,
        name=node.name,
        map_id=node.map_id,
        node_type=node.node_type.value,
        status=status,
        exit_node_ids=tuple(exit_node_ids),
        map_x=node.position[0] if node.position is not None else None,
        map_y=node.position[1] if node.position is not None else None,
        direction=_route_direction(nodes[current_id], node),
        location_id=world_location_id_for_node(node),
        memory_summary=_node_memory_summary(company, node),
        memory_notes=_node_memory_notes(company, session, node),
        inventory_rewards=_node_inventory_rewards(node),
        supply_rewards=_node_supply_rewards(node),
        reputation_reward=_node_reputation_reward(node),
        coin_reward=_node_coin_reward(node),
        inventory_requirements=_node_inventory_requirements(node),
        supply_costs=_node_supply_costs(node),
        action_summaries=_node_action_summaries(
            node,
            nodes,
            completed_action_ids=completed_action_ids,
        ),
        current=current,
        known=known,
        visited=visited,
        cleared=cleared,
        safe_return=node.safe_return,
        quest_marker=_dungeon_quest_marker(
            node,
            quest_target_node_ids=quest_target_node_ids or set(),
        ),
    )


def _node_memory_summary(
    company: CompanyState,
    node: ExpeditionNodeDefinition,
) -> str:
    location_id = world_location_id_for_node(node)
    memory = company.world_memory.get(location_id)
    if memory is None:
        return ""
    pieces = [
        "discovered" if node.id in memory.discovered_node_ids else "",
        "threat cleared" if node.id in memory.cleared_threat_node_ids else "",
        f"{memory.visit_count} location visits" if memory.visit_count else "",
    ]
    if node.known_route_unlock and node.known_route_unlock in memory.unlocked_shortcut_ids:
        pieces.append("shortcut charted")
    return _join_detail(*pieces)


def _node_memory_notes(
    company: CompanyState,
    session: Any,
    node: ExpeditionNodeDefinition,
) -> tuple[str, ...]:
    notes: list[str] = []
    report = getattr(session, "report", None)
    if report is not None:
        notes.extend(
            signal.message
            for signal in getattr(report, "event_signals", ())
            if signal.node_id == node.id
        )
    notes.extend(entry.summary for entry in company.company_timeline if entry.node_id == node.id)
    notes.extend(memory.summary for memory in company.hero_memories if memory.node_id == node.id)
    return tuple(_dedupe_lines(notes)[-3:])


def _node_inventory_rewards(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    rewards = dict(node.loot)
    for action in node.actions:
        for item_id, quantity in action.loot.items():
            rewards[item_id] = rewards.get(item_id, 0) + quantity
    return tuple(sorted((item_id, quantity) for item_id, quantity in rewards.items() if quantity))


def _node_supply_rewards(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    rewards = dict(node.supply_rewards)
    for action in node.actions:
        for supply_id, quantity in action.supply_rewards.items():
            rewards[supply_id] = rewards.get(supply_id, 0) + quantity
    return tuple(
        sorted((supply_id, quantity) for supply_id, quantity in rewards.items() if quantity)
    )


def _node_reputation_reward(node: ExpeditionNodeDefinition) -> int:
    return node.reputation_reward + sum(action.reputation_reward for action in node.actions)


def _node_coin_reward(node: ExpeditionNodeDefinition) -> int:
    return node.coin_reward + sum(action.coin_reward for action in node.actions)


def _node_inventory_requirements(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    requirements: dict[str, int] = {}
    for action in node.actions:
        for item_id, quantity in action.inventory_requirements.items():
            requirements[item_id] = max(requirements.get(item_id, 0), quantity)
    return tuple(sorted(requirements.items()))


def _node_supply_costs(
    node: ExpeditionNodeDefinition,
) -> tuple[tuple[str, int], ...]:
    costs: dict[str, int] = {}
    for action in node.actions:
        for supply_id, quantity in action.supply_costs.items():
            costs[supply_id] = max(costs.get(supply_id, 0), quantity)
    return tuple(sorted(costs.items()))


def _node_action_summaries(
    node: ExpeditionNodeDefinition,
    nodes: Mapping[str, ExpeditionNodeDefinition],
    *,
    completed_action_ids: set[str],
) -> tuple[str, ...]:
    lines: list[str] = []
    for action in node.actions:
        state = "completed" if room_action_key(node.id, action.id) in completed_action_ids else ""
        reward_detail = _action_reward_detail(action)
        detail = _join_detail(
            state,
            f"needs {_quantity_detail(action.inventory_requirements)}"
            if action.inventory_requirements
            else "",
            f"costs {_quantity_detail(action.supply_costs)}" if action.supply_costs else "",
            f"yields {reward_detail}" if reward_detail else "",
            _revealed_exit_detail(action.reveal_exits, nodes),
        )
        lines.append(f"{action.label}: {detail}" if detail else action.label)
    return tuple(lines)


def _quantity_detail(values: Mapping[str, int]) -> str:
    return ", ".join(
        f"{quantity} {_display_id(item_id)}"
        for item_id, quantity in sorted(values.items())
        if quantity
    )


def _action_reward_detail(action: ExpeditionRoomActionDefinition) -> str:
    lines: list[str] = []
    if action.reputation_reward:
        lines.append(f"{action.reputation_reward} reputation")
    if action.coin_reward:
        lines.append(f"{action.coin_reward} Coin")
    lines.extend(
        f"{quantity} {_display_id(item_id)}" for item_id, quantity in sorted(action.loot.items())
    )
    lines.extend(
        f"{quantity} {_display_id(supply_id)}"
        for supply_id, quantity in sorted(action.supply_rewards.items())
    )
    return ", ".join(lines)


def _revealed_exit_detail(
    exit_node_ids: Sequence[str],
    nodes: Mapping[str, ExpeditionNodeDefinition],
) -> str:
    if not exit_node_ids:
        return ""
    labels = [nodes[exit_id].name if exit_id in nodes else exit_id for exit_id in exit_node_ids]
    return f"reveals {', '.join(labels)}"


def _dedupe_lines(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _display_id(value: str) -> str:
    return value.replace("_", " ")


def _direction_between_positions(
    from_position: tuple[int, int],
    to_position: tuple[int, int],
) -> str:
    dx = to_position[0] - from_position[0]
    dy = to_position[1] - from_position[1]
    if dx == 0 and dy == 0:
        return ""
    if dx == 0 and dy < 0:
        return "North"
    if dx > 0 and dy == 0:
        return "East"
    if dx == 0 and dy > 0:
        return "South"
    if dx < 0 and dy == 0:
        return "West"
    vertical = "North" if dy < 0 else "South" if dy > 0 else ""
    horizontal = "East" if dx > 0 else "West" if dx < 0 else ""
    return " ".join(part for part in (vertical, horizontal) if part)


def _route_direction(
    current: ExpeditionNodeDefinition,
    node: ExpeditionNodeDefinition,
) -> str:
    if current.position is None or node.position is None or current.id == node.id:
        return ""
    return _direction_between_positions(current.position, node.position)


def _room_action_visible(
    company: CompanyState,
    action: ExpeditionRoomActionDefinition,
) -> bool:
    if not action.requires_active_contracts:
        return True
    return all(
        contract_id in company.active_contract_ids
        for contract_id in action.requires_active_contracts
    )


def _dungeon_action_view(
    action: ExpeditionRoomActionDefinition,
    company: CompanyState,
    *,
    current_node_id: str,
    current_cleared: bool,
    completed_action_ids: set[str],
) -> DungeonActionView:
    action_key = room_action_key(current_node_id, action.id)
    if action.once and action_key in completed_action_ids:
        state = "completed"
    elif action.requires_cleared and not current_cleared:
        state = "requires cleared room"
    elif any(
        company.inventory.get(item_id, 0) < quantity
        for item_id, quantity in action.inventory_requirements.items()
    ):
        state = "missing item"
    elif any(
        company.supplies.get(supply_id, 0) < quantity
        for supply_id, quantity in action.supply_costs.items()
    ):
        state = "missing supplies"
    else:
        state = "available"
    return DungeonActionView(
        action_id=action.id,
        label=action.label,
        description=action.description,
        state=state,
        requirements=tuple(sorted(action.inventory_requirements.items())),
        cost=tuple(sorted(action.supply_costs.items())),
        reward=tuple(_action_reward_lines(action)),
    )


def _join_detail(*pieces: str) -> str:
    return "  |  ".join(piece.strip() for piece in pieces if piece.strip())


def _action_reward_lines(action: ExpeditionRoomActionDefinition) -> list[str]:
    lines: list[str] = []
    if action.reputation_reward:
        lines.append(f"{action.reputation_reward} reputation")
    if action.coin_reward:
        lines.append(f"{action.coin_reward} Coin")
    lines.extend(f"{quantity} {item_id}" for item_id, quantity in sorted(action.loot.items()))
    lines.extend(
        f"{quantity} {supply_id}" for supply_id, quantity in sorted(action.supply_rewards.items())
    )
    return lines


def _node_name(
    nodes: Mapping[str, ExpeditionNodeDefinition],
    node_id: str,
    room_names: Mapping[str, str] | None = None,
) -> str:
    node = nodes.get(node_id)
    if node is None:
        if room_names is not None and node_id in room_names:
            return room_names[node_id]
        return node_id.replace("_", " ").title()
    return node.name


def _room_action_name(nodes: Mapping[str, ExpeditionNodeDefinition], action_key: str) -> str:
    node_id, _separator, action_id = action_key.partition(":")
    node = nodes.get(node_id)
    if node is None:
        return action_key
    action = next((candidate for candidate in node.actions if candidate.id == action_id), None)
    if action is None:
        return action_key
    return f"{node.name}: {action.label}"
