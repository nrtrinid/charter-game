"""One-shot splitter for game.app.views monolith into package modules."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/game/app/views.py"
PKG = ROOT / "src/game/app/views"

COMMON_HEADER = '''\
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

'''

MODULES: dict[str, set[str]] = {
    "constants": {
        "EMPTY_SLOT_VALUE",
        "QUEST_PICKUP_NODE_ID",
        "BLACKWOOD_CAVE_TARGET_NODE_ID",
        "BLACKWOOD_BOSS_TARGET_NODE_ID",
        "BREACH_TARGET_NODE_ID",
    },
    "formatting": {"_dedupe", "_signed", "_dedupe_lines", "_join_detail"},
    "art": {
        "_combatant_art_lines",
        "_combatant_art_frames",
        "_combatant_art_frame_impacts",
        "_art_frame_impacts",
        "_art_frame_holds",
        "_combatant_art_asset",
        "_hero_art_asset",
        "_dungeon_node_art_asset",
        "_generated_maze_art_key",
        "_art_lines",
        "_art_display_name",
        "_art_glyph",
        "_art_mini_lines",
        "_art_mini_frames",
        "_derive_mini_lines",
        "_art_frames",
    },
    "combat": {
        "CombatActorView",
        "CombatSkillOption",
        "CombatTargetOption",
        "CombatMoveOption",
        "CombatEnemyIntentView",
        "CombatReactionOption",
        "CombatTurnOrderEntry",
        "CombatView",
        "build_combat_view",
        "_slot_ordered_combatants",
        "_formation_preview_slots",
        "_move_preview_slots",
        "_formation_slot_summaries",
        "_enemy_intent_view",
        "_turn_order_entries",
        "_reaction_options",
        "_reaction_label",
        "_target_sort_key",
        "_combatant_view",
        "_slot_display",
    },
    "regional": {
        "WorldLocationView",
        "WorldDestinationView",
        "WorldView",
        "ArrivalBriefView",
        "RegionalMapView",
        "ShellStatusView",
        "build_shell_status",
        "build_world_view",
        "build_regional_map_view",
        "build_regional_render_view",
        "build_regional_arrival_context",
        "_world_location_view",
        "_current_world_location_id",
        "_current_regional_node_id",
        "_known_world_location_ids",
        "_travel_destinations",
        "_location_contract_lines",
        "_contract_state_label",
        "_destination",
        "_contract_context",
        "_world_breadcrumb",
        "_arrival_change_lines",
    },
    "town": {
        "ContractBoardEntryView",
        "TownUpgradeView",
        "TownDashboardView",
        "SupplyShopView",
        "RelicBrokerView",
        "DeepSurgeryCandidateView",
        "DeepSurgeryView",
        "build_town_dashboard",
        "_contract_board_entries",
        "_contract_board_is_visible",
        "_contract_board_entry",
        "_upgrade_entries",
        "build_supply_shop_view",
        "build_relic_broker_view",
        "build_deep_surgery_view",
        "_upgrade_effect_summary",
    },
    "hero": {
        "HeroListEntry",
        "HeroSheetTraitView",
        "HeroSheetFreshMemoryView",
        "HeroSheetMemoryEntryView",
        "HeroSheetSignalView",
        "HeroSheetView",
        "RosterSectionView",
        "MemorialEntryView",
        "FormationSlotView",
        "FormationView",
        "GearItemView",
        "GearHeroView",
        "GearInventoryView",
        "RecruitOfferView",
        "RecruitOffersView",
        "build_roster_sections",
        "build_memorial_entries",
        "build_formation_view",
        "build_gear_inventory_view",
        "build_hero_sheet_view",
        "_gear_item_views",
        "_gear_hero_views",
        "build_recruit_offers_view",
        "preview_assign_hero",
        "_formation_protection_line",
        "_hero_abnormal_status",
        "hero_protection_line",
        "_hero_entry",
        "_hero_memories",
        "_sheet_trait",
        "_hero_roster_state",
        "_signal_label",
        "_player_memory_summary",
        "_hero_condition",
        "_stat_bonus_summary",
        "_trait_label",
        "_life_state_labels",
        "_skill_description",
        "_skill_intent",
        "build_hero_portrait_view",
        "_TownProtector",
        "_town_protectors",
    },
    "dungeon": {
        "DungeonRoomView",
        "DungeonMapNodeView",
        "DungeonActionView",
        "DungeonView",
        "ExpeditionReportView",
        "build_dungeon_view",
        "build_expedition_report_view",
        "_frontier_preview_exit_view",
        "_dungeon_quest_marker",
        "_quest_target_node_ids",
        "_active_hunt_contract_ids",
        "_active_scout_contract_ids",
        "_dungeon_node_view",
        "_node_memory_summary",
        "_node_memory_notes",
        "_node_inventory_rewards",
        "_node_supply_rewards",
        "_node_reputation_reward",
        "_node_coin_reward",
        "_node_inventory_requirements",
        "_node_supply_costs",
        "_node_action_summaries",
        "_quantity_detail",
        "_action_reward_detail",
        "_revealed_exit_detail",
        "_display_id",
        "_direction_between_positions",
        "_route_direction",
        "_room_action_visible",
        "_dungeon_action_view",
        "_action_reward_lines",
        "_node_name",
        "_room_action_name",
        "_ordered_route_ids",
        "_route_sort_key",
        "_compass_rank",
        "_item_deltas",
        "_report_change_lines",
        "_hero_outcome_line",
    },
}

CROSS_IMPORTS: dict[str, list[str]] = {
    "formatting": [],
    "constants": [],
    "art": [],
    "dungeon": [
        "from game.app.views.art import _art_lines, _dungeon_node_art_asset",
        "from game.app.views.constants import (",
        "    BLACKWOOD_BOSS_TARGET_NODE_ID,",
        "    BLACKWOOD_CAVE_TARGET_NODE_ID,",
        "    BREACH_TARGET_NODE_ID,",
        "    QUEST_PICKUP_NODE_ID,",
        ")",
        "from game.app.views.formatting import _dedupe, _dedupe_lines, _join_detail, _signed",
    ],
    "combat": [
        "from game.app.views.art import (",
        "    _art_display_name,",
        "    _art_frame_holds,",
        "    _art_frame_impacts,",
        "    _art_frames,",
        "    _art_glyph,",
        "    _art_lines,",
        "    _art_mini_frames,",
        "    _art_mini_lines,",
        "    _combatant_art_asset,",
        ")",
        "from game.app.views.formatting import _join_detail",
        "from game.app.views.hero import _trait_label",
    ],
    "regional": [
        "from game.app.views.dungeon import DungeonMapNodeView, DungeonRoomView, DungeonView",
        "from game.app.views.formatting import _dedupe, _join_detail",
    ],
    "town": [
        "from game.app.views.formatting import _join_detail",
        "from game.app.views.hero import (",
        "    HeroListEntry,",
        "    _hero_entry,",
        "    _hero_memories,",
        "    _trait_label,",
        ")",
    ],
    "hero": [
        "from game.app.views.art import (",
        "    _art_display_name,",
        "    _art_frame_holds,",
        "    _art_frame_impacts,",
        "    _art_frames,",
        "    _art_glyph,",
        "    _art_lines,",
        "    _art_mini_frames,",
        "    _art_mini_lines,",
        "    _derive_mini_lines,",
        "    _hero_art_asset,",
        ")",
        "from game.app.views.combat import CombatActorView",
        "from game.app.views.constants import EMPTY_SLOT_VALUE",
        "from game.app.views.formatting import _join_detail",
        "from game.app.views.town import _town_protectors",
    ],
}

PUBLIC_EXPORTS = sorted(
    {
        name
        for names in MODULES.values()
        for name in names
        if not name.startswith("_") or name == "_TownProtector"
    }
    - {"_TownProtector"}
)


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    name_to_module: dict[str, str] = {}
    for mod, names in MODULES.items():
        for name in names:
            name_to_module[name] = mod

    chunks: dict[str, list[str]] = {mod: [] for mod in MODULES}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in name_to_module:
                    mod = name_to_module[target.id]
                    start = node.lineno - 1
                    end = node.end_lineno or node.lineno
                    chunks[mod].append("".join(lines[start:end]))
            continue
        name = getattr(node, "name", None)
        if name is None or name not in name_to_module:
            if name:
                raise SystemExit(f"Unmapped top-level definition: {name}")
            continue
        mod = name_to_module[name]
        start = node.lineno - 1
        if node.decorator_list:
            start = node.decorator_list[0].lineno - 1
        end = node.end_lineno or node.lineno
        chunks[mod].append("".join(lines[start:end]))

    PKG.mkdir(exist_ok=True)

    for mod, parts in chunks.items():
        if not parts:
            raise SystemExit(f"No content for module {mod}")
        body = "\n\n".join(part.rstrip() for part in parts) + "\n"
        cross = "\n".join(CROSS_IMPORTS.get(mod, []))
        if cross:
            cross = cross + "\n\n"
        module_path = PKG / f"{mod}.py"
        if mod == "constants":
            module_text = (
                '"""Shared view-layer constants."""\n\nfrom __future__ import annotations\n\n'
                + body
            )
        elif mod == "formatting":
            module_text = (
                '"""Shared string formatting helpers for view builders."""\n\n'
                "from __future__ import annotations\n\n"
                "from collections.abc import Sequence\n\n"
                + body
            )
        else:
            module_text = COMMON_HEADER + cross + body
        module_path.write_text(module_text, encoding="utf-8", newline="\n")
        print(f"wrote {module_path.name}: {len(parts)} definitions")

    init = PKG / "__init__.py"
    init_parts = [
        '"""App-facing view models for terminal rendering (package barrel)."""',
        "",
        "from __future__ import annotations",
        "",
        "from game.app.actions import ScreenAction, ScreenActionKind, ScreenActionRisk",
        "from game.app.views.art import *",
        "from game.app.views.combat import *",
        "from game.app.views.constants import *",
        "from game.app.views.dungeon import *",
        "from game.app.views.formatting import *",
        "from game.app.views.hero import *",
        "from game.app.views.regional import *",
        "from game.app.views.town import *",
        "",
        "__all__ = [",
    ]
    for name in PUBLIC_EXPORTS:
        init_parts.append(f'    "{name}",')
    init_parts.extend(
        [
            '    "ScreenAction",',
            '    "ScreenActionKind",',
            '    "ScreenActionRisk",',
            "]",
            "",
        ]
    )
    init.write_text("\n".join(init_parts), encoding="utf-8", newline="\n")
    print("wrote __init__.py")

    SRC.unlink()
    print("removed views.py")


if __name__ == "__main__":
    main()
