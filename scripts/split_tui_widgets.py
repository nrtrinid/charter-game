"""One-shot splitter for game.ui.tui_widgets monolith into package modules."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/game/ui/tui_widgets.py"
PKG = ROOT / "src/game/ui/tui_widgets"

IMPORT_BLOCK_START = 3
IMPORT_BLOCK_END = 36

CLASS_TO_MODULE: dict[str, str] = {
    "StatusHeader": "shell",
    "BodyPane": "shell",
    "DetailPane": "shell",
    "LogPane": "shell",
    "CommandDock": "shell",
    "ExpeditionProgressStrip": "town",
    "FormationBoard": "town",
    "TownDashboardPanel": "town",
    "YardPanel": "town",
    "PackPanel": "town",
    "GearLockerPanel": "town",
    "RelicBrokerPanel": "town",
    "SupplyShopPanel": "town",
    "CompanyPanel": "town",
    "DungeonMapPanel": "dungeon",
    "DungeonRoomPanel": "dungeon",
    "ExpeditionReportPanel": "dungeon",
    "CombatPanel": "combat",
}

FUNCTION_TO_MODULE: dict[str, str] = {
    "formation_slot_faces_inward": "town",
    "format_meta_line": "shell",
    "primary_hotkey": "shell",
    "portrait_detail_lines": "combat",
    "_dock_label": "shell",
    "_show_dock_warning": "shell",
    "_is_boss_route_action": "shell",
    "_flat_command_lines": "shell",
    "_dock_command_line": "shell",
    "_shortcut_line": "shell",
    "_dock_help_lines": "shell",
    "_wide_dock_text": "shell",
    "_fit_dock_line": "shell",
    "_formation_preview_text": "town",
    "_formation_preview_row": "town",
    "_preview_name": "town",
    "_slot_name": "town",
    "_interactable_hint": "dungeon",
    "_interactable_target": "dungeon",
    "_contract_summary_line": "dungeon",
    "_contract_reward_summary": "dungeon",
    "_upgrade_summary_line": "dungeon",
    "_report_brief_lines": "dungeon",
    "_reward_lines": "dungeon",
    "_delta_lines": "dungeon",
    "_signed": "dungeon",
    "_objective_lines": "dungeon",
    "_action_number_for_value": "dungeon",
    "_map_inventory_line": "dungeon",
    "_map_node_detail_lines": "dungeon",
    "_node_inventory_brief": "dungeon",
    "_quantity_line": "dungeon",
    "_label_identifier": "dungeon",
    "_minimap_lines": "dungeon",
    "_highlight_minimap_nodes": "dungeon",
    "_minimap_current_node_labels": "dungeon",
    "_minimap_highlighted_node_labels": "dungeon",
    "_coordinate_minimap_lines": "dungeon",
    "_coordinate_full_map_lines": "dungeon",
    "_minimap_viewport_anchor": "dungeon",
    "_map_nodes_share_drawable_edge": "dungeon",
    "_full_map_lines": "dungeon",
    "_spatial_map_nodes": "dungeon",
    "_draw_map_edge": "dungeon",
    "_draw_map_edge_between": "dungeon",
    "_draw_map_horizontal_connector": "dungeon",
    "_draw_map_vertical_connector": "dungeon",
    "_draw_map_label": "dungeon",
    "_draw_map_label_at": "dungeon",
    "_draw_map_overflow_markers": "dungeon",
    "_map_edge_key": "dungeon",
    "_map_canvas_position": "dungeon",
    "_put_map_char": "dungeon",
    "_minimap_branch_lines": "dungeon",
    "_compass_exits": "dungeon",
    "_visited_branch_lines": "dungeon",
    "_minimap_node_label": "dungeon",
    "_is_quest_marker_node": "dungeon",
    "_map_status_label": "dungeon",
    "_short_map_name": "dungeon",
    "_actor_marker": "dungeon",
    "_grid_cell": "combat",
    "_grid_text": "combat",
    "_styled_cell_value": "combat",
    "_combat_cell_style": "combat",
    "_intent_style": "combat",
    "_soft_intent_style": "combat",
    "_intent_label": "combat",
    "_combat_status_line": "combat",
    "_combat_cell_figure": "combat",
    "_actor_grid_glyph": "combat",
    "_actor_sprite": "combat",
    "_turn_order_rail": "combat",
    "_turn_order_chip": "combat",
    "_short_order_name": "combat",
    "_turn_line": "combat",
    "_pressure_lines": "combat",
    "_names_for_actor_ids": "combat",
    "_focused_actor_id": "combat",
    "_selected_skill_name": "combat",
    "_lines_or_none": "combat",
    "_portrait_detail_lines": "combat",
    "_portrait_badges": "combat",
    "_portrait_badge_style": "combat",
    "_portrait_effect_lines": "combat",
    "_card_line": "combat",
    "_markup_safe_visible": "combat",
    "_death_art_lines": "combat",
    "_actor_state_detail_lines": "combat",
    "_actor_effect_lines": "combat",
    "_display_state": "combat",
    "_normalized_state": "combat",
    "_dedupe_text": "combat",
}

MODULE_DOC = {
    "constants": "Shared layout and animation constants for TUI widgets.",
    "animation": "Animation and portrait art helpers for TUI widgets.",
    "shell": "Shell chrome widgets (header, panes, command dock).",
    "town": "Town and formation Textual widgets.",
    "dungeon": "Dungeon map, room, and report Textual widgets.",
    "combat": "Combat panel and beat-rendering helpers.",
}


def import_block(source: str) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[IMPORT_BLOCK_START - 1 : IMPORT_BLOCK_END])


def constants_block(source: str) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[37:79])


def segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines(keepends=True)
    end = getattr(node, "end_lineno", None) or node.lineno
    return "".join(lines[node.lineno - 1 : end])


def module_for_function(name: str, lineno: int) -> str:
    if name in FUNCTION_TO_MODULE:
        return FUNCTION_TO_MODULE[name]
    if lineno < 688:
        return "animation"
    if lineno < 838:
        return "shell"
    if lineno < 1197:
        return "town"
    if lineno < 1465:
        return "dungeon"
    if lineno < 1947:
        return "combat"
    if lineno < 3421:
        return "combat"
    if lineno < 4238:
        return "dungeon"
    return "combat"


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = import_block(source)
    consts = constants_block(source)

    chunks: dict[str, list[str]] = {
        "constants": [consts],
        "animation": [],
        "shell": [],
        "town": [],
        "dungeon": [],
        "combat": [],
    }

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            mod = CLASS_TO_MODULE[node.name]
            chunks[mod].append(segment(source, node))
        elif isinstance(node, ast.FunctionDef):
            mod = module_for_function(node.name, node.lineno)
            chunks[mod].append(segment(source, node))

    PKG.mkdir(parents=True, exist_ok=True)

    (PKG / "constants.py").write_text(
        f'"""{MODULE_DOC["constants"]}"""\n\nfrom __future__ import annotations\n\n'
        + consts
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    for mod in ("animation", "shell", "town", "dungeon", "combat"):
        body = "\n".join(chunk.rstrip() for chunk in chunks[mod]) + "\n"
        extra_imports = ""
        if mod == "animation":
            extra_imports = "from game.ui.tui_widgets.constants import *\n"
        elif mod == "shell":
            extra_imports = "from game.ui.tui_widgets.constants import COMMAND_LABEL_WIDTH\n"
        elif mod == "town":
            extra_imports = (
                "from game.ui.tui_widgets.constants import PARTY_COMBAT_ROWS\n"
                "from game.ui.tui_widgets.combat import (\n"
                "    _grid_cell,\n"
                "    _grid_text,\n"
                "    _lines_or_none,\n"
                "    _mini_side_rows,\n"
                ")\n"
                "from game.ui.tui_widgets.dungeon import (\n"
                "    _contract_summary_line,\n"
                "    _objective_lines,\n"
                "    _upgrade_summary_line,\n"
                ")\n"
                "from game.ui.tui_widgets.shell import format_meta_line\n"
            )
        elif mod == "dungeon":
            extra_imports = (
                "from game.ui.tui_widgets.animation import _compact_art_lines\n"
                "from game.ui.tui_widgets.combat import _lines_or_none\n"
                "from game.ui.tui_widgets.shell import format_meta_line\n"
            )
        elif mod == "combat":
            extra_imports = (
                "from game.ui.tui_widgets.constants import *\n"
                "from game.ui.tui_widgets.animation import (\n"
                "    _animation_art_lines,\n"
                "    _authored_action_frame_count,\n"
                "    _beat_callouts,\n"
                "    _beat_hp_overrides,\n"
                "    _beat_motion_offsets,\n"
                "    _beat_pulse_styles,\n"
                "    _beat_status_overrides,\n"
                "    _compact_art_lines,\n"
                "    _held_frame_index,\n"
                "    _idle_frame_hold,\n"
                "    _knockback_direction,\n"
                "    _knockback_distance,\n"
                "    _marked_art_lines,\n"
                "    _portrait_animation_art_lines,\n"
                "    _portrait_card_lines,\n"
                "    _portrait_display_art_lines,\n"
                "    _procedural_animation_art_lines,\n"
                "    _staged_animation_cue,\n"
                "    _team_direction,\n"
                ")\n"
                "from game.ui.tui_widgets.shell import format_meta_line, primary_hotkey\n"
            )
        content = (
            f'"""{MODULE_DOC[mod]}"""\n\n'
            "from __future__ import annotations\n\n"
            + imports
            + "\n"
            + extra_imports
            + "\n"
            + body
        )
        (PKG / f"{mod}.py").write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {mod}.py: {len(chunks[mod])} chunks")

    init = '''\
"""Textual widgets used by the fullscreen frontend (package barrel)."""

from __future__ import annotations

from game.ui.tui_widgets.animation import _compact_art_lines
from game.ui.tui_widgets.combat import (
    CombatPanel,
    _mini_art_lines,
    _mini_slot_nudge,
    _normalize_mini_lines,
    portrait_detail_lines,
)
from game.ui.tui_widgets.dungeon import (
    DungeonMapPanel,
    DungeonRoomPanel,
    ExpeditionReportPanel,
)
from game.ui.tui_widgets.shell import (
    BodyPane,
    CommandDock,
    DetailPane,
    LogPane,
    StatusHeader,
    format_meta_line,
    primary_hotkey,
)
from game.ui.tui_widgets.town import (
    CompanyPanel,
    ExpeditionProgressStrip,
    FormationBoard,
    GearLockerPanel,
    PackPanel,
    RelicBrokerPanel,
    SupplyShopPanel,
    TownDashboardPanel,
    YardPanel,
    formation_slot_faces_inward,
)

__all__ = [
    "BodyPane",
    "CombatPanel",
    "CommandDock",
    "CompanyPanel",
    "DetailPane",
    "DungeonMapPanel",
    "DungeonRoomPanel",
    "ExpeditionProgressStrip",
    "ExpeditionReportPanel",
    "FormationBoard",
    "GearLockerPanel",
    "LogPane",
    "PackPanel",
    "RelicBrokerPanel",
    "StatusHeader",
    "SupplyShopPanel",
    "TownDashboardPanel",
    "YardPanel",
    "_compact_art_lines",
    "_mini_art_lines",
    "_mini_slot_nudge",
    "_normalize_mini_lines",
    "formation_slot_faces_inward",
    "format_meta_line",
    "portrait_detail_lines",
    "primary_hotkey",
]
'''
    (PKG / "__init__.py").write_text(init, encoding="utf-8", newline="\n")
    print("done")


if __name__ == "__main__":
    main()
