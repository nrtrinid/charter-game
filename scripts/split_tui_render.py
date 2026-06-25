"""One-shot splitter for game.ui.tui render methods into tui_render package."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUI = ROOT / "src/game/ui/tui.py"
PKG = ROOT / "src/game/ui/tui_render"

SHOW_RENAMES = {
    "_show_main": "show_main",
    "_show_company": "show_company",
    "_show_save_load": "show_save_load",
    "_show_help": "show_help",
    "_show_current_place": "show_current_place",
    "_show_system": "show_system",
    "_show_confirm": "show_confirm",
    "_show_name_prompt": "show_name_prompt",
    "_show_regional_place": "show_regional_place",
    "_show_regional_map": "show_regional_map",
    "_show_world_map": "show_world_map",
    "_show_regional_interactions": "show_regional_interactions",
    "_show_arrival_brief": "show_arrival_brief",
    "_show_town": "show_town",
    "_show_town_submenu": "show_town_submenu",
    "_show_roster": "show_roster",
    "_show_supplies": "show_supplies",
    "_show_pack": "show_pack",
    "_show_company_summary": "show_company_summary",
    "_show_gear_locker": "show_gear_locker",
    "_show_hero_sheet": "show_hero_sheet",
    "_show_hero_memories": "show_hero_memories",
    "_show_hero_gear": "show_hero_gear",
    "_show_ledger": "show_ledger",
    "_show_memorial": "show_memorial",
    "_show_recruiting": "show_recruiting",
    "_show_recruiting_hire": "show_recruiting_hire",
    "_show_deep_surgery": "show_deep_surgery",
    "_show_supply_shop": "show_supply_shop",
    "_show_supply_buy": "show_supply_buy",
    "_show_relic_broker": "show_relic_broker",
    "_show_formation": "show_formation",
    "_show_assign_hero": "show_assign_hero",
    "_show_expedition": "show_expedition",
    "_show_playback": "show_playback",
    "_show_dungeon": "show_dungeon",
    "_show_dungeon_map": "show_dungeon_map",
    "_show_dungeon_interactions": "show_dungeon_interactions",
    "_show_expedition_report": "show_expedition_report",
    "_show_report_view": "show_report_view",
    "_show_breach": "show_breach",
    "_show_combat_command": "show_combat_command",
    "_show_combat_skill": "show_combat_skill",
    "_show_combat_view": "show_combat_view",
    "_show_combat_target": "show_combat_target",
    "_show_resolution": "show_resolution",
    "_show_enemy_turn": "show_enemy_turn",
}

MODULES: dict[str, dict[str, object]] = {
    "shell": {
        "class": "ShellRender",
        "methods": [
            "_show_main",
            "_show_company",
            "_show_save_load",
            "_show_help",
            "_show_current_place",
            "_show_system",
            "_show_confirm",
            "_show_name_prompt",
            "_reset_ui_session",
        ],
    },
    "regional": {
        "class": "RegionalRender",
        "methods": [
            "_show_regional_place",
            "_show_regional_map",
            "_show_world_map",
            "_show_regional_interactions",
            "_show_arrival_brief",
            "_regional_place_text",
            "_regional_log_text",
            "_regional_map_text",
            "_regional_navigation_actions",
            "_regional_map_display_actions",
            "_arrival_brief_text",
            "_world_map_text",
            "_travel_destination_label",
            "_place_name",
            "_available_regional_room_action_commands",
            "_regional_move_lands_on_place",
            "_focused_regional_node_id",
        ],
    },
    "town": {
        "class": "TownRender",
        "methods": [
            "_show_town",
            "_render_town_view",
            "_show_town_submenu",
            "_town_hub_actions",
            "_town_submenu_actions",
            "_town_service_action",
            "_town_back_label",
            "_brief_text",
            "_first_visit_hint",
            "_town_hub_body",
            "_town_yard_body",
            "_town_gate_text",
            "_town_market_text",
            "_town_recovery_text",
            "_deep_surgery_text",
            "_town_quartermaster_text",
            "_town_recruitment_text",
            "_town_records_text",
            "_contract_board_text",
            "_contract_reward_summary",
            "_upgrade_board_text",
            "_show_roster",
            "_roster_hero_actions",
            "_show_supplies",
            "_show_pack",
            "_pack_actions",
            "_pack_text",
            "_show_company_summary",
            "_company_summary_actions",
            "_show_gear_locker",
            "_gear_locker_actions",
            "_show_hero_sheet",
            "_show_hero_memories",
            "_show_hero_gear",
            "_hero_sheet_actions",
            "_hero_gear_actions",
            "_gear_locker_text",
            "_hero_sheet_text",
            "_hero_gear_text",
            "_hero_quirks_memories_text",
            "_sheet_trait_lines",
            "_fresh_memory_lines",
            "_permanent_memory_lines",
            "_career_signal_lines",
            "_hero_portrait_actor",
            "_formation_portrait_actors",
            "_formation_board_text",
            "_company_summary_body",
            "_formation_detail_text",
            "_hero_sheet_preview_detail",
            "_hero_sheet_section_detail",
            "_hero_sheet_detail_view",
            "_hero_sheet_quirk_line",
            "_hero_sheet_memory_line",
            "_show_ledger",
            "_show_memorial",
            "_show_recruiting",
            "_render_recruiting_view",
            "_show_recruiting_hire",
            "_current_recruiting_view",
            "_recruiting_text",
            "_recruit_offer_detail",
            "_recruit_fit_line",
            "_show_deep_surgery",
            "_show_supply_shop",
            "_render_supply_shop_view",
            "_show_supply_buy",
            "_supply_shop_text",
            "_show_relic_broker",
            "_selection_actions",
            "_show_formation",
            "_render_formation_view",
            "_show_assign_hero",
            "_roster_text",
            "_gear_action_detail",
            "_supply_action_detail",
            "_pack_gear_detail",
            "_contract_action_detail",
            "_contract_dock_help_text",
        ],
    },
    "dungeon": {
        "class": "DungeonRender",
        "methods": [
            "_show_expedition",
            "_begin_expedition",
            "_start_playback",
            "_meaningful_playback_events",
            "_opening_enemy_response_playback_event_ids",
            "_show_playback",
            "_finish_playback",
            "_show_dungeon",
            "_show_dungeon_map",
            "_show_dungeon_interactions",
            "_dungeon_navigation_actions",
            "_available_room_action_commands",
            "_blocked_room_actions",
            "_blocked_room_actions_reason",
            "_renumbered_action",
            "_show_expedition_report",
            "_show_report_view",
            "_show_breach",
            "_playback_log_text",
            "_dungeon_log_text",
            "_dungeon_place_text",
            "_dungeon_detail_text",
            "_dungeon_interact_detail_text",
            "_dungeon_route_detail_text",
            "_dungeon_room_action_detail_text",
            "_active_dungeon_minimap_text",
            "_focused_dungeon_node_id",
            "_post_combat_continue_label",
            "_post_combat_continue_description",
            "_expedition_progress_strip",
            "_room_action_notice",
            "_room_action_state_reason",
            "_quantity_list",
            "_quantity_label",
            "_sentence_case",
            "_route_dock_help_text",
        ],
    },
    "combat": {
        "class": "CombatRender",
        "methods": [
            "_show_combat_command",
            "_show_combat_skill",
            "_show_combat_view",
            "_track_combat_turn_handoff",
            "_show_combat_target",
            "_show_resolution",
            "_show_enemy_turn",
            "_show_opening_enemy_response_if_needed",
            "_combat_text",
            "_combat_beat_text",
            "_combat_event_beats",
            "_current_beat_text",
            "_current_beat_animation_last_frame",
            "_combat_detail_text",
            "_combatant_lines",
            "_beat_text",
        ],
    },
}

IMPORT_BLOCK_START = 4  # 1-based line after docstring in tui.py
IMPORT_BLOCK_END = 150

MODULE_HEADER = '''\
"""TUI screen rendering for {domain}."""

from __future__ import annotations

from dataclasses import dataclass

from game.ui.tui_render.protocol import TuiRenderHost

'''


def extract_methods(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CharterApp":
            return {
                n.name: n
                for n in node.body
                if isinstance(n, ast.FunctionDef)
            }
    raise RuntimeError("CharterApp not found")


def method_source(source: str, func: ast.FunctionDef) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[func.lineno - 1 : func.end_lineno])


def transform_method(
    body: str,
    old_name: str,
    local_methods: set[str],
) -> str:
    new_name = SHOW_RENAMES.get(old_name, old_name)
    body = re.sub(
        rf"^    def {re.escape(old_name)}\(",
        f"    def {new_name}(",
        body,
        count=1,
    )
    body = body.replace("self.", "self.app.")
    for method in sorted(local_methods, key=len, reverse=True):
        renamed = SHOW_RENAMES.get(method, method)
        body = body.replace(f"self.app.{method}", f"self.{renamed}")
        body = body.replace(f"self.app.{renamed}", f"self.{renamed}")
    return body


def import_block(source: str) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[IMPORT_BLOCK_START - 1 : IMPORT_BLOCK_END])


def build_module(
    domain: str,
    class_name: str,
    method_names: list[str],
    all_methods: dict[str, ast.FunctionDef],
    source: str,
) -> str:
    local = set(method_names)
    parts = [
        MODULE_HEADER.format(domain=domain),
        import_block(source),
        "\n",
        f"@dataclass\nclass {class_name}:\n    app: TuiRenderHost\n\n",
    ]
    for name in method_names:
        if name not in all_methods:
            raise KeyError(f"{name} missing from CharterApp in {domain}")
        raw = method_source(source, all_methods[name])
        parts.append(transform_method(raw, name, local))
        parts.append("\n")
    return "".join(parts)


def write_protocol() -> None:
    content = '''\
"""Protocol for TUI render host (CharterApp surface used by render modules)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from game.app.controller import AppController
from game.app.views import (
    CombatView,
    DungeonView,
    ExpeditionReportView,
    FormationView,
    GearInventoryView,
    RecruitOffersView,
    RegionalMapView,
    RelicBrokerView,
    ScreenAction,
    SupplyShopView,
)
from game.core.events import GameEvent
from game.core.hci import HciResultAnalysis
from textual.css.query import NoMatches
from textual.widgets import Input


class TuiRenderHost(Protocol):
    controller: AppController
    screen_state: str
    screen_title: str
    body_text: str
    message: str
    save_path: Path
    actions: tuple[ScreenAction, ...]
    focused_command_index: int
    playback_index: int
    playback_hci: HciResultAnalysis | None
    current_combat_view: CombatView | None
    current_combat_phase: str
    selected_skill_id: str | None
    pending_enemy_events: list[GameEvent]
    pending_enemy_beats: list[list[GameEvent]]
    pending_enemy_view: CombatView | None
    pending_resolution_hci: HciResultAnalysis | None
    opening_enemy_response_session_key: str
    pending_confirm: str | None
    pending_slot_label: str
    pending_gear_return_state: str
    pending_gear_locker_return_state: str
    pending_formation_return_state: str
    pending_help_return_state: str
    pending_regional_return_state: str
    pending_world_map_return_state: str
    current_regional_view: RegionalMapView | None
    current_dungeon_view: DungeonView | None
    current_report_view: ExpeditionReportView | None
    current_gear_view: GearInventoryView | None
    current_supply_shop_view: SupplyShopView | None
    current_relic_broker_view: RelicBrokerView | None
    idle_animation_frame: int
    beat_animation_frame: int
    current_beat_events: list[GameEvent]
    current_beat_deferred_events: list[GameEvent]
    current_beat_view: CombatView | None
    current_beat_title: str
    visited_screens: set[str]
    input_mode: str | None
    _current_formation_view: FormationView | None
    _company_summary_objective: str
    _company_summary_roster_lines: list[str]

    def _show_screen(
        self,
        state: str,
        title: str,
        body: str,
        actions: Sequence[ScreenAction],
        *,
        message: str = "",
        log: str = "",
    ) -> None: ...

    def _record_events(self, events: list[GameEvent]) -> None: ...
    def _hide_name_input(self) -> None: ...
    def query_one(self, selector: str, expect_type: type[Input]) -> Input: ...

    def _show_main(self, message: str = "") -> None: ...
    def _show_town(self, message: str = "", hci: HciResultAnalysis | None = None) -> None: ...
    def _show_system(self, message: str = "", hci: HciResultAnalysis | None = None) -> None: ...
    def _show_current_place(
        self, message: str = "", hci: HciResultAnalysis | None = None
    ) -> None: ...
    def _show_regional_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
        return_to: str | None = None,
    ) -> None: ...
    def _show_dungeon(self, message: str = "", hci: HciResultAnalysis | None = None) -> None: ...
    def _show_combat_view(
        self,
        view: CombatView,
        *,
        phase: str = "command",
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _show_resolution(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _show_enemy_turn(self) -> None: ...
    def _show_expedition_report(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _begin_expedition(
        self,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
        direct_to_dungeon: bool = False,
    ) -> None: ...
    def _start_playback(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None: ...
    def _finish_playback(self, message: str = "") -> None: ...
    def _renumbered_action(
        self,
        action: ScreenAction,
        number: int,
        *,
        label: str | None = None,
        default: bool | None = None,
    ) -> ScreenAction: ...
    def _result_log_text(
        self,
        events: Sequence[GameEvent],
        hci: HciResultAnalysis | None,
    ) -> str: ...
    def _events_text(self, events: Sequence[GameEvent]) -> str: ...
    def _party_watch_text(self) -> str: ...
    def _hero_lines(self, heroes: Sequence[Any]) -> str: ...
    def _hero_gear_summary(self, hero: Any) -> str: ...
    def _generic_action_detail(self, action: ScreenAction) -> str: ...
    def _is_safe_default(self, action: ScreenAction) -> bool: ...
    def _render(self) -> None: ...

    @property
    def focused_action(self) -> ScreenAction | None: ...
'''
    (PKG / "protocol.py").write_text(content, encoding="utf-8", newline="\n")


def write_init() -> None:
    content = '''\
"""TUI screen rendering modules (package barrel)."""

from __future__ import annotations

from game.ui.tui_render.combat import CombatRender
from game.ui.tui_render.dungeon import DungeonRender
from game.ui.tui_render.regional import RegionalRender
from game.ui.tui_render.shell import ShellRender
from game.ui.tui_render.town import TownRender

__all__ = [
    "CombatRender",
    "DungeonRender",
    "RegionalRender",
    "ShellRender",
    "TownRender",
]
'''
    (PKG / "__init__.py").write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    source = TUI.read_text(encoding="utf-8")
    all_methods = extract_methods(source)
    PKG.mkdir(parents=True, exist_ok=True)
    write_protocol()
    for module_name, spec in MODULES.items():
        content = build_module(
            module_name,
            str(spec["class"]),
            list(spec["methods"]),
            all_methods,
            source,
        )
        out = PKG / f"{module_name}.py"
        out.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {out.name}: {len(spec['methods'])} methods")
    write_init()
    print("done")


if __name__ == "__main__":
    main()
