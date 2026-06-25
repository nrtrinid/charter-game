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
from game.ui.tui_widgets.formation import formation_slot_faces_inward
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
