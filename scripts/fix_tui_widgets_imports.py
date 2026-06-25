from pathlib import Path

PKG = Path("src/game/ui/tui_widgets")
for name in ["animation.py", "shell.py", "town.py", "dungeon.py", "combat.py"]:
    p = PKG / name
    t = p.read_text(encoding="utf-8")
    t = t.replace(
        "from __future__ import annotations\n\nfrom __future__ import annotations\n",
        "from __future__ import annotations\n",
    )
    t = t.replace(
        "from game.ui.tui_widgets import constants\n",
        "from game.ui.tui_widgets.constants import *\n",
    )
    p.write_text(t, encoding="utf-8", newline="\n")

init = PKG / "__init__.py"
init.write_text(
    '''\
"""Textual widgets used by the fullscreen frontend (package barrel)."""

from __future__ import annotations

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
from game.ui.tui_widgets.animation import _compact_art_lines

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
''',
    encoding="utf-8",
    newline="\n",
)
print("fixed")
