"""TUI action handler modules."""

from game.ui.tui_handlers.combat import CombatHandlers
from game.ui.tui_handlers.dungeon import DungeonHandlers
from game.ui.tui_handlers.regional import RegionalHandlers
from game.ui.tui_handlers.shell import ShellHandlers
from game.ui.tui_handlers.town import TownHandlers

__all__ = [
    "CombatHandlers",
    "DungeonHandlers",
    "RegionalHandlers",
    "ShellHandlers",
    "TownHandlers",
]
