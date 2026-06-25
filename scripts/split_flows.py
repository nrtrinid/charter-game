"""One-shot splitter for game.app.flows monolith into package modules."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/game/app/flows.py"
PKG = ROOT / "src/game/app/flows"

HEADER = '''\
"""Application orchestration flows used by AppController."""

from __future__ import annotations

'''

# Line ranges are 1-based inclusive from original flows.py
SLICES: dict[str, tuple[int, int, str]] = {
    "base.py": (
        171,
        269,
        "",
    ),
    "town.py": (
        272,
        724,
        "from game.app.flows.base import ControllerFlow\n\n",
    ),
    "expedition.py": (
        727,
        828,
        (
            "from game.app.flows.base import ControllerFlow\n"
            "from game.app.flows.dungeon import DungeonFlow\n"
            "from game.app.flows.manual_combat import MANUAL_STAGE_SHALLOW_CAVE\n\n"
        ),
    ),
    "dungeon.py": (
        831,
        1123,
        "from game.app.flows.base import ControllerFlow\n\n",
    ),
    "manual_combat.py": (
        167,
        168,
        "",
    ),
}

MANUAL_COMBAT_CLASS = (1126, 1582)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    import_block = "".join(lines[4:165])  # lines 5-165 (imports only)

    PKG.mkdir(parents=True, exist_ok=True)

    for filename, (start, end, extra) in SLICES.items():
        if filename == "manual_combat.py":
            body = "".join(lines[start - 1 : end]) + "\n\n"
            body += "".join(lines[MANUAL_COMBAT_CLASS[0] - 1 : MANUAL_COMBAT_CLASS[1]])
            content = HEADER + import_block + extra + body
        else:
            body = "".join(lines[start - 1 : end])
            content = HEADER + import_block + extra + body
        (PKG / filename).write_text(content, encoding="utf-8", newline="\n")

    init = '''\
"""Application orchestration flows (package barrel)."""

from __future__ import annotations

from game.app.flows.base import ControllerFlow
from game.app.flows.dungeon import DungeonFlow
from game.app.flows.expedition import ExpeditionFlow
from game.app.flows.manual_combat import (
    MANUAL_STAGE_CAVE_BOSS,
    MANUAL_STAGE_SHALLOW_CAVE,
    ManualCombatFlow,
)
from game.app.flows.town import TownFlow

__all__ = [
    "ControllerFlow",
    "DungeonFlow",
    "ExpeditionFlow",
    "MANUAL_STAGE_CAVE_BOSS",
    "MANUAL_STAGE_SHALLOW_CAVE",
    "ManualCombatFlow",
    "TownFlow",
]
'''
    (PKG / "__init__.py").write_text(init, encoding="utf-8", newline="\n")
    print(f"Wrote package under {PKG}")


if __name__ == "__main__":
    main()
