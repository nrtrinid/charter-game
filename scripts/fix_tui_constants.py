"""Post-process: create tui_constants and patch render imports."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUI = ROOT / "src/game/ui/tui.py"
RENDER_DIR = ROOT / "src/game/ui/tui_render"

RENDER_IMPORT = """from game.ui.tui_constants import (
    BREACH_PENDING_FLAG,
    DEFAULT_COMPANY_NAME,
    DEFAULT_SAVE_PATH,
    _enemy_ai_controls_text,
    _enemy_ai_mode_label,
    _next_enemy_ai_mode,
    _route_summary_line,
    _route_warning_line,
)
"""


def main() -> None:
    constants_src = TUI.read_text(encoding="utf-8")
    start = constants_src.index("DEFAULT_SAVE_PATH")
    end = constants_src.index("class CharterApp")
    block = constants_src[start:end].strip() + "\n"

    header = (
        '"""Shared TUI constants and route/AI label helpers."""\n\n'
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n"
        "from typing import Any\n\n"
        "from game.app.views import ScreenActionRisk\n"
        "from game.combat.enemy_decision import (\n"
        "    PRODUCTION_ENEMY_AI_MODE_DESCRIPTIONS,\n"
        "    PRODUCTION_ENEMY_AI_MODE_LABELS,\n"
        "    SUPPORTED_PRODUCTION_ENEMY_AI_MODES,\n"
        "    production_enemy_movement_mode,\n"
        "    production_enemy_wait_mode,\n"
        ")\n\n"
    )
    (ROOT / "src/game/ui/tui_constants.py").write_text(
        header + block, encoding="utf-8", newline="\n"
    )

    anchor = "from game.ui.tui_render.protocol import TuiRenderHost\n"
    for path in RENDER_DIR.glob("*.py"):
        if path.name in {"protocol.py", "__init__.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "tui_constants" in text:
            continue
        text = text.replace(anchor, anchor + "\n" + RENDER_IMPORT + "\n")
        path.write_text(text, encoding="utf-8", newline="\n")

    tui = TUI.read_text(encoding="utf-8")
    if "from game.ui.tui_constants import" not in tui:
        tui = tui.replace(block + "\n\n", "")
        wound_anchor = "from game.ui.wounds import mortal_wound_badge\n"
        tui_import = (
            "\nfrom game.ui.tui_constants import (\n"
            "    BEAT_ANIMATION_LAST_FRAME,\n"
            "    BEAT_ANIMATION_START_FRAME,\n"
            "    BEAT_IDLE_CYCLE,\n"
            "    BREACH_PENDING_FLAG,\n"
            "    DEFAULT_COMPANY_NAME,\n"
            "    DEFAULT_SAVE_PATH,\n"
            "    GLOBAL_SHORTCUT_SCREENS,\n"
            "    GLOBAL_SHORTCUT_TEXT,\n"
            "    TURN_FLASH_LAST_FRAME,\n"
            "    UNSAFE_DEFAULT_RISKS,\n"
            "    _enemy_ai_controls_text,\n"
            "    _enemy_ai_mode_label,\n"
            "    _enemy_ai_mode_text,\n"
            "    _enemy_timing_label,\n"
            "    _next_enemy_ai_mode,\n"
            "    _route_direction_label,\n"
            "    _route_exception_tag,\n"
            "    _route_summary_line,\n"
            "    _route_warning_line,\n"
            ")\n"
        )
        tui = tui.replace(wound_anchor, wound_anchor + tui_import)
        TUI.write_text(tui, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
