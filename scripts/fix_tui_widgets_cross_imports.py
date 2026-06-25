"""Add cross-module imports for split tui_widgets package."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PKG = Path("src/game/ui/tui_widgets")
MODULES = ["animation", "shell", "town", "dungeon", "combat"]

# Explicit cross-module symbol exports (private helpers shared across domains).
CROSS_IMPORTS: dict[str, dict[str, list[str]]] = {
    "town": {
        "combat": ["_grid_cell", "_grid_text", "_lines_or_none", "_mini_side_rows"],
        "dungeon": [
            "_contract_summary_line",
            "_contract_reward_summary",
            "_objective_lines",
            "_upgrade_summary_line",
        ],
        "shell": ["format_meta_line"],
    },
    "dungeon": {
        "animation": ["_compact_art_lines"],
        "combat": ["_lines_or_none"],
        "shell": ["format_meta_line"],
    },
    "combat": {
        "animation": [
            "_animation_art_lines",
            "_authored_action_frame_count",
            "_beat_callouts",
            "_beat_hp_overrides",
            "_beat_motion_offsets",
            "_beat_pulse_styles",
            "_beat_status_overrides",
            "_compact_art_lines",
            "_held_frame_index",
            "_idle_frame_hold",
            "_knockback_direction",
            "_knockback_distance",
            "_marked_art_lines",
            "_portrait_animation_art_lines",
            "_portrait_card_lines",
            "_portrait_display_art_lines",
            "_procedural_animation_art_lines",
            "_staged_animation_cue",
            "_team_direction",
        ],
        "shell": ["format_meta_line", "primary_hotkey"],
    },
    "shell": {},
    "animation": {},
}


def defined_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def inject_imports(path: Path, imports: dict[str, list[str]]) -> None:
    text = path.read_text(encoding="utf-8")
    # Remove broken star import from constants if present
    text = re.sub(
        r"from game\.ui\.tui_widgets\.constants import \*\n",
        "",
        text,
    )
    if "from game.ui.tui_widgets.constants import" not in text:
        const_block = "from game.ui.tui_widgets.constants import *\n"
    else:
        const_block = ""

    import_lines: list[str] = []
    if const_block:
        import_lines.append(const_block)
    for src_mod, symbols in imports.items():
        if not symbols:
            continue
        joined = ",\n    ".join(symbols)
        import_lines.append(
            f"from game.ui.tui_widgets.{src_mod} import (\n    {joined},\n)\n"
        )

    # Remove old cross-import blocks
    text = re.sub(
        r"from game\.ui\.tui_widgets\.(?:combat|dungeon|shell|animation)"
        r" import[^\n]*(?:\n    [^\n]+)*\n\)",
        "",
        text,
    )
    text = re.sub(
        r"from game\.ui\.tui_widgets\.(?:combat|dungeon|shell|animation) import [^\n]+\n",
        "",
        text,
    )

    anchor = "from game.ui.wounds import mortal_wound_badge\n"
    if anchor in text:
        block = "\n" + "".join(import_lines) + "\n"
        text = text.replace(anchor, anchor + block)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    for mod in MODULES:
        path = PKG / f"{mod}.py"
        inject_imports(path, CROSS_IMPORTS.get(mod, {}))
        print(f"patched {mod}.py")
    print("done")


if __name__ == "__main__":
    main()
