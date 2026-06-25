"""Remove extracted render methods from tui.py and add thin delegates."""

from __future__ import annotations

import ast
from pathlib import Path

from split_tui_render import MODULES, SHOW_RENAMES

ROOT = Path(__file__).resolve().parents[1]
TUI = ROOT / "src/game/ui/tui.py"

RENDER_ATTR = {
    "shell": "_shell_render",
    "regional": "_regional_render",
    "town": "_town_render",
    "dungeon": "_dungeon_render",
    "combat": "_combat_render",
}

METHOD_TO_MODULE: dict[str, str] = {}
for module_name, spec in MODULES.items():
    for method in spec["methods"]:
        METHOD_TO_MODULE[method] = module_name

EXTRACTED = set(METHOD_TO_MODULE)


def method_signature(func: ast.FunctionDef) -> str:
    args = ast.unparse(func.args)
    ret = f" -> {ast.unparse(func.returns)}" if func.returns else ""
    return f"({args}){ret}"


def call_expression(func: ast.FunctionDef) -> str:
    call_parts: list[str] = []
    for arg in func.args.args[1:]:
        call_parts.append(arg.arg)
    for arg in func.args.kwonlyargs:
        call_parts.append(f"{arg.arg}={arg.arg}")
    if func.args.vararg:
        call_parts.append(f"*{func.args.vararg.arg}")
    if func.args.kwarg:
        call_parts.append(f"**{func.args.kwarg.arg}")
    return ", ".join(call_parts)


def main() -> None:
    source = TUI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    charter: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CharterApp":
            charter = node
            break
    if charter is None:
        raise RuntimeError("CharterApp not found")

    methods: dict[str, ast.FunctionDef] = {
        n.name: n for n in charter.body if isinstance(n, ast.FunctionDef)
    }

    delegate_lines: list[str] = []
    for method in sorted(EXTRACTED, key=lambda m: methods[m].lineno):
        func = methods[method]
        module = METHOD_TO_MODULE[method]
        render = RENDER_ATTR[module]
        target = SHOW_RENAMES.get(method, method)
        sig = method_signature(func)
        call = call_expression(func)
        if func.returns is not None:
            delegate_lines.append(
                f"    def {method}{sig}:\n"
                f"        return self.{render}.{target}({call})\n"
            )
        else:
            delegate_lines.append(
                f"    def {method}{sig}:\n"
                f"        self.{render}.{target}({call})\n"
            )

    remove_ranges: list[tuple[int, int]] = []
    for method in EXTRACTED:
        func = methods[method]
        remove_ranges.append((func.lineno, func.end_lineno or func.lineno))
    remove_ranges.sort(reverse=True)
    new_lines = list(lines)
    for start, end in remove_ranges:
        del new_lines[start - 1 : end]

    patched = "".join(new_lines)

    import_anchor = "from game.ui.tui_handlers import ("
    render_import = (
        "from game.ui.tui_render import (\n"
        "    CombatRender,\n"
        "    DungeonRender,\n"
        "    RegionalRender,\n"
        "    ShellRender,\n"
        "    TownRender,\n"
        ")\n"
    )
    if "from game.ui.tui_render import" not in patched:
        idx = patched.find(import_anchor)
        end = patched.find(")", idx) + 2
        patched = patched[:end] + "\n" + render_import + patched[end:]

    init_anchor = "        self._shell_handlers = ShellHandlers(self)\n"
    init_block = (
        "        self._shell_render = ShellRender(self)\n"
        "        self._regional_render = RegionalRender(self)\n"
        "        self._town_render = TownRender(self)\n"
        "        self._dungeon_render = DungeonRender(self)\n"
        "        self._combat_render = CombatRender(self)\n"
    )
    if "_shell_render" not in patched:
        patched = patched.replace(init_anchor, init_anchor + init_block)

    anchor = "    def _screen_descriptors(self)"
    delegate_block = "\n".join(delegate_lines) + "\n\n"
    if delegate_block.strip() not in patched:
        patched = patched.replace(anchor, delegate_block + anchor)

    TUI.write_text(patched, encoding="utf-8", newline="\n")
    print(f"removed {len(EXTRACTED)} methods, added {len(delegate_lines)} delegates")


if __name__ == "__main__":
    main()
