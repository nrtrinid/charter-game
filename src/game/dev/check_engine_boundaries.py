"""Check engine layers for forbidden game.ui imports."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ENGINE_PACKAGES = ("combat", "campaign", "expedition", "core", "content", "data")
UI_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+game\.ui(?:\.\w+)*\s+import|import\s+game\.ui(?:\.\w+)*)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class BoundaryViolation:
    path: Path
    line_number: int
    line: str


def iter_engine_python_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    game_root = project_root / "src" / "game"
    for package in ENGINE_PACKAGES:
        package_root = game_root / package
        if package_root.is_dir():
            files.extend(sorted(package_root.rglob("*.py")))
    return files


def find_ui_import_violations(
    project_root: Path,
    paths: list[Path] | None = None,
) -> list[BoundaryViolation]:
    targets = paths if paths is not None else iter_engine_python_files(project_root)
    violations: list[BoundaryViolation] = []
    for path in targets:
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in UI_IMPORT_RE.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            line = text.splitlines()[line_number - 1].strip()
            violations.append(
                BoundaryViolation(path=path, line_number=line_number, line=line)
            )
    return violations


def format_violations(violations: list[BoundaryViolation], project_root: Path) -> str:
    if not violations:
        return "Boundary check: no game.ui imports in engine packages."
    lines = ["Boundary check FAILED: game.ui imports in engine packages:"]
    for violation in violations:
        rel = violation.path.relative_to(project_root)
        lines.append(f"  {rel}:{violation.line_number}: {violation.line}")
    return "\n".join(lines)


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if engine code imports game.ui.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="dungeon-party-game root",
    )
    args = parser.parse_args(argv)
    project_root = args.project_root or project_root_from_here()
    violations = find_ui_import_violations(project_root)
    print(format_violations(violations, project_root))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
