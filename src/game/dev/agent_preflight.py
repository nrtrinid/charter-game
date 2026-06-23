"""Agent session preflight: git snapshot, task-block guess, verify hints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from game.dev.check_engine_boundaries import (
    find_ui_import_violations,
    format_violations,
    project_root_from_here,
)

BLOCK_VERIFY_TESTS: dict[str, tuple[str, ...]] = {
    "combat-rules": (
        "tests/test_manual_combat.py",
        "tests/test_targeting.py",
        "tests/test_formation.py",
    ),
    "campaign-town": ("tests/test_town_loop.py",),
    "expedition-dungeon": (
        "tests/test_dungeon.py",
        "tests/test_generated_maze.py",
    ),
    "textual-ui": ("tests/test_tui.py",),
    "cli-legacy": ("tests/test_cli.py", "tests/test_main.py"),
    "content-yaml": ("tests/test_main.py",),
    "save-load": ("tests/test_save_load.py",),
    "ai-lab-balance": (
        "tests/test_ai_oracle.py",
        "tests/test_policy_band_report.py",
    ),
    "app-commands": ("tests/test_hci_substrate.py",),
    "docs-only": ("tests/test_main.py",),
}

AREA_LABELS: tuple[tuple[str, str], ...] = (
    ("src/game/combat/", "combat"),
    ("src/game/campaign/", "campaign"),
    ("src/game/expedition/", "expedition"),
    ("src/game/ui/", "ui"),
    ("src/game/app/", "app"),
    ("src/game/data/", "data"),
    ("src/game/dev/", "dev"),
    ("src/game/core/", "core"),
    ("data/", "data-yaml"),
    ("tests/", "tests"),
    ("docs/", "docs"),
    ("project_sources/", "project_sources"),
)


@dataclass(frozen=True)
class GitSnapshot:
    branch: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True)
class ReportContext:
    branch: str | None
    changed_paths: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]
    blocks: tuple[str, ...]
    suggested_tests: tuple[str, ...]
    warnings: tuple[str, ...]


def run_git(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_git_snapshot(project_root: Path) -> GitSnapshot | None:
    branch = run_git(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = run_git(project_root, "status", "--porcelain")
    if branch is None or status is None:
        return None
    changed: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip().strip('"')
        changed.append(path.replace("\\", "/"))
    return GitSnapshot(branch=branch, changed_paths=tuple(changed))


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def path_matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern) or fnmatch(path, pattern.rstrip("/") + "/**")
    return fnmatch(path, pattern) or path.endswith("/" + pattern) or path == pattern


def classify_path(path: str) -> tuple[str, ...]:
    blocks: list[str] = []
    if path_matches(path, "src/game/campaign/save_load.py") or path.endswith("/company.py"):
        blocks.append("save-load")
    if path_matches(path, "src/game/combat/") or path.endswith("/manual_combat.py"):
        blocks.append("combat-rules")
    if path_matches(path, "src/game/campaign/") or path_matches(
        path, "data/town.yaml"
    ) or path_matches(path, "data/gear.yaml"):
        blocks.append("campaign-town")
    if path_matches(path, "src/game/expedition/") or path_matches(
        path, "data/expeditions.yaml"
    ) or path_matches(path, "data/world.yaml"):
        blocks.append("expedition-dungeon")
    if path_matches(path, "src/game/ui/tui*.py") or path.endswith(
        "/tui_widgets.py"
    ) or path.endswith("/tui_models.py"):
        blocks.append("textual-ui")
    if path.endswith("/cli.py") or path.endswith("/hci_text.py") or path.endswith("/wounds.py"):
        blocks.append("cli-legacy")
    if path.startswith("data/") and path.endswith(".yaml"):
        blocks.append("content-yaml")
    if path_matches(path, "src/game/data/"):
        blocks.append("content-yaml")
    if path_matches(path, "src/game/dev/") or path_matches(path, "docs/dev/"):
        blocks.append("ai-lab-balance")
    if any(
        path.endswith(f"/{name}")
        for name in ("commands.py", "controller.py", "flows.py", "actions.py", "views.py")
    ) and path.startswith("src/game/app/"):
        blocks.append("app-commands")
    if (
        path.endswith(".md")
        or path.startswith("project_sources/")
        or path.endswith("/CHANGELOG.md")
    ):
        blocks.append("docs-only")
    if path.endswith("/screens.py"):
        blocks.append("textual-ui")
        blocks.append("cli-legacy")
    return tuple(dict.fromkeys(blocks))


def group_changed_paths(changed_paths: Sequence[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in changed_paths:
        normalized = normalize_path(path)
        label = "other"
        for prefix, area in AREA_LABELS:
            if normalized.startswith(prefix):
                label = area
                break
        groups.setdefault(label, []).append(normalized)
    return groups


def suggest_tests(blocks: Sequence[str]) -> list[str]:
    tests: list[str] = []
    for block in blocks:
        for test_path in BLOCK_VERIFY_TESTS.get(block, ()):
            if test_path not in tests:
                tests.append(test_path)
    return tests


def build_warnings(
    project_root: Path,
    changed_paths: Sequence[str],
    blocks: Sequence[str],
    suggested_tests: Sequence[str],
) -> list[str]:
    warnings: list[str] = []
    normalized = [normalize_path(path) for path in changed_paths]

    yaml_changed = any(
        path.startswith("data/") and path.endswith(".yaml") for path in normalized
    )
    if (
        yaml_changed
        and "tests/test_main.py" not in suggested_tests
        and "content-yaml" not in blocks
    ):
        warnings.append(
            "data/*.yaml changed - run loader coverage (tests/test_main.py) and domain tests."
        )

    save_touched = any(
        "save_load.py" in path or path.endswith("/company.py") for path in normalized
    )
    if save_touched and "tests/test_save_load.py" not in suggested_tests:
        warnings.append("Save shape touched - include tests/test_save_load.py.")

    ui_changed = any(path.startswith("src/game/ui/") for path in normalized)
    if ui_changed and "tests/test_tui.py" not in suggested_tests and "textual-ui" in blocks:
        warnings.append("Textual UI changed — prefer .\\rtk.ps1 tui (may take >5 minutes).")

    engine_changed = [
        project_root / path
        for path in normalized
        if path.startswith("src/game/")
        and any(
            path.startswith(f"src/game/{pkg}/")
            for pkg in ("combat", "campaign", "expedition", "core", "content", "data")
        )
    ]
    if engine_changed:
        violations = find_ui_import_violations(project_root, engine_changed)
        if violations:
            warnings.append(format_violations(violations, project_root))

    combat_changed = any(path.startswith("src/game/combat/") for path in normalized)
    if combat_changed:
        warnings.append("Combat changed - keep previews/rules out of src/game/ui.")

    return warnings


def collect_report_context(
    project_root: Path,
    snapshot: GitSnapshot | None,
) -> ReportContext:
    if snapshot is None:
        changed_paths: tuple[str, ...] = ()
        branch = None
    else:
        changed_paths = snapshot.changed_paths
        branch = snapshot.branch

    groups_raw = group_changed_paths(changed_paths)
    groups = {area: tuple(paths) for area, paths in groups_raw.items()}

    blocks: list[str] = []
    for path in changed_paths:
        for block in classify_path(path):
            if block not in blocks:
                blocks.append(block)

    suggested_tests = tuple(suggest_tests(blocks))
    warnings = tuple(
        build_warnings(project_root, changed_paths, blocks, suggested_tests)
    )
    return ReportContext(
        branch=branch,
        changed_paths=changed_paths,
        groups=groups,
        blocks=tuple(blocks),
        suggested_tests=suggested_tests,
        warnings=warnings,
    )


def run_rtk(project_root: Path, task: str, extra: Sequence[str] = ()) -> int:
    rtk = project_root / "rtk.ps1"
    if not rtk.is_file():
        print(f"Missing {rtk}")
        return 1
    command = ["pwsh", "-NoProfile", "-File", str(rtk), task, *extra]
    completed = subprocess.run(command, cwd=project_root)
    return completed.returncode


def print_report(project_root: Path, snapshot: GitSnapshot | None, run_boundaries: bool) -> int:
    context = collect_report_context(project_root, snapshot)
    print("Agent preflight")
    print(f"Project: {project_root}")
    if context.branch is None:
        print("Git: unavailable (not a repo or git missing)")
    else:
        print(f"Branch: {context.branch}")
        print(f"Changed paths: {len(context.changed_paths)}")
        if context.changed_paths:
            for area in sorted(context.groups):
                print(f"  [{area}]")
                for path in context.groups[area]:
                    print(f"    - {path}")
        else:
            print("  (clean working tree)")

    if context.blocks:
        print(f"Suggested AGENT_CONTEXT_MAP block(s): {', '.join(context.blocks)}")
        print("  Read: docs/AGENT_CONTEXT_MAP.md")
    else:
        print("Suggested AGENT_CONTEXT_MAP block: (none - clean tree or unclear)")
        print("  Read: AGENTS.md, then docs/AGENT_CONTEXT_MAP.md if scope is unclear")

    if context.suggested_tests:
        test_args = " ".join(context.suggested_tests)
        print("Suggested verify:")
        print("  .\\rtk.ps1 check")
        print(f"  .\\rtk.ps1 test {test_args}")
    else:
        print("Suggested verify:")
        print("  .\\rtk.ps1 check")
        print("  .\\rtk.ps1 quick")

    if context.warnings:
        print("Warnings:")
        for warning in context.warnings:
            for line in warning.splitlines():
                print(f"  {line}")

    if run_boundaries:
        violations = find_ui_import_violations(project_root)
        print(format_violations(violations, project_root))
        if violations:
            return 1
    return 0


def print_scout_bundle(
    project_root: Path,
    context: ReportContext,
    task: str | None,
) -> None:
    print("=== CHARTER SCOUT BUNDLE (Stage 1) ===")
    print("Do not implement. Output SCOUT_PACKET per prompts/scout_packet_schema.md")
    print("Workflow: prompts/agent_workflow.md")
    print()
    print("TASK:")
    print(task.strip() if task else "<paste your one-line task here>")
    print()
    print("--- PREFLIGHT ---")
    if context.branch is None:
        print("Git: unavailable")
    else:
        print(f"Branch: {context.branch}")
        print(f"Changed paths: {len(context.changed_paths)}")
        for area in sorted(context.groups):
            print(f"  [{area}]")
            for path in context.groups[area]:
                print(f"    - {path}")
    if context.blocks:
        print(f"Suggested blocks: {', '.join(context.blocks)}")
    else:
        print("Suggested blocks: (none)")
    if context.suggested_tests:
        print(f"Suggested tests: {' '.join(context.suggested_tests)}")
    if context.warnings:
        print("Warnings:")
        for warning in context.warnings:
            for line in warning.splitlines():
                print(f"  {line}")
    print("--- END PREFLIGHT ---")
    print()
    print("SCOUT INSTRUCTIONS (cheap model):")
    print("- Read docs/AGENT_CONTEXT_MAP.md for suggested block(s) only")
    print("- Use rg and narrow reads; respect .agentignore")
    print("- Fill prompts/scout_packet_schema.md and return YAML SCOUT_PACKET")
    print()
    print("NEXT: prompts/drift_check_prompt.md (smart model, after SCOUT_PACKET)")
    print("=== END SCOUT BUNDLE ===")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent session preflight for Charter.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="dungeon-party-game root",
    )
    parser.add_argument(
        "--boundaries",
        action="store_true",
        help="Run full engine boundary scan (game.ui imports)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run .\\rtk.ps1 check and suggested focused tests after the report",
    )
    parser.add_argument(
        "--scout",
        action="store_true",
        help="Print Stage 1 scout bundle (preflight + task) for copy-paste",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task text to embed in scout bundle",
    )
    args = parser.parse_args(argv)
    project_root = args.project_root or project_root_from_here()

    snapshot = collect_git_snapshot(project_root)
    if args.scout:
        context = collect_report_context(project_root, snapshot)
        print_scout_bundle(project_root, context, args.task)
        return 0

    report_code = print_report(project_root, snapshot, run_boundaries=args.boundaries)

    if args.verify:
        print("\nRunning verify...")
        check_code = run_rtk(project_root, "check")
        if check_code != 0:
            return check_code
        blocks: list[str] = []
        if snapshot:
            for path in snapshot.changed_paths:
                blocks.extend(classify_path(path))
        suggested_tests = suggest_tests(tuple(dict.fromkeys(blocks)))
        if suggested_tests:
            test_code = run_rtk(project_root, "test", suggested_tests)
            return test_code if test_code != 0 else report_code
        quick_code = run_rtk(project_root, "quick")
        return quick_code if quick_code != 0 else report_code

    return report_code


if __name__ == "__main__":
    sys.exit(main())
