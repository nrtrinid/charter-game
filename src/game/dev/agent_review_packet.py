"""Lightweight review packet for agent ergonomics."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from game.dev.agent_preflight import collect_git_snapshot, collect_report_context
from game.dev.check_engine_boundaries import project_root_from_here


@dataclass(frozen=True)
class CmdResult:
    code: int
    out: str


def run_git(project_root: Path, *args: str) -> CmdResult:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return CmdResult(1, f"(git unavailable: {exc})")
    out = (completed.stdout or "").strip()
    err = (completed.stderr or "").strip()
    combined = "\n".join(part for part in (out, err) if part)
    return CmdResult(completed.returncode, combined or "(no output)")


def format_list(lines: list[str]) -> str:
    if not lines:
        return "- (none)"
    return "\n".join(f"- {line}" for line in lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a lightweight review packet (Markdown).")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="dungeon-party-game root",
    )
    args = parser.parse_args(argv)
    project_root = args.project_root or project_root_from_here()

    status = run_git(project_root, "status", "--short")
    diffstat = run_git(project_root, "diff", "--stat")

    snapshot = collect_git_snapshot(project_root)
    context = collect_report_context(project_root, snapshot)

    suggested_blocks = list(context.blocks)
    suggested_tests = list(context.suggested_tests)
    warnings = list(context.warnings)

    print("# Review packet")
    print()
    print(f"- Project: `{project_root}`")
    print(f"- Branch: `{context.branch}`" if context.branch else "- Branch: (unknown)")
    print()

    print("## Git status")
    print()
    print("```text")
    print(status.out)
    print("```")
    print()

    print("## Diffstat")
    print()
    print("```text")
    print(diffstat.out)
    print("```")
    print()

    print("## Suggested context blocks")
    print()
    print(format_list(suggested_blocks))
    print()

    print("## Suggested focused tests")
    print()
    print(format_list(suggested_tests))
    print()

    print("## Risks / warnings")
    print()
    print(format_list([w.splitlines()[0] for w in warnings]))
    print()

    print("## Skipped checks")
    print()
    print("- (none; this command does not run tests, lint, types, or boundaries)")
    print()

    print("## Next step")
    print()
    if suggested_tests:
        test_args = " ".join(suggested_tests)
        print(f"- Run: `.\\rtk.ps1 check` and `.\\rtk.ps1 test {test_args}`")
    else:
        print("- Run: `.\\rtk.ps1 check` and `.\\rtk.ps1 quick`")
    print("- If engine packages changed: run `.\\rtk.ps1 boundaries`")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
