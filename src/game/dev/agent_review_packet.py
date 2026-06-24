"""Lightweight review packet for agent ergonomics."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from game.dev.agent_doctor import FreshnessFinding, collect_freshness_findings
from game.dev.agent_preflight import collect_git_snapshot, collect_report_context
from game.dev.check_engine_boundaries import project_root_from_here

HANDOFF_TEMPLATE = """Changed:
Tests:
Docs:
Risks:
Did not touch:
Next safe step:"""


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


def format_findings(findings: list[FreshnessFinding]) -> str:
    if not findings:
        return "- (none)"
    lines: list[str] = []
    for finding in findings:
        first_line = finding.message.splitlines()[0]
        prefix = finding.level.upper()
        lines.append(f"- [{prefix}] {first_line}")
    return "\n".join(lines)


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
    changed_paths = snapshot.changed_paths if snapshot else ()
    context = collect_report_context(project_root, snapshot)
    freshness = list(collect_freshness_findings(project_root, changed_paths))

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

    print("## Doctor / freshness")
    print()
    print(format_findings(freshness))
    print()

    print("## Skipped checks")
    print()
    print("- Doctor is advisory only; it does not run pytest, ruff, mypy, or boundaries.")
    print("- Run `.\\rtk.ps1 doctor` separately for a full freshness report.")
    print()

    print("## Handoff template")
    print()
    print("```markdown")
    print(HANDOFF_TEMPLATE)
    print("```")
    print()

    print("## Next step")
    print()
    print("- Run: `.\\rtk.ps1 doctor` then fill the handoff template above.")
    if suggested_tests:
        test_args = " ".join(suggested_tests)
        print(f"- Run: `.\\rtk.ps1 check` and `.\\rtk.ps1 test {test_args}`")
    else:
        print("- Run: `.\\rtk.ps1 check` and `.\\rtk.ps1 quick`")
    print("- If engine packages changed: run `.\\rtk.ps1 boundaries`")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
