from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from game.dev.agent_doctor import collect_freshness_findings
from game.dev.agent_doctor import main as doctor_main


def test_doc_link_missing(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("[missing](does-not-exist.md)\n", encoding="utf-8")
    findings = collect_freshness_findings(tmp_path, ())
    codes = {finding.code for finding in findings}
    assert "doc_link_missing" in codes


def test_doc_link_ok(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("[readme](README.md)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    findings = collect_freshness_findings(tmp_path, ())
    assert not any(finding.code == "doc_link_missing" for finding in findings)


def test_dangerous_path_touched(tmp_path: Path) -> None:
    findings = collect_freshness_findings(tmp_path, (".env",))
    assert any(finding.code == "dangerous_path_touched" for finding in findings)


def test_stale_rtk_task(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Run .\\rtk.ps1 not-a-task\n", encoding="utf-8")
    findings = collect_freshness_findings(tmp_path, ())
    assert any(finding.code == "stale_rtk_task" for finding in findings)


def test_boundaries_recommended(tmp_path: Path) -> None:
    findings = collect_freshness_findings(
        tmp_path, ("src/game/combat/foo.py",)
    )
    assert any(finding.code == "boundaries_recommended" for finding in findings)


def test_doctor_cli_prints(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = doctor_main(["--project-root", str(tmp_path)])
    output = buffer.getvalue()
    assert code == 0
    assert "Agent doctor" in output
