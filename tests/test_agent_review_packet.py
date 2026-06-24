from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from game.dev.agent_review_packet import main as review_packet_main


def test_review_packet_prints_sections(tmp_path: Path) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = review_packet_main(["--project-root", str(tmp_path)])
    output = buffer.getvalue()
    assert code == 0
    assert "# Review packet" in output
    assert "## Git status" in output
    assert "## Diffstat" in output
    assert "## Suggested context blocks" in output
    assert "## Doctor / freshness" in output
    assert "## Handoff template" in output
    assert "## Next step" in output


def test_review_packet_on_real_repo() -> None:
    project_root = Path(__file__).resolve().parents[1]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = review_packet_main(["--project-root", str(project_root)])
    output = buffer.getvalue()
    assert code == 0
    assert "dungeon-party-game" in output or str(project_root) in output
