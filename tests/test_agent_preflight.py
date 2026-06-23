from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from game.dev.agent_preflight import (
    ReportContext,
    classify_path,
    collect_report_context,
    parse_status_path,
    print_scout_bundle,
    suggest_tests,
)
from game.dev.check_engine_boundaries import find_ui_import_violations


def test_classify_combat_path() -> None:
    assert "combat-rules" in classify_path("src/game/combat/targeting.py")


def test_parse_status_path_modified_src_file() -> None:
    assert parse_status_path(" M src/game/combat/targeting.py") == (
        "src/game/combat/targeting.py"
    )


def test_parse_status_path_untracked() -> None:
    assert parse_status_path("?? prompts/foo.md") == "prompts/foo.md"


def test_parse_status_path_rename() -> None:
    assert parse_status_path('R  old.py -> new.py') == "new.py"


def test_parse_status_path_strip_regression() -> None:
    line = " M src/game/combat/targeting.py"
    assert parse_status_path(line) == "src/game/combat/targeting.py"
    assert line.strip()[3:] == "rc/game/combat/targeting.py"


def test_classify_town_yaml_prefers_campaign_and_content() -> None:
    blocks = classify_path("data/town.yaml")
    assert "campaign-town" in blocks
    assert "content-yaml" in blocks


def test_classify_save_load_path() -> None:
    assert "save-load" in classify_path("src/game/campaign/save_load.py")


def test_classify_ai_lab_path() -> None:
    assert "ai-lab-balance" in classify_path("src/game/dev/ai_lab.py")


def test_suggest_tests_merges_blocks() -> None:
    tests = suggest_tests(["combat-rules", "save-load"])
    assert "tests/test_manual_combat.py" in tests
    assert "tests/test_save_load.py" in tests


def test_boundary_scan_finds_forbidden_import(tmp_path: Path) -> None:
    engine_file = tmp_path / "src" / "game" / "combat" / "bad.py"
    engine_file.parent.mkdir(parents=True)
    engine_file.write_text("from game.ui.cli import something\n", encoding="utf-8")
    violations = find_ui_import_violations(tmp_path, [engine_file])
    assert len(violations) == 1
    assert violations[0].line_number == 1


def test_boundary_scan_passes_on_real_repo() -> None:
    project_root = Path(__file__).resolve().parents[1]
    violations = find_ui_import_violations(project_root)
    assert violations == []


def test_collect_report_context_from_paths() -> None:
    project_root = Path(__file__).resolve().parents[1]
    from game.dev.agent_preflight import GitSnapshot

    snapshot = GitSnapshot(
        branch="main",
        changed_paths=("src/game/combat/targeting.py", "data/town.yaml"),
    )
    context = collect_report_context(project_root, snapshot)
    assert "combat-rules" in context.blocks
    assert "campaign-town" in context.blocks
    assert context.suggested_tests


def test_print_scout_bundle_includes_task() -> None:
    project_root = Path(__file__).resolve().parents[1]
    context = ReportContext(
        branch="main",
        changed_paths=("src/game/combat/targeting.py",),
        groups={"combat": ("src/game/combat/targeting.py",)},
        blocks=("combat-rules",),
        suggested_tests=("tests/test_targeting.py",),
        warnings=(),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_scout_bundle(project_root, context, "Tune bandit targeting")
    output = buffer.getvalue()
    assert "CHARTER SCOUT BUNDLE" in output
    assert "Tune bandit targeting" in output
    assert "combat-rules" in output
    assert "scout_packet_schema.md" in output
