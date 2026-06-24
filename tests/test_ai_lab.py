from __future__ import annotations

import pytest

from game.dev import ai_lab

pytestmark = pytest.mark.slow


def test_ai_lab_enemy_packages_command_outputs_package_health(capsys) -> None:
    result = ai_lab.main(
        [
            "enemy-packages",
            "--route",
            "opening_critical_path",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Enemy AI Training Harness" in text
    assert "Package Health:" in text


def test_ai_lab_generated_route_command_outputs_route_lab(capsys) -> None:
    result = ai_lab.main(
        [
            "generated-route",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
            "--profile",
            "breach_probe",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Generated Route Lab" in text
    assert "Envelope generated_maze_scout:" in text


def test_ai_lab_route_sweep_ranks_profiles(capsys) -> None:
    result = ai_lab.main(
        [
            "route-sweep",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
            "--profile",
            "breach_probe",
            "--profile",
            "tight_probe",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Route Sweep" in text
    assert "breach_probe:" in text
    assert "tight_probe:" in text


def test_ai_lab_enemy_sweep_outputs_decision_cards(capsys) -> None:
    result = ai_lab.main(
        [
            "enemy-sweep",
            "--route",
            "opening_critical_path",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Enemy Counterfactual Sweep" in text
    assert "recommendation" in text
    assert "confidence" in text
    assert "gates" in text


def test_ai_lab_discover_tactics_outputs_tactic_report(capsys) -> None:
    result = ai_lab.main(
        [
            "discover-tactics",
            "--package",
            "maw_package",
            "--route",
            "opening_critical_path",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
            "--eval-hero-policy",
            "mixed",
            "--emphasis-scale",
            "1.0",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Tactic Discovery: maw_package" in text
    assert "Discovered tactic candidates:" in text
    assert "pattern:" in text
    assert "risks:" in text
    assert "decision:" in text


def test_ai_lab_balance_breach_fights_dry_run_outputs_candidates(capsys) -> None:
    result = ai_lab.main(
        [
            "balance-breach-fights",
            "--dry-run",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Breach Fight Balance Lab" in text
    assert "Ranked candidates:" in text
    assert "Applied: no" in text or "Selected: none" in text


def test_ai_lab_oracle_report_command_runs(capsys) -> None:
    result = ai_lab.main(
        [
            "oracle-report",
            "--route",
            "opening_critical_path",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "AI Lab Oracle Report" in text


def test_ai_lab_oracle_report_output_structure(capsys) -> None:
    result = ai_lab.main(
        [
            "oracle-report",
            "--route",
            "opening_pressure_path",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Oracle mode: score-only heuristic (depth-0)" in text
    assert "enemy actions checked:" in text
    assert "oracle misses:" in text
    assert "Counterplay:" in text
    assert "Findings:" in text
    assert "noncompletion:" in text


def test_ai_lab_enemy_packages_still_works(capsys) -> None:
    result = ai_lab.main(
        [
            "enemy-packages",
            "--route",
            "opening_critical_path",
            "--seeds",
            "1",
            "--max-rounds",
            "1",
        ]
    )

    text = capsys.readouterr().out

    assert result == 0
    assert "Enemy AI Training Harness" in text
    assert "Package Health:" in text
