from __future__ import annotations

from game.dev.route_lab import (
    GeneratedRouteLabConfig,
    RouteLabConfig,
    format_generated_route_lab_summary,
    format_route_lab_summary,
    run_generated_route_lab,
    run_route_lab,
    score_route_envelope,
)
from game.dev.train_enemy_ai import RoutePressureSummary
from tests.conftest import get_definitions


def test_authored_route_lab_summary_includes_completion_and_envelope_score() -> None:
    summary = run_route_lab(
        RouteLabConfig(
            definitions=get_definitions(),
            route_id="opening_critical_path",
            seeds=1,
            max_rounds=1,
        )
    )
    text = format_route_lab_summary(summary)

    assert summary.route_id == "opening_critical_path"
    assert summary.envelope_score.envelope_id == "critical_path"
    assert "Route Lab" in text
    assert "Completion:" in text
    assert "Envelope critical_path:" in text


def test_generated_route_lab_is_seed_deterministic() -> None:
    definitions = get_definitions()
    config = GeneratedRouteLabConfig(
        definitions=definitions,
        seeds=2,
        max_rounds=1,
        strategy_id="take_all_optional",
        pressure_profile_id="breach_probe",
    )

    first = run_generated_route_lab(config)
    second = run_generated_route_lab(config)

    assert [
        [encounter.encounter_id for encounter in run.encounters] for run in first.runs
    ] == [
        [encounter.encounter_id for encounter in run.encounters] for run in second.runs
    ]
    assert first.envelope_score == second.envelope_score


def test_generated_route_summary_includes_recipe_shape_and_final_condition() -> None:
    summary = run_generated_route_lab(
        GeneratedRouteLabConfig(
            definitions=get_definitions(),
            seeds=1,
            max_rounds=1,
            strategy_id="take_all_optional",
            pressure_profile_id="marked_hunt",
        )
    )
    text = format_generated_route_lab_summary(summary)

    assert summary.pressure_profile_id == "marked_hunt"
    assert summary.envelope_score.envelope_id == "generated_maze_hunt"
    assert summary.runs[0].route.recipe is not None
    assert summary.runs[0].route.recipe.include_hunt
    assert "Generated Route Lab" in text
    assert "Route shape:" in text
    assert "Envelope generated_maze_hunt:" in text


def test_generated_route_lab_does_not_mutate_global_definitions_or_memory() -> None:
    definitions = get_definitions()
    encounter_ids = tuple(sorted(definitions.encounters))

    run_generated_route_lab(
        GeneratedRouteLabConfig(
            definitions=definitions,
            seeds=1,
            max_rounds=1,
            pressure_profile_id="breach_probe",
        )
    )

    assert tuple(sorted(definitions.encounters)) == encounter_ids


def test_route_envelope_completion_above_target_band_warns_not_passes() -> None:
    score = score_route_envelope(
        RoutePressureSummary(
            route_count=10,
            completed_count=10,
            failed_at_counts={},
            average_reward=0,
            average_final_hero_hp=12,
            average_final_hero_effort=2,
            average_downs=0,
            average_deaths=0,
            average_mortal_wounds=0,
            average_hp_entering_cave_mini_boss=0,
            average_hp_leaving_cave_mini_boss=0,
        ),
        envelope_id="generated_maze_scout",
    )

    assert score.status == "WARN"
    assert "completion above target band" in score.warnings


def test_route_envelope_inside_target_bands_passes() -> None:
    score = score_route_envelope(
        RoutePressureSummary(
            route_count=10,
            completed_count=8,
            failed_at_counts={},
            average_reward=0,
            average_final_hero_hp=12,
            average_final_hero_effort=2,
            average_downs=0,
            average_deaths=0,
            average_mortal_wounds=0,
            average_hp_entering_cave_mini_boss=16,
            average_hp_leaving_cave_mini_boss=8,
        ),
        envelope_id="critical_path",
    )

    assert score.status == "PASS"
    assert score.warnings == ()


def test_route_envelope_severe_under_completion_fails() -> None:
    score = score_route_envelope(
        RoutePressureSummary(
            route_count=10,
            completed_count=2,
            failed_at_counts={"shallow_cave": 8},
            average_reward=0,
            average_final_hero_hp=0,
            average_final_hero_effort=0,
            average_downs=0,
            average_deaths=0,
            average_mortal_wounds=0,
            average_hp_entering_cave_mini_boss=0,
            average_hp_leaving_cave_mini_boss=0,
        ),
        envelope_id="critical_path",
    )

    assert score.status == "FAIL"
    assert "completion below target band" in score.warnings


def test_route_envelope_doomed_boss_entry_fails() -> None:
    score = score_route_envelope(
        RoutePressureSummary(
            route_count=10,
            completed_count=8,
            failed_at_counts={},
            average_reward=0,
            average_final_hero_hp=5,
            average_final_hero_effort=1,
            average_downs=0,
            average_deaths=0,
            average_mortal_wounds=0,
            average_hp_entering_cave_mini_boss=3,
            average_hp_leaving_cave_mini_boss=0,
        ),
        envelope_id="critical_path",
    )

    assert score.status == "FAIL"
    assert "party reaches boss already doomed" in score.warnings
