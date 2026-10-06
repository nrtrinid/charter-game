from __future__ import annotations

from pathlib import Path

import pytest

from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import _load_frozen_policies
from game.dev.ml_cp6 import PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.dev.ml_cp9 import _observation_audit, _sample_sources
from game.ml.counterfactual import OutcomePreference, stable_digest
from game.ml.models import LinearCandidateScorer
from game.ml.route_policy_learning import (
    ORACLE_FEATURE_NAMES,
    ORACLE_SCHEMA_ID,
    PERSISTENT_FEATURE_NAMES,
    PRIMARY_FEATURE_NAMES,
    PRIMARY_SCHEMA_ID,
    FrozenRouteRankerPolicy,
    RouteNormalization,
    RouteOutcome,
    collect_route_source_decisions,
    compare_route_outcomes,
    run_route_counterfactual_branch,
    run_route_trajectory,
)
from game.ml.route_termination import run_corrected_stateful_route


def _route(route_id: str):
    return next(route for route in PRIMARY_ROUTES if route.route_id == route_id)


def _outcome(
    *,
    enemy: bool = False,
    completed: bool = True,
    hp: int = 40,
    effort: int = 20,
    downed: int = 0,
    deaths: int = 0,
    wounds: int = 0,
    strain_marks: int = 0,
    status: str | None = None,
) -> RouteOutcome:
    if status is None:
        status = "ROUTE_FAILED_ENEMY_VICTORY" if enemy else "ROUTE_COMPLETED"
    return RouteOutcome(
        terminal_status=status,
        enemy_victory=enemy,
        route_completed=completed,
        final_hero_hp=hp,
        final_hero_effort=effort,
        final_downed=downed,
        hero_deaths=deaths,
        mortal_wounds=wounds,
        strain_marks=strain_marks,
        final_company_digest="test",
    )


def test_primary_schema_adds_only_visible_persistent_aggregates() -> None:
    assert len(PRIMARY_FEATURE_NAMES) == 90
    assert len(PERSISTENT_FEATURE_NAMES) == 9
    assert "route_encounter_index" not in PRIMARY_FEATURE_NAMES
    assert "route_encounters_remaining" not in PRIMARY_FEATURE_NAMES
    assert "route_progress_fraction" not in PRIMARY_FEATURE_NAMES
    assert len(ORACLE_FEATURE_NAMES) == 93
    assert ORACLE_FEATURE_NAMES[-3:] == (
        "route_encounter_index",
        "route_encounters_remaining",
        "route_progress_fraction",
    )


def test_existing_observation_audit_preserves_zero_prescriptive_and_future_features() -> None:
    audit = _observation_audit()["existing_schema"]
    assert audit["feature_count"] == 81
    assert audit["prescriptive_count"] == 0
    assert audit["identity_leakage_count"] == 0
    assert audit["future_information_count"] == 0


def test_route_preference_uses_victory_then_pareto_without_scalarization() -> None:
    win = _outcome(enemy=True, completed=False)
    completion = _outcome()
    assert compare_route_outcomes(win, completion) == OutcomePreference.FIRST
    assert (
        compare_route_outcomes(win, _outcome(enemy=True, completed=False)) == OutcomePreference.TIE
    )

    worse_for_heroes = _outcome(hp=30, effort=15, wounds=2)
    better_for_heroes = _outcome(hp=40, effort=20, wounds=1)
    assert compare_route_outcomes(worse_for_heroes, better_for_heroes) == OutcomePreference.FIRST

    crossed = _outcome(hp=30, effort=25, wounds=1)
    assert compare_route_outcomes(crossed, better_for_heroes) == OutcomePreference.INCOMPARABLE


def test_real_retreat_never_enters_completion_pareto_ordering() -> None:
    retreat = _outcome(
        completed=False,
        hp=1,
        effort=0,
        wounds=9,
        status="ROUTE_ENDED_PRODUCTION_RETREAT",
    )
    completion = _outcome(hp=50, effort=20)
    assert compare_route_outcomes(retreat, completion) == OutcomePreference.INCOMPARABLE
    assert compare_route_outcomes(retreat, retreat) == OutcomePreference.INCOMPARABLE


def test_old_seed_route_trajectory_matches_cp7_production_faithful_result() -> None:
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    route = _route("fresh_hunt")
    expected = run_corrected_stateful_route(
        definitions,
        route,
        root_seed=40001,
        enemy_policy_name="pi2_cp4",
        enemy_policy=policies["pi2_cp4"],
        hero_policy_id="company_survival",
        safety_budget=SAFETY_BUDGET,
    )
    outcome, _digest, _encounters, _local = run_route_trajectory(
        definitions,
        route,
        root_seed=40001,
        enemy_policy_name="pi2_cp4",
        enemy_policy=policies["pi2_cp4"],
        hero_policy_id="company_survival",
        safety_budget=SAFETY_BUDGET,
    )
    assert outcome.terminal_status == expected["route_terminal_status"]
    assert outcome.final_company_digest == stable_digest(expected["final_company_state"])


def test_old_seed_source_action_control_replays_entire_route_exactly() -> None:
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    route = _route("authored_cave")
    sources = collect_route_source_decisions(
        definitions,
        route,
        root_seed=40001,
        source_policy=policies["pi2_cp4"],
        source_policy_name="pi2_cp4",
        hero_policy_id="company_survival",
        split="old_seed_test",
        safety_budget=SAFETY_BUDGET,
    )
    assert sources
    source = sources[0]
    result = run_route_counterfactual_branch(
        definitions,
        source,
        source.capture.source_choice,
        continuation_policy=policies["pi2_cp4"],
        continuation_policy_name="pi2_cp4",
        safety_budget=SAFETY_BUDGET,
    )
    assert result.reached_target
    assert result.replay_state_digest == source.capture.state_digest
    assert result.replay_rng_digest == source.capture.rng_digest
    assert result.replay_candidates == source.capture.candidates
    assert result.control_exact
    assert result.route_digest == source.baseline_route_digest


def test_sampling_is_deterministic_on_consumed_seed_metadata() -> None:
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    sources = collect_route_source_decisions(
        definitions,
        _route("light_to_hunt"),
        root_seed=40001,
        source_policy=policies["pi2_cp4"],
        source_policy_name="pi2_cp4",
        hero_policy_id="company_survival",
        split="old_seed_test",
        safety_budget=SAFETY_BUDGET,
    )
    count = min(5, len(sources))
    first = _sample_sources(sources, count)
    second = _sample_sources(tuple(reversed(sources)), count)
    assert [item.capture.state_digest for item in first] == [
        item.capture.state_digest for item in second
    ]


def test_primary_policy_rejects_route_context_oracle_normalization() -> None:
    normalization = RouteNormalization(
        feature_names=ORACLE_FEATURE_NAMES,
        means=tuple(0.0 for _ in ORACLE_FEATURE_NAMES),
        scales=tuple(1.0 for _ in ORACLE_FEATURE_NAMES),
        zero_variance=tuple(False for _ in ORACLE_FEATURE_NAMES),
        schema_id=ORACLE_SCHEMA_ID,
    )
    model = LinearCandidateScorer(len(ORACLE_FEATURE_NAMES))
    with pytest.raises(ValueError, match="primary tactical schema"):
        FrozenRouteRankerPolicy(model, normalization, policy_id="oracle")


def test_cp9_completed_artifact_root_is_ignored_generated_output() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "artifacts/ml/" in gitignore
    assert PRIMARY_SCHEMA_ID == "enemy-route-outcome-candidate-v1"
