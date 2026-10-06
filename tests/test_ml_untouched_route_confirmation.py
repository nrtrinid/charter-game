from __future__ import annotations

from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import _load_frozen_policies
from game.dev.ml_cp6 import PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.dev.ml_cp10 import _load_selected_pi4
from game.dev.ml_cp11 import (
    CONFIRMATION_SEEDS,
    EXPECTED_CASE_COUNT,
    EXPECTED_GATE_MODE,
    EXPECTED_MARGIN_THRESHOLD,
    EXPECTED_PI2_SHA,
    EXPECTED_PI4_SHA,
    PRIMARY_HERO_POLICY,
    _policy_identity,
    classify_confirmation,
    effect_compatibility,
)
from game.ml.conservative_route_policy import ConservativeRoutePolicy
from game.ml.route_policy_learning import RecordingRoutePolicy, run_route_trajectory


def _summary(*, diff: float, p: float, first_only: int, second_only: int) -> dict[str, float | int]:
    return {
        "win_rate_difference": diff,
        "mcnemar_exact_two_sided_p": p,
        "first_only_wins": first_only,
        "second_only_wins": second_only,
    }


def _fresh_pi4(anchor, ranker, support, selected) -> ConservativeRoutePolicy:
    return ConservativeRoutePolicy(
        anchor_policy=anchor,
        route_policy=ranker,
        support_profile=support,
        gate_mode=str(selected["gate_mode"]),
        margin_threshold=float(selected["margin_threshold"]),
        policy_id="pi4_conservative",
    )


def test_cp11_seed_and_route_authority_are_exact() -> None:
    assert CONFIRMATION_SEEDS[0] == 63001
    assert CONFIRMATION_SEEDS[-1] == 63064
    assert len(CONFIRMATION_SEEDS) == 64
    assert len(PRIMARY_ROUTES) == 6
    assert EXPECTED_CASE_COUNT == 384
    assert PRIMARY_HERO_POLICY == "company_survival"


def test_confirmation_classification_rules_are_frozen() -> None:
    assert (
        classify_confirmation(_summary(diff=0.05, p=0.01, first_only=3, second_only=12))
        == "CLEARLY_CONFIRMED"
    )
    assert (
        classify_confirmation(_summary(diff=0.05, p=0.20, first_only=3, second_only=12))
        == "SUPPORTED_BUT_UNDERPOWERED"
    )
    assert (
        classify_confirmation(_summary(diff=0.0, p=1.0, first_only=4, second_only=4))
        == "NO_REPLICATION"
    )
    assert (
        classify_confirmation(_summary(diff=-0.05, p=0.20, first_only=12, second_only=3))
        == "REGRESSION_SIGNAL"
    )
    assert (
        classify_confirmation(_summary(diff=-0.05, p=0.01, first_only=12, second_only=3))
        == "REFUTED"
    )


def test_cp10_effect_compatibility_rule_is_deterministic() -> None:
    assert effect_compatibility(0.06, [0.01, 0.11]) == "YES"
    assert effect_compatibility(-0.20, [-0.30, -0.10]) == "NO"
    assert effect_compatibility(-0.02, [-0.05, 0.01]) == "INCONCLUSIVE"


def test_frozen_policy_identity_matches_cp10_authority() -> None:
    identity = _policy_identity()
    assert identity["pi2"]["model_sha256"] == EXPECTED_PI2_SHA
    assert identity["pi4"]["model_sha256"] == EXPECTED_PI4_SHA
    assert identity["pi4"]["gate_mode"] == EXPECTED_GATE_MODE
    assert identity["pi4"]["margin_threshold"] == EXPECTED_MARGIN_THRESHOLD


def test_recording_wrapper_does_not_change_consumed_pi2_or_pi4_route() -> None:
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    ranker, _normalization, support, selected = _load_selected_pi4()
    route = next(route for route in PRIMARY_ROUTES if route.route_id == "fresh_hunt")
    seed = 72001

    raw_pi2_outcome, raw_pi2_digest, _encounters, _local = run_route_trajectory(
        definitions,
        route,
        root_seed=seed,
        enemy_policy_name="pi2_cp4",
        enemy_policy=policies["pi2_cp4"],
        hero_policy_id=PRIMARY_HERO_POLICY,
        safety_budget=SAFETY_BUDGET,
    )
    pi2_recorder = RecordingRoutePolicy(policies["pi2_cp4"])
    wrapped_pi2_outcome, wrapped_pi2_digest, _encounters, _local = run_route_trajectory(
        definitions,
        route,
        root_seed=seed,
        enemy_policy_name="pi2_cp4",
        enemy_policy=policies["pi2_cp4"],
        hero_policy_id=PRIMARY_HERO_POLICY,
        safety_budget=SAFETY_BUDGET,
        recording_policy=pi2_recorder,
    )
    assert wrapped_pi2_outcome == raw_pi2_outcome
    assert wrapped_pi2_digest == raw_pi2_digest
    assert pi2_recorder.captures

    raw_pi4 = _fresh_pi4(policies["pi2_cp4"], ranker, support, selected)
    raw_pi4_outcome, raw_pi4_digest, _encounters, _local = run_route_trajectory(
        definitions,
        route,
        root_seed=seed,
        enemy_policy_name="pi4_conservative",
        enemy_policy=raw_pi4,
        hero_policy_id=PRIMARY_HERO_POLICY,
        safety_budget=SAFETY_BUDGET,
    )
    wrapped_pi4 = _fresh_pi4(policies["pi2_cp4"], ranker, support, selected)
    pi4_recorder = RecordingRoutePolicy(wrapped_pi4)
    wrapped_pi4_outcome, wrapped_pi4_digest, _encounters, _local = run_route_trajectory(
        definitions,
        route,
        root_seed=seed,
        enemy_policy_name="pi4_conservative",
        enemy_policy=wrapped_pi4,
        hero_policy_id=PRIMARY_HERO_POLICY,
        safety_budget=SAFETY_BUDGET,
        recording_policy=pi4_recorder,
    )
    assert wrapped_pi4_outcome == raw_pi4_outcome
    assert wrapped_pi4_digest == raw_pi4_digest
    assert pi4_recorder.captures
    assert len(wrapped_pi4.audit) == len(pi4_recorder.captures)
