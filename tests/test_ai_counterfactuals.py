from __future__ import annotations

import pytest

from game.dev.ai_counterfactuals import (
    NO_EFFECT,
    PROMISING,
    REGRESSION,
    CounterfactualVariant,
    DecisionWeightOverride,
    SkillOverride,
    _recommendation,
    _recommendation_order,
    apply_counterfactual_variant,
    format_counterfactual_sweep,
    run_counterfactual_sweep,
)
from game.dev.train_enemy_ai import TrainingRunConfig
from tests.conftest import get_definitions


def test_skill_override_clones_definitions_without_mutating_baseline() -> None:
    definitions = get_definitions()
    original_damage = definitions.skills["maw_slam"].damage

    changed = apply_counterfactual_variant(
        definitions,
        CounterfactualVariant(
            variant_id="maw_slam_plus_1",
            skill_overrides=(
                SkillOverride(
                    "maw_slam",
                    {"damage": original_damage + 1},
                ),
            ),
        ),
    )

    assert changed is not definitions
    assert changed.skills["maw_slam"].damage == original_damage + 1
    assert definitions.skills["maw_slam"].damage == original_damage


def test_unknown_override_id_fails_clearly() -> None:
    definitions = get_definitions()

    with pytest.raises(ValueError, match="Unknown skill override id: missing"):
        apply_counterfactual_variant(
            definitions,
            CounterfactualVariant(
                variant_id="missing",
                skill_overrides=(SkillOverride("missing", {"damage": 1}),),
            ),
        )


def test_weight_only_counterfactual_runs_with_policy_override() -> None:
    definitions = get_definitions()
    config = TrainingRunConfig(
        definitions=definitions,
        encounter_ids=("road_bandits",),
        seeds=1,
        max_rounds=1,
    )

    summary = run_counterfactual_sweep(
        config,
        (
            CounterfactualVariant(
                variant_id="weight_only",
                decision_weight_overrides=(DecisionWeightOverride("damage_pressure", 99),),
            ),
        ),
    )

    assert summary.results[0].summary.learned_weights["damage_pressure"] == 99


def test_counterfactual_sweep_is_deterministic_and_formats_results() -> None:
    definitions = get_definitions()
    original_damage = definitions.skills["maw_slam"].damage
    config = TrainingRunConfig(
        definitions=definitions,
        route_id="opening_critical_path",
        seeds=1,
        max_rounds=1,
    )
    variants = (
        CounterfactualVariant(
            variant_id="maw_slam_plus_1",
            skill_overrides=(SkillOverride("maw_slam", {"damage": 4}),),
        ),
    )

    first = run_counterfactual_sweep(config, variants)
    second = run_counterfactual_sweep(config, variants)
    text = format_counterfactual_sweep(first)

    assert [result.variant_id for result in first.results] == [
        result.variant_id for result in second.results
    ]
    assert first.results[0].score == second.results[0].score
    assert "Enemy Counterfactual Sweep" in text
    assert "maw_slam_plus_1" in text
    assert "recommendation" in text
    assert "confidence" in text
    assert "paired seeds" in text
    assert "gates" in text
    assert "package health" in text
    assert "route envelope" in text
    assert definitions.skills["maw_slam"].damage == original_damage


def test_zero_delta_counterfactual_is_no_effect() -> None:
    definitions = get_definitions()
    summary = run_counterfactual_sweep(
        TrainingRunConfig(
            definitions=definitions,
            route_id="opening_critical_path",
            seeds=1,
            max_rounds=1,
        ),
        (CounterfactualVariant(variant_id="no_op"),),
    )

    result = summary.results[0]

    assert result.recommendation == NO_EFFECT
    assert result.score == 0
    assert result.reward_delta == 0
    assert result.completion_delta == 0


def test_no_effect_recommendation_ranks_below_actual_improvement() -> None:
    assert _recommendation_order(PROMISING) < _recommendation_order(NO_EFFECT)


def test_package_noise_below_threshold_is_no_effect() -> None:
    recommendation, reasons = _recommendation(
        reward_delta=0,
        completion_delta=0,
        package_health_delta=0,
        package_metric_delta=5,
        route_envelope_delta=0,
    )

    assert recommendation == NO_EFFECT
    assert reasons == ("no meaningful metric delta",)


def test_reward_improvement_without_regression_is_promising() -> None:
    recommendation, reasons = _recommendation(
        reward_delta=250,
        completion_delta=0,
        package_health_delta=0,
        package_metric_delta=0,
        route_envelope_delta=0,
    )

    assert recommendation == PROMISING
    assert "pressure reward improved" in reasons


def test_reward_regression_is_regression() -> None:
    recommendation, reasons = _recommendation(
        reward_delta=-250,
        completion_delta=0,
        package_health_delta=0,
        package_metric_delta=0,
        route_envelope_delta=0,
    )

    assert recommendation == REGRESSION
    assert "pressure reward regressed" in reasons
