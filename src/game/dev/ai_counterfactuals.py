"""Dev-only counterfactual sweeps for enemy/package research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from game.content.definitions import GameDefinitions
from game.dev.ai_packages import PackageReport, evaluate_enemy_packages
from game.dev.route_lab import score_route_envelope
from game.dev.train_enemy_ai import TrainingRunConfig, TrainingRunSummary, run_training_harness

RECOMMENDED = "RECOMMENDED"
PROMISING = "PROMISING"
MIXED = "MIXED"
NO_EFFECT = "NO_EFFECT"
REGRESSION = "REGRESSION"
MEANINGFUL_REWARD_DELTA = 100
MEANINGFUL_PACKAGE_METRIC_DELTA = 10


@dataclass(frozen=True)
class SkillOverride:
    skill_id: str
    updates: Mapping[str, Any]


@dataclass(frozen=True)
class EnemyOverride:
    enemy_id: str
    updates: Mapping[str, Any]


@dataclass(frozen=True)
class DecisionWeightOverride:
    feature_name: str
    weight: int


@dataclass(frozen=True)
class CounterfactualVariant:
    variant_id: str
    description: str = ""
    skill_overrides: tuple[SkillOverride, ...] = ()
    enemy_overrides: tuple[EnemyOverride, ...] = ()
    decision_weight_overrides: tuple[DecisionWeightOverride, ...] = ()


@dataclass(frozen=True)
class CounterfactualResult:
    variant_id: str
    description: str
    summary: TrainingRunSummary
    package_report: PackageReport
    score: int
    reward_delta: int
    completion_delta: int
    package_health_delta: int = 0
    package_metric_delta: int = 0
    route_envelope_delta: int = 0
    recommendation: str = NO_EFFECT
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterfactualSweepSummary:
    baseline_summary: TrainingRunSummary
    baseline_package_report: PackageReport
    results: tuple[CounterfactualResult, ...]


def apply_counterfactual_variant(
    definitions: GameDefinitions,
    variant: CounterfactualVariant,
) -> GameDefinitions:
    skills = dict(definitions.skills)
    enemies = dict(definitions.enemies)
    for skill_override in variant.skill_overrides:
        if skill_override.skill_id not in skills:
            raise ValueError(f"Unknown skill override id: {skill_override.skill_id}")
        skills[skill_override.skill_id] = skills[skill_override.skill_id].model_copy(
            update=dict(skill_override.updates)
        )
    for enemy_override in variant.enemy_overrides:
        if enemy_override.enemy_id not in enemies:
            raise ValueError(f"Unknown enemy override id: {enemy_override.enemy_id}")
        enemies[enemy_override.enemy_id] = enemies[enemy_override.enemy_id].model_copy(
            update=dict(enemy_override.updates)
        )
    return replace(
        definitions,
        skills_file=definitions.skills_file.model_copy(update={"skills": skills}),
        enemies_file=definitions.enemies_file.model_copy(update={"enemies": enemies}),
    )


def run_counterfactual_sweep(
    config: TrainingRunConfig,
    variants: Sequence[CounterfactualVariant],
) -> CounterfactualSweepSummary:
    baseline = run_training_harness(config)
    baseline_report = evaluate_enemy_packages(baseline)
    results = []
    for variant in variants:
        variant_definitions = apply_counterfactual_variant(
            baseline_definition(config, baseline),
            variant,
        )
        candidate_config = replace(config, definitions=variant_definitions)
        if variant.decision_weight_overrides:
            candidate_config = replace(
                candidate_config,
                learned_weight_overrides={
                    override.feature_name: override.weight
                    for override in variant.decision_weight_overrides
                },
            )
        candidate = run_training_harness(candidate_config)
        package_report = evaluate_enemy_packages(candidate)
        comparison = _compare_counterfactual(
            baseline,
            baseline_report,
            candidate,
            package_report,
        )
        results.append(
            CounterfactualResult(
                variant_id=variant.variant_id,
                description=variant.description,
                summary=candidate,
                package_report=package_report,
                score=comparison.score,
                reward_delta=candidate.learned_total_reward - baseline.learned_total_reward,
                completion_delta=_completion_count(candidate) - _completion_count(baseline),
                package_health_delta=comparison.package_health_delta,
                package_metric_delta=comparison.package_metric_delta,
                route_envelope_delta=comparison.route_envelope_delta,
                recommendation=comparison.recommendation,
                reasons=comparison.reasons,
            )
        )
    return CounterfactualSweepSummary(
        baseline_summary=baseline,
        baseline_package_report=baseline_report,
        results=tuple(
            sorted(
                results,
                key=lambda result: (
                    _recommendation_order(result.recommendation),
                    -result.score,
                    -result.reward_delta,
                    result.variant_id,
                ),
            )
        ),
    )


def format_counterfactual_sweep(summary: CounterfactualSweepSummary) -> str:
    from game.dev.ai_decisions import (
        build_balance_decision_report,
        format_balance_decision_report,
    )

    return format_balance_decision_report(build_balance_decision_report(summary))


def baseline_definition(
    config: TrainingRunConfig,
    baseline: TrainingRunSummary,
) -> GameDefinitions:
    if config.definitions is not None:
        return config.definitions
    if baseline.heuristic_episodes:
        # The summary intentionally does not retain definitions. Load through the normal
        # harness path for CLI use where config.definitions is omitted.
        from game.data.loaders import load_game_definitions

        return load_game_definitions()
    from game.data.loaders import load_game_definitions

    return load_game_definitions()


@dataclass(frozen=True)
class _CounterfactualComparison:
    score: int
    recommendation: str
    reasons: tuple[str, ...]
    package_health_delta: int
    package_metric_delta: int
    route_envelope_delta: int


def _compare_counterfactual(
    baseline: TrainingRunSummary,
    baseline_report: PackageReport,
    candidate: TrainingRunSummary,
    package_report: PackageReport,
) -> _CounterfactualComparison:
    reward_delta = candidate.learned_total_reward - baseline.learned_total_reward
    completion_delta = _completion_count(candidate) - _completion_count(baseline)
    package_health_delta = _package_health_delta(baseline_report, package_report)
    package_metric_delta = _package_metric_delta(baseline_report, package_report)
    route_envelope_delta = _route_envelope_delta(baseline, candidate)
    score = (
        reward_delta
        + completion_delta * 250
        + package_health_delta * 100
        + package_metric_delta
        + route_envelope_delta * 100
    )
    recommendation, reasons = _recommendation(
        reward_delta=reward_delta,
        completion_delta=completion_delta,
        package_health_delta=package_health_delta,
        package_metric_delta=package_metric_delta,
        route_envelope_delta=route_envelope_delta,
    )
    if recommendation == NO_EFFECT:
        score = 0
    return _CounterfactualComparison(
        score=score,
        recommendation=recommendation,
        reasons=reasons,
        package_health_delta=package_health_delta,
        package_metric_delta=package_metric_delta,
        route_envelope_delta=route_envelope_delta,
    )


def _recommendation(
    *,
    reward_delta: int,
    completion_delta: int,
    package_health_delta: int,
    package_metric_delta: int,
    route_envelope_delta: int,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    reward_meaningful = abs(reward_delta) >= MEANINGFUL_REWARD_DELTA
    metric_meaningful = abs(package_metric_delta) >= MEANINGFUL_PACKAGE_METRIC_DELTA
    any_meaningful = any(
        (
            reward_meaningful,
            completion_delta != 0,
            package_health_delta != 0,
            metric_meaningful,
            route_envelope_delta != 0,
        )
    )
    if not any_meaningful:
        return NO_EFFECT, ("no meaningful metric delta",)
    regressions = []
    improvements = []
    if reward_delta <= -MEANINGFUL_REWARD_DELTA:
        regressions.append("pressure reward regressed")
    elif reward_delta >= MEANINGFUL_REWARD_DELTA:
        improvements.append("pressure reward improved")
    if completion_delta < 0:
        regressions.append("route completions regressed")
    elif completion_delta > 0:
        improvements.append("route completions improved")
    if package_health_delta < 0:
        regressions.append("package health regressed")
    elif package_health_delta > 0:
        improvements.append("package health improved")
    if route_envelope_delta < 0:
        regressions.append("route envelope regressed")
    elif route_envelope_delta > 0:
        improvements.append("route envelope improved")
    if metric_meaningful and package_metric_delta > 0:
        improvements.append("key package metrics improved")
    elif metric_meaningful and package_metric_delta < 0:
        regressions.append("key package metrics regressed")

    reasons.extend(improvements)
    reasons.extend(regressions)
    if regressions and improvements:
        return MIXED, tuple(reasons)
    if regressions:
        return REGRESSION, tuple(reasons)
    if completion_delta > 0 or package_health_delta > 0 or route_envelope_delta > 0:
        return RECOMMENDED, tuple(reasons)
    return PROMISING, tuple(reasons)


def _package_health_delta(
    baseline_report: PackageReport,
    candidate_report: PackageReport,
) -> int:
    baseline = {result.package_id: result.status for result in baseline_report.results}
    return sum(
        _package_status_value(result.status)
        - _package_status_value(baseline.get(result.package_id, "WARN"))
        for result in candidate_report.results
    )


def _package_metric_delta(
    baseline_report: PackageReport,
    candidate_report: PackageReport,
) -> int:
    baseline = {result.package_id: result.metric_values for result in baseline_report.results}
    delta = 0
    for result in candidate_report.results:
        baseline_metrics = baseline.get(result.package_id, {})
        for metric_id in _KEY_PACKAGE_METRICS:
            before = baseline_metrics.get(metric_id)
            after = result.metric_values.get(metric_id)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                delta += _metric_direction(metric_id) * int(after - before)
    return delta


def _route_envelope_delta(
    baseline: TrainingRunSummary,
    candidate: TrainingRunSummary,
) -> int:
    if not baseline.route_id or not baseline.learned_route_results:
        return 0
    envelope_id = _default_route_envelope_id(baseline.route_id)
    baseline_score = score_route_envelope(
        baseline.learned_route_summary,
        envelope_id=envelope_id,
    )
    candidate_score = score_route_envelope(
        candidate.learned_route_summary,
        envelope_id=envelope_id,
    )
    return _route_status_value(candidate_score.status) - _route_status_value(
        baseline_score.status
    )


_KEY_PACKAGE_METRICS = {
    "grabs",
    "grab_to_bite_same_target",
    "support_grabs",
    "bone_soldier_guarded_boss",
    "marks_applied",
    "exploited_by_enemy_hit",
    "vulnerable_payoffs",
    "ignored_marked_legal_attacks",
}


def _metric_direction(metric_id: str) -> int:
    if metric_id == "ignored_marked_legal_attacks":
        return -1
    return 1


def _package_status_value(status: str) -> int:
    return {
        "FAIL": -2,
        "WARN": -1,
        "OK_LOW_USE": 0,
        "PASS": 1,
    }.get(status, -1)


def _route_status_value(status: str) -> int:
    return {
        "FAIL": -2,
        "WARN": -1,
        "PASS": 1,
    }.get(status, -1)


def _recommendation_order(recommendation: str) -> int:
    return {
        RECOMMENDED: 0,
        PROMISING: 1,
        MIXED: 2,
        NO_EFFECT: 3,
        REGRESSION: 4,
    }.get(recommendation, 5)


def _default_route_envelope_id(route_id: str) -> str:
    if route_id == "opening_pressure_path":
        return "optional_pressure_path"
    return "critical_path"


def _completion_count(summary: TrainingRunSummary) -> int:
    if summary.learned_route_results:
        return sum(1 for result in summary.learned_route_results if result.completed)
    return summary.learned_victors.get("heroes", 0)


__all__ = [
    "CounterfactualResult",
    "CounterfactualSweepSummary",
    "CounterfactualVariant",
    "DecisionWeightOverride",
    "EnemyOverride",
    "SkillOverride",
    "apply_counterfactual_variant",
    "format_counterfactual_sweep",
    "run_counterfactual_sweep",
]
