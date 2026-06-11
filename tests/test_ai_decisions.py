from __future__ import annotations

from game.combat.enemy_learning import EnemyDecisionEpisode, EnemyPressureMetrics
from game.dev.ai_counterfactuals import (
    MIXED,
    NO_EFFECT,
    PROMISING,
    RECOMMENDED,
    REGRESSION,
    CounterfactualResult,
    CounterfactualSweepSummary,
)
from game.dev.ai_decisions import (
    GATE_FAIL,
    build_balance_decision_report,
    format_balance_decision_report,
    paired_seed_delta,
)
from game.dev.ai_packages import PackageHealthResult, PackageReport
from game.dev.train_enemy_ai import RouteTrainingResult, TrainingRunSummary


def test_zero_delta_variant_is_no_effect_with_failed_meaningful_delta_gate() -> None:
    baseline = _summary(episodes=(_episode(seed=1, reward=100, final_hp=12),))
    result = _result(
        summary=baseline,
        reward_delta=0,
        completion_delta=0,
        package_health_delta=0,
        package_metric_delta=0,
        route_envelope_delta=0,
        recommendation=NO_EFFECT,
        reasons=("no meaningful metric delta",),
    )

    report = build_balance_decision_report(_sweep(baseline, _package_report(), result))
    decision = report.decisions[0]

    assert decision.recommendation == NO_EFFECT
    assert decision.confidence == "high"
    assert decision.score == 0
    assert _gate_status(decision, "meaningful_delta_present") == GATE_FAIL


def test_clear_improvement_is_promising_or_recommended_when_gates_pass() -> None:
    baseline = _summary(episodes=(_episode(seed=1, reward=100, final_hp=14),))
    candidate = _summary(episodes=(_episode(seed=1, reward=500, final_hp=8),))
    result = _result(
        summary=candidate,
        reward_delta=400,
        completion_delta=0,
        package_health_delta=0,
        package_metric_delta=20,
        route_envelope_delta=0,
        recommendation=PROMISING,
        reasons=("pressure reward improved", "key package metrics improved"),
    )

    decision = build_balance_decision_report(
        _sweep(baseline, _package_report(), result)
    ).decisions[0]

    assert decision.recommendation in {PROMISING, RECOMMENDED}
    assert all(gate.status != GATE_FAIL for gate in decision.gates)
    assert decision.paired_seed_delta.seeds_improved == 1


def test_reward_improvement_with_package_health_regression_is_mixed() -> None:
    baseline = _summary(episodes=(_episode(seed=1, reward=100, final_hp=14),))
    candidate = _summary(episodes=(_episode(seed=1, reward=500, final_hp=8),))
    result = _result(
        summary=candidate,
        package_report=_package_report(status="WARN"),
        reward_delta=400,
        package_health_delta=-2,
        package_metric_delta=20,
        recommendation=MIXED,
        reasons=("pressure reward improved", "package health regressed"),
    )

    decision = build_balance_decision_report(
        _sweep(baseline, _package_report(status="PASS"), result)
    ).decisions[0]

    assert decision.recommendation == MIXED
    assert _gate_status(decision, "package_health_not_regressed") == GATE_FAIL


def test_reward_regression_is_regression() -> None:
    baseline = _summary(episodes=(_episode(seed=1, reward=500, final_hp=8),))
    candidate = _summary(episodes=(_episode(seed=1, reward=100, final_hp=14),))
    result = _result(
        summary=candidate,
        reward_delta=-400,
        package_metric_delta=-20,
        recommendation=REGRESSION,
        reasons=("pressure reward regressed",),
    )

    decision = build_balance_decision_report(
        _sweep(baseline, _package_report(), result)
    ).decisions[0]

    assert decision.recommendation == REGRESSION
    assert decision.confidence == "low"


def test_route_envelope_failure_blocks_recommendation() -> None:
    baseline = _summary(
        route_id="opening_critical_path",
        route_results=tuple(_route_result(seed, completed=True) for seed in range(1, 11)),
    )
    candidate = _summary(
        route_id="opening_critical_path",
        route_results=tuple(
            _route_result(seed, completed=False, failed_at="shallow_cave")
            for seed in range(1, 11)
        ),
    )
    result = _result(
        summary=candidate,
        reward_delta=500,
        completion_delta=-10,
        package_metric_delta=20,
        route_envelope_delta=-3,
        recommendation=MIXED,
        reasons=("pressure reward improved", "route envelope regressed"),
    )

    decision = build_balance_decision_report(
        _sweep(baseline, _package_report(), result)
    ).decisions[0]

    assert _gate_status(decision, "route_envelope_not_fail") == GATE_FAIL
    assert decision.recommendation not in {PROMISING, RECOMMENDED}
    assert decision.candidate_envelope_status == "FAIL"


def test_package_health_regression_gate_blocks_recommendation() -> None:
    baseline = _summary(episodes=(_episode(seed=1, reward=100, final_hp=12),))
    candidate = _summary(episodes=(_episode(seed=1, reward=500, final_hp=8),))
    result = _result(
        summary=candidate,
        package_report=_package_report(status="FAIL"),
        reward_delta=400,
        package_health_delta=-3,
        package_metric_delta=20,
        recommendation=MIXED,
    )

    decision = build_balance_decision_report(
        _sweep(baseline, _package_report(status="PASS"), result)
    ).decisions[0]

    assert _gate_status(decision, "package_health_not_regressed") == GATE_FAIL
    assert decision.recommendation == MIXED


def test_paired_seed_comparison_counts_improved_worsened_and_unchanged() -> None:
    baseline = _summary(
        route_id="opening_critical_path",
        route_results=(
            _route_result(1, reward=100, final_hp=12),
            _route_result(2, reward=300, final_hp=8),
            _route_result(3, reward=100, final_hp=12),
        ),
    )
    candidate = _summary(
        route_id="opening_critical_path",
        route_results=(
            _route_result(1, reward=400, final_hp=6),
            _route_result(2, reward=100, final_hp=12),
            _route_result(3, reward=100, final_hp=12),
        ),
    )

    paired = paired_seed_delta(baseline, candidate)

    assert paired.seeds_compared == 3
    assert paired.seeds_improved == 1
    assert paired.seeds_worsened == 1
    assert paired.seeds_unchanged == 1
    assert paired.average_reward_delta == 100 / 3
    assert paired.average_final_hp_delta == -2 / 3


def test_decision_report_format_includes_card_fields() -> None:
    baseline = _summary(episodes=(_episode(seed=1, reward=100, final_hp=12),))
    result = _result(summary=baseline, recommendation=NO_EFFECT)

    text = format_balance_decision_report(
        build_balance_decision_report(_sweep(baseline, _package_report(), result))
    )

    assert "recommendation NO_EFFECT" in text
    assert "confidence high" in text
    assert "paired seeds" in text
    assert "gates" in text
    assert "reason:" in text


def _gate_status(decision, gate_id: str) -> str:
    return next(gate.status for gate in decision.gates if gate.gate_id == gate_id)


def _episode(
    *,
    encounter_id: str = "road_bandits",
    seed: int = 1,
    reward: int = 0,
    final_hp: int = 10,
) -> EnemyDecisionEpisode:
    return EnemyDecisionEpisode(
        encounter_id=encounter_id,
        encounter_name=encounter_id,
        seed=seed,
        records=(),
        final_victor="heroes",
        total_reward=reward,
        metrics=EnemyPressureMetrics(final_hero_hp_total=final_hp),
    )


def _route_result(
    seed: int,
    *,
    completed: bool = True,
    failed_at: str | None = None,
    reward: int = 100,
    final_hp: int = 12,
) -> RouteTrainingResult:
    episode = _episode(seed=seed, reward=reward, final_hp=final_hp)
    return RouteTrainingResult(
        route_id="opening_critical_path",
        seed=seed,
        episodes=(episode,),
        encounters=(),
        completed=completed,
        failed_at_encounter_id=None if completed else failed_at or "cave_mini_boss",
        final_hero_hp_total=final_hp,
        final_hero_effort_total=2,
    )


def _summary(
    *,
    episodes: tuple[EnemyDecisionEpisode, ...] = (),
    route_results: tuple[RouteTrainingResult, ...] = (),
    route_id: str = "",
) -> TrainingRunSummary:
    learned_episodes = episodes or tuple(
        episode for result in route_results for episode in result.episodes
    )
    return TrainingRunSummary(
        encounter_ids=("road_bandits",),
        seed_count=max(1, len(episodes) or len(route_results)),
        hero_policy_id="mixed",
        evaluation_hero_policy_ids=("mixed",),
        policy_scope_ids=("global",),
        preset_id="fresh",
        route_id=route_id,
        enemy_wait_mode="none",
        enemy_movement_mode="recovery_only",
        heuristic_episodes=learned_episodes,
        learned_episodes=learned_episodes,
        heuristic_route_results=route_results,
        learned_route_results=route_results,
        learned_weights={},
        encounter_breakdowns=(),
        policy_evaluations=(),
    )


def _package_report(status: str = "PASS") -> PackageReport:
    return PackageReport(
        route_id="",
        preset_id="fresh",
        hero_policy_id="mixed",
        seed_count=1,
        results=(
            PackageHealthResult(
                package_id="maw_package",
                status=status,
                details=("synthetic",),
                metric_values={"grabs": 1},
            ),
        ),
    )


def _result(
    *,
    summary: TrainingRunSummary,
    package_report: PackageReport | None = None,
    reward_delta: int = 0,
    completion_delta: int = 0,
    package_health_delta: int = 0,
    package_metric_delta: int = 0,
    route_envelope_delta: int = 0,
    recommendation: str = NO_EFFECT,
    reasons: tuple[str, ...] = (),
) -> CounterfactualResult:
    return CounterfactualResult(
        variant_id="variant",
        description="synthetic variant",
        summary=summary,
        package_report=package_report or _package_report(),
        score=(
            reward_delta
            + completion_delta * 250
            + package_health_delta * 100
            + package_metric_delta
            + route_envelope_delta * 100
        ),
        reward_delta=reward_delta,
        completion_delta=completion_delta,
        package_health_delta=package_health_delta,
        package_metric_delta=package_metric_delta,
        route_envelope_delta=route_envelope_delta,
        recommendation=recommendation,
        reasons=reasons,
    )


def _sweep(
    baseline: TrainingRunSummary,
    baseline_report: PackageReport,
    result: CounterfactualResult,
) -> CounterfactualSweepSummary:
    return CounterfactualSweepSummary(
        baseline_summary=baseline,
        baseline_package_report=baseline_report,
        results=(result,),
    )
