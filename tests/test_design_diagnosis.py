from __future__ import annotations

from game.dev.ai_decisions import GATE_FAIL, GATE_PASS, GATE_WARN
from game.dev.design_diagnosis import (
    SUMMARY_HEALTHY,
    SUMMARY_INSUFFICIENT_DATA,
    SUMMARY_PROMISING_PACKAGE,
    VERDICT_BLOCKED,
    VERDICT_HEALTHY,
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_NEEDS_TUNING,
    VERDICT_PROMISING,
    build_design_diagnosis,
    format_design_diagnosis,
)
from game.dev.maze_mark_guard_package import (
    DIAG_PAYOFF_NEVER_FIRES,
    DIAG_SMART_POLICY_ANSWERS_PACKAGE,
    DIAG_WARD_ROLE_INVISIBLE,
    GATE_INFO,
    MazeMarkGuardMetrics,
    PackageContractGateResult,
)
from game.dev.policy_band_report import (
    DIAG_TOO_PUNISHING,
    SCENARIO_GENERATED_SCOUT,
    BalanceGateResult,
    PolicyBandReport,
    PolicyRunMetrics,
    format_policy_band_report,
)


def _metric(
    hero_policy_id: str = "mixed",
    completion_rate: float = 0.5,
    *,
    maze_package: MazeMarkGuardMetrics | None = None,
) -> PolicyRunMetrics:
    return PolicyRunMetrics(
        hero_policy_id=hero_policy_id,
        completion_rate=completion_rate,
        victor_counts={"heroes": int(completion_rate * 100)},
        average_downs=0.0,
        average_deaths=0.0,
        average_final_hp=0.0,
        average_final_effort=0.0,
        marks_applied=0,
        marks_exploited=0,
        vulnerable_payoffs=0,
        maze_package=maze_package,
    )


def _report(
    *,
    gates: tuple[BalanceGateResult, ...] = (),
    package_gates: tuple[PackageContractGateResult, ...] = (),
    overall_status: str = GATE_PASS,
    package_overall_status: str = GATE_PASS,
    package_insufficient_samples: bool = False,
    scenario_kind: str = SCENARIO_GENERATED_SCOUT,
    policy_metrics: tuple[PolicyRunMetrics, ...] | None = None,
) -> PolicyBandReport:
    return PolicyBandReport(
        scenario_id="breach_probe",
        scenario_name="breach_probe",
        scenario_kind=scenario_kind,
        seed_count=3,
        policy_metrics=policy_metrics or (_metric(),),
        gates=gates,
        overall_status=overall_status,
        package_gates=package_gates,
        package_overall_status=package_overall_status,
        package_insufficient_samples=package_insufficient_samples,
    )


def test_healthy_report_produces_healthy_verdict() -> None:
    card = build_design_diagnosis(_report())
    assert card.verdict == VERDICT_HEALTHY
    assert card.problem_labels == ()
    assert card.summary == SUMMARY_HEALTHY
    text = "\n".join(format_design_diagnosis(card))
    assert "Problems:" not in text


def test_route_hard_fail_produces_blocked_verdict() -> None:
    card = build_design_diagnosis(
        _report(
            gates=(
                BalanceGateResult(
                    "scout_completion",
                    GATE_FAIL,
                    "15%",
                    DIAG_TOO_PUNISHING,
                ),
            ),
            overall_status=GATE_FAIL,
        )
    )
    assert card.verdict == VERDICT_BLOCKED
    assert DIAG_TOO_PUNISHING in card.problem_labels


def test_route_warn_only_with_package_pass_produces_promising() -> None:
    card = build_design_diagnosis(
        _report(
            gates=(
                BalanceGateResult(
                    "scout_completion",
                    GATE_WARN,
                    "25%",
                    DIAG_TOO_PUNISHING,
                ),
            ),
            overall_status=GATE_WARN,
            package_overall_status=GATE_PASS,
        )
    )
    assert card.verdict == VERDICT_PROMISING


def test_package_warn_with_route_pass_produces_promising() -> None:
    card = build_design_diagnosis(
        _report(
            overall_status=GATE_PASS,
            package_gates=(
                PackageContractGateResult(
                    "ward_role_invisible",
                    GATE_WARN,
                    "ward_actions=0",
                    DIAG_WARD_ROLE_INVISIBLE,
                ),
                PackageContractGateResult(
                    "payoff_never_fires",
                    GATE_WARN,
                    "mixed setup=2 true_payoff=0",
                    DIAG_PAYOFF_NEVER_FIRES,
                ),
            ),
            package_overall_status=GATE_WARN,
        )
    )
    assert card.verdict == VERDICT_PROMISING
    assert card.summary == SUMMARY_PROMISING_PACKAGE
    assert DIAG_WARD_ROLE_INVISIBLE in card.problem_labels
    assert DIAG_PAYOFF_NEVER_FIRES in card.problem_labels


def test_both_route_and_package_warn_produces_needs_tuning() -> None:
    card = build_design_diagnosis(
        _report(
            gates=(
                BalanceGateResult(
                    "scout_completion",
                    GATE_WARN,
                    "25%",
                    DIAG_TOO_PUNISHING,
                ),
            ),
            overall_status=GATE_WARN,
            package_gates=(
                PackageContractGateResult(
                    "payoff_never_fires",
                    GATE_WARN,
                    "mixed setup=2 true_payoff=0",
                    DIAG_PAYOFF_NEVER_FIRES,
                ),
            ),
            package_overall_status=GATE_WARN,
        )
    )
    assert card.verdict == VERDICT_NEEDS_TUNING


def test_package_never_sets_up_maps_to_cause_and_experiment() -> None:
    card = build_design_diagnosis(
        _report(
            package_gates=(
                PackageContractGateResult(
                    "package_never_sets_up",
                    GATE_FAIL,
                    "no maze mark/guard setup observed",
                    "package_never_sets_up",
                ),
            ),
            package_overall_status=GATE_FAIL,
        )
    )
    assert "package_never_sets_up" in card.problem_labels
    assert any("setup actor" in cause for cause in card.likely_causes)
    assert any("setup-capable enemy" in exp for exp in card.suggested_experiments)


def test_payoff_never_fires_maps_to_cause_and_experiment() -> None:
    card = build_design_diagnosis(
        _report(
            package_gates=(
                PackageContractGateResult(
                    "payoff_never_fires",
                    GATE_WARN,
                    "mixed setup=2 true_payoff=0",
                    DIAG_PAYOFF_NEVER_FIRES,
                ),
            ),
            package_overall_status=GATE_WARN,
        )
    )
    assert DIAG_PAYOFF_NEVER_FIRES in card.problem_labels
    assert any("payoff is not connecting" in cause for cause in card.likely_causes)
    assert any("stalker payoff enemies" in exp for exp in card.suggested_experiments)


def test_smart_policy_answers_package_is_informational_only() -> None:
    card = build_design_diagnosis(
        _report(
            package_gates=(
                PackageContractGateResult(
                    "smart_policy_answers_package",
                    GATE_INFO,
                    "anti_mark suppressed payoff without trivializing route completion",
                    DIAG_SMART_POLICY_ANSWERS_PACKAGE,
                    diagnostic_only=True,
                ),
            ),
        )
    )
    assert card.verdict == VERDICT_HEALTHY
    assert DIAG_SMART_POLICY_ANSWERS_PACKAGE not in card.problem_labels
    assert any("positive signal" in exp for exp in card.suggested_experiments)


def test_insufficient_samples_produces_insufficient_data_verdict() -> None:
    card = build_design_diagnosis(
        _report(
            package_insufficient_samples=True,
            policy_metrics=(
                _metric(maze_package=MazeMarkGuardMetrics(maze_episodes=1)),
            ),
        )
    )
    assert card.verdict == VERDICT_INSUFFICIENT_DATA
    assert card.confidence == "low"
    assert card.summary == SUMMARY_INSUFFICIENT_DATA


def test_format_policy_band_report_includes_design_diagnosis_section() -> None:
    text = format_policy_band_report(_report())
    assert "Design diagnosis:" in text
    assert "Verdict:" in text
    assert "Summary:" in text
