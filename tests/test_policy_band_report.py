from __future__ import annotations

from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionRuntimeContext,
    EnemyDecisionTrace,
)
from game.combat.enemy_learning import (
    SUPPORTED_HERO_POLICY_IDS,
    EnemyDecisionEpisode,
    EnemyDecisionRecord,
    EnemyPressureMetrics,
    GuardFlowMetrics,
    MarkFlowMetrics,
)
from game.dev.ai_decisions import GATE_FAIL, GATE_PASS, GATE_WARN
from game.dev.maze_mark_guard_package import (
    DIAG_NAIVE_GETS_PUNISHED,
    DIAG_PAYOFF_NEVER_FIRES,
    DIAG_SMART_POLICY_ANSWERS_PACKAGE,
    DIAG_SMART_POLICY_DELETES_PACKAGE,
    DIAG_WARD_ROLE_INVISIBLE,
    MazeMarkGuardMetrics,
    MazePackageThresholds,
    PolicyPackageInput,
    aggregate_maze_mark_guard_metrics,
    episode_has_true_payoff,
    evaluate_maze_package_gates,
    format_maze_package_section,
    package_overall_status,
)
from game.dev.policy_band_report import (
    DIAG_NAIVE_ONLY_PRESSURE,
    DIAG_PACKAGE_PAYOFF_SUPPRESSED,
    DIAG_SMART_POLICY_TRIVIALIZES,
    DIAG_TOO_EASY,
    DIAG_TOO_PUNISHING,
    SCENARIO_GENERATED_HUNT,
    SCENARIO_GENERATED_SCOUT,
    BalanceGateResult,
    PolicyBandReport,
    PolicyBandThresholds,
    PolicyRunMetrics,
    _build_policy_band_report,
    evaluate_policy_band,
    format_policy_band_report,
    policy_metrics_from_generated_summary,
    policy_metrics_from_training_evaluation,
    run_authored_route_policy_band,
)
from game.dev.route_lab import GeneratedRouteLabSummary, RouteEnvelopeScore, RouteLabConfig
from game.dev.train_enemy_ai import TrainingPolicyEvaluation
from tests.conftest import get_definitions


def _metric(
    hero_policy_id: str,
    completion_rate: float,
    *,
    marks_applied: int = 0,
    marks_exploited: int = 0,
    vulnerable_payoffs: int = 0,
    maze_package: MazeMarkGuardMetrics | None = None,
    ward_role_package: MazeMarkGuardMetrics | None = None,
) -> PolicyRunMetrics:
    return PolicyRunMetrics(
        hero_policy_id=hero_policy_id,
        completion_rate=completion_rate,
        victor_counts={"heroes": int(completion_rate * 100)},
        average_downs=0.0,
        average_deaths=0.0,
        average_final_hp=0.0,
        average_final_effort=0.0,
        marks_applied=marks_applied,
        marks_exploited=marks_exploited,
        vulnerable_payoffs=vulnerable_payoffs,
        maze_package=maze_package,
        ward_role_package=ward_role_package,
    )


def _six_policy_metrics(**overrides: float) -> tuple[PolicyRunMetrics, ...]:
    return tuple(
        _metric(policy_id, overrides.get(policy_id, 0.5))
        for policy_id in SUPPORTED_HERO_POLICY_IDS
    )


def _gate(gates: tuple[BalanceGateResult, ...], gate_id: str) -> BalanceGateResult:
    return next(gate for gate in gates if gate.gate_id == gate_id)


def _maze_record(skill_id: str) -> EnemyDecisionRecord:
    candidate = EnemyDecisionCandidate(
        skill_id=skill_id,
        target_id="hero_front",
        score=0,
        skill_order=0,
        target_order=0,
        skill_tags=frozenset(),
        features={},
    )
    trace = EnemyDecisionTrace(
        enemy_id="enemy",
        runtime_context=EnemyDecisionRuntimeContext(),
        candidates=(candidate,),
        chosen=candidate,
    )
    return EnemyDecisionRecord(
        enemy_id="enemy",
        round_number=1,
        action_index=0,
        trace=trace,
        chosen_skill_id=skill_id,
        chosen_target_id="hero_front",
        chosen_features={},
        events=(),
    )


def _maze_episode(
    *,
    encounter_id: str = "generated_maze_pattern_cell",
    records: tuple[EnemyDecisionRecord, ...] = (),
    mark_flow: MarkFlowMetrics | None = None,
    guard_flow: GuardFlowMetrics | None = None,
    forced_movement: int = 0,
) -> EnemyDecisionEpisode:
    metrics = EnemyPressureMetrics(
        forced_movement=forced_movement,
        mark_flow=mark_flow or MarkFlowMetrics(),
        guard_flow=guard_flow or GuardFlowMetrics(),
    )
    return EnemyDecisionEpisode(
        encounter_id=encounter_id,
        encounter_name=encounter_id,
        seed=1,
        records=records,
        final_victor="heroes",
        total_reward=0,
        metrics=metrics,
    )


def _package_input(
    hero_policy_id: str,
    completion_rate: float,
    maze_package: MazeMarkGuardMetrics,
    *,
    ward_role_package: MazeMarkGuardMetrics | None = None,
) -> PolicyPackageInput:
    return PolicyPackageInput(
        hero_policy_id=hero_policy_id,
        completion_rate=completion_rate,
        maze_package=maze_package,
        ward_role_package=ward_role_package,
    )


def test_aggregate_policy_metrics_from_synthetic_inputs() -> None:
    summary = GeneratedRouteLabSummary(
        breach_id="shallow_cave_breach",
        seed_count=2,
        hero_policy_id="mixed",
        preset_id="fresh",
        strategy_id="mainline",
        pressure_profile_id="breach_probe",
        runs=(),
        envelope_score=RouteEnvelopeScore(
            envelope_id="generated_maze_scout",
            status=GATE_PASS,
            score=100,
        ),
    )
    empty_metrics = policy_metrics_from_generated_summary(summary)
    assert empty_metrics.completion_rate == 0.0
    assert empty_metrics.hero_policy_id == "mixed"

    evaluation = TrainingPolicyEvaluation(
        policy_scope_id="global",
        evaluation_hero_policy_id="damage_race",
        episodes=(),
        route_results=(),
        learned_weights={},
        encounter_breakdowns=(),
    )
    training_metrics = policy_metrics_from_training_evaluation(evaluation)
    assert training_metrics.hero_policy_id == "damage_race"
    assert training_metrics.completion_rate == 0.0


def test_scout_completion_pass_warn_fail() -> None:
    thresholds = PolicyBandThresholds()

    def scout_gate(rate: float) -> BalanceGateResult:
        return _gate(
            evaluate_policy_band(
                scenario_kind=SCENARIO_GENERATED_SCOUT,
                policy_metrics=(_metric("mixed", rate),),
                thresholds=thresholds,
            ),
            "scout_completion",
        )

    assert scout_gate(0.50).status == GATE_PASS
    assert scout_gate(0.25).status == GATE_WARN
    assert scout_gate(0.25).diagnostic_label == DIAG_TOO_PUNISHING
    assert scout_gate(0.85).status == GATE_WARN
    assert scout_gate(0.85).diagnostic_label == DIAG_TOO_EASY
    assert scout_gate(0.15).status == GATE_FAIL
    assert scout_gate(0.95).status == GATE_FAIL


def test_hunt_completion_pass_warn_fail() -> None:
    thresholds = PolicyBandThresholds()

    def hunt_gate(rate: float) -> BalanceGateResult:
        return _gate(
            evaluate_policy_band(
                scenario_kind=SCENARIO_GENERATED_HUNT,
                policy_metrics=(_metric("mixed", rate),),
                thresholds=thresholds,
            ),
            "hunt_completion",
        )

    assert hunt_gate(0.30).status == GATE_PASS
    assert hunt_gate(0.07).status == GATE_WARN
    assert hunt_gate(0.07).diagnostic_label == DIAG_TOO_PUNISHING
    assert hunt_gate(0.70).status == GATE_WARN
    assert hunt_gate(0.70).diagnostic_label == DIAG_TOO_EASY
    assert hunt_gate(0.03).status == GATE_FAIL
    assert hunt_gate(0.80).status == GATE_FAIL


def test_policy_spread_warn_and_fail() -> None:
    thresholds = PolicyBandThresholds()
    warn_metrics = (
        _metric("naive", 0.10),
        _metric("mixed", 0.50),
    )
    warn_gate = _gate(
        evaluate_policy_band(
            scenario_kind=SCENARIO_GENERATED_SCOUT,
            policy_metrics=warn_metrics,
            thresholds=thresholds,
        ),
        "policy_spread",
    )
    assert warn_gate.status == GATE_WARN
    assert warn_gate.detail == "40pt"

    fail_metrics = (
        _metric("naive", 0.05),
        _metric("conservative", 0.70),
    )
    fail_gate = _gate(
        evaluate_policy_band(
            scenario_kind=SCENARIO_GENERATED_SCOUT,
            policy_metrics=fail_metrics,
            thresholds=thresholds,
        ),
        "policy_spread",
    )
    assert fail_gate.status == GATE_FAIL
    assert fail_gate.diagnostic_label == DIAG_SMART_POLICY_TRIVIALIZES


def test_smart_beats_mixed_warning() -> None:
    thresholds = PolicyBandThresholds()
    metrics = _six_policy_metrics(mixed=0.40, anti_mark=0.66, conservative=0.50)
    gate = _gate(
        evaluate_policy_band(
            scenario_kind=SCENARIO_GENERATED_SCOUT,
            policy_metrics=metrics,
            thresholds=thresholds,
        ),
        "smart_beats_mixed",
    )
    assert gate.status == GATE_WARN
    assert gate.diagnostic_label == DIAG_SMART_POLICY_TRIVIALIZES


def test_package_payoff_suppressed_warning() -> None:
    thresholds = PolicyBandThresholds()
    metrics = _six_policy_metrics(mixed=0.40, anti_mark=0.70)
    metrics = tuple(
        _metric("anti_mark", 0.70)
        if item.hero_policy_id == "anti_mark"
        else item
        for item in metrics
    )
    gate = _gate(
        evaluate_policy_band(
            scenario_kind=SCENARIO_GENERATED_HUNT,
            policy_metrics=metrics,
            thresholds=thresholds,
        ),
        "package_payoff_suppressed",
    )
    assert gate.status == GATE_WARN
    assert gate.diagnostic_label == DIAG_PACKAGE_PAYOFF_SUPPRESSED


def test_naive_only_pressure_label() -> None:
    thresholds = PolicyBandThresholds()
    metrics = (
        _metric("naive", 0.10),
        _metric("damage_race", 0.50),
        _metric("mixed", 0.50),
    )
    gate = _gate(
        evaluate_policy_band(
            scenario_kind=SCENARIO_GENERATED_SCOUT,
            policy_metrics=metrics,
            thresholds=thresholds,
        ),
        "policy_spread",
    )
    assert gate.status == GATE_WARN
    assert gate.diagnostic_label == DIAG_NAIVE_ONLY_PRESSURE


def test_overall_status_worst_gate_wins() -> None:
    report = PolicyBandReport(
        scenario_id="test",
        scenario_name="test",
        scenario_kind=SCENARIO_GENERATED_SCOUT,
        seed_count=1,
        policy_metrics=(_metric("mixed", 0.15),),
        gates=(
            BalanceGateResult("scout_completion", GATE_FAIL, "15%", DIAG_TOO_PUNISHING),
            BalanceGateResult("policy_spread", GATE_PASS, "0pt"),
        ),
        overall_status=GATE_FAIL,
    )
    assert report.overall_status == GATE_FAIL


def test_authored_route_policy_band_produces_eight_policies() -> None:
    report = run_authored_route_policy_band(
        RouteLabConfig(
            definitions=get_definitions(),
            route_id="opening_critical_path",
            seeds=1,
            max_rounds=1,
        )
    )
    assert len(report.policy_metrics) == len(SUPPORTED_HERO_POLICY_IDS)
    assert {metrics.hero_policy_id for metrics in report.policy_metrics} == set(
        SUPPORTED_HERO_POLICY_IDS
    )


def test_maze_package_metrics_aggregate_from_synthetic_episodes() -> None:
    setup_episode = _maze_episode(
        records=(_maze_record("splinter_mark"), _maze_record("ward_pattern")),
        mark_flow=MarkFlowMetrics(
            marks_applied=1,
            exploited_by_enemy_hit=1,
            total_damage_to_marked=4,
        ),
        guard_flow=GuardFlowMetrics(guard_uses=1, guard_damage_blocked=2),
    )
    counterplay_episode = _maze_episode(
        encounter_id="generated_maze_stalker",
        records=(_maze_record("mark_the_pattern"),),
        mark_flow=MarkFlowMetrics(marks_applied=1, ignored_marked_legal_attacks=2),
    )
    ignored_episode = _maze_episode(encounter_id="other_encounter")

    metrics = aggregate_maze_mark_guard_metrics(
        (setup_episode, counterplay_episode, ignored_episode)
    )
    assert metrics.maze_episodes == 2
    assert metrics.episodes_with_setup == 2
    assert metrics.mark_setup_attempts == 2
    assert metrics.marks_applied == 2
    assert metrics.ward_actions == 1
    assert metrics.guard_uses == 1
    assert metrics.package_payoff_episodes == 1
    assert metrics.setup_without_payoff_episodes == 1
    assert metrics.ignored_marked_legal_attacks == 2


def test_true_payoff_requires_setup_plus_mark_flow_signals() -> None:
    raw_stalker = _maze_episode(records=(_maze_record("stalker_cut"),))
    assert not episode_has_true_payoff(raw_stalker)

    setup_payoff = _maze_episode(
        records=(_maze_record("splinter_mark"),),
        mark_flow=MarkFlowMetrics(marks_applied=1, exploited_by_enemy_hit=1),
    )
    assert episode_has_true_payoff(setup_payoff)


def test_payoff_never_fires_warning_triggers() -> None:
    mixed_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        marks_applied=2,
    )
    inputs = (
        _package_input("mixed", 0.50, mixed_package),
        _package_input("naive", 0.40, mixed_package),
    )
    gates = evaluate_maze_package_gates(inputs)
    payoff_gate = next(gate for gate in gates if gate.gate_id == "payoff_never_fires")
    assert payoff_gate.status == GATE_WARN
    assert payoff_gate.diagnostic_label == DIAG_PAYOFF_NEVER_FIRES


def test_smart_policy_deletes_package_warning_triggers() -> None:
    naive_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        package_payoff_episodes=2,
    )
    anti_package = MazeMarkGuardMetrics(maze_episodes=3, episodes_with_setup=1)
    mixed_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        package_payoff_episodes=1,
    )
    inputs = (
        _package_input("mixed", 0.40, mixed_package),
        _package_input("naive", 0.35, naive_package),
        _package_input("anti_mark", 0.70, anti_package),
    )
    gates = evaluate_maze_package_gates(inputs)
    delete_gate = next(
        gate for gate in gates if gate.gate_id == "smart_policy_deletes_package"
    )
    assert delete_gate.status == GATE_WARN
    assert delete_gate.diagnostic_label == DIAG_SMART_POLICY_DELETES_PACKAGE


def test_smart_policy_answers_package_is_informational_only() -> None:
    mixed_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        package_payoff_episodes=1,
    )
    anti_package = MazeMarkGuardMetrics(maze_episodes=3, episodes_with_setup=1)
    inputs = (
        _package_input("mixed", 0.50, mixed_package),
        _package_input("anti_mark", 0.55, anti_package),
    )
    gates = evaluate_maze_package_gates(inputs)
    info_gate = next(
        gate for gate in gates if gate.gate_id == "smart_policy_answers_package"
    )
    assert info_gate.diagnostic_only
    assert info_gate.diagnostic_label == DIAG_SMART_POLICY_ANSWERS_PACKAGE
    assert package_overall_status(gates) == GATE_PASS


def test_insufficient_samples_skips_package_gates() -> None:
    mixed_package = MazeMarkGuardMetrics(maze_episodes=1, episodes_with_setup=1)
    inputs = (_package_input("mixed", 0.50, mixed_package),)
    gates = evaluate_maze_package_gates(inputs)
    assert gates == ()

    lines = format_maze_package_section(
        inputs,
        gates,
        insufficient_samples=True,
        thresholds=MazePackageThresholds(min_package_episodes=2),
    )
    text = "\n".join(lines)
    assert "insufficient samples" in text
    assert "maze_episodes=1" in text
    assert "skipped (insufficient samples)" in text


def test_ward_role_invisible_uses_ward_actions_only() -> None:
    ward_role = MazeMarkGuardMetrics(
        maze_episodes=2,
        guard_uses=3,
        ward_actions=0,
    )
    mixed_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        marks_applied=2,
        package_payoff_episodes=1,
    )
    inputs = (
        _package_input(
            "mixed",
            0.50,
            mixed_package,
            ward_role_package=ward_role,
        ),
    )
    gates = evaluate_maze_package_gates(inputs)
    ward_gate = next(gate for gate in gates if gate.gate_id == "ward_role_invisible")
    assert ward_gate.status == GATE_WARN
    assert ward_gate.diagnostic_label == DIAG_WARD_ROLE_INVISIBLE


def test_naive_package_pressure_recognized() -> None:
    naive_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        package_payoff_episodes=1,
    )
    mixed_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        package_payoff_episodes=1,
    )
    inputs = (
        _package_input("mixed", 0.50, mixed_package),
        _package_input("naive", 0.20, naive_package),
        _package_input("anti_mark", 0.55, mixed_package),
        _package_input("conservative", 0.50, mixed_package),
    )
    gates = evaluate_maze_package_gates(inputs)
    naive_gate = next(gate for gate in gates if gate.gate_id == "naive_gets_punished")
    assert naive_gate.status == GATE_WARN
    assert naive_gate.diagnostic_label == DIAG_NAIVE_GETS_PUNISHED


def test_format_policy_band_report_includes_tactical_policy() -> None:
    report = _build_policy_band_report(
        scenario_id="breach_probe",
        scenario_name="breach_probe",
        scenario_kind=SCENARIO_GENERATED_SCOUT,
        seed_count=3,
        policy_metrics=_six_policy_metrics(),
        thresholds=PolicyBandThresholds(),
    )
    text = format_policy_band_report(report)
    assert "tactical" in text
    assert "company_survival" in text


def test_format_policy_band_report_includes_maze_package_section() -> None:
    mixed_package = MazeMarkGuardMetrics(
        maze_episodes=3,
        episodes_with_setup=2,
        package_payoff_episodes=1,
        ward_actions=1,
    )
    report = _build_policy_band_report(
        scenario_id="breach_probe",
        scenario_name="breach_probe",
        scenario_kind=SCENARIO_GENERATED_SCOUT,
        seed_count=3,
        policy_metrics=(
            _metric(
                "mixed",
                0.50,
                maze_package=mixed_package,
                ward_role_package=MazeMarkGuardMetrics(maze_episodes=2, ward_actions=1),
            ),
        ),
        thresholds=PolicyBandThresholds(),
    )
    text = format_policy_band_report(report)
    assert "Maze package" in text
    assert "true_payoff" in text
    assert "ward_actions" in text
    assert "Package gates:" in text
    assert "Package overall:" in text


def test_build_policy_band_report_marks_insufficient_samples() -> None:
    mixed_package = MazeMarkGuardMetrics(maze_episodes=1, episodes_with_setup=1)
    report = _build_policy_band_report(
        scenario_id="breach_probe",
        scenario_name="breach_probe",
        scenario_kind=SCENARIO_GENERATED_SCOUT,
        seed_count=1,
        policy_metrics=(_metric("mixed", 0.50, maze_package=mixed_package),),
        thresholds=PolicyBandThresholds(),
    )
    assert report.package_insufficient_samples
    assert report.package_gates == ()
    assert report.package_overall_status == GATE_PASS
    assert "insufficient samples" in format_policy_band_report(report)
