from __future__ import annotations

from types import MappingProxyType

from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionRuntimeContext,
    EnemyDecisionTrace,
    _choose_best_candidate,
    choose_heuristic_enemy_candidate,
)
from game.combat.enemy_learning import (
    EnemyDecisionEpisode,
    EnemyDecisionRecord,
    EnemyPressureMetrics,
)
from game.core.events import DamageEvent
from game.dev.ai_oracle import (
    CounterplayFinding,
    CounterplayMetrics,
    OracleReport,
    aggregate_counterplay_metrics,
    analyze_oracle_record,
    evaluate_counterplay_findings,
    format_oracle_report,
)
from game.dev.route_lab import RouteEnvelopeScore
from game.dev.train_enemy_ai import TrainingRunSummary


def _candidate(
    skill_id: str,
    target_id: str,
    score: int,
    *,
    skill_order: int = 0,
    target_order: int = 0,
) -> EnemyDecisionCandidate:
    return EnemyDecisionCandidate(
        skill_id=skill_id,
        target_id=target_id,
        score=score,
        skill_order=skill_order,
        target_order=target_order,
        skill_tags=frozenset(),
        features=MappingProxyType({}),
    )


def _record(
    *,
    chosen_skill_id: str,
    chosen_target_id: str,
    candidates: tuple[EnemyDecisionCandidate, ...],
) -> EnemyDecisionRecord:
    trace = EnemyDecisionTrace(
        enemy_id="enemy_1",
        runtime_context=EnemyDecisionRuntimeContext(),
        candidates=candidates,
        chosen=candidates[0] if candidates else None,
    )
    return EnemyDecisionRecord(
        enemy_id="enemy_1",
        round_number=2,
        action_index=0,
        trace=trace,
        chosen_skill_id=chosen_skill_id,
        chosen_target_id=chosen_target_id,
        chosen_features={},
        events=(),
        enemy_class_id="cave_maw_brute",
    )


def test_choose_heuristic_enemy_candidate_matches_internal() -> None:
    candidates = (
        _candidate("maw_slam", "front_left", 10, skill_order=0),
        _candidate("drag_forward", "front_left", 20, skill_order=1),
    )

    assert choose_heuristic_enemy_candidate(candidates) == _choose_best_candidate(candidates)


def test_analyze_oracle_record_detects_miss() -> None:
    candidates = (
        _candidate("maw_slam", "front_left", 10),
        _candidate("drag_forward", "front_left", 25),
    )
    record = _record(
        chosen_skill_id="maw_slam",
        chosen_target_id="front_left",
        candidates=candidates,
    )

    miss = analyze_oracle_record(
        record,
        encounter_id="cave_mini_boss",
        package_id="maw_package",
        miss_threshold=1,
    )

    assert miss is not None
    assert miss.oracle_skill_id == "drag_forward"
    assert miss.delta == 15
    assert "skill_mismatch" in miss.reason_codes


def test_analyze_oracle_record_no_miss_when_matching() -> None:
    candidates = (
        _candidate("maw_slam", "front_left", 10),
        _candidate("drag_forward", "front_left", 5),
    )
    record = _record(
        chosen_skill_id="maw_slam",
        chosen_target_id="front_left",
        candidates=candidates,
    )

    assert (
        analyze_oracle_record(
            record,
            encounter_id="cave_mini_boss",
            package_id="maw_package",
            miss_threshold=1,
        )
        is None
    )


def test_aggregate_counterplay_metrics_skill_dominance() -> None:
    episode = EnemyDecisionEpisode(
        encounter_id="shallow_cave",
        encounter_name="Shallow Cave",
        seed=1,
        records=(
            _record(
                chosen_skill_id="maw_slam",
                chosen_target_id="front_left",
                candidates=(_candidate("maw_slam", "front_left", 1),),
            ),
            _record(
                chosen_skill_id="maw_slam",
                chosen_target_id="front_right",
                candidates=(_candidate("maw_slam", "front_right", 1),),
            ),
            _record(
                chosen_skill_id="drag_forward",
                chosen_target_id="front_left",
                candidates=(_candidate("drag_forward", "front_left", 1),),
            ),
        ),
        final_victor="heroes",
        total_reward=0,
        metrics=EnemyPressureMetrics(rounds_elapsed=4, skill_uses={"maw_slam": 2}),
    )
    summary = _training_summary(episodes=(episode,))

    metrics = aggregate_counterplay_metrics(summary)

    assert metrics.dominant_skill_id == "maw_slam"
    assert metrics.dominant_skill_rate == 2 / 3
    assert metrics.same_skill_spam_rate == 0.5


def test_aggregate_counterplay_metrics_optional_fields_none() -> None:
    episode = EnemyDecisionEpisode(
        encounter_id="shallow_cave",
        encounter_name="Shallow Cave",
        seed=1,
        records=(),
        final_victor="heroes",
        total_reward=0,
    )
    summary = _training_summary(episodes=(episode,), route_id="")

    metrics = aggregate_counterplay_metrics(summary)
    text = format_oracle_report(
        OracleReport(
            route_id="",
            seed_count=1,
            hero_policy_id="mixed",
            preset_id="fresh",
            metrics=metrics,
        )
    )

    assert metrics.damage_to_downed_rate is None
    assert metrics.first_downed_round_avg is None
    assert metrics.noncompletion_count is None
    assert "damage to downed: n/a" in text
    assert "first downed round avg: n/a" in text
    assert "noncompletion: n/a" in text


def test_aggregate_counterplay_metrics_damage_to_downed_uses_hp_before() -> None:
    candidates = (_candidate("maw_slam", "front_left", 1),)
    trace = EnemyDecisionTrace(
        enemy_id="enemy_1",
        runtime_context=EnemyDecisionRuntimeContext(),
        candidates=candidates,
        chosen=candidates[0],
    )
    record = EnemyDecisionRecord(
        enemy_id="enemy_1",
        round_number=3,
        action_index=0,
        trace=trace,
        chosen_skill_id="maw_slam",
        chosen_target_id="front_left",
        chosen_features={},
        events=(
            DamageEvent(
                message="hit",
                source_id="enemy_1",
                target_id="front_left",
                amount=2,
                hp_before=0,
            ),
        ),
    )
    episode = EnemyDecisionEpisode(
        encounter_id="shallow_cave",
        encounter_name="Shallow Cave",
        seed=1,
        records=(record,),
        final_victor="heroes",
        total_reward=0,
    )
    summary = _training_summary(episodes=(episode,))

    metrics = aggregate_counterplay_metrics(summary)

    assert metrics.damage_to_downed_count == 1
    assert metrics.damage_to_downed_rate == 1.0


def test_evaluate_findings_dominant_skill_warn() -> None:
    metrics = CounterplayMetrics(
        total_enemy_actions=10,
        dominant_skill_id="maw_slam",
        dominant_skill_rate=0.40,
        run_count=1,
    )

    findings = evaluate_counterplay_findings(metrics, envelope_score=None)
    dominant = next(item for item in findings if item.finding_id == "dominant_skill_high")

    assert dominant.status == "WARN"
    assert "maw_slam" in dominant.detail


def test_evaluate_findings_route_reuses_envelope() -> None:
    metrics = CounterplayMetrics(total_enemy_actions=1, run_count=1)
    envelope = RouteEnvelopeScore(
        envelope_id="critical_path",
        status="FAIL",
        score=40,
        warnings=("completion below target band",),
    )

    findings = evaluate_counterplay_findings(metrics, envelope_score=envelope)
    punishing = next(item for item in findings if item.finding_id == "route_too_punishing")

    assert punishing.status == "FAIL"
    assert "completion below target band" in punishing.detail


def test_evaluate_findings_route_too_clean_not_confusing_on_warn_envelope() -> None:
    metrics = CounterplayMetrics(total_enemy_actions=1, run_count=1)
    envelope = RouteEnvelopeScore(
        envelope_id="optional_pressure_path",
        status="WARN",
        score=70,
        warnings=("party reaches boss too healthy",),
    )

    findings = evaluate_counterplay_findings(metrics, envelope_score=envelope)
    too_clean = next(item for item in findings if item.finding_id == "route_too_clean")

    assert too_clean.status == "PASS"
    assert "route not too clean" in too_clean.detail


def test_evaluate_findings_route_bimodal_collapse_fail() -> None:
    metrics = CounterplayMetrics(total_enemy_actions=1, run_count=1)
    envelope = RouteEnvelopeScore(
        envelope_id="optional_pressure_path",
        status="FAIL",
        score=40,
        warnings=(
            "route_bimodal_collapse: completion 24%; noncompletion 38/50; "
            "boss-entry 35.4 among survivors exceeds max 26; final HP 6.0",
        ),
    )

    findings = evaluate_counterplay_findings(metrics, envelope_score=envelope)
    bimodal = next(item for item in findings if item.finding_id == "route_bimodal_collapse")
    punishing = next(item for item in findings if item.finding_id == "route_too_punishing")

    assert bimodal.status == "FAIL"
    assert "noncompletion 38/50" in bimodal.detail
    assert "route_bimodal_collapse" in punishing.detail
    assert "party reaches boss too healthy" not in punishing.detail


def test_evaluate_findings_route_bimodal_collapse_warn() -> None:
    metrics = CounterplayMetrics(total_enemy_actions=1, run_count=1)
    envelope = RouteEnvelopeScore(
        envelope_id="optional_pressure_path",
        status="WARN",
        score=70,
        warnings=(
            "route_bimodal_collapse: completion 70%; noncompletion 3/10; "
            "boss-entry 30.0 among survivors exceeds max 26; final HP 8.0",
        ),
    )

    findings = evaluate_counterplay_findings(metrics, envelope_score=envelope)
    bimodal = next(item for item in findings if item.finding_id == "route_bimodal_collapse")

    assert bimodal.status == "WARN"


def test_format_oracle_report_uses_noncompletion_label() -> None:
    report = OracleReport(
        route_id="opening_pressure_path",
        seed_count=2,
        hero_policy_id="mixed",
        preset_id="fresh",
        metrics=CounterplayMetrics(
            total_enemy_actions=4,
            noncompletion_count=3,
            noncompletion_rate=0.75,
            pre_boss_failure_count=1,
            run_count=2,
        ),
    )

    text = format_oracle_report(report)

    assert "noncompletion: 3 (75%)" in text
    assert "pre-boss failures: 1" in text


def test_format_oracle_report_stable_structure() -> None:
    report = OracleReport(
        route_id="opening_pressure_path",
        seed_count=2,
        hero_policy_id="mixed",
        preset_id="fresh",
        metrics=CounterplayMetrics(
            total_enemy_actions=4,
            oracle_miss_count=1,
            oracle_miss_rate=0.25,
            average_miss_delta=3.0,
            largest_miss=("enemy_1", "maw_slam", "drag_forward", 3),
            dominant_skill_id="maw_slam",
            dominant_skill_rate=0.5,
            run_count=2,
        ),
        findings=(
            CounterplayFinding(
                finding_id="oracle_misses_high",
                status="WARN",
                detail="score-only heuristic oracle misses on 25.0% of enemy actions.",
            ),
        ),
    )

    text = format_oracle_report(report)

    assert "AI Lab Oracle Report" in text
    assert "Oracle mode: score-only heuristic (depth-0)" in text
    assert "enemy actions checked: 4" in text
    assert "oracle misses: 1" in text
    assert "Counterplay:" in text
    assert "Findings:" in text
    assert "oracle_misses_high" in text


def _training_summary(
    *,
    episodes: tuple[EnemyDecisionEpisode, ...],
    route_id: str = "opening_pressure_path",
) -> TrainingRunSummary:
    return TrainingRunSummary(
        encounter_ids=("shallow_cave",),
        seed_count=1,
        hero_policy_id="mixed",
        evaluation_hero_policy_ids=("mixed",),
        policy_scope_ids=("global",),
        preset_id="fresh",
        route_id=route_id,
        enemy_wait_mode="none",
        enemy_movement_mode="recovery_only",
        heuristic_episodes=(),
        learned_episodes=episodes,
        heuristic_route_results=(),
        learned_route_results=(),
        learned_weights={},
        encounter_breakdowns=(),
        policy_evaluations=(),
    )
