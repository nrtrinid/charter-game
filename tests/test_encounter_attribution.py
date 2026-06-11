from __future__ import annotations

from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionRuntimeContext,
    EnemyDecisionTrace,
)
from game.combat.enemy_learning import (
    EnemyDecisionEpisode,
    EnemyDecisionRecord,
    EnemyPressureMetrics,
    GuardFlowMetrics,
    MarkFlowMetrics,
)
from game.dev.ai_decisions import GATE_PASS, GATE_WARN
from game.dev.encounter_attribution import (
    LABEL_RAW_STALKER_WITHOUT_PAYOFF,
    LABEL_SETUP_WITHOUT_PAYOFF,
    LABEL_STRONG_PACKAGE_PAYOFF,
    LABEL_WARD_ROLE_INVISIBLE,
    EncounterAttributionReport,
    aggregate_encounter_attribution,
    evaluate_encounter_attribution,
    format_encounter_attribution_section,
)
from game.dev.maze_mark_guard_package import (
    GATE_INFO,
    MazeMarkGuardMetrics,
    episode_has_true_payoff,
)
from game.dev.policy_band_report import (
    SCENARIO_GENERATED_SCOUT,
    PolicyBandReport,
    PolicyRunMetrics,
    format_policy_band_report,
)


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
    forced_movement: int = 0,
) -> EnemyDecisionEpisode:
    metrics = EnemyPressureMetrics(
        forced_movement=forced_movement,
        mark_flow=mark_flow or MarkFlowMetrics(),
        guard_flow=GuardFlowMetrics(),
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


def test_aggregate_groups_by_maze_package_encounter_ids_only() -> None:
    episodes = (
        _maze_episode(encounter_id="generated_maze_probe"),
        _maze_episode(encounter_id="other_encounter"),
        _maze_episode(encounter_id="generated_maze_stalker"),
    )
    report = aggregate_encounter_attribution(episodes, policy_id="mixed")
    encounter_ids = {row.encounter_id for row in report.encounters}
    assert encounter_ids == {"generated_maze_probe", "generated_maze_stalker"}
    assert "other_encounter" not in encounter_ids


def test_aggregate_counts_setup_and_true_payoff_per_encounter() -> None:
    payoff_episode = _maze_episode(
        encounter_id="generated_maze_pattern_cell",
        records=(_maze_record("splinter_mark"),),
        mark_flow=MarkFlowMetrics(marks_applied=1, exploited_by_enemy_hit=1),
    )
    setup_only = _maze_episode(
        encounter_id="generated_maze_pattern_cell",
        records=(_maze_record("mark_the_pattern"),),
        mark_flow=MarkFlowMetrics(marks_applied=1),
    )
    report = aggregate_encounter_attribution(
        (payoff_episode, setup_only),
        policy_id="mixed",
    )
    row = report.encounters[0]
    assert row.encounter_id == "generated_maze_pattern_cell"
    assert row.episode_count == 2
    assert row.setup_episodes == 2
    assert row.payoff_episodes == 1
    assert episode_has_true_payoff(payoff_episode)
    assert not episode_has_true_payoff(setup_only)


def test_raw_stalker_alone_does_not_count_as_payoff() -> None:
    episode = _maze_episode(
        encounter_id="generated_maze_stalker",
        records=(_maze_record("stalker_cut"),),
    )
    report = aggregate_encounter_attribution((episode,), policy_id="mixed")
    row = report.encounters[0]
    assert row.stalker_cut_uses == 1
    assert row.setup_episodes == 0
    assert row.payoff_episodes == 0


def test_setup_without_payoff_finding_triggers() -> None:
    episode = _maze_episode(
        encounter_id="generated_maze_pattern_cell",
        records=(_maze_record("splinter_mark"),),
        mark_flow=MarkFlowMetrics(marks_applied=1),
    )
    report = aggregate_encounter_attribution((episode,), policy_id="mixed")
    findings = evaluate_encounter_attribution(report)
    labels = {finding.label for finding in findings}
    assert LABEL_SETUP_WITHOUT_PAYOFF in labels
    assert findings[0].severity == GATE_WARN


def test_raw_stalker_without_payoff_finding_triggers() -> None:
    episode = _maze_episode(
        encounter_id="generated_maze_stalker",
        records=(
            _maze_record("splinter_mark"),
            _maze_record("stalker_cut"),
        ),
        mark_flow=MarkFlowMetrics(marks_applied=1),
    )
    report = aggregate_encounter_attribution((episode,), policy_id="mixed")
    findings = evaluate_encounter_attribution(report)
    assert any(finding.label == LABEL_RAW_STALKER_WITHOUT_PAYOFF for finding in findings)


def test_ward_role_invisible_only_for_ward_role_encounters() -> None:
    stalker_episode = _maze_episode(encounter_id="generated_maze_stalker")
    probe_episode = _maze_episode(encounter_id="generated_maze_probe")
    stalker_report = aggregate_encounter_attribution(
        (stalker_episode,),
        policy_id="mixed",
    )
    probe_report = aggregate_encounter_attribution(
        (probe_episode,),
        policy_id="mixed",
    )
    stalker_labels = {f.label for f in evaluate_encounter_attribution(stalker_report)}
    probe_labels = {f.label for f in evaluate_encounter_attribution(probe_report)}
    assert LABEL_WARD_ROLE_INVISIBLE in stalker_labels
    assert LABEL_WARD_ROLE_INVISIBLE not in probe_labels


def test_strong_package_payoff_is_info_and_setup_visible_not_emitted() -> None:
    episodes = tuple(
        _maze_episode(
            encounter_id="generated_maze_pattern_cell",
            records=(_maze_record("splinter_mark"),),
            mark_flow=MarkFlowMetrics(marks_applied=1, exploited_by_enemy_hit=1),
        )
        for _ in range(3)
    ) + (
        _maze_episode(
            encounter_id="generated_maze_pattern_cell",
            records=(_maze_record("splinter_mark"),),
            mark_flow=MarkFlowMetrics(marks_applied=1),
        ),
    )
    report = aggregate_encounter_attribution(episodes, policy_id="mixed")
    findings = evaluate_encounter_attribution(report)
    labels = {finding.label for finding in findings}
    assert LABEL_STRONG_PACKAGE_PAYOFF in labels
    assert "setup_visible" not in labels
    strong = next(f for f in findings if f.label == LABEL_STRONG_PACKAGE_PAYOFF)
    assert strong.severity == GATE_INFO


def test_format_includes_encounter_ids_and_key_metrics() -> None:
    report = EncounterAttributionReport(
        policy_id="mixed",
        encounters=(
            aggregate_encounter_attribution(
                (
                    _maze_episode(
                        encounter_id="generated_maze_pattern_cell",
                        records=(_maze_record("ward_pattern"),),
                        mark_flow=MarkFlowMetrics(marks_applied=1, total_damage_to_marked=3),
                    ),
                ),
                policy_id="mixed",
            ).encounters[0],
        ),
    )
    text = "\n".join(format_encounter_attribution_section((report,), ()))
    assert "Encounter attribution:" in text
    assert "generated_maze_pattern_cell" in text
    assert "marked_dmg=3" in text
    assert "ward=1" in text


def test_format_policy_band_report_includes_encounter_attribution() -> None:
    attribution = aggregate_encounter_attribution(
        (
            _maze_episode(
                encounter_id="generated_maze_pattern_cell",
                records=(_maze_record("splinter_mark"),),
                mark_flow=MarkFlowMetrics(marks_applied=1),
            ),
        ),
        policy_id="mixed",
    )
    maze_package = MazeMarkGuardMetrics(
        maze_episodes=1,
        episodes_with_setup=1,
    )
    report = PolicyBandReport(
        scenario_id="breach_probe",
        scenario_name="breach_probe",
        scenario_kind=SCENARIO_GENERATED_SCOUT,
        seed_count=1,
        policy_metrics=(
            PolicyRunMetrics(
                hero_policy_id="mixed",
                completion_rate=0.5,
                victor_counts={"heroes": 50},
                average_downs=0.0,
                average_deaths=0.0,
                average_final_hp=0.0,
                average_final_effort=0.0,
                marks_applied=1,
                marks_exploited=0,
                vulnerable_payoffs=0,
                maze_package=maze_package,
                encounter_attribution=attribution,
            ),
        ),
        gates=(),
        overall_status=GATE_PASS,
    )
    text = format_policy_band_report(report)
    assert "Encounter attribution:" in text
    assert "Design diagnosis:" in text
    assert text.index("Encounter attribution:") < text.index("Design diagnosis:")
