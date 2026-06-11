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
    HeroPolicyActionRecord,
    MarkFlowMetrics,
)
from game.dev.ai_decisions import GATE_PASS, GATE_WARN
from game.dev.hero_policy_audit import (
    LABEL_EFFORT_LOW_VALUE,
    LABEL_GENERIC_KILLS_OVER_PAYOFF_PRESSURE,
    LABEL_HEALTHY_PACKAGE_RESPONSE,
    LABEL_IGNORED_KILLABLES,
    LABEL_MARK_PRESSURE_PERSISTS,
    LABEL_MARK_RESPONSE_NOT_WORKING,
    LABEL_PACKAGE_FIXATION,
    LABEL_PAYOFF_EXPOSURE_HIGH,
    HeroPolicyAuditReport,
    aggregate_hero_policy_audit,
    evaluate_hero_policy_audit,
    format_hero_policy_audit_section,
)
from game.dev.maze_mark_guard_package import GATE_INFO, MazeMarkGuardMetrics
from game.dev.policy_band_report import (
    SCENARIO_GENERATED_SCOUT,
    PolicyBandReport,
    PolicyRunMetrics,
    format_policy_band_report,
)


def _hero_action(**overrides: object) -> HeroPolicyActionRecord:
    values: dict[str, object] = {
        "hero_id": "hero_front",
        "round_number": 1,
        "skill_id": "strike",
        "target_id": "enemy_1",
        "effort_cost": 0,
        "estimated_damage": 2,
        "killable": False,
        "is_heal": False,
        "killable_opportunities": 0,
        "ignored_killable_opportunity": False,
        "package_target": "",
        "marked_hero_present": False,
        "produced_kill": False,
        "target_hp_remaining": 3,
    }
    values.update(overrides)
    return HeroPolicyActionRecord(**values)  # type: ignore[arg-type]


def _episode_with_hero_actions(
    *actions: HeroPolicyActionRecord,
    marked_damage: int = 0,
    encounter_id: str = "generated_maze_probe",
) -> EnemyDecisionEpisode:
    return EnemyDecisionEpisode(
        encounter_id=encounter_id,
        encounter_name=encounter_id,
        seed=1,
        records=(),
        final_victor="heroes",
        total_reward=0,
        metrics=EnemyPressureMetrics(
            mark_flow=MarkFlowMetrics(total_damage_to_marked=marked_damage),
        ),
        hero_actions=actions,
    )


def test_aggregate_kill_and_effort_counters() -> None:
    episodes = (
        _episode_with_hero_actions(
            _hero_action(
                effort_cost=2,
                estimated_damage=4,
                killable=True,
                produced_kill=True,
                killable_opportunities=1,
            ),
            _hero_action(
                effort_cost=1,
                estimated_damage=3,
                killable=False,
                killable_opportunities=2,
            ),
            _hero_action(effort_cost=0, estimated_damage=1, is_heal=True),
        ),
    )
    report = aggregate_hero_policy_audit(episodes, policy_id="mixed")
    metrics = report.metrics
    assert metrics.total_hero_actions == 3
    assert metrics.kills == 1
    assert metrics.healing_actions == 1
    assert metrics.offensive_actions == 2
    assert metrics.effort_spent == 3
    assert metrics.effort_actions == 2
    assert metrics.effort_kill_actions == 1
    assert metrics.nonlethal_effort_damage == 1
    assert metrics.zero_effort_useful_attacks == 0


def test_package_target_counts_reflect_chosen_target_role() -> None:
    episodes = (
        _episode_with_hero_actions(
            _hero_action(
                package_target="payoff",
                estimated_damage=3,
                target_id="enemy_payoff",
            ),
            _hero_action(
                package_target="setup",
                estimated_damage=2,
                target_id="enemy_setup",
            ),
            _hero_action(
                package_target="",
                estimated_damage=2,
                target_id="enemy_filler",
            ),
        ),
        _episode_with_hero_actions(
            _hero_action(estimated_damage=1, target_id="enemy_other"),
            encounter_id="generated_maze_pattern_cell",
        ),
    )
    report = aggregate_hero_policy_audit(episodes, policy_id="mixed")
    metrics = report.metrics
    assert metrics.package_target_attacks == 2
    assert metrics.payoff_enemy_attacks == 1
    assert metrics.setup_enemy_attacks == 1
    assert metrics.nonlethal_package_attacks == 2


def test_ignored_killables_finding_triggers() -> None:
    actions = tuple(
        _hero_action(
            killable_opportunities=2,
            ignored_killable_opportunity=True,
            estimated_damage=2,
        )
        for _ in range(4)
    )
    report = aggregate_hero_policy_audit(
        (_episode_with_hero_actions(*actions),),
        policy_id="tactical",
    )
    findings = evaluate_hero_policy_audit((report,))
    labels = {finding.label for finding in findings}
    assert LABEL_IGNORED_KILLABLES in labels
    ignored = next(f for f in findings if f.label == LABEL_IGNORED_KILLABLES)
    assert ignored.severity == GATE_WARN


def test_effort_low_value_finding_triggers() -> None:
    actions = (
        _hero_action(
            effort_cost=2,
            estimated_damage=3,
            killable=False,
            produced_kill=False,
        ),
        _hero_action(
            effort_cost=1,
            estimated_damage=2,
            killable=False,
            produced_kill=False,
        ),
        _hero_action(
            effort_cost=1,
            estimated_damage=4,
            killable=True,
            produced_kill=True,
        ),
        _hero_action(
            effort_cost=1,
            estimated_damage=2,
            killable=False,
            produced_kill=False,
        ),
    )
    report = aggregate_hero_policy_audit(
        (_episode_with_hero_actions(*actions),),
        policy_id="tactical",
    )
    findings = evaluate_hero_policy_audit((report,))
    labels = {finding.label for finding in findings}
    assert LABEL_EFFORT_LOW_VALUE in labels
    assert all(f.severity == GATE_WARN for f in findings if f.label == LABEL_EFFORT_LOW_VALUE)


def test_aggregate_package_low_hp_and_followup_kills() -> None:
    episode = _episode_with_hero_actions(
        _hero_action(
            package_target="payoff",
            target_id="enemy_payoff",
            estimated_damage=3,
            target_hp_remaining=1,
        ),
        _hero_action(
            target_id="enemy_payoff",
            estimated_damage=2,
            killable=True,
            produced_kill=True,
            target_hp_remaining=0,
        ),
        _hero_action(
            package_target="setup",
            target_id="enemy_setup",
            estimated_damage=4,
            target_hp_remaining=6,
        ),
    )
    report = aggregate_hero_policy_audit((episode,), policy_id="tactical")
    assert report.metrics.nonlethal_package_low_hp_created == 1
    assert report.metrics.package_followup_kills == 1


def test_normalized_marked_damage_properties() -> None:
    episode = _episode_with_hero_actions(
        _hero_action(),
        _hero_action(),
        marked_damage=100,
    )
    report = aggregate_hero_policy_audit((episode, episode), policy_id="mixed")
    assert report.marked_damage_total == 200
    assert report.marked_dmg_per_episode == 100.0
    assert report.marked_dmg_per_hero_action == 50.0


def test_package_fixation_finding_triggers() -> None:
    actions = tuple(
        _hero_action(
            package_target="payoff",
            estimated_damage=3,
            killable=False,
        )
        for _ in range(4)
    )
    tactical = aggregate_hero_policy_audit(
        (_episode_with_hero_actions(*actions),),
        policy_id="tactical",
    )
    mixed = aggregate_hero_policy_audit((), policy_id="mixed")
    findings = evaluate_hero_policy_audit(
        (mixed, tactical),
        completion_by_policy={"mixed": 0.50, "tactical": 0.30},
    )
    labels = {finding.label for finding in findings}
    assert LABEL_PACKAGE_FIXATION in labels
    fixation = next(f for f in findings if f.label == LABEL_PACKAGE_FIXATION)
    assert "pkg_nl_low_hp=" in fixation.detail


def test_mark_pressure_persists_when_better_completion() -> None:
    mixed = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(),
                _hero_action(),
                marked_damage=100,
            ),
        ),
        policy_id="mixed",
    )
    anti_mark = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(package_target="payoff", estimated_damage=2),
                _hero_action(package_target="setup", estimated_damage=2),
                marked_damage=130,
            ),
        ),
        policy_id="anti_mark",
    )
    findings = evaluate_hero_policy_audit(
        (mixed, anti_mark),
        completion_by_policy={"mixed": 0.54, "anti_mark": 0.62},
    )
    labels = {finding.label for finding in findings}
    assert LABEL_MARK_PRESSURE_PERSISTS in labels
    assert LABEL_MARK_RESPONSE_NOT_WORKING not in labels
    persists = next(f for f in findings if f.label == LABEL_MARK_PRESSURE_PERSISTS)
    assert "marked_dmg/act=" in persists.detail
    assert "completion" in persists.detail


def test_mark_response_not_working_when_worse_completion() -> None:
    mixed = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(),
                _hero_action(),
                marked_damage=100,
            ),
        ),
        policy_id="mixed",
    )
    tactical = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(package_target="payoff", estimated_damage=2),
                _hero_action(package_target="setup", estimated_damage=2),
                marked_damage=120,
            ),
        ),
        policy_id="tactical",
    )
    findings = evaluate_hero_policy_audit(
        (mixed, tactical),
        completion_by_policy={"mixed": 0.54, "tactical": 0.38},
    )
    mark_finding = next(
        f for f in findings if f.label == LABEL_MARK_RESPONSE_NOT_WORKING
    )
    assert "marked_dmg/act=" in mark_finding.detail
    assert mark_finding.severity == GATE_WARN


def test_mark_finding_fallback_without_completion_context() -> None:
    mixed = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(),
                _hero_action(),
                marked_damage=100,
            ),
        ),
        policy_id="mixed",
    )
    tactical = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(package_target="payoff", estimated_damage=2),
                _hero_action(package_target="setup", estimated_damage=2),
                marked_damage=120,
            ),
        ),
        policy_id="tactical",
    )
    findings = evaluate_hero_policy_audit((mixed, tactical))
    mark_finding = next(
        f for f in findings if f.label == LABEL_MARK_RESPONSE_NOT_WORKING
    )
    assert "outcome context unavailable" in mark_finding.detail


def test_healthy_package_response_when_lower_mark_and_better_completion() -> None:
    mixed = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(),
                _hero_action(),
                marked_damage=200,
            ),
        ),
        policy_id="mixed",
    )
    specialist = aggregate_hero_policy_audit(
        (
            _episode_with_hero_actions(
                _hero_action(package_target="payoff", estimated_damage=2),
                _hero_action(package_target="setup", estimated_damage=2),
                marked_damage=100,
            ),
        ),
        policy_id="company_survival",
    )
    findings = evaluate_hero_policy_audit(
        (mixed, specialist),
        completion_by_policy={"mixed": 0.54, "company_survival": 0.60},
    )
    labels = {finding.label for finding in findings}
    assert LABEL_HEALTHY_PACKAGE_RESPONSE in labels
    assert LABEL_MARK_RESPONSE_NOT_WORKING not in labels


def test_no_finding_depends_on_package_followup_kills() -> None:
    episode = _episode_with_hero_actions(
        _hero_action(
            package_target="payoff",
            estimated_damage=3,
            target_hp_remaining=1,
        ),
        _hero_action(
            target_id="enemy_1",
            estimated_damage=2,
            killable=True,
            produced_kill=True,
        ),
    )
    tactical = aggregate_hero_policy_audit((episode,), policy_id="tactical")
    mixed = aggregate_hero_policy_audit((), policy_id="mixed")
    findings = evaluate_hero_policy_audit(
        (mixed, tactical),
        completion_by_policy={"mixed": 0.5, "tactical": 0.3},
    )
    for finding in findings:
        assert "followup" not in finding.label
        assert "followup" not in finding.detail.lower()


def test_healthy_package_response_info_when_marked_damage_lower_than_mixed() -> None:
    mixed = aggregate_hero_policy_audit(
        (_episode_with_hero_actions(marked_damage=100),),
        policy_id="mixed",
    )
    anti_mark_actions = (
        _hero_action(package_target="payoff", estimated_damage=2),
        _hero_action(package_target="setup", estimated_damage=2),
    )
    anti_mark = aggregate_hero_policy_audit(
        (_episode_with_hero_actions(*anti_mark_actions, marked_damage=50),),
        policy_id="anti_mark",
    )
    findings = evaluate_hero_policy_audit(
        (mixed, anti_mark),
        completion_by_policy={"mixed": 0.5, "anti_mark": 0.6},
    )
    labels = {finding.label for finding in findings}
    assert LABEL_HEALTHY_PACKAGE_RESPONSE in labels
    healthy = next(f for f in findings if f.label == LABEL_HEALTHY_PACKAGE_RESPONSE)
    assert healthy.severity == GATE_INFO
    assert healthy.policy_id == "anti_mark"


def _enemy_record(skill_id: str) -> EnemyDecisionRecord:
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


def _episode_with_records(
    *actions: HeroPolicyActionRecord,
    records: tuple[EnemyDecisionRecord, ...] = (),
    marked_damage: int = 0,
) -> EnemyDecisionEpisode:
    return EnemyDecisionEpisode(
        encounter_id="generated_maze_stalker",
        encounter_name="generated_maze_stalker",
        seed=1,
        records=records,
        final_victor="heroes",
        total_reward=0,
        metrics=EnemyPressureMetrics(
            mark_flow=MarkFlowMetrics(total_damage_to_marked=marked_damage),
        ),
        hero_actions=actions,
    )


def test_threat_exposure_aggregates_kills_by_package_role() -> None:
    episodes = (
        _episode_with_records(
            _hero_action(
                package_target="payoff",
                target_id="payoff_enemy",
                produced_kill=True,
                killable=True,
            ),
            _hero_action(
                package_target="setup",
                target_id="setup_enemy",
                produced_kill=True,
                killable=True,
            ),
            _hero_action(
                package_target="",
                target_id="filler_enemy",
                produced_kill=True,
                killable=True,
            ),
            _hero_action(package_target="payoff", target_id="payoff_enemy"),
        ),
    )
    report = aggregate_hero_policy_audit(episodes, policy_id="company_survival")
    exposure = report.threat_exposure
    assert exposure.payoff_kills == 1
    assert exposure.setup_kills == 1
    assert exposure.generic_kills == 1
    assert exposure.payoff_attacks == 2
    assert exposure.setup_attacks == 1
    assert exposure.generic_attacks == 1


def test_threat_exposure_counts_enemy_payoff_skill_uses() -> None:
    episodes = (
        _episode_with_records(
            _hero_action(estimated_damage=2),
            records=(
                _enemy_record("stalker_cut"),
                _enemy_record("stalker_hook"),
                _enemy_record("stalker_cut"),
            ),
        ),
    )
    report = aggregate_hero_policy_audit(episodes, policy_id="mixed")
    exposure = report.threat_exposure
    assert exposure.stalker_cut_uses == 2
    assert exposure.stalker_hook_uses == 1
    assert exposure.enemy_payoff_skill_uses == 3


def test_threat_exposure_detects_generic_kills_over_payoff_pressure() -> None:
    mixed = aggregate_hero_policy_audit(
        (
            _episode_with_records(
                _hero_action(
                    package_target="payoff",
                    produced_kill=True,
                    killable=True,
                ),
                _hero_action(estimated_damage=2),
                records=(_enemy_record("stalker_cut"),),
            ),
        ),
        policy_id="mixed",
    )
    company_actions = (
        _hero_action(
            package_target="payoff",
            target_id="payoff_enemy",
            estimated_damage=2,
        ),
        _hero_action(
            package_target="",
            target_id="filler_1",
            produced_kill=True,
            killable=True,
        ),
        _hero_action(
            package_target="",
            target_id="filler_2",
            produced_kill=True,
            killable=True,
        ),
        _hero_action(
            package_target="",
            target_id="filler_3",
            produced_kill=True,
            killable=True,
        ),
    )
    company = aggregate_hero_policy_audit(
        (
            _episode_with_records(
                *company_actions,
                records=(
                    _enemy_record("stalker_cut"),
                    _enemy_record("stalker_cut"),
                    _enemy_record("stalker_cut"),
                    _enemy_record("stalker_cut"),
                ),
            ),
        ),
        policy_id="company_survival",
    )
    findings = evaluate_hero_policy_audit(
        (mixed, company),
        completion_by_policy={"mixed": 0.5, "company_survival": 0.2},
    )
    labels = {finding.label for finding in findings}
    assert LABEL_GENERIC_KILLS_OVER_PAYOFF_PRESSURE in labels
    assert company.threat_exposure.generic_kill_while_payoff_alive == 3


def test_threat_exposure_detects_payoff_exposure_high() -> None:
    mixed = aggregate_hero_policy_audit(
        (
            _episode_with_records(
                _hero_action(),
                records=(_enemy_record("stalker_cut"),),
            ),
        ),
        policy_id="mixed",
    )
    company = aggregate_hero_policy_audit(
        (
            _episode_with_records(
                _hero_action(),
                records=(
                    _enemy_record("stalker_cut"),
                    _enemy_record("stalker_cut"),
                ),
            ),
        ),
        policy_id="company_survival",
    )
    findings = evaluate_hero_policy_audit((mixed, company))
    labels = {finding.label for finding in findings}
    assert LABEL_PAYOFF_EXPOSURE_HIGH in labels


def test_formatter_includes_threat_exposure_section() -> None:
    episodes = (
        _episode_with_records(
            _hero_action(
                package_target="payoff",
                produced_kill=True,
                killable=True,
            ),
            records=(_enemy_record("stalker_cut"),),
        ),
    )
    reports = (
        aggregate_hero_policy_audit(episodes, policy_id="mixed"),
        aggregate_hero_policy_audit(episodes, policy_id="company_survival"),
    )
    text = "\n".join(format_hero_policy_audit_section(reports, ()))
    assert "Threat exposure:" in text
    assert "setup_kills=" in text
    assert "payoff_kills=" in text
    assert "generic_kills=" in text
    assert "payoff_actions/act=" in text


def test_formatter_includes_mixed_tactical_anti_mark_lines() -> None:
    def _report(policy_id: str, actions: int, kills: int) -> HeroPolicyAuditReport:
        episode = _episode_with_hero_actions(
            *(
                _hero_action(produced_kill=i == 0, killable_opportunities=1)
                for i in range(actions)
            ),
            marked_damage=10 * actions,
        )
        return aggregate_hero_policy_audit((episode,), policy_id=policy_id)

    reports = (
        _report("mixed", 3, 1),
        _report("tactical", 4, 2),
        _report("anti_mark", 2, 0),
        _report("company_survival", 5, 2),
    )
    text = "\n".join(format_hero_policy_audit_section(reports, ()))
    assert "Hero policy audit:" in text
    assert "mixed:" in text
    assert "tactical:" in text
    assert "anti_mark:" in text
    assert "company_survival:" in text
    assert (
        text.index("mixed:")
        < text.index("tactical:")
        < text.index("anti_mark:")
        < text.index("company_survival:")
    )
    assert "marked_dmg/ep=" in text
    assert "marked_dmg/act=" in text
    assert "pkg_nl_low_hp=" in text


def test_format_policy_band_report_includes_hero_policy_audit() -> None:
    mixed_actions = (
        _hero_action(killable_opportunities=1),
        _hero_action(package_target="payoff", estimated_damage=2),
    )
    tactical_actions = tuple(
        _hero_action(
            killable_opportunities=2,
            ignored_killable_opportunity=True,
            estimated_damage=2,
        )
        for _ in range(4)
    )
    mixed_audit = aggregate_hero_policy_audit(
        (_episode_with_hero_actions(*mixed_actions, marked_damage=80),),
        policy_id="mixed",
    )
    tactical_audit = aggregate_hero_policy_audit(
        (_episode_with_hero_actions(*tactical_actions, marked_damage=40),),
        policy_id="tactical",
    )
    maze_package = MazeMarkGuardMetrics(maze_episodes=1)
    report = PolicyBandReport(
        scenario_id="breach_probe",
        scenario_name="breach_probe",
        scenario_kind=SCENARIO_GENERATED_SCOUT,
        seed_count=1,
        policy_metrics=(
            PolicyRunMetrics(
                hero_policy_id="mixed",
                completion_rate=0.5,
                victor_counts={"heroes": 1},
                average_downs=0.0,
                average_deaths=0.0,
                average_final_hp=0.0,
                average_final_effort=0.0,
                marks_applied=0,
                marks_exploited=0,
                vulnerable_payoffs=0,
                maze_package=maze_package,
                hero_policy_audit=mixed_audit,
            ),
            PolicyRunMetrics(
                hero_policy_id="tactical",
                completion_rate=0.3,
                victor_counts={"heroes": 1},
                average_downs=0.0,
                average_deaths=0.0,
                average_final_hp=0.0,
                average_final_effort=0.0,
                marks_applied=0,
                marks_exploited=0,
                vulnerable_payoffs=0,
                maze_package=maze_package,
                hero_policy_audit=tactical_audit,
            ),
        ),
        gates=(),
        overall_status=GATE_PASS,
    )
    text = format_policy_band_report(report)
    assert "Hero policy audit:" in text
    assert "Design diagnosis:" in text
    assert text.index("Hero policy audit:") < text.index("Design diagnosis:")
    assert "mixed:" in text
    assert "tactical:" in text
    assert "Audit findings:" in text
