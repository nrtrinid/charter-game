from __future__ import annotations

from dataclasses import replace

from game.combat.enemy_learning import (
    BossSequenceMetrics,
    BossTargetingMetrics,
    MarkFlowMetrics,
)
from game.dev.ai_packages import (
    DEFAULT_PACKAGE_CONTRACTS,
    evaluate_enemy_packages,
    format_package_report,
)
from game.dev.train_enemy_ai import PolicyPressureSummary, TrainingRunConfig, run_training_harness


def test_package_contracts_instantiate() -> None:
    assert {contract.package_id for contract in DEFAULT_PACKAGE_CONTRACTS} == {
        "maw_package",
        "bandit_kill_lane",
        "wolf_mark",
    }
    assert all(contract.discovery_feature_ids for contract in DEFAULT_PACKAGE_CONTRACTS)
    assert all(contract.primary_success_metrics for contract in DEFAULT_PACKAGE_CONTRACTS)
    assert all(contract.tactic_metric_ids for contract in DEFAULT_PACKAGE_CONTRACTS)


def test_package_report_formats_health_statuses() -> None:
    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=1,
        )
    )

    report = evaluate_enemy_packages(summary)
    text = format_package_report(report)

    assert "Package Health:" in text
    assert "bandit_kill_lane:" in text
    assert any(result.status in {"PASS", "WARN", "FAIL"} for result in report.results)


def test_maw_package_health_warns_on_low_grab_to_bite_conversion() -> None:
    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=("cave_mini_boss",),
            seeds=1,
            max_rounds=1,
        )
    )
    poor_boss = replace(
        summary.encounter_breakdowns[0],
        learned=_replace_policy_boss_metrics(
            summary.encounter_breakdowns[0].learned,
            BossSequenceMetrics(grab_uses=4, bite_uses=1, grab_to_bite_same_target=0),
            BossTargetingMetrics(grab_target_classes={"field_surgeon": 4}),
        ),
    )
    summary = replace(summary, encounter_breakdowns=(poor_boss,))

    result = evaluate_enemy_packages(summary).results[0]

    assert result.package_id == "maw_package"
    assert result.status == "WARN"
    assert "low grab->bite conversion" in "; ".join(result.details)


def test_bandit_package_health_fails_on_ignored_marked_spike() -> None:
    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=1,
        )
    )
    poor_mark = replace(
        summary.encounter_breakdowns[0],
        learned=_replace_policy_mark_metrics(
            summary.encounter_breakdowns[0].learned,
            MarkFlowMetrics(
                marks_applied=4,
                exploited_by_enemy_hit=1,
                vulnerable_payoffs=0,
                ignored_marked_legal_attacks=8,
            ),
        ),
    )
    summary = replace(summary, encounter_breakdowns=(poor_mark,))

    result = evaluate_enemy_packages(summary).results[1]

    assert result.package_id == "bandit_kill_lane"
    assert result.status == "FAIL"


def test_wolf_low_mark_use_is_not_automatic_failure_when_ally_reach_is_low() -> None:
    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=("wolf_pack",),
            seeds=1,
            max_rounds=1,
        )
    )
    low_use = replace(
        summary.encounter_breakdowns[0],
        learned=_replace_policy_mark_metrics(
            summary.encounter_breakdowns[0].learned,
            MarkFlowMetrics(marks_applied=1, mark_ally_reach_total=0, mark_ally_reach_count=1),
        ),
    )
    summary = replace(summary, encounter_breakdowns=(low_use,))

    result = evaluate_enemy_packages(summary).results[2]
    text = format_package_report(evaluate_enemy_packages(summary))

    assert result.package_id == "wolf_mark"
    assert result.status == "OK_LOW_USE"
    assert "low Mark usage acceptable" in text


def test_wolf_low_mark_use_with_high_ally_reach_warns() -> None:
    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=("wolf_pack",),
            seeds=1,
            max_rounds=1,
        )
    )
    low_use_high_reach = replace(
        summary.encounter_breakdowns[0],
        learned=_replace_policy_mark_metrics(
            summary.encounter_breakdowns[0].learned,
            MarkFlowMetrics(
                marks_applied=1,
                mark_ally_reach_total=5,
                mark_ally_reach_count=1,
            ),
        ),
    )
    summary = replace(summary, encounter_breakdowns=(low_use_high_reach,))

    result = evaluate_enemy_packages(summary).results[2]

    assert result.package_id == "wolf_mark"
    assert result.status == "WARN"
    assert "available ally reach" in "; ".join(result.details)


def test_wolf_ignored_marked_legal_spike_fails() -> None:
    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=("wolf_pack",),
            seeds=1,
            max_rounds=1,
        )
    )
    ignored_spike = replace(
        summary.encounter_breakdowns[0],
        learned=_replace_policy_mark_metrics(
            summary.encounter_breakdowns[0].learned,
            MarkFlowMetrics(
                marks_applied=3,
                exploited_by_enemy_hit=0,
                vulnerable_payoffs=0,
                ignored_marked_legal_attacks=5,
                mark_ally_reach_total=5,
                mark_ally_reach_count=1,
            ),
        ),
    )
    summary = replace(summary, encounter_breakdowns=(ignored_spike,))

    result = evaluate_enemy_packages(summary).results[2]

    assert result.package_id == "wolf_mark"
    assert result.status == "FAIL"


def _replace_policy_boss_metrics(
    summary: PolicyPressureSummary,
    boss_sequence: BossSequenceMetrics,
    boss_targeting: BossTargetingMetrics,
) -> PolicyPressureSummary:
    return replace(
        summary,
        episode_count=1,
        boss_sequence=boss_sequence,
        boss_targeting=boss_targeting,
    )


def _replace_policy_mark_metrics(
    summary: PolicyPressureSummary,
    mark_flow: MarkFlowMetrics,
) -> PolicyPressureSummary:
    return replace(
        summary,
        episode_count=1,
        mark_flow=mark_flow,
        marks_applied=mark_flow.marks_applied,
        marks_exploited=mark_flow.exploited_by_enemy_hit,
    )
