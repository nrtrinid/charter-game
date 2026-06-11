from __future__ import annotations

import pytest

from game.dev.ai_counterfactuals import MIXED, RECOMMENDED
from game.dev.ai_decisions import BalanceDecision, PairedSeedDelta
from game.dev.ai_packages import (
    DEFAULT_PACKAGE_CONTRACTS,
    PackageHealthResult,
    PackageReport,
)
from game.dev.ai_tactics import (
    TacticDiscoveryConfig,
    _adjust_decision_for_package_identity,
    _candidate_variants,
    _pattern_lines,
    format_tactic_discovery_report,
    run_tactic_discovery,
)
from game.dev.train_enemy_ai import TrainingRunConfig, run_training_harness
from tests.conftest import get_definitions


def test_tactic_discovery_rejects_unknown_package() -> None:
    with pytest.raises(ValueError, match="Unknown package id: missing_package"):
        run_tactic_discovery(
            TacticDiscoveryConfig(package_id="missing_package", seeds=1, max_rounds=1)
        )


def test_package_emphasis_variants_are_deterministic() -> None:
    definitions = get_definitions()
    config = TacticDiscoveryConfig(
        package_id="maw_package",
        definitions=definitions,
        route_id="opening_critical_path",
        seeds=1,
        max_rounds=1,
        policy_scope_ids=("global", "boss"),
        emphasis_scales=(1.0, 1.5),
    )
    baseline = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            route_id=config.route_id,
            seeds=1,
            max_rounds=1,
            policy_scope_ids=config.policy_scope_ids,
            preset_id=config.preset_id,
            hero_policy_id=config.hero_policy_id,
        )
    )
    contract = _contract("maw_package")

    first = _candidate_variants(config, contract, baseline)
    second = _candidate_variants(config, contract, baseline)

    assert first == second
    assert [variant[0] for variant in first] == [
        "global_maw_package_x1_0",
        "global_maw_package_x1_5",
        "boss_maw_package_x1_0",
        "boss_maw_package_x1_5",
    ]
    assert all("maw_grab_setup" in variant[3] for variant in first)


def test_tactic_discovery_report_formats_patterns_risks_and_decisions() -> None:
    report = run_tactic_discovery(
        TacticDiscoveryConfig(
            package_id="maw_package",
            route_id="opening_critical_path",
            seeds=1,
            max_rounds=1,
            policy_scope_ids=("global", "boss"),
            emphasis_scales=(1.0,),
            evaluation_hero_policy_ids=("mixed",),
        )
    )
    text = format_tactic_discovery_report(report)

    assert report.package_contract.package_id == "maw_package"
    assert report.candidates
    assert "Tactic Discovery: maw_package" in text
    assert "Discovered tactic candidates:" in text
    assert "pattern:" in text
    assert "risks:" in text
    assert "decision:" in text
    assert "Evaluation opponents: mixed" in text


def test_tactic_discovery_does_not_mutate_definitions() -> None:
    definitions = get_definitions()
    original_tags = tuple(definitions.skills["drag_forward"].tags)

    run_tactic_discovery(
        TacticDiscoveryConfig(
            package_id="maw_package",
            definitions=definitions,
            route_id="opening_critical_path",
            seeds=1,
            max_rounds=1,
            emphasis_scales=(1.0,),
            evaluation_hero_policy_ids=("mixed",),
        )
    )

    assert tuple(definitions.skills["drag_forward"].tags) == original_tags


def test_maw_pattern_summary_reports_package_metric_deltas() -> None:
    contract = _contract("maw_package")
    before = _report(
        contract.package_id,
        {
            "support_grabs": 2,
            "grab_to_bite_same_target": 1,
            "direct_front_bites": 5,
            "bone_soldier_guarded_boss": 0,
            "support_grab_downs": 0,
        },
    )
    after = _report(
        contract.package_id,
        {
            "support_grabs": 5,
            "grab_to_bite_same_target": 3,
            "direct_front_bites": 4,
            "bone_soldier_guarded_boss": 2,
            "support_grab_downs": 1,
        },
    )

    patterns = _pattern_lines(contract, before, after)

    assert "support_grabs +3.0" in patterns
    assert "grab_to_bite_same_target +2.0" in patterns
    assert "direct_front_bites -1.0" in patterns
    assert "bone_soldier_guarded_boss +2.0" in patterns


def test_bandit_pattern_summary_reports_mark_flow_deltas() -> None:
    contract = _contract("bandit_kill_lane")
    before = _report(
        contract.package_id,
        {
            "marks_applied": 2,
            "exploited_by_enemy_hit": 1,
            "vulnerable_payoffs": 0,
            "marked_downs": 0,
            "ignored_marked_legal_attacks": 4,
        },
    )
    after = _report(
        contract.package_id,
        {
            "marks_applied": 3,
            "exploited_by_enemy_hit": 4,
            "vulnerable_payoffs": 2,
            "marked_downs": 1,
            "ignored_marked_legal_attacks": 1,
        },
    )

    patterns = _pattern_lines(contract, before, after)

    assert "marks_applied +1.0" in patterns
    assert "exploited_by_enemy_hit +3.0" in patterns
    assert "vulnerable_payoffs +2.0" in patterns
    assert "ignored_marked_legal_attacks -3.0" in patterns


def test_generic_package_contract_uses_custom_tactic_metrics() -> None:
    contract = DEFAULT_PACKAGE_CONTRACTS[0].__class__(
        package_id="future_package",
        enemy_ids=("enemy",),
        setup_actions=("setup",),
        payoff_actions=("payoff",),
        tactic_metric_ids=("setup_count", "payoff_count"),
        primary_success_metrics=("payoff_count",),
        discovery_feature_ids=("future_feature",),
    )
    before = _report(contract.package_id, {"setup_count": 1, "payoff_count": 0})
    after = _report(contract.package_id, {"setup_count": 2, "payoff_count": 3})

    patterns = _pattern_lines(contract, before, after)

    assert patterns == ("payoff_count +3.0",)


def test_primary_package_metric_regression_downgrades_recommendation() -> None:
    contract = _contract("bandit_kill_lane")
    before = _report(
        contract.package_id,
        {"exploited_by_enemy_hit": 4, "vulnerable_payoffs": 2, "marked_downs": 1},
    )
    after = _report(
        contract.package_id,
        {"exploited_by_enemy_hit": 2, "vulnerable_payoffs": 1, "marked_downs": 1},
    )
    decision = BalanceDecision(
        variant_id="candidate",
        description="candidate",
        recommendation=RECOMMENDED,
        confidence="high",
        gates=(),
        deltas=(),
        paired_seed_delta=PairedSeedDelta(),
        reasons=(),
        score=100,
    )

    adjusted = _adjust_decision_for_package_identity(
        contract,
        decision,
        before,
        after,
    )

    assert adjusted.recommendation == MIXED
    assert "primary package metrics regressed" in adjusted.reasons[-1]


def _contract(package_id: str):
    return next(
        contract
        for contract in DEFAULT_PACKAGE_CONTRACTS
        if contract.package_id == package_id
    )


def _report(package_id: str, metrics: dict[str, int | float]) -> PackageReport:
    return PackageReport(
        route_id="",
        preset_id="fresh",
        hero_policy_id="mixed",
        seed_count=1,
        results=(
            PackageHealthResult(
                package_id=package_id,
                status="PASS",
                details=("synthetic",),
                metric_values=metrics,
            ),
        ),
    )
