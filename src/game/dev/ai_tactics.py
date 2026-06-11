"""Dev-only tactic discovery reports for enemy AI packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from game.combat.enemy_learning import SUPPORTED_HERO_POLICY_IDS
from game.content.definitions import GameDefinitions
from game.dev.ai_counterfactuals import (
    MIXED,
    NO_EFFECT,
    PROMISING,
    RECOMMENDED,
    REGRESSION,
    CounterfactualResult,
    CounterfactualSweepSummary,
    _compare_counterfactual,
    _completion_count,
)
from game.dev.ai_decisions import BalanceDecision, build_balance_decision_report
from game.dev.ai_packages import (
    DEFAULT_PACKAGE_CONTRACTS,
    EnemyPackageContract,
    PackageReport,
    evaluate_enemy_packages,
)
from game.dev.train_enemy_ai import (
    SUPPORTED_POLICY_SCOPE_IDS,
    SUPPORTED_PRESET_IDS,
    SUPPORTED_ROUTE_IDS,
    TrainingRunConfig,
    TrainingRunSummary,
    _encounter_breakdowns,
    run_training_harness,
)

_DEFAULT_EMPHASIS_SCALES = (1.0, 1.25, 1.5)
_SEEDED_FEATURE_WEIGHT = 10
_MAX_DISPLAY_CANDIDATES = 5


@dataclass(frozen=True)
class TacticDiscoveryConfig:
    package_id: str
    route_id: str = "opening_pressure_path"
    seeds: int = 50
    max_rounds: int = 20
    hero_policy_id: str = "mixed"
    evaluation_hero_policy_ids: tuple[str, ...] = SUPPORTED_HERO_POLICY_IDS
    preset_id: str = "attrition"
    policy_scope_ids: tuple[str, ...] = ("global", "boss", "per_role")
    emphasis_scales: tuple[float, ...] = _DEFAULT_EMPHASIS_SCALES
    enemy_wait_mode: str = "none"
    enemy_movement_mode: str = "recovery_only"
    definitions: GameDefinitions | None = None


@dataclass(frozen=True)
class TacticCandidate:
    candidate_id: str
    description: str
    policy_scope_id: str
    emphasis_scale: float
    weight_overrides: Mapping[str, int]
    summary: TrainingRunSummary
    package_report: PackageReport
    decision: BalanceDecision
    patterns: tuple[str, ...]
    risks: tuple[str, ...]


@dataclass(frozen=True)
class TacticDiscoveryReport:
    package_contract: EnemyPackageContract
    config: TacticDiscoveryConfig
    baseline_summary: TrainingRunSummary
    baseline_package_report: PackageReport
    candidates: tuple[TacticCandidate, ...]


def run_tactic_discovery(config: TacticDiscoveryConfig) -> TacticDiscoveryReport:
    contract = _package_contract(config.package_id)
    route_id = _validate_supported(config.route_id, SUPPORTED_ROUTE_IDS, "route")
    preset_id = _validate_supported(config.preset_id, SUPPORTED_PRESET_IDS, "preset")
    policy_scope_ids = _validate_policy_scopes(config.policy_scope_ids)
    evaluation_hero_policy_ids = _validate_hero_policies(config.evaluation_hero_policy_ids)
    normalized = TacticDiscoveryConfig(
        package_id=contract.package_id,
        route_id=route_id,
        seeds=config.seeds,
        max_rounds=config.max_rounds,
        hero_policy_id=_validate_supported(
            config.hero_policy_id,
            SUPPORTED_HERO_POLICY_IDS,
            "hero policy",
        ),
        evaluation_hero_policy_ids=evaluation_hero_policy_ids,
        preset_id=preset_id,
        policy_scope_ids=policy_scope_ids,
        emphasis_scales=tuple(config.emphasis_scales) or _DEFAULT_EMPHASIS_SCALES,
        enemy_wait_mode=config.enemy_wait_mode,
        enemy_movement_mode=config.enemy_movement_mode,
        definitions=config.definitions,
    )
    baseline = run_training_harness(
        TrainingRunConfig(
            definitions=normalized.definitions,
            route_id=normalized.route_id,
            seeds=normalized.seeds,
            max_rounds=normalized.max_rounds,
            hero_policy_id=normalized.hero_policy_id,
            evaluation_hero_policy_ids=normalized.evaluation_hero_policy_ids,
            policy_scope_ids=policy_scope_ids,
            preset_id=normalized.preset_id,
            enemy_wait_mode=normalized.enemy_wait_mode,
            enemy_movement_mode=normalized.enemy_movement_mode,
        )
    )
    baseline_report = evaluate_enemy_packages(baseline)
    candidates = tuple(
        _candidate_for_variant(normalized, contract, baseline, baseline_report, variant)
        for variant in _candidate_variants(normalized, contract, baseline)
    )
    return TacticDiscoveryReport(
        package_contract=contract,
        config=normalized,
        baseline_summary=baseline,
        baseline_package_report=baseline_report,
        candidates=tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    _recommendation_order(candidate.decision.recommendation),
                    -candidate.decision.score,
                    candidate.candidate_id,
                ),
            )
        ),
    )


def format_tactic_discovery_report(report: TacticDiscoveryReport) -> str:
    config = report.config
    lines = [
        f"Tactic Discovery: {report.package_contract.package_id}",
        f"Route: {config.route_id}",
        f"Seeds: {config.seeds}",
        f"Training opponent: {config.hero_policy_id}",
        f"Evaluation opponents: {', '.join(config.evaluation_hero_policy_ids)}",
        f"Preset: {config.preset_id}",
        f"Enemy wait mode: {config.enemy_wait_mode}",
        f"Enemy movement mode: {config.enemy_movement_mode}",
        "Discovered tactic candidates:",
    ]
    if not report.candidates:
        lines.append("  none")
        return "\n".join(lines)
    for index, candidate in enumerate(report.candidates[:_MAX_DISPLAY_CANDIDATES], start=1):
        decision = candidate.decision
        patterns = "; ".join(candidate.patterns) if candidate.patterns else "no pattern delta"
        risks = "; ".join(candidate.risks) if candidate.risks else "none"
        lines.extend(
            [
                f"  {index}. {candidate.candidate_id}:",
                f"    recommendation {decision.recommendation}",
                f"    confidence {decision.confidence}",
                (
                    "    route envelope "
                    f"{decision.baseline_envelope_status} -> "
                    f"{decision.candidate_envelope_status}"
                ),
                f"    pattern: {patterns}",
                f"    risks: {risks}",
                f"    decision: {_decision_text(decision.recommendation)}",
                f"    {candidate.description}",
            ]
        )
    lines.append("Robustness:")
    lines.extend(_robustness_lines(report.candidates[:1]))
    return "\n".join(lines)


def _candidate_for_variant(
    config: TacticDiscoveryConfig,
    contract: EnemyPackageContract,
    baseline: TrainingRunSummary,
    baseline_report: PackageReport,
    variant: tuple[str, str, float, Mapping[str, int]],
) -> TacticCandidate:
    candidate_id, scope_id, scale, overrides = variant
    summary = run_training_harness(
        TrainingRunConfig(
            definitions=config.definitions,
            route_id=config.route_id,
            seeds=config.seeds,
            max_rounds=config.max_rounds,
            hero_policy_id=config.hero_policy_id,
            evaluation_hero_policy_ids=config.evaluation_hero_policy_ids,
            policy_scope_ids=(scope_id,),
            preset_id=config.preset_id,
            learned_weight_overrides=overrides,
            enemy_wait_mode=config.enemy_wait_mode,
            enemy_movement_mode=config.enemy_movement_mode,
        )
    )
    scoped_summary = _summary_for_scope(summary, scope_id, config.hero_policy_id)
    package_report = evaluate_enemy_packages(scoped_summary)
    comparison = _compare_counterfactual(
        baseline,
        baseline_report,
        scoped_summary,
        package_report,
    )
    result = CounterfactualResult(
        variant_id=candidate_id,
        description=_candidate_description(contract, scope_id, scale),
        summary=scoped_summary,
        package_report=package_report,
        score=comparison.score,
        reward_delta=scoped_summary.learned_total_reward - baseline.learned_total_reward,
        completion_delta=_completion_count(scoped_summary) - _completion_count(baseline),
        package_health_delta=comparison.package_health_delta,
        package_metric_delta=comparison.package_metric_delta,
        route_envelope_delta=comparison.route_envelope_delta,
        recommendation=comparison.recommendation,
        reasons=comparison.reasons,
    )
    decision = build_balance_decision_report(
        CounterfactualSweepSummary(
            baseline_summary=baseline,
            baseline_package_report=baseline_report,
            results=(result,),
        )
    ).decisions[0]
    decision = _adjust_decision_for_package_identity(
        contract,
        decision,
        baseline_report,
        package_report,
    )
    return TacticCandidate(
        candidate_id=candidate_id,
        description=result.description,
        policy_scope_id=scope_id,
        emphasis_scale=scale,
        weight_overrides=dict(overrides),
        summary=summary,
        package_report=package_report,
        decision=decision,
        patterns=_pattern_lines(contract, baseline_report, package_report),
        risks=_risk_lines(contract, decision, baseline_report, package_report, summary),
    )


def _candidate_variants(
    config: TacticDiscoveryConfig,
    contract: EnemyPackageContract,
    baseline: TrainingRunSummary,
) -> tuple[tuple[str, str, float, Mapping[str, int]], ...]:
    variants: list[tuple[str, str, float, Mapping[str, int]]] = []
    feature_ids = contract.discovery_feature_ids
    if not feature_ids:
        return ()
    allowed_scopes = _package_scopes(contract, config.policy_scope_ids)
    for scope_id in allowed_scopes:
        base_weights = _weights_for_scope(baseline, scope_id)
        for scale in config.emphasis_scales:
            overrides = _scaled_feature_overrides(base_weights, feature_ids, scale)
            variants.append(
                (
                    f"{scope_id}_{contract.package_id}_x{_scale_label(scale)}",
                    scope_id,
                    scale,
                    overrides,
                )
            )
    return tuple(variants)


def _summary_for_scope(
    summary: TrainingRunSummary,
    scope_id: str,
    hero_policy_id: str,
) -> TrainingRunSummary:
    for evaluation in summary.policy_evaluations:
        if (
            evaluation.policy_scope_id == scope_id
            and evaluation.evaluation_hero_policy_id == hero_policy_id
        ):
            return replace(
                summary,
                learned_episodes=evaluation.episodes,
                learned_route_results=evaluation.route_results,
                learned_weights=evaluation.learned_weights,
                encounter_breakdowns=_encounter_breakdowns(
                    summary.encounter_ids,
                    summary.heuristic_episodes,
                    evaluation.episodes,
                ),
            )
    raise ValueError(f"Missing tactic evaluation for scope: {scope_id}")


def _scaled_feature_overrides(
    base_weights: Mapping[str, int],
    feature_ids: tuple[str, ...],
    scale: float,
) -> dict[str, int]:
    return {
        feature_id: max(
            1,
            int(round(max(_SEEDED_FEATURE_WEIGHT, base_weights.get(feature_id, 0)) * scale)),
        )
        for feature_id in feature_ids
    }


def _weights_for_scope(summary: TrainingRunSummary, scope_id: str) -> Mapping[str, int]:
    if scope_id == "global":
        return summary.learned_weights
    for evaluation in summary.policy_evaluations:
        if (
            evaluation.policy_scope_id == scope_id
            and evaluation.evaluation_hero_policy_id == summary.hero_policy_id
        ):
            return evaluation.learned_weights
    return summary.learned_weights


def _package_scopes(
    contract: EnemyPackageContract,
    selected_scopes: tuple[str, ...],
) -> tuple[str, ...]:
    if contract.package_id == "maw_package":
        preferred: tuple[str, ...] = ("global", "boss")
    elif contract.package_id in {"bandit_kill_lane", "wolf_mark"}:
        preferred = ("global", "per_role")
    else:
        preferred = ("global",)
    return tuple(scope for scope in preferred if scope in selected_scopes)


def _pattern_lines(
    contract: EnemyPackageContract,
    baseline_report: PackageReport,
    candidate_report: PackageReport,
) -> tuple[str, ...]:
    before = _package_result(baseline_report, contract.package_id).metric_values
    after = _package_result(candidate_report, contract.package_id).metric_values
    if contract.package_id == "maw_package":
        metric_ids: tuple[str, ...] = (
            "support_grabs",
            "grab_to_bite_same_target",
            "direct_front_bites",
            "bone_soldier_guarded_boss",
            "support_grab_downs",
        )
    elif contract.package_id in {"bandit_kill_lane", "wolf_mark"}:
        metric_ids = (
            "marks_applied",
            "exploited_by_enemy_hit",
            "vulnerable_payoffs",
            "marked_downs",
            "ignored_marked_legal_attacks",
        )
    else:
        metric_ids = contract.primary_success_metrics or contract.tactic_metric_ids
    lines = tuple(
        _metric_delta_text(metric_id, before.get(metric_id, 0), after.get(metric_id, 0))
        for metric_id in metric_ids
        if _metric_delta_value(before.get(metric_id, 0), after.get(metric_id, 0)) != 0
    )
    return lines or ("package metrics unchanged",)


def _risk_lines(
    contract: EnemyPackageContract,
    decision: BalanceDecision,
    baseline_report: PackageReport,
    candidate_report: PackageReport,
    summary: TrainingRunSummary,
) -> tuple[str, ...]:
    risks: list[str] = []
    if decision.candidate_envelope_status == "FAIL":
        risks.append("route envelope failed")
    if decision.recommendation in {MIXED, REGRESSION}:
        risks.append("decision gates found regression risk")
    before = _package_result(baseline_report, contract.package_id).metric_values
    after = _package_result(candidate_report, contract.package_id).metric_values
    for metric_id in contract.degeneracy_metrics:
        if _metric_delta_value(before.get(metric_id, 0), after.get(metric_id, 0)) > 0:
            risks.append(f"{metric_id} increased")
    for metric_id in contract.primary_success_metrics:
        if _metric_delta_value(before.get(metric_id, 0), after.get(metric_id, 0)) < 0:
            risks.append(f"{metric_id} decreased")
    if _only_naive_improved(summary):
        risks.append("improvement appears limited to naive hero policy")
    return tuple(risks)


def _adjust_decision_for_package_identity(
    contract: EnemyPackageContract,
    decision: BalanceDecision,
    baseline_report: PackageReport,
    candidate_report: PackageReport,
) -> BalanceDecision:
    if decision.recommendation not in {RECOMMENDED, PROMISING}:
        return decision
    before = _package_result(baseline_report, contract.package_id).metric_values
    after = _package_result(candidate_report, contract.package_id).metric_values
    regressions = tuple(
        metric_id
        for metric_id in contract.primary_success_metrics
        if _metric_delta_value(before.get(metric_id, 0), after.get(metric_id, 0)) < 0
    )
    if not regressions:
        return decision
    return replace(
        decision,
        recommendation=MIXED,
        reasons=(
            *decision.reasons,
            f"primary package metrics regressed: {', '.join(regressions)}",
        ),
    )


def _robustness_lines(candidates: Sequence[TacticCandidate]) -> list[str]:
    if not candidates:
        return ["  none"]
    candidate = candidates[0]
    lines = []
    for evaluation in candidate.summary.policy_evaluations:
        if evaluation.policy_scope_id != candidate.policy_scope_id:
            continue
        route_text = ""
        if candidate.summary.route_id:
            route = evaluation.route_summary
            route_text = f"; routes {route.completed_count}/{route.route_count}"
        lines.append(
            "  "
            f"{candidate.candidate_id} vs {evaluation.evaluation_hero_policy_id}: "
            f"reward {evaluation.total_reward}; victors {_format_counts(evaluation.victor_counts)}"
            f"{route_text}"
        )
    return lines or ["  none"]


def _candidate_description(
    contract: EnemyPackageContract,
    scope_id: str,
    scale: float,
) -> str:
    return (
        f"Emphasize {contract.package_id} discovery features at x{scale:.2f} "
        f"for {scope_id} learned policy scope."
    )


def _decision_text(recommendation: str) -> str:
    if recommendation in {RECOMMENDED, PROMISING}:
        return "worth manual playtest"
    if recommendation == MIXED:
        return "inspect before playtest"
    if recommendation == NO_EFFECT:
        return "do not promote"
    return "do not promote"


def _package_contract(package_id: str) -> EnemyPackageContract:
    for contract in DEFAULT_PACKAGE_CONTRACTS:
        if contract.package_id == package_id:
            return contract
    raise ValueError(f"Unknown package id: {package_id}")


def _package_result(report: PackageReport, package_id: str):
    for result in report.results:
        if result.package_id == package_id:
            return result
    raise ValueError(f"Missing package report: {package_id}")


def _metric_delta_text(metric_id: str, before, after) -> str:
    delta = _metric_delta_value(before, after)
    return f"{metric_id} {delta:+}"


def _metric_delta_value(before, after) -> float:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return 0.0
    return float(after - before)


def _only_naive_improved(summary: TrainingRunSummary) -> bool:
    if len(summary.policy_evaluations) <= 1:
        return False
    rewards = {
        evaluation.evaluation_hero_policy_id: evaluation.total_reward
        for evaluation in summary.policy_evaluations
    }
    naive = rewards.get("naive", 0)
    others = [reward for policy_id, reward in rewards.items() if policy_id != "naive"]
    return bool(others) and naive > 0 and all(reward <= 0 for reward in others)


def _scale_label(scale: float) -> str:
    return str(scale).replace(".", "_")


def _validate_supported(value: str, supported: Sequence[str], label: str) -> str:
    if value in supported:
        return value
    raise ValueError(f"Unknown {label}: {value}")


def _validate_policy_scopes(scope_ids: tuple[str, ...]) -> tuple[str, ...]:
    scopes = tuple(scope_ids) or ("global",)
    missing = tuple(scope for scope in scopes if scope not in SUPPORTED_POLICY_SCOPE_IDS)
    if missing:
        raise ValueError(f"Unknown policy scope: {missing[0]}")
    return scopes


def _validate_hero_policies(policy_ids: tuple[str, ...]) -> tuple[str, ...]:
    policies = tuple(policy_ids) or SUPPORTED_HERO_POLICY_IDS
    missing = tuple(policy for policy in policies if policy not in SUPPORTED_HERO_POLICY_IDS)
    if missing:
        raise ValueError(f"Unknown hero policy: {missing[0]}")
    return policies


def _recommendation_order(recommendation: str) -> int:
    return {
        RECOMMENDED: 0,
        PROMISING: 1,
        MIXED: 2,
        NO_EFFECT: 3,
        REGRESSION: 4,
    }.get(recommendation, 5)


def _format_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


__all__ = [
    "TacticCandidate",
    "TacticDiscoveryConfig",
    "TacticDiscoveryReport",
    "format_tactic_discovery_report",
    "run_tactic_discovery",
]
