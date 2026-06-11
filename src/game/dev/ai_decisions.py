"""Dev-only balance decision cards for AI lab counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass

from game.dev.ai_counterfactuals import (
    MIXED,
    NO_EFFECT,
    PROMISING,
    RECOMMENDED,
    REGRESSION,
    CounterfactualResult,
    CounterfactualSweepSummary,
)
from game.dev.ai_packages import PackageReport
from game.dev.route_lab import RouteEnvelopeScore, score_route_envelope
from game.dev.train_enemy_ai import RouteTrainingResult, TrainingRunSummary

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

GATE_PASS = "PASS"
GATE_WARN = "WARN"
GATE_FAIL = "FAIL"

MEANINGFUL_REWARD_DELTA = 100
STRONG_REWARD_DELTA = 1000


@dataclass(frozen=True)
class MetricDelta:
    metric_id: str
    before: int | float | str
    after: int | float | str
    delta: int | float | str


@dataclass(frozen=True)
class PromotionGate:
    gate_id: str
    status: str
    reason: str


@dataclass(frozen=True)
class PairedSeedDelta:
    seeds_compared: int = 0
    seeds_improved: int = 0
    seeds_worsened: int = 0
    seeds_unchanged: int = 0
    route_completions_gained: int = 0
    route_completions_lost: int = 0
    average_reward_delta: float = 0.0
    average_final_hp_delta: float = 0.0


@dataclass(frozen=True)
class BalanceDecision:
    variant_id: str
    description: str
    recommendation: str
    confidence: str
    gates: tuple[PromotionGate, ...]
    deltas: tuple[MetricDelta, ...]
    paired_seed_delta: PairedSeedDelta
    reasons: tuple[str, ...]
    score: int
    baseline_envelope_status: str = "n/a"
    candidate_envelope_status: str = "n/a"


@dataclass(frozen=True)
class BalanceDecisionReport:
    baseline_reward: int
    baseline_route_completions: int
    decisions: tuple[BalanceDecision, ...]


def build_balance_decision_report(
    sweep_summary: CounterfactualSweepSummary,
) -> BalanceDecisionReport:
    decisions = tuple(
        _decision_for_result(sweep_summary, result)
        for result in sweep_summary.results
    )
    return BalanceDecisionReport(
        baseline_reward=sweep_summary.baseline_summary.learned_total_reward,
        baseline_route_completions=_completion_count(sweep_summary.baseline_summary),
        decisions=tuple(
            sorted(
                decisions,
                key=lambda decision: (
                    _recommendation_order(decision.recommendation),
                    -decision.score,
                    decision.variant_id,
                ),
            )
        ),
    )


def format_balance_decision_report(report: BalanceDecisionReport) -> str:
    lines = [
        "Enemy Counterfactual Sweep",
        (
            "Baseline learned reward: "
            f"{report.baseline_reward}; "
            f"routes {report.baseline_route_completions}"
        ),
        "Ranked variants:",
    ]
    if not report.decisions:
        lines.append("  none")
    for decision in report.decisions:
        gate_counts = _gate_counts(decision.gates)
        reason = "; ".join(decision.reasons) if decision.reasons else "no reason recorded"
        lines.extend(
            [
                f"  {decision.variant_id}:",
                f"    recommendation {decision.recommendation}",
                f"    confidence {decision.confidence}",
                f"    score {decision.score:+}",
                f"    {_format_delta_line(decision.deltas)}",
                (
                    "    route envelope "
                    f"{decision.baseline_envelope_status} -> "
                    f"{decision.candidate_envelope_status}"
                ),
                (
                    "    paired seeds "
                    f"improved={decision.paired_seed_delta.seeds_improved}, "
                    f"worsened={decision.paired_seed_delta.seeds_worsened}, "
                    f"unchanged={decision.paired_seed_delta.seeds_unchanged}, "
                    f"reward_delta={decision.paired_seed_delta.average_reward_delta:.1f}, "
                    f"final_hp_delta={decision.paired_seed_delta.average_final_hp_delta:.1f}"
                ),
                (
                    "    gates "
                    f"{gate_counts.get(GATE_PASS, 0)} PASS, "
                    f"{gate_counts.get(GATE_WARN, 0)} WARN, "
                    f"{gate_counts.get(GATE_FAIL, 0)} FAIL"
                ),
                f"    reason: {reason}",
            ]
        )
        for gate in decision.gates:
            if gate.status != GATE_PASS:
                lines.append(f"    gate {gate.gate_id}: {gate.status} - {gate.reason}")
        if decision.description:
            lines.append(f"    {decision.description}")
    return "\n".join(lines)


def paired_seed_delta(
    baseline: TrainingRunSummary,
    candidate: TrainingRunSummary,
) -> PairedSeedDelta:
    if baseline.learned_route_results or candidate.learned_route_results:
        return _paired_route_seed_delta(
            baseline.learned_route_results,
            candidate.learned_route_results,
        )
    return _paired_episode_seed_delta(baseline, candidate)


def _decision_for_result(
    sweep_summary: CounterfactualSweepSummary,
    result: CounterfactualResult,
) -> BalanceDecision:
    paired = paired_seed_delta(sweep_summary.baseline_summary, result.summary)
    baseline_envelope = _route_envelope_score(sweep_summary.baseline_summary)
    candidate_envelope = _route_envelope_score(result.summary)
    gates = _promotion_gates(
        result,
        sweep_summary.baseline_package_report,
        result.package_report,
        candidate_envelope,
    )
    recommendation = _decision_recommendation(result, gates, paired)
    confidence = _decision_confidence(result, recommendation, paired)
    reasons = _decision_reasons(result, gates, paired, recommendation)
    return BalanceDecision(
        variant_id=result.variant_id,
        description=result.description,
        recommendation=recommendation,
        confidence=confidence,
        gates=gates,
        deltas=_metric_deltas(result),
        paired_seed_delta=paired,
        reasons=reasons,
        score=0 if recommendation == NO_EFFECT else result.score,
        baseline_envelope_status=baseline_envelope.status if baseline_envelope else "n/a",
        candidate_envelope_status=candidate_envelope.status if candidate_envelope else "n/a",
    )


def _promotion_gates(
    result: CounterfactualResult,
    baseline_report: PackageReport,
    candidate_report: PackageReport,
    candidate_envelope: RouteEnvelopeScore | None,
) -> tuple[PromotionGate, ...]:
    gates = [
        _route_envelope_gate(candidate_envelope),
        _package_health_gate(baseline_report, candidate_report),
        _meaningful_delta_gate(result),
        _major_failure_shift_gate(result),
        PromotionGate(
            "no_hidden_spike_warning",
            GATE_PASS,
            "current telemetry has no deeper hidden-spike detector",
        ),
    ]
    return tuple(gates)


def _route_envelope_gate(candidate_envelope: RouteEnvelopeScore | None) -> PromotionGate:
    if candidate_envelope is None:
        return PromotionGate("route_envelope_not_fail", GATE_PASS, "no route envelope")
    if candidate_envelope.status == GATE_FAIL:
        return PromotionGate("route_envelope_not_fail", GATE_FAIL, "route envelope failed")
    return PromotionGate(
        "route_envelope_not_fail",
        GATE_PASS,
        f"route envelope is {candidate_envelope.status}",
    )


def _package_health_gate(
    baseline_report: PackageReport,
    candidate_report: PackageReport,
) -> PromotionGate:
    baseline = {result.package_id: result.status for result in baseline_report.results}
    severe: list[str] = []
    warnings: list[str] = []
    for result in candidate_report.results:
        before = baseline.get(result.package_id, "WARN")
        after = result.status
        if _status_value(after) < _status_value(before):
            text = f"{result.package_id} {before}->{after}"
            if before in {"PASS", "OK_LOW_USE"} and after in {"WARN", "FAIL"}:
                severe.append(text)
            else:
                warnings.append(text)
    if severe:
        return PromotionGate(
            "package_health_not_regressed",
            GATE_FAIL,
            ", ".join(severe),
        )
    if warnings:
        return PromotionGate(
            "package_health_not_regressed",
            GATE_WARN,
            ", ".join(warnings),
        )
    return PromotionGate("package_health_not_regressed", GATE_PASS, "package health held")


def _meaningful_delta_gate(result: CounterfactualResult) -> PromotionGate:
    meaningful = any(
        (
            abs(result.reward_delta) >= MEANINGFUL_REWARD_DELTA,
            result.completion_delta != 0,
            result.package_health_delta != 0,
            abs(result.package_metric_delta) >= 10,
            result.route_envelope_delta != 0,
        )
    )
    if not meaningful:
        return PromotionGate(
            "meaningful_delta_present",
            GATE_FAIL,
            "no meaningful metric delta",
        )
    return PromotionGate("meaningful_delta_present", GATE_PASS, "meaningful delta present")


def _major_failure_shift_gate(result: CounterfactualResult) -> PromotionGate:
    if result.completion_delta < 0:
        return PromotionGate(
            "no_major_failure_shift",
            GATE_WARN,
            f"route completions changed by {result.completion_delta}",
        )
    return PromotionGate("no_major_failure_shift", GATE_PASS, "no completion regression")


def _decision_recommendation(
    result: CounterfactualResult,
    gates: tuple[PromotionGate, ...],
    paired: PairedSeedDelta,
) -> str:
    failed_gates = [gate for gate in gates if gate.status == GATE_FAIL]
    warning_gates = [gate for gate in gates if gate.status == GATE_WARN]
    has_improvement = _has_improvement(result, paired)
    has_regression = _has_regression(result, paired)
    if result.recommendation == NO_EFFECT or any(
        gate.gate_id == "meaningful_delta_present" and gate.status == GATE_FAIL
        for gate in gates
    ):
        return NO_EFFECT
    if failed_gates:
        return MIXED if has_improvement else REGRESSION
    if has_improvement and not has_regression and not warning_gates:
        return RECOMMENDED if _strong_paired_direction(paired) else PROMISING
    if has_improvement and (has_regression or warning_gates):
        return MIXED
    if has_regression:
        return REGRESSION
    return result.recommendation


def _decision_confidence(
    result: CounterfactualResult,
    recommendation: str,
    paired: PairedSeedDelta,
) -> str:
    if recommendation == NO_EFFECT:
        return CONFIDENCE_HIGH
    if recommendation == REGRESSION and (
        abs(result.reward_delta) >= STRONG_REWARD_DELTA
        or result.package_health_delta < 0
        or result.route_envelope_delta < 0
    ):
        return CONFIDENCE_HIGH
    changed = paired.seeds_improved + paired.seeds_worsened
    if changed >= 3:
        dominant = max(paired.seeds_improved, paired.seeds_worsened)
        if dominant / changed >= 0.7:
            return CONFIDENCE_HIGH
        return CONFIDENCE_MEDIUM
    if abs(result.reward_delta) >= STRONG_REWARD_DELTA:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _decision_reasons(
    result: CounterfactualResult,
    gates: tuple[PromotionGate, ...],
    paired: PairedSeedDelta,
    recommendation: str,
) -> tuple[str, ...]:
    reasons = list(result.reasons)
    for gate in gates:
        if gate.status != GATE_PASS and gate.reason not in reasons:
            reasons.append(gate.reason)
    if paired.seeds_improved or paired.seeds_worsened:
        reasons.append(
            "paired seeds "
            f"{paired.seeds_improved} improved/"
            f"{paired.seeds_worsened} worsened"
        )
    if recommendation == NO_EFFECT and not reasons:
        reasons.append("no meaningful metric delta")
    return tuple(reasons)


def _metric_deltas(result: CounterfactualResult) -> tuple[MetricDelta, ...]:
    return (
        MetricDelta("reward", "baseline", "candidate", result.reward_delta),
        MetricDelta("completion", "baseline", "candidate", result.completion_delta),
        MetricDelta(
            "package_health",
            "baseline",
            "candidate",
            result.package_health_delta,
        ),
        MetricDelta(
            "package_metrics",
            "baseline",
            "candidate",
            result.package_metric_delta,
        ),
        MetricDelta(
            "route_envelope",
            "baseline",
            "candidate",
            result.route_envelope_delta,
        ),
    )


def _paired_route_seed_delta(
    baseline_results: tuple[RouteTrainingResult, ...],
    candidate_results: tuple[RouteTrainingResult, ...],
) -> PairedSeedDelta:
    baseline = {result.seed: result for result in baseline_results}
    candidate = {result.seed: result for result in candidate_results}
    reward_deltas: list[int] = []
    hp_deltas: list[int] = []
    improved = worsened = unchanged = gained = lost = 0
    for seed in sorted(set(baseline).intersection(candidate)):
        before = baseline[seed]
        after = candidate[seed]
        reward_delta = after.total_reward - before.total_reward
        hp_delta = after.final_hero_hp_total - before.final_hero_hp_total
        reward_deltas.append(reward_delta)
        hp_deltas.append(hp_delta)
        seed_gained = 0
        seed_lost = 0
        if after.completed and not before.completed:
            gained += 1
            seed_gained = 1
        elif before.completed and not after.completed:
            lost += 1
            seed_lost = 1
        direction = _paired_direction(
            reward_delta,
            hp_delta,
            gained_delta=seed_gained,
            lost_delta=seed_lost,
        )
        if direction > 0:
            improved += 1
        elif direction < 0:
            worsened += 1
        else:
            unchanged += 1
    return PairedSeedDelta(
        seeds_compared=len(reward_deltas),
        seeds_improved=improved,
        seeds_worsened=worsened,
        seeds_unchanged=unchanged,
        route_completions_gained=gained,
        route_completions_lost=lost,
        average_reward_delta=_average(reward_deltas),
        average_final_hp_delta=_average(hp_deltas),
    )


def _paired_episode_seed_delta(
    baseline: TrainingRunSummary,
    candidate: TrainingRunSummary,
) -> PairedSeedDelta:
    before = {
        (episode.encounter_id, episode.seed): episode
        for episode in baseline.learned_episodes
    }
    after = {
        (episode.encounter_id, episode.seed): episode
        for episode in candidate.learned_episodes
    }
    reward_deltas: list[int] = []
    hp_deltas: list[int] = []
    improved = worsened = unchanged = 0
    for key in sorted(set(before).intersection(after)):
        reward_delta = after[key].total_reward - before[key].total_reward
        hp_delta = (
            after[key].metrics.final_hero_hp_total
            - before[key].metrics.final_hero_hp_total
        )
        reward_deltas.append(reward_delta)
        hp_deltas.append(hp_delta)
        direction = _paired_direction(reward_delta, hp_delta)
        if direction > 0:
            improved += 1
        elif direction < 0:
            worsened += 1
        else:
            unchanged += 1
    return PairedSeedDelta(
        seeds_compared=len(reward_deltas),
        seeds_improved=improved,
        seeds_worsened=worsened,
        seeds_unchanged=unchanged,
        average_reward_delta=_average(reward_deltas),
        average_final_hp_delta=_average(hp_deltas),
    )


def _paired_direction(
    reward_delta: int,
    final_hp_delta: int,
    *,
    gained_delta: int = 0,
    lost_delta: int = 0,
) -> int:
    if gained_delta:
        return 1
    if lost_delta:
        return -1
    if abs(reward_delta) >= MEANINGFUL_REWARD_DELTA:
        return 1 if reward_delta > 0 else -1
    if abs(final_hp_delta) >= 2:
        return 1 if final_hp_delta < 0 else -1
    return 0


def _route_envelope_score(summary: TrainingRunSummary) -> RouteEnvelopeScore | None:
    if not summary.route_id or not summary.learned_route_results:
        return None
    return score_route_envelope(
        summary.learned_route_summary,
        envelope_id=_default_route_envelope_id(summary.route_id),
    )


def _default_route_envelope_id(route_id: str) -> str:
    if route_id == "opening_pressure_path":
        return "optional_pressure_path"
    return "critical_path"


def _has_improvement(result: CounterfactualResult, paired: PairedSeedDelta) -> bool:
    return any(
        (
            result.reward_delta >= MEANINGFUL_REWARD_DELTA,
            result.completion_delta > 0,
            result.package_health_delta > 0,
            result.package_metric_delta >= 10,
            result.route_envelope_delta > 0,
            paired.seeds_improved > paired.seeds_worsened,
        )
    )


def _has_regression(result: CounterfactualResult, paired: PairedSeedDelta) -> bool:
    return any(
        (
            result.reward_delta <= -MEANINGFUL_REWARD_DELTA,
            result.completion_delta < 0,
            result.package_health_delta < 0,
            result.package_metric_delta <= -10,
            result.route_envelope_delta < 0,
            paired.seeds_worsened > paired.seeds_improved,
        )
    )


def _strong_paired_direction(paired: PairedSeedDelta) -> bool:
    changed = paired.seeds_improved + paired.seeds_worsened
    return changed >= 3 and paired.seeds_improved / changed >= 0.7


def _status_value(status: str) -> int:
    return {
        "FAIL": -2,
        "WARN": -1,
        "OK_LOW_USE": 0,
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


def _completion_count(summary: TrainingRunSummary) -> int:
    if summary.learned_route_results:
        return sum(1 for result in summary.learned_route_results if result.completed)
    return summary.learned_victors.get("heroes", 0)


def _gate_counts(gates: tuple[PromotionGate, ...]) -> dict[str, int]:
    counts = {GATE_PASS: 0, GATE_WARN: 0, GATE_FAIL: 0}
    for gate in gates:
        counts[gate.status] = counts.get(gate.status, 0) + 1
    return counts


def _format_delta_line(deltas: tuple[MetricDelta, ...]) -> str:
    values = {delta.metric_id: delta.delta for delta in deltas}
    return (
        f"reward {int(values.get('reward', 0)):+}; "
        f"completion {int(values.get('completion', 0)):+}; "
        f"package health {int(values.get('package_health', 0)):+}; "
        f"package metrics {int(values.get('package_metrics', 0)):+}; "
        f"route envelope {int(values.get('route_envelope', 0)):+}"
    )


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


__all__ = [
    "BalanceDecision",
    "BalanceDecisionReport",
    "MetricDelta",
    "PairedSeedDelta",
    "PromotionGate",
    "build_balance_decision_report",
    "format_balance_decision_report",
    "paired_seed_delta",
]
