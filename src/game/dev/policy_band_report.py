"""Cross-policy robustness gates for route and breach lab scenarios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from game.combat.enemy_learning import SUPPORTED_HERO_POLICY_IDS
from game.dev.ai_decisions import GATE_FAIL, GATE_PASS, GATE_WARN
from game.dev.breach_balance_lab import BreachFightBalanceConfig
from game.dev.encounter_attribution import (
    EncounterAttributionReport,
    aggregate_encounter_attribution,
)
from game.dev.hero_policy_audit import (
    HeroPolicyAuditReport,
    aggregate_hero_policy_audit,
)
from game.dev.maze_mark_guard_package import (
    WARD_ROLE_ENCOUNTER_IDS,
    MazeMarkGuardMetrics,
    MazePackageThresholds,
    PackageContractGateResult,
    PolicyPackageInput,
    aggregate_maze_mark_guard_by_encounter,
    aggregate_maze_mark_guard_metrics,
    evaluate_maze_package_gates,
    format_maze_package_section,
    mixed_maze_episode_count,
)
from game.dev.maze_mark_guard_package import (
    package_overall_status as maze_package_overall_status,
)
from game.dev.route_lab import (
    GeneratedRouteLabConfig,
    GeneratedRouteLabSummary,
    RouteLabConfig,
    run_generated_route_lab,
)
from game.dev.train_enemy_ai import (
    TrainingPolicyEvaluation,
    TrainingRunConfig,
    run_training_harness,
)

SCENARIO_AUTHORED_ROUTE = "authored_route"
SCENARIO_GENERATED_SCOUT = "generated_scout"
SCENARIO_GENERATED_HUNT = "generated_hunt"

DIAG_TOO_PUNISHING = "too_punishing"
DIAG_TOO_EASY = "too_easy"
DIAG_SMART_POLICY_TRIVIALIZES = "smart_policy_trivializes"
DIAG_NAIVE_ONLY_PRESSURE = "naive_only_pressure"
DIAG_PACKAGE_PAYOFF_SUPPRESSED = "package_payoff_suppressed"

_GATE_STATUS_RANK = {GATE_PASS: 0, GATE_WARN: 1, GATE_FAIL: 2}


@dataclass(frozen=True)
class PolicyBandThresholds:
    scout_target_min: float = 0.30
    scout_target_max: float = 0.80
    scout_fail_min: float = 0.20
    scout_fail_max: float = 0.90
    hunt_target_min: float = 0.10
    hunt_target_max: float = 0.60
    hunt_fail_min: float = 0.05
    hunt_fail_max: float = 0.75
    spread_warn: int = 40
    spread_fail: int = 60
    smart_beats_mixed_warn: int = 25
    min_marks_applied: int = 1
    min_marks_exploited: int = 1
    min_vulnerable_payoffs: int = 1


@dataclass(frozen=True)
class PolicyRunMetrics:
    hero_policy_id: str
    completion_rate: float
    victor_counts: Mapping[str, int]
    average_downs: float
    average_deaths: float
    average_final_hp: float
    average_final_effort: float
    marks_applied: int
    marks_exploited: int
    vulnerable_payoffs: int
    maze_package: MazeMarkGuardMetrics | None = None
    ward_role_package: MazeMarkGuardMetrics | None = None
    encounter_attribution: EncounterAttributionReport | None = None
    hero_policy_audit: HeroPolicyAuditReport | None = None


@dataclass(frozen=True)
class BalanceGateResult:
    gate_id: str
    status: str
    detail: str
    diagnostic_label: str | None = None


@dataclass(frozen=True)
class PolicyBandReport:
    scenario_id: str
    scenario_name: str
    scenario_kind: str
    seed_count: int
    policy_metrics: tuple[PolicyRunMetrics, ...]
    gates: tuple[BalanceGateResult, ...]
    overall_status: str
    package_gates: tuple[PackageContractGateResult, ...] = ()
    package_overall_status: str = GATE_PASS
    package_insufficient_samples: bool = False


@dataclass(frozen=True)
class PolicyBandPairReport:
    scout: PolicyBandReport
    hunt: PolicyBandReport
    overall_status: str


def policy_metrics_from_training_evaluation(
    evaluation: TrainingPolicyEvaluation,
) -> PolicyRunMetrics:
    route_summary = evaluation.route_summary
    if route_summary.route_count:
        completion_rate = route_summary.completed_count / route_summary.route_count
        average_final_hp = route_summary.average_final_hero_hp
        average_final_effort = route_summary.average_final_hero_effort
        average_downs = route_summary.average_downs
        average_deaths = route_summary.average_deaths
    else:
        episode_count = len(evaluation.episodes)
        completion_rate = (
            evaluation.victor_counts.get("heroes", 0) / episode_count
            if episode_count
            else 0.0
        )
        average_final_hp = 0.0
        average_final_effort = 0.0
        average_downs = 0.0
        average_deaths = 0.0
    marks_applied, marks_exploited, vulnerable_payoffs = _mark_totals_from_episodes(
        evaluation.episodes
    )
    return PolicyRunMetrics(
        hero_policy_id=evaluation.evaluation_hero_policy_id,
        completion_rate=completion_rate,
        victor_counts=dict(evaluation.victor_counts),
        average_downs=average_downs,
        average_deaths=average_deaths,
        average_final_hp=average_final_hp,
        average_final_effort=average_final_effort,
        marks_applied=marks_applied,
        marks_exploited=marks_exploited,
        vulnerable_payoffs=vulnerable_payoffs,
    )


def policy_metrics_from_generated_summary(
    summary: GeneratedRouteLabSummary,
) -> PolicyRunMetrics:
    runs = summary.runs
    if runs:
        completion_rate = sum(1 for run in runs if run.completed) / len(runs)
        average_final_hp = sum(run.final_hero_hp_total for run in runs) / len(runs)
        average_final_effort = sum(run.final_hero_effort_total for run in runs) / len(
            runs
        )
        average_downs = sum(
            sum(episode.metrics.hero_downs for episode in run.episodes) for run in runs
        ) / len(runs)
        average_deaths = sum(
            sum(episode.metrics.hero_deaths for episode in run.episodes) for run in runs
        ) / len(runs)
    else:
        completion_rate = 0.0
        average_final_hp = 0.0
        average_final_effort = 0.0
        average_downs = 0.0
        average_deaths = 0.0
    episodes = tuple(episode for run in runs for episode in run.episodes)
    victor_counts: dict[str, int] = {}
    for episode in episodes:
        victor = episode.final_victor
        victor_counts[victor] = victor_counts.get(victor, 0) + 1
    marks_applied, marks_exploited, vulnerable_payoffs = _mark_totals_from_episodes(
        episodes
    )
    maze_package = aggregate_maze_mark_guard_metrics(episodes)
    ward_role_package = aggregate_maze_mark_guard_by_encounter(
        episodes, WARD_ROLE_ENCOUNTER_IDS
    )
    encounter_attribution = aggregate_encounter_attribution(
        episodes,
        policy_id=summary.hero_policy_id,
    )
    hero_policy_audit = aggregate_hero_policy_audit(
        episodes,
        policy_id=summary.hero_policy_id,
    )
    return PolicyRunMetrics(
        hero_policy_id=summary.hero_policy_id,
        completion_rate=completion_rate,
        victor_counts=victor_counts,
        average_downs=average_downs,
        average_deaths=average_deaths,
        average_final_hp=average_final_hp,
        average_final_effort=average_final_effort,
        marks_applied=marks_applied,
        marks_exploited=marks_exploited,
        vulnerable_payoffs=vulnerable_payoffs,
        maze_package=maze_package,
        ward_role_package=ward_role_package,
        encounter_attribution=encounter_attribution,
        hero_policy_audit=hero_policy_audit,
    )


def evaluate_policy_band(
    *,
    scenario_kind: str,
    policy_metrics: Sequence[PolicyRunMetrics],
    thresholds: PolicyBandThresholds | None = None,
) -> tuple[BalanceGateResult, ...]:
    thresholds = thresholds or PolicyBandThresholds()
    gates: list[BalanceGateResult] = []
    if scenario_kind == SCENARIO_GENERATED_SCOUT:
        gates.append(
            _scout_completion_gate(_mixed_completion(policy_metrics), thresholds)
        )
    elif scenario_kind == SCENARIO_GENERATED_HUNT:
        gates.append(
            _hunt_completion_gate(_mixed_completion(policy_metrics), thresholds)
        )
    if len(policy_metrics) >= 2:
        gates.append(_policy_spread_gate(policy_metrics, thresholds))
        gates.append(_smart_beats_mixed_gate(policy_metrics, thresholds))
    if scenario_kind in (SCENARIO_GENERATED_SCOUT, SCENARIO_GENERATED_HUNT):
        gates.append(_package_payoff_suppressed_gate(policy_metrics, thresholds))
    return tuple(gates)


def run_authored_route_policy_band(
    config: RouteLabConfig,
    *,
    thresholds: PolicyBandThresholds | None = None,
) -> PolicyBandReport:
    thresholds = thresholds or PolicyBandThresholds()
    training_summary = run_training_harness(
        TrainingRunConfig(
            definitions=config.definitions,
            route_id=config.route_id,
            seeds=config.seeds,
            max_rounds=config.max_rounds,
            hero_policy_id="mixed",
            evaluation_hero_policy_ids=SUPPORTED_HERO_POLICY_IDS,
            policy_scope_ids=("global",),
            preset_id=config.preset_id,
            enemy_wait_mode=config.enemy_wait_mode,
            enemy_movement_mode=config.enemy_movement_mode,
        )
    )
    policy_metrics = tuple(
        policy_metrics_from_training_evaluation(evaluation)
        for evaluation in training_summary.policy_evaluations
        if evaluation.policy_scope_id == "global"
    )
    return _build_policy_band_report(
        scenario_id=config.route_id,
        scenario_name=config.route_id,
        scenario_kind=SCENARIO_AUTHORED_ROUTE,
        seed_count=config.seeds,
        policy_metrics=policy_metrics,
        thresholds=thresholds,
    )


def run_generated_route_policy_band(
    config: GeneratedRouteLabConfig,
    *,
    scenario_kind: str,
    thresholds: PolicyBandThresholds | None = None,
) -> PolicyBandReport:
    thresholds = thresholds or PolicyBandThresholds()
    policy_metrics: list[PolicyRunMetrics] = []
    for hero_policy_id in SUPPORTED_HERO_POLICY_IDS:
        summary = run_generated_route_lab(
            replace(config, hero_policy_id=hero_policy_id)
        )
        policy_metrics.append(policy_metrics_from_generated_summary(summary))
    scenario_name = config.pressure_profile_id or scenario_kind
    return _build_policy_band_report(
        scenario_id=scenario_name,
        scenario_name=scenario_name,
        scenario_kind=scenario_kind,
        seed_count=config.seeds,
        policy_metrics=tuple(policy_metrics),
        thresholds=thresholds,
    )


def run_breach_policy_band_pair(
    config: BreachFightBalanceConfig,
    *,
    thresholds: PolicyBandThresholds | None = None,
) -> PolicyBandPairReport:
    thresholds = thresholds or PolicyBandThresholds()
    base_config = GeneratedRouteLabConfig(
        breach_id="shallow_cave_breach",
        seeds=config.seeds,
        max_rounds=config.max_rounds,
        preset_id=config.preset_id,
        strategy_id=config.strategy_id,
        definitions=config.definitions,
    )
    scout = run_generated_route_policy_band(
        replace(base_config, pressure_profile_id="breach_probe"),
        scenario_kind=SCENARIO_GENERATED_SCOUT,
        thresholds=thresholds,
    )
    hunt = run_generated_route_policy_band(
        replace(base_config, pressure_profile_id="marked_hunt"),
        scenario_kind=SCENARIO_GENERATED_HUNT,
        thresholds=thresholds,
    )
    return PolicyBandPairReport(
        scout=scout,
        hunt=hunt,
        overall_status=_worst_gate_status(
            scout.overall_status,
            hunt.overall_status,
        ),
    )


def format_policy_band_report(report: PolicyBandReport) -> str:
    lines = [
        "Policy Band Report",
        f"Scenario: {report.scenario_kind} ({report.scenario_name})",
        f"Seeds: {report.seed_count}",
        "Per-policy:",
    ]
    for metrics in report.policy_metrics:
        lines.append(
            f"  {metrics.hero_policy_id}: completion={metrics.completion_rate:.0%} "
            f"victors={metrics.victor_counts} downs={metrics.average_downs:.1f} "
            f"deaths={metrics.average_deaths:.1f} hp={metrics.average_final_hp:.1f} "
            f"effort={metrics.average_final_effort:.1f} marks={metrics.marks_applied} "
            f"exploited={metrics.marks_exploited} "
            f"vulnerable={metrics.vulnerable_payoffs}"
        )
    lines.append("Gates:")
    for gate in report.gates:
        label = f" {gate.diagnostic_label}" if gate.diagnostic_label else ""
        lines.append(f"  {gate.gate_id}: {gate.status} ({gate.detail}){label}")
    lines.append(f"Overall: {report.overall_status}")
    if report.scenario_kind in (SCENARIO_GENERATED_SCOUT, SCENARIO_GENERATED_HUNT):
        package_inputs = _policy_package_inputs(report.policy_metrics)
        lines.extend(
            format_maze_package_section(
                package_inputs,
                report.package_gates,
                insufficient_samples=report.package_insufficient_samples,
            )
        )
    if report.scenario_kind in (SCENARIO_GENERATED_SCOUT, SCENARIO_GENERATED_HUNT):
        from game.dev.encounter_attribution import (
            EncounterAttributionFinding,
            evaluate_encounter_attribution,
            format_encounter_attribution_section,
        )

        attribution_reports = _encounter_attribution_reports_for_format(report)
        if attribution_reports:
            attribution_findings: list[EncounterAttributionFinding] = []
            for attribution_report in attribution_reports:
                attribution_findings.extend(
                    evaluate_encounter_attribution(attribution_report)
                )
            lines.extend(
                format_encounter_attribution_section(
                    attribution_reports,
                    tuple(attribution_findings),
                )
            )

        from game.dev.hero_policy_audit import (
            evaluate_hero_policy_audit,
            format_hero_policy_audit_section,
        )

        audit_reports = _hero_policy_audit_reports_for_format(report)
        if audit_reports:
            completion_by_policy = {
                metrics.hero_policy_id: metrics.completion_rate
                for metrics in report.policy_metrics
            }
            audit_findings = evaluate_hero_policy_audit(
                audit_reports,
                completion_by_policy=completion_by_policy,
            )
            lines.extend(
                format_hero_policy_audit_section(audit_reports, audit_findings)
            )

    from game.dev.design_diagnosis import build_design_diagnosis, format_design_diagnosis

    lines.extend(format_design_diagnosis(build_design_diagnosis(report)))
    return "\n".join(lines)


def format_policy_band_pair_report(report: PolicyBandPairReport) -> str:
    return "\n\n".join(
        (
            format_policy_band_report(report.scout),
            format_policy_band_report(report.hunt),
            f"Pair overall: {report.overall_status}",
        )
    )


def _build_policy_band_report(
    *,
    scenario_id: str,
    scenario_name: str,
    scenario_kind: str,
    seed_count: int,
    policy_metrics: tuple[PolicyRunMetrics, ...],
    thresholds: PolicyBandThresholds,
) -> PolicyBandReport:
    gates = evaluate_policy_band(
        scenario_kind=scenario_kind,
        policy_metrics=policy_metrics,
        thresholds=thresholds,
    )
    package_gates: tuple[PackageContractGateResult, ...] = ()
    package_overall = GATE_PASS
    package_insufficient_samples = False
    if scenario_kind in (SCENARIO_GENERATED_SCOUT, SCENARIO_GENERATED_HUNT):
        package_inputs = _policy_package_inputs(policy_metrics)
        maze_thresholds = MazePackageThresholds(
            smart_beats_mixed_warn=thresholds.smart_beats_mixed_warn,
        )
        package_insufficient_samples = (
            mixed_maze_episode_count(package_inputs)
            < maze_thresholds.min_package_episodes
        )
        if package_insufficient_samples:
            package_gates = ()
            package_overall = GATE_PASS
        else:
            package_gates = evaluate_maze_package_gates(
                package_inputs,
                thresholds=maze_thresholds,
            )
            package_overall = maze_package_overall_status(package_gates)
    return PolicyBandReport(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        scenario_kind=scenario_kind,
        seed_count=seed_count,
        policy_metrics=policy_metrics,
        gates=gates,
        overall_status=_worst_gate_status(*(gate.status for gate in gates)),
        package_gates=package_gates,
        package_overall_status=package_overall,
        package_insufficient_samples=package_insufficient_samples,
    )


def _policy_package_inputs(
    policy_metrics: Sequence[PolicyRunMetrics],
) -> tuple[PolicyPackageInput, ...]:
    return tuple(
        PolicyPackageInput(
            hero_policy_id=metrics.hero_policy_id,
            completion_rate=metrics.completion_rate,
            maze_package=metrics.maze_package or MazeMarkGuardMetrics(),
            ward_role_package=metrics.ward_role_package,
        )
        for metrics in policy_metrics
        if metrics.maze_package is not None
    )


def _hero_policy_audit_reports_for_format(
    report: PolicyBandReport,
) -> tuple[HeroPolicyAuditReport, ...]:
    reports: list[HeroPolicyAuditReport] = []
    for metrics in report.policy_metrics:
        if metrics.hero_policy_audit is None:
            continue
        if metrics.hero_policy_id in (
            "mixed",
            "tactical",
            "anti_mark",
            "company_survival",
        ):
            reports.append(metrics.hero_policy_audit)
        elif metrics.hero_policy_audit.metrics.total_hero_actions > 0:
            reports.append(metrics.hero_policy_audit)
    return tuple(reports)


def _encounter_attribution_reports_for_format(
    report: PolicyBandReport,
) -> tuple[EncounterAttributionReport, ...]:
    reports: list[EncounterAttributionReport] = []
    for metrics in report.policy_metrics:
        if metrics.encounter_attribution is None or not metrics.encounter_attribution.encounters:
            continue
        if metrics.hero_policy_id == "mixed":
            reports.append(metrics.encounter_attribution)
            continue
        if any(
            row.setup_episodes > 0 or row.payoff_episodes > 0
            for row in metrics.encounter_attribution.encounters
        ):
            reports.append(metrics.encounter_attribution)
    return tuple(reports)


def _mark_totals_from_episodes(episodes: Sequence) -> tuple[int, int, int]:
    marks_applied = 0
    marks_exploited = 0
    vulnerable_payoffs = 0
    for episode in episodes:
        metrics = episode.metrics
        marks_applied += int(metrics.marks_applied)
        marks_exploited += int(metrics.mark_flow.exploited_by_enemy_hit)
        vulnerable_payoffs += int(metrics.mark_flow.vulnerable_payoffs)
    return marks_applied, marks_exploited, vulnerable_payoffs


def _mixed_completion(policy_metrics: Sequence[PolicyRunMetrics]) -> float:
    for metrics in policy_metrics:
        if metrics.hero_policy_id == "mixed":
            return metrics.completion_rate
    return 0.0


def _metrics_for(
    policy_metrics: Sequence[PolicyRunMetrics],
    hero_policy_id: str,
) -> PolicyRunMetrics | None:
    for metrics in policy_metrics:
        if metrics.hero_policy_id == hero_policy_id:
            return metrics
    return None


def _completion_for(
    policy_metrics: Sequence[PolicyRunMetrics],
    hero_policy_id: str,
) -> float:
    metrics = _metrics_for(policy_metrics, hero_policy_id)
    return metrics.completion_rate if metrics is not None else 0.0


def _scout_completion_gate(
    completion_rate: float,
    thresholds: PolicyBandThresholds,
) -> BalanceGateResult:
    detail = f"{completion_rate * 100:.0f}%"
    if completion_rate < thresholds.scout_fail_min:
        return BalanceGateResult(
            "scout_completion",
            GATE_FAIL,
            detail,
            DIAG_TOO_PUNISHING,
        )
    if completion_rate > thresholds.scout_fail_max:
        return BalanceGateResult(
            "scout_completion",
            GATE_FAIL,
            detail,
            DIAG_TOO_EASY,
        )
    if completion_rate < thresholds.scout_target_min:
        return BalanceGateResult(
            "scout_completion",
            GATE_WARN,
            detail,
            DIAG_TOO_PUNISHING,
        )
    if completion_rate > thresholds.scout_target_max:
        return BalanceGateResult(
            "scout_completion",
            GATE_WARN,
            detail,
            DIAG_TOO_EASY,
        )
    return BalanceGateResult("scout_completion", GATE_PASS, detail)


def _hunt_completion_gate(
    completion_rate: float,
    thresholds: PolicyBandThresholds,
) -> BalanceGateResult:
    detail = f"{completion_rate * 100:.0f}%"
    if completion_rate < thresholds.hunt_fail_min:
        return BalanceGateResult(
            "hunt_completion",
            GATE_FAIL,
            detail,
            DIAG_TOO_PUNISHING,
        )
    if completion_rate > thresholds.hunt_fail_max:
        return BalanceGateResult(
            "hunt_completion",
            GATE_FAIL,
            detail,
            DIAG_TOO_EASY,
        )
    if completion_rate < thresholds.hunt_target_min:
        return BalanceGateResult(
            "hunt_completion",
            GATE_WARN,
            detail,
            DIAG_TOO_PUNISHING,
        )
    if completion_rate > thresholds.hunt_target_max:
        return BalanceGateResult(
            "hunt_completion",
            GATE_WARN,
            detail,
            DIAG_TOO_EASY,
        )
    return BalanceGateResult("hunt_completion", GATE_PASS, detail)


def _policy_spread_gate(
    policy_metrics: Sequence[PolicyRunMetrics],
    thresholds: PolicyBandThresholds,
) -> BalanceGateResult:
    completions = {
        metrics.hero_policy_id: metrics.completion_rate for metrics in policy_metrics
    }
    spread_points = (max(completions.values()) - min(completions.values())) * 100
    detail = f"{spread_points:.0f}pt"
    if spread_points >= thresholds.spread_fail:
        status = GATE_FAIL
    elif spread_points >= thresholds.spread_warn:
        status = GATE_WARN
    else:
        return BalanceGateResult("policy_spread", GATE_PASS, detail)
    max_policy = max(completions, key=lambda policy_id: completions[policy_id])
    min_policy = min(completions, key=lambda policy_id: completions[policy_id])
    diagnostic: str | None = None
    if max_policy in ("anti_mark", "conservative"):
        diagnostic = DIAG_SMART_POLICY_TRIVIALIZES
    elif min_policy == "naive":
        diagnostic = DIAG_NAIVE_ONLY_PRESSURE
    return BalanceGateResult(
        "policy_spread",
        status,
        detail,
        diagnostic,
    )


def _smart_beats_mixed_gate(
    policy_metrics: Sequence[PolicyRunMetrics],
    thresholds: PolicyBandThresholds,
) -> BalanceGateResult:
    mixed = _completion_for(policy_metrics, "mixed")
    smart_best = max(
        _completion_for(policy_metrics, "anti_mark"),
        _completion_for(policy_metrics, "conservative"),
    )
    delta_points = (smart_best - mixed) * 100
    detail = f"{delta_points:.0f}pt"
    if delta_points >= thresholds.smart_beats_mixed_warn:
        return BalanceGateResult(
            "smart_beats_mixed",
            GATE_WARN,
            detail,
            DIAG_SMART_POLICY_TRIVIALIZES,
        )
    return BalanceGateResult("smart_beats_mixed", GATE_PASS, detail)


def _package_payoff_suppressed_gate(
    policy_metrics: Sequence[PolicyRunMetrics],
    thresholds: PolicyBandThresholds,
) -> BalanceGateResult:
    anti_mark = _metrics_for(policy_metrics, "anti_mark")
    if anti_mark is None:
        return BalanceGateResult("package_payoff_suppressed", GATE_PASS, "n/a")
    mixed = _completion_for(policy_metrics, "mixed")
    delta_points = (anti_mark.completion_rate - mixed) * 100
    detail = f"{delta_points:.0f}pt"
    if delta_points < thresholds.smart_beats_mixed_warn:
        return BalanceGateResult("package_payoff_suppressed", GATE_PASS, detail)
    if (
        anti_mark.marks_applied >= thresholds.min_marks_applied
        or anti_mark.marks_exploited >= thresholds.min_marks_exploited
        or anti_mark.vulnerable_payoffs >= thresholds.min_vulnerable_payoffs
    ):
        return BalanceGateResult("package_payoff_suppressed", GATE_PASS, detail)
    return BalanceGateResult(
        "package_payoff_suppressed",
        GATE_WARN,
        detail,
        DIAG_PACKAGE_PAYOFF_SUPPRESSED,
    )


def _worst_gate_status(*statuses: str) -> str:
    if not statuses:
        return GATE_PASS
    return max(statuses, key=lambda status: _GATE_STATUS_RANK[status])
