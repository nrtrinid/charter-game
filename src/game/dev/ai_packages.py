"""Dev-only enemy package health contracts and reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from game.dev.train_enemy_ai import EncounterTrainingBreakdown, TrainingRunSummary

PACKAGE_PASS = "PASS"
PACKAGE_WARN = "WARN"
PACKAGE_FAIL = "FAIL"
PACKAGE_OK_LOW_USE = "OK_LOW_USE"
PackageHealthStatus = str


@dataclass(frozen=True)
class PackageMetricThreshold:
    metric_id: str
    warn_min: float | None = None
    pass_min: float | None = None
    fail_above: float | None = None
    note: str = ""


@dataclass(frozen=True)
class EnemyPackageContract:
    package_id: str
    enemy_ids: tuple[str, ...]
    setup_actions: tuple[str, ...]
    payoff_actions: tuple[str, ...]
    support_actions: tuple[str, ...] = ()
    intended_counterplay: tuple[str, ...] = ()
    hero_policy_ids: tuple[str, ...] = ("mixed", "survival", "anti_mark", "conservative")
    preset_ids: tuple[str, ...] = ("fresh", "attrition")
    encounter_ids: tuple[str, ...] = ()
    route_ids: tuple[str, ...] = ()
    success_metrics: tuple[str, ...] = ()
    failure_metrics: tuple[str, ...] = ()
    degeneracy_metrics: tuple[str, ...] = ()
    thresholds: tuple[PackageMetricThreshold, ...] = ()
    tactic_metric_ids: tuple[str, ...] = ()
    primary_success_metrics: tuple[str, ...] = ()
    discovery_feature_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackageHealthResult:
    package_id: str
    status: PackageHealthStatus
    details: tuple[str, ...]
    metric_values: Mapping[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class PackageReport:
    route_id: str
    preset_id: str
    hero_policy_id: str
    seed_count: int
    results: tuple[PackageHealthResult, ...]


DEFAULT_PACKAGE_CONTRACTS: tuple[EnemyPackageContract, ...] = (
    EnemyPackageContract(
        package_id="maw_package",
        enemy_ids=("cave_maw_brute",),
        setup_actions=("drag_forward",),
        payoff_actions=("maw_slam",),
        support_actions=("dead_guard",),
        intended_counterplay=("move", "guard", "kill_maw", "kill_guard"),
        encounter_ids=("cave_mini_boss", "generated_maze_hunt"),
        route_ids=("opening_critical_path", "opening_pressure_path"),
        success_metrics=(
            "grabs",
            "grab_to_bite_same_target",
            "support_grabs",
            "bone_soldier_guarded_boss",
        ),
        failure_metrics=("boss_killed_before_first_bite", "grabbed_target_escaped_before_bite"),
        thresholds=(
            PackageMetricThreshold("grabs", warn_min=1),
            PackageMetricThreshold("grab_to_bite_rate", warn_min=0.20),
            PackageMetricThreshold("support_grabs", warn_min=1),
        ),
        tactic_metric_ids=(
            "grabs",
            "grab_to_bite_same_target",
            "support_grabs",
            "support_grab_downs",
            "direct_front_bites",
            "bone_soldier_guarded_boss",
        ),
        primary_success_metrics=(
            "grab_to_bite_same_target",
            "support_grabs",
            "bone_soldier_guarded_boss",
        ),
        degeneracy_metrics=(
            "direct_front_bites",
            "grabbed_target_escaped_before_bite",
        ),
        discovery_feature_ids=(
            "maw_grab_setup",
            "maw_bite_payoff",
            "maw_grab_high_value_support",
            "maw_grab_down_threat",
            "maw_grab_bite_expected_collapse",
            "boss_guard_package",
            "drag_forward",
        ),
    ),
    EnemyPackageContract(
        package_id="bandit_kill_lane",
        enemy_ids=("bandit_lookout", "bandit_slinger", "bandit_cutthroat"),
        setup_actions=("pinning_shot",),
        payoff_actions=("bandit_blade", "sling_stone"),
        intended_counterplay=("move", "guard", "kill_spotter", "kill_payoff_enemy"),
        encounter_ids=("road_bandits",),
        route_ids=("opening_pressure_path",),
        success_metrics=("marks_applied", "exploited_by_enemy_hit", "vulnerable_payoffs"),
        failure_metrics=("ignored_marked_legal_attacks",),
        thresholds=(
            PackageMetricThreshold("marks_applied", warn_min=1),
            PackageMetricThreshold("ignored_marked_legal_attacks", fail_above=3),
        ),
        tactic_metric_ids=(
            "marks_applied",
            "exploited_by_enemy_hit",
            "multi_hit_focus",
            "vulnerable_payoffs",
            "marked_downs",
            "ignored_marked_legal_attacks",
        ),
        primary_success_metrics=(
            "exploited_by_enemy_hit",
            "vulnerable_payoffs",
            "marked_downs",
        ),
        degeneracy_metrics=("ignored_marked_legal_attacks",),
        discovery_feature_ids=(
            "bandit_mark_collapse",
            "bandit_mark_kill_lane",
            "bandit_marked_attack",
            "bandit_marked_payoff",
            "mark_ally_reach_count",
            "mark_expected_followup_damage",
            "mark_payoff_attacker_count",
        ),
    ),
    EnemyPackageContract(
        package_id="wolf_mark",
        enemy_ids=("wolf", "alpha_wolf"),
        setup_actions=("howl_mark",),
        payoff_actions=("pack_bite",),
        intended_counterplay=("guard", "kill_alpha", "break_reach"),
        encounter_ids=("wolf_pack",),
        route_ids=("opening_pressure_path",),
        success_metrics=("marks_applied", "average_ally_reach"),
        failure_metrics=("ignored_marked_legal_attacks",),
        tactic_metric_ids=(
            "marks_applied",
            "average_ally_reach",
            "exploited_by_enemy_hit",
            "vulnerable_payoffs",
            "ignored_marked_legal_attacks",
        ),
        primary_success_metrics=("marks_applied", "exploited_by_enemy_hit"),
        degeneracy_metrics=("ignored_marked_legal_attacks",),
        discovery_feature_ids=(
            "mark_ally_reach_count",
            "mark_expected_followup_damage",
            "mark_payoff_attacker_count",
            "vulnerable_payoff",
            "marked_focus",
        ),
    ),
)


def evaluate_enemy_packages(
    summary: TrainingRunSummary,
    contracts: Sequence[EnemyPackageContract] = DEFAULT_PACKAGE_CONTRACTS,
) -> PackageReport:
    return PackageReport(
        route_id=summary.route_id,
        preset_id=summary.preset_id,
        hero_policy_id=summary.hero_policy_id,
        seed_count=summary.seed_count,
        results=tuple(evaluate_package_health(contract, summary) for contract in contracts),
    )


def evaluate_package_health(
    contract: EnemyPackageContract,
    summary: TrainingRunSummary,
) -> PackageHealthResult:
    breakdowns = tuple(
        breakdown
        for breakdown in summary.encounter_breakdowns
        if breakdown.encounter_id in contract.encounter_ids
    )
    if not breakdowns:
        return PackageHealthResult(
            package_id=contract.package_id,
            status=PACKAGE_WARN,
            details=("no matching encounter data",),
            metric_values={"encounters": ",".join(contract.encounter_ids)},
        )
    if contract.package_id == "maw_package":
        return _evaluate_maw_package(contract, breakdowns)
    if contract.package_id == "bandit_kill_lane":
        return _evaluate_bandit_package(contract, breakdowns)
    if contract.package_id == "wolf_mark":
        return _evaluate_wolf_package(contract, breakdowns)
    return PackageHealthResult(
        package_id=contract.package_id,
        status=PACKAGE_WARN,
        details=("no evaluator registered for package",),
        metric_values={},
    )


def format_package_report(report: PackageReport) -> str:
    lines = [
        "Package Health:",
        (
            "  "
            f"route={report.route_id or 'isolated encounters'}; "
            f"preset={report.preset_id}; "
            f"hero_policy={report.hero_policy_id}; "
            f"seeds={report.seed_count}"
        ),
    ]
    for result in report.results:
        details = "; ".join(result.details) if result.details else "no details"
        metrics = _format_metric_values(result.metric_values)
        lines.append(f"  {result.package_id}: {result.status}")
        lines.append(f"    {details}")
        if metrics:
            lines.append(f"    {metrics}")
    return "\n".join(lines)


def _evaluate_maw_package(
    contract: EnemyPackageContract,
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> PackageHealthResult:
    metrics = _maw_metric_values(breakdowns)
    details: list[str] = []
    status = PACKAGE_PASS

    if metrics["episodes"] == 0:
        status = PACKAGE_WARN
        details.append("no learned episodes")
    elif metrics["grabs"] == 0 and metrics["bites"] == 0:
        status = PACKAGE_FAIL
        details.append("boss package produced no grabs or bites")
    else:
        if metrics["grabs"] == 0:
            status = _worse_status(status, PACKAGE_WARN)
            details.append("no Drag Forward usage")
        if metrics["bites"] == 0:
            status = _worse_status(status, PACKAGE_WARN)
            details.append("no Maw Slam/Bite usage")
        if metrics["grabs"] >= 3 and metrics["grab_to_bite_rate"] < _threshold(
            contract,
            "grab_to_bite_rate",
            0.20,
        ):
            status = _worse_status(status, PACKAGE_WARN)
            details.append("low grab->bite conversion")
        if metrics["grabs"] >= 3 and metrics["support_grabs"] == 0:
            status = _worse_status(status, PACKAGE_WARN)
            details.append("no support-target grabs observed")
        if (
            metrics["cave_mini_boss_episodes"] >= 5
            and metrics["bone_soldier_guarded_boss"] == 0
        ):
            status = _worse_status(status, PACKAGE_WARN)
            details.append("no Bone Soldier guard support observed")

    if not details:
        details.append(
            "grabs, bite payoff, support targeting, and guard diagnostics are in band"
        )
    return PackageHealthResult(contract.package_id, status, tuple(details), metrics)


def _evaluate_bandit_package(
    contract: EnemyPackageContract,
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> PackageHealthResult:
    metrics = _mark_metric_values(breakdowns)
    details: list[str] = []
    status = PACKAGE_PASS
    if metrics["episodes"] == 0:
        status = PACKAGE_WARN
        details.append("no learned episodes")
    elif metrics["marks_applied"] == 0 and metrics["episodes"] >= 2:
        status = PACKAGE_FAIL
        details.append("no Mark setup observed")
    if metrics["ignored_marked_legal_attacks"] >= _threshold(
        contract,
        "ignored_marked_legal_attacks",
        3,
    ) and metrics["ignored_marked_legal_attacks"] > (
        metrics["exploited_by_enemy_hit"] + metrics["vulnerable_payoffs"]
    ):
        status = PACKAGE_FAIL
        details.append("ignored legal marked targets outpaced payoff")
    if metrics["marks_applied"] >= 3 and metrics["vulnerable_payoffs"] == 0:
        status = _worse_status(status, PACKAGE_WARN)
        details.append("marks were set but no vulnerable payoff was observed")
    if not details:
        details.append("marked hits/payoffs are present and ignored legal marks are controlled")
    return PackageHealthResult(contract.package_id, status, tuple(details), metrics)


def _evaluate_wolf_package(
    contract: EnemyPackageContract,
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> PackageHealthResult:
    metrics = _mark_metric_values(breakdowns)
    details: list[str] = []
    status = PACKAGE_PASS
    if metrics["episodes"] == 0:
        return PackageHealthResult(
            contract.package_id,
            PACKAGE_WARN,
            ("no learned episodes",),
            metrics,
        )
    low_ally_reach = metrics["average_ally_reach"] < 1.0
    low_mark_use = metrics["marks_applied"] <= max(1, int(metrics["episodes"] * 0.25))
    if low_mark_use and low_ally_reach:
        return PackageHealthResult(
            contract.package_id,
            PACKAGE_OK_LOW_USE,
            ("low Mark usage acceptable because average ally reach is low",),
            metrics,
        )
    if metrics["ignored_marked_legal_attacks"] > (
        metrics["exploited_by_enemy_hit"] + metrics["vulnerable_payoffs"] + 2
    ):
        status = PACKAGE_FAIL
        details.append("ignored legal marked targets spiked")
    if low_mark_use and not low_ally_reach:
        status = _worse_status(status, PACKAGE_WARN)
        details.append("low wolf Mark usage despite available ally reach")
    if not details:
        details.append("wolf Mark behavior is consistent with available follow-up")
    return PackageHealthResult(contract.package_id, status, tuple(details), metrics)


def _maw_metric_values(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> dict[str, float | int]:
    episodes = sum(breakdown.learned.episode_count for breakdown in breakdowns)
    cave_episodes = sum(
        breakdown.learned.episode_count
        for breakdown in breakdowns
        if breakdown.encounter_id == "cave_mini_boss"
    )
    sequence = [breakdown.learned.boss_sequence for breakdown in breakdowns]
    targeting = [breakdown.learned.boss_targeting for breakdown in breakdowns]
    grabs = sum(metric.grab_uses for metric in sequence)
    conversions = sum(metric.grab_to_bite_same_target for metric in sequence)
    return {
        "episodes": episodes,
        "cave_mini_boss_episodes": cave_episodes,
        "grabs": grabs,
        "bites": sum(metric.bite_uses for metric in sequence),
        "grab_to_bite_same_target": conversions,
        "grab_to_bite_rate": conversions / grabs if grabs else 0.0,
        "support_grabs": sum(metric.support_grabs for metric in targeting),
        "support_grab_downs": sum(metric.support_grab_downs for metric in targeting),
        "direct_front_bites": sum(metric.direct_front_bites for metric in targeting),
        "bone_soldier_guarded_boss": sum(metric.bone_soldier_guarded_boss for metric in sequence),
    }


def _mark_metric_values(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> dict[str, float | int]:
    episodes = sum(breakdown.learned.episode_count for breakdown in breakdowns)
    metrics = [breakdown.learned.mark_flow for breakdown in breakdowns]
    reach_total = sum(metric.mark_ally_reach_total for metric in metrics)
    reach_count = sum(metric.mark_ally_reach_count for metric in metrics)
    return {
        "episodes": episodes,
        "marks_applied": sum(metric.marks_applied for metric in metrics),
        "exploited_by_enemy_hit": sum(metric.exploited_by_enemy_hit for metric in metrics),
        "multi_hit_focus": sum(metric.multi_hit_focus for metric in metrics),
        "vulnerable_payoffs": sum(metric.vulnerable_payoffs for metric in metrics),
        "marked_downs": sum(metric.marked_downs for metric in metrics),
        "ignored_marked_legal_attacks": sum(
            metric.ignored_marked_legal_attacks for metric in metrics
        ),
        "average_ally_reach": reach_total / reach_count if reach_count else 0.0,
    }


def _threshold(
    contract: EnemyPackageContract,
    metric_id: str,
    fallback: float,
) -> float:
    for threshold in contract.thresholds:
        if threshold.metric_id == metric_id:
            if threshold.fail_above is not None:
                return threshold.fail_above
            if threshold.warn_min is not None:
                return threshold.warn_min
            if threshold.pass_min is not None:
                return threshold.pass_min
    return fallback


def _worse_status(
    current: PackageHealthStatus,
    candidate: PackageHealthStatus,
) -> PackageHealthStatus:
    order = {
        PACKAGE_PASS: 0,
        PACKAGE_OK_LOW_USE: 1,
        PACKAGE_WARN: 2,
        PACKAGE_FAIL: 3,
    }
    return candidate if order[candidate] > order[current] else current


def _format_metric_values(values: Mapping[str, float | int | str]) -> str:
    parts: list[str] = []
    for key, value in sorted(values.items()):
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


__all__ = [
    "DEFAULT_PACKAGE_CONTRACTS",
    "EnemyPackageContract",
    "PackageHealthResult",
    "PackageMetricThreshold",
    "PackageReport",
    "evaluate_enemy_packages",
    "evaluate_package_health",
    "format_package_report",
]
