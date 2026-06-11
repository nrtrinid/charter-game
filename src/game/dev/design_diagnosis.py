"""Deterministic design diagnosis cards for policy-band lab reports."""

from __future__ import annotations

from dataclasses import dataclass

from game.dev.ai_decisions import GATE_FAIL, GATE_PASS, GATE_WARN
from game.dev.maze_mark_guard_package import (
    DIAG_NAIVE_GETS_PUNISHED,
    DIAG_PACKAGE_NEVER_SETS_UP,
    DIAG_PAYOFF_NEVER_FIRES,
    DIAG_SMART_POLICY_ANSWERS_PACKAGE,
    DIAG_SMART_POLICY_DELETES_PACKAGE,
    DIAG_WARD_ROLE_INVISIBLE,
    GATE_INFO,
)
from game.dev.policy_band_report import (
    DIAG_NAIVE_ONLY_PRESSURE,
    DIAG_PACKAGE_PAYOFF_SUPPRESSED,
    DIAG_SMART_POLICY_TRIVIALIZES,
    DIAG_TOO_EASY,
    DIAG_TOO_PUNISHING,
    PolicyBandReport,
)

VERDICT_HEALTHY = "HEALTHY"
VERDICT_PROMISING = "PROMISING"
VERDICT_NEEDS_TUNING = "NEEDS_TUNING"
VERDICT_BLOCKED = "BLOCKED"
VERDICT_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

SUMMARY_HEALTHY = (
    "Route and package metrics are within current policy-band expectations."
)
SUMMARY_INSUFFICIENT_DATA = (
    "Too few Maze episodes to diagnose package health; route gates still apply."
)
SUMMARY_PROMISING_PACKAGE = "needs package visibility"
SUMMARY_PROMISING_ROUTE = "route pressure needs policy-band alignment"
SUMMARY_NEEDS_TUNING = "multiple route/package warnings need coordinated tuning"
SUMMARY_BLOCKED = "hard gate failure blocks promotion until addressed"


@dataclass(frozen=True)
class DiagnosisRuleResult:
    label: str
    cause: str
    experiment: str
    severity: str


@dataclass(frozen=True)
class DesignDiagnosisCard:
    scenario_id: str
    verdict: str
    summary: str
    problem_labels: tuple[str, ...]
    likely_causes: tuple[str, ...]
    suggested_experiments: tuple[str, ...]
    confidence: str = "medium"


@dataclass(frozen=True)
class _DiagnosisRule:
    cause: str
    experiments: tuple[str, ...]
    severity: str


DIAGNOSIS_RULES: dict[str, _DiagnosisRule] = {
    DIAG_PACKAGE_NEVER_SETS_UP: _DiagnosisRule(
        cause=(
            "Maze setup actor or setup skill may not be present, legal, or prioritized."
        ),
        experiments=(
            "Verify generated Maze encounter recipes include a setup-capable enemy.",
            "Check setup skill legality and enemy decision priority.",
        ),
        severity="fail",
    ),
    DIAG_PAYOFF_NEVER_FIRES: _DiagnosisRule(
        cause=(
            "Setup appears, but payoff is not connecting to mark-flow, marked damage, "
            "vulnerable payoff, or forced movement."
        ),
        experiments=(
            "Check whether stalker payoff enemies appear after setup.",
            "Check target legality after marks are applied.",
            (
                "Try giving the payoff enemy safer positioning or earlier initiative "
                "in a controlled candidate."
            ),
        ),
        severity="warn",
    ),
    DIAG_SMART_POLICY_DELETES_PACKAGE: _DiagnosisRule(
        cause="Smart anti-mark play may erase the package too completely.",
        experiments=(
            "Add a secondary pressure axis that is not only mark-based.",
            "Make counterplay reduce danger rather than completely removing it.",
            "Add attrition or formation pressure that still matters when marks are handled.",
        ),
        severity="warn",
    ),
    DIAG_SMART_POLICY_ANSWERS_PACKAGE: _DiagnosisRule(
        cause=(
            "Smart play is suppressing payoff while route completion remains reasonable."
        ),
        experiments=(
            "Treat this as a positive signal unless completion becomes too high.",
            "Watch for future regressions where anti_mark becomes a free win.",
        ),
        severity="info",
    ),
    DIAG_NAIVE_GETS_PUNISHED: _DiagnosisRule(
        cause=(
            "Package is successfully punishing greedy or low-counterplay behavior."
        ),
        experiments=(
            "Treat as healthy if smart policies still take damage/resources.",
            (
                "If naive deaths spike too high, add a softer teaching/probe room "
                "before full package pressure."
            ),
        ),
        severity="warn",
    ),
    DIAG_WARD_ROLE_INVISIBLE: _DiagnosisRule(
        cause="Ward-role enemy is not using ward_pattern often enough.",
        experiments=(
            "Check whether ward-role enemies survive to act.",
            "Check whether ward_pattern is legal and prioritized.",
            (
                "Try placing the ward enemy in a protected slot in one candidate "
                "encounter."
            ),
        ),
        severity="warn",
    ),
    DIAG_TOO_PUNISHING: _DiagnosisRule(
        cause="Route pressure exceeds current policy-band expectations.",
        experiments=(
            "Reduce early encounter pressure.",
            "Move the hardest package room later.",
            "Add a lower-pressure probe room before payoff rooms.",
        ),
        severity="warn",
    ),
    DIAG_TOO_EASY: _DiagnosisRule(
        cause="Route pressure is below current policy-band expectations.",
        experiments=(
            "Increase package frequency slightly.",
            "Add a second pressure axis.",
            "Move payoff earlier only if scout/hunt identity supports it.",
        ),
        severity="warn",
    ),
    DIAG_SMART_POLICY_TRIVIALIZES: _DiagnosisRule(
        cause="Conservative or anti_mark policy may have a dominant answer.",
        experiments=(
            "Add mixed pressure that still respects counterplay.",
            "Ensure smart play wins safer, not free.",
        ),
        severity="warn",
    ),
    DIAG_NAIVE_ONLY_PRESSURE: _DiagnosisRule(
        cause=(
            "Content may only punish bad play while smart policies walk through."
        ),
        experiments=(
            "Add pressure that survives basic counterplay.",
            "Avoid simply increasing raw damage.",
        ),
        severity="warn",
    ),
    DIAG_PACKAGE_PAYOFF_SUPPRESSED: _DiagnosisRule(
        cause=(
            "Anti-mark route completion is high while mark payoff signals stay low."
        ),
        experiments=(
            "Treat this as a watch signal unless completion becomes a free win.",
            "Watch for regressions where anti_mark trivializes the route entirely.",
        ),
        severity="warn",
    ),
}

_GATE_ID_ALIASES: dict[str, str] = {
    "package_never_sets_up": DIAG_PACKAGE_NEVER_SETS_UP,
    "payoff_never_fires": DIAG_PAYOFF_NEVER_FIRES,
    "smart_policy_deletes_package": DIAG_SMART_POLICY_DELETES_PACKAGE,
    "smart_policy_answers_package": DIAG_SMART_POLICY_ANSWERS_PACKAGE,
    "naive_gets_punished": DIAG_NAIVE_GETS_PUNISHED,
    "ward_role_invisible": DIAG_WARD_ROLE_INVISIBLE,
    "too_punishing": DIAG_TOO_PUNISHING,
    "too_easy": DIAG_TOO_EASY,
    "smart_policy_trivializes": DIAG_SMART_POLICY_TRIVIALIZES,
    "naive_only_pressure": DIAG_NAIVE_ONLY_PRESSURE,
    "package_payoff_suppressed": DIAG_PACKAGE_PAYOFF_SUPPRESSED,
}


def build_design_diagnosis(report: PolicyBandReport) -> DesignDiagnosisCard:
    problem_labels = _collect_problem_labels(report)
    info_labels = _collect_info_labels(report)
    verdict = _resolve_verdict(report)
    summary = _resolve_summary(report, verdict)
    confidence = "low" if report.package_insufficient_samples else "medium"
    causes, experiments = _map_rules(problem_labels + info_labels)
    return DesignDiagnosisCard(
        scenario_id=report.scenario_id,
        verdict=verdict,
        summary=summary,
        problem_labels=problem_labels,
        likely_causes=causes,
        suggested_experiments=experiments,
        confidence=confidence,
    )


def format_design_diagnosis(card: DesignDiagnosisCard) -> list[str]:
    lines = [
        "Design diagnosis:",
        f"  Verdict: {card.verdict}",
        f"  Summary: {card.summary}",
    ]
    if card.problem_labels:
        lines.append("  Problems:")
        for label in card.problem_labels:
            lines.append(f"    - {label}")
    if card.likely_causes:
        lines.append("  Likely causes:")
        for cause in card.likely_causes:
            lines.append(f"    - {cause}")
    if card.suggested_experiments:
        lines.append("  Suggested experiments:")
        for index, experiment in enumerate(card.suggested_experiments, start=1):
            lines.append(f"    {index}. {experiment}")
    return lines


def _collect_problem_labels(report: PolicyBandReport) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for gate in report.gates:
        if gate.status not in (GATE_WARN, GATE_FAIL):
            continue
        label = gate.diagnostic_label or gate.gate_id
        if label not in seen:
            labels.append(label)
            seen.add(label)
    for package_gate in report.package_gates:
        if package_gate.diagnostic_only or package_gate.status == GATE_INFO:
            continue
        if package_gate.status not in (GATE_WARN, GATE_FAIL):
            continue
        label = package_gate.diagnostic_label or package_gate.gate_id
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return tuple(labels)


def _collect_info_labels(report: PolicyBandReport) -> tuple[str, ...]:
    labels: list[str] = []
    for gate in report.package_gates:
        if not gate.diagnostic_only and gate.status != GATE_INFO:
            continue
        label = gate.diagnostic_label or gate.gate_id
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _route_has_fail(report: PolicyBandReport) -> bool:
    return any(gate.status == GATE_FAIL for gate in report.gates)


def _route_has_warn(report: PolicyBandReport) -> bool:
    return any(gate.status == GATE_WARN for gate in report.gates)


def _package_has_fail(report: PolicyBandReport) -> bool:
    return any(
        gate.status == GATE_FAIL and not gate.diagnostic_only
        for gate in report.package_gates
    )


def _package_has_warn(report: PolicyBandReport) -> bool:
    return any(
        gate.status == GATE_WARN and not gate.diagnostic_only
        for gate in report.package_gates
    )


def _resolve_verdict(report: PolicyBandReport) -> str:
    if _route_has_fail(report) or _package_has_fail(report):
        return VERDICT_BLOCKED
    if report.package_insufficient_samples:
        return VERDICT_INSUFFICIENT_DATA
    route_warn = _route_has_warn(report)
    package_warn = _package_has_warn(report)
    if not route_warn and not package_warn:
        return VERDICT_HEALTHY
    route_clean = report.overall_status == GATE_PASS
    package_clean = report.package_overall_status == GATE_PASS
    if (route_clean and package_warn) or (package_clean and route_warn):
        return VERDICT_PROMISING
    return VERDICT_NEEDS_TUNING


def _resolve_summary(report: PolicyBandReport, verdict: str) -> str:
    if verdict == VERDICT_HEALTHY:
        return SUMMARY_HEALTHY
    if verdict == VERDICT_INSUFFICIENT_DATA:
        return SUMMARY_INSUFFICIENT_DATA
    if verdict == VERDICT_BLOCKED:
        return SUMMARY_BLOCKED
    if verdict == VERDICT_PROMISING:
        if report.overall_status == GATE_PASS and _package_has_warn(report):
            return SUMMARY_PROMISING_PACKAGE
        if report.package_overall_status == GATE_PASS and _route_has_warn(report):
            return SUMMARY_PROMISING_ROUTE
    if verdict == VERDICT_NEEDS_TUNING:
        return SUMMARY_NEEDS_TUNING
    return SUMMARY_HEALTHY


def _lookup_rule(label: str) -> _DiagnosisRule | None:
    key = _GATE_ID_ALIASES.get(label, label)
    return DIAGNOSIS_RULES.get(key)


def _map_rules(
    labels: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    causes: list[str] = []
    experiments: list[str] = []
    seen_causes: set[str] = set()
    seen_experiments: set[str] = set()
    for label in labels:
        rule = _lookup_rule(label)
        if rule is None:
            continue
        if rule.cause not in seen_causes:
            causes.append(rule.cause)
            seen_causes.add(rule.cause)
        for experiment in rule.experiments:
            if experiment not in seen_experiments:
                experiments.append(experiment)
                seen_experiments.add(experiment)
    return tuple(causes), tuple(experiments)
