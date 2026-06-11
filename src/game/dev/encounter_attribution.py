"""Per-encounter package attribution for generated Maze policy-band reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from game.combat.enemy_learning import EnemyDecisionEpisode
from game.dev.ai_decisions import GATE_WARN
from game.dev.maze_mark_guard_package import (
    GATE_INFO,
    MAZE_PACKAGE_ENCOUNTER_IDS,
    WARD_ROLE_ENCOUNTER_IDS,
    MazeMarkGuardMetrics,
    aggregate_maze_mark_guard_by_encounter,
)

LABEL_RAW_STALKER_WITHOUT_PAYOFF = "raw_stalker_without_payoff"
LABEL_SETUP_WITHOUT_PAYOFF = "setup_without_payoff"
LABEL_STRONG_PACKAGE_PAYOFF = "strong_package_payoff"
LABEL_WARD_ROLE_INVISIBLE = "ward_role_invisible"

_SEVERITY_RANK = {GATE_WARN: 0, GATE_INFO: 1}


@dataclass(frozen=True)
class EncounterPackageAttribution:
    encounter_id: str
    episode_count: int
    setup_episodes: int
    payoff_episodes: int
    ward_actions: int
    stalker_cut_uses: int
    stalker_hook_uses: int
    marks_applied: int
    exploited_by_enemy_hit: int
    vulnerable_payoffs: int
    marked_damage: int
    forced_movement_after_setup: int
    ignored_marked_legal_attacks: int
    setup_without_payoff_episodes: int


@dataclass(frozen=True)
class EncounterAttributionReport:
    policy_id: str
    encounters: tuple[EncounterPackageAttribution, ...]


@dataclass(frozen=True)
class EncounterAttributionFinding:
    encounter_id: str
    label: str
    detail: str
    severity: str


def aggregate_encounter_attribution(
    episodes: Sequence[EnemyDecisionEpisode],
    *,
    policy_id: str,
) -> EncounterAttributionReport:
    rows: list[EncounterPackageAttribution] = []
    for encounter_id in MAZE_PACKAGE_ENCOUNTER_IDS:
        metrics = aggregate_maze_mark_guard_by_encounter(episodes, (encounter_id,))
        if metrics.maze_episodes == 0:
            continue
        rows.append(_metrics_to_attribution(encounter_id, metrics))
    return EncounterAttributionReport(policy_id=policy_id, encounters=tuple(rows))


def evaluate_encounter_attribution(
    report: EncounterAttributionReport,
) -> tuple[EncounterAttributionFinding, ...]:
    findings: list[EncounterAttributionFinding] = []
    for row in report.encounters:
        findings.extend(_findings_for_row(row))
    return tuple(_sort_findings(findings))


def format_encounter_attribution_section(
    reports: Sequence[EncounterAttributionReport],
    findings: Sequence[EncounterAttributionFinding],
) -> list[str]:
    if not reports:
        return []

    lines = ["Encounter attribution:"]
    for report in reports:
        if not report.encounters:
            continue
        lines.append(f"  {report.policy_id}:")
        for row in report.encounters:
            lines.append(
                f"    {row.encounter_id}: episodes={row.episode_count} "
                f"setup={row.setup_episodes} "
                f"payoff={row.payoff_episodes}/{row.setup_episodes} "
                f"ward={row.ward_actions} stalker_cut={row.stalker_cut_uses} "
                f"hook={row.stalker_hook_uses} marked_dmg={row.marked_damage} "
                f"forced={row.forced_movement_after_setup}"
            )

    if findings:
        lines.append("Encounter findings:")
        for finding in findings:
            lines.append(
                f"  {finding.encounter_id}: {finding.label} {finding.severity} - "
                f"{finding.detail}"
            )
    return lines


def _metrics_to_attribution(
    encounter_id: str,
    metrics: MazeMarkGuardMetrics,
) -> EncounterPackageAttribution:
    return EncounterPackageAttribution(
        encounter_id=encounter_id,
        episode_count=metrics.maze_episodes,
        setup_episodes=metrics.episodes_with_setup,
        payoff_episodes=metrics.package_payoff_episodes,
        ward_actions=metrics.ward_actions,
        stalker_cut_uses=metrics.stalker_cut_uses,
        stalker_hook_uses=metrics.stalker_hook_uses,
        marks_applied=metrics.marks_applied,
        exploited_by_enemy_hit=metrics.exploited_by_enemy_hit,
        vulnerable_payoffs=metrics.vulnerable_payoffs,
        marked_damage=metrics.marked_damage,
        forced_movement_after_setup=metrics.forced_movement_after_setup,
        ignored_marked_legal_attacks=metrics.ignored_marked_legal_attacks,
        setup_without_payoff_episodes=metrics.setup_without_payoff_episodes,
    )


def _findings_for_row(
    row: EncounterPackageAttribution,
) -> list[EncounterAttributionFinding]:
    findings: list[EncounterAttributionFinding] = []
    low_sample_suffix = " (low sample)" if row.episode_count == 1 else ""

    if row.setup_episodes > 0 and row.payoff_episodes == 0:
        findings.append(
            EncounterAttributionFinding(
                row.encounter_id,
                LABEL_SETUP_WITHOUT_PAYOFF,
                (
                    f"setup appeared but payoff failed in "
                    f"{row.setup_without_payoff_episodes}/{row.setup_episodes} "
                    f"setup episodes{low_sample_suffix}"
                ),
                GATE_WARN,
            )
        )

    stalker_uses = row.stalker_cut_uses + row.stalker_hook_uses
    if stalker_uses > 0 and row.payoff_episodes == 0:
        findings.append(
            EncounterAttributionFinding(
                row.encounter_id,
                LABEL_RAW_STALKER_WITHOUT_PAYOFF,
                f"stalker skills used without true payoff signals{low_sample_suffix}",
                GATE_WARN,
            )
        )

    if (
        row.encounter_id in WARD_ROLE_ENCOUNTER_IDS
        and row.episode_count > 0
        and row.ward_actions == 0
    ):
        findings.append(
            EncounterAttributionFinding(
                row.encounter_id,
                LABEL_WARD_ROLE_INVISIBLE,
                f"ward-role encounter had no ward_pattern actions{low_sample_suffix}",
                GATE_WARN,
            )
        )

    if row.setup_episodes >= 2 and row.payoff_episodes / row.setup_episodes >= 0.75:
        findings.append(
            EncounterAttributionFinding(
                row.encounter_id,
                LABEL_STRONG_PACKAGE_PAYOFF,
                (
                    "encounter likely carrying package pressure "
                    f"({row.payoff_episodes}/{row.setup_episodes})"
                ),
                GATE_INFO,
            )
        )

    return findings


def _sort_findings(
    findings: Sequence[EncounterAttributionFinding],
) -> list[EncounterAttributionFinding]:
    encounter_order = {encounter_id: index for index, encounter_id in enumerate(
        MAZE_PACKAGE_ENCOUNTER_IDS
    )}

    def sort_key(finding: EncounterAttributionFinding) -> tuple[int, int]:
        return (
            _SEVERITY_RANK.get(finding.severity, 99),
            encounter_order.get(finding.encounter_id, 99),
        )

    return sorted(findings, key=sort_key)
