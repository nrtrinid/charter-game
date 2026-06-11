"""Hero policy action-quality audit for generated-route policy-band reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from game.combat.enemy_learning import EnemyDecisionEpisode, HeroPolicyActionRecord
from game.dev.ai_decisions import GATE_WARN
from game.dev.maze_mark_guard_package import GATE_INFO, PAYOFF_SKILL_IDS

LABEL_EFFORT_LOW_VALUE = "effort_low_value"
LABEL_GENERIC_KILLS_OVER_PAYOFF_PRESSURE = "generic_kills_over_payoff_pressure"
LABEL_HEALTHY_THREAT_REMOVAL = "healthy_threat_removal"
LABEL_HEALTHY_PACKAGE_RESPONSE = "healthy_package_response"
LABEL_IGNORED_KILLABLES = "ignored_killables"
LABEL_LOW_KILL_CONVERSION = "low_kill_conversion"
LABEL_MARK_PRESSURE_PERSISTS = "mark_pressure_persists_despite_better_outcome"
LABEL_MARK_RESPONSE_NOT_WORKING = "mark_response_not_working"
LABEL_PACKAGE_FIXATION = "package_fixation"
LABEL_PAYOFF_EXPOSURE_HIGH = "payoff_exposure_high"
LABEL_SETUP_PRESSURE_UNANSWERED = "setup_pressure_unanswered"

_COMPLETION_MATERIALITY_THRESHOLD = 0.05
_PAYOFF_EXPOSURE_MATERIALITY_RATIO = 1.25
_GENERIC_KILL_WHILE_PAYOFF_ALIVE_MIN = 3
_SETUP_ALIVE_ACTION_RATE_THRESHOLD = 0.35

_DEFAULT_DISPLAY_POLICIES = ("mixed", "tactical", "anti_mark", "company_survival")

_SEVERITY_RANK = {GATE_WARN: 0, GATE_INFO: 1}


@dataclass(frozen=True)
class HeroPolicyAuditMetrics:
    total_hero_actions: int = 0
    offensive_actions: int = 0
    healing_actions: int = 0
    kills: int = 0
    killable_opportunities: int = 0
    killable_opportunities_taken: int = 0
    ignored_killable_opportunities: int = 0
    effort_spent: int = 0
    effort_actions: int = 0
    effort_kill_actions: int = 0
    nonlethal_effort_damage: int = 0
    zero_effort_useful_attacks: int = 0
    package_target_attacks: int = 0
    setup_enemy_attacks: int = 0
    payoff_enemy_attacks: int = 0
    nonlethal_package_attacks: int = 0
    nonlethal_package_low_hp_created: int = 0
    package_followup_kills: int = 0
    marked_hero_turns: int = 0
    enemies_left_low_hp: int = 0

    @property
    def kill_conversion(self) -> float:
        if self.killable_opportunities <= 0:
            return 0.0
        return self.kills / self.killable_opportunities

    @property
    def ignored_killable_rate(self) -> float:
        if self.killable_opportunities <= 0:
            return 0.0
        return self.ignored_killable_opportunities / self.killable_opportunities

    @property
    def effort_kill_rate(self) -> float:
        if self.effort_actions <= 0:
            return 0.0
        return self.effort_kill_actions / self.effort_actions

    @property
    def package_fixation_rate(self) -> float:
        if self.package_target_attacks <= 0:
            return 0.0
        return self.nonlethal_package_attacks / self.package_target_attacks


@dataclass(frozen=True)
class HeroThreatExposureMetrics:
    setup_kills: int = 0
    payoff_kills: int = 0
    generic_kills: int = 0
    setup_attacks: int = 0
    payoff_attacks: int = 0
    generic_attacks: int = 0
    setup_killable_opportunities: int = 0
    payoff_killable_opportunities: int = 0
    generic_killable_opportunities: int = 0
    payoff_enemies_left_alive_after_actions: int = 0
    setup_enemies_left_alive_after_actions: int = 0
    stalker_cut_uses: int = 0
    stalker_hook_uses: int = 0
    enemy_payoff_skill_uses: int = 0
    generic_kill_while_payoff_alive: int = 0
    generic_kill_while_setup_alive: int = 0

    @property
    def enemy_payoff_actions_seen(self) -> int:
        return self.enemy_payoff_skill_uses

    def payoff_actions_per_hero_action(self, total_hero_actions: int) -> float:
        return self.enemy_payoff_skill_uses / max(1, total_hero_actions)

    def payoff_actions_per_episode(self, episode_count: int) -> float:
        return self.enemy_payoff_skill_uses / max(1, episode_count)


@dataclass(frozen=True)
class HeroPolicyAuditReport:
    policy_id: str
    metrics: HeroPolicyAuditMetrics
    episode_count: int
    marked_damage_total: int
    threat_exposure: HeroThreatExposureMetrics = HeroThreatExposureMetrics()

    @property
    def marked_dmg_per_episode(self) -> float:
        return self.marked_damage_total / max(1, self.episode_count)

    @property
    def marked_dmg_per_hero_action(self) -> float:
        return self.marked_damage_total / max(1, self.metrics.total_hero_actions)


@dataclass(frozen=True)
class HeroPolicyAuditFinding:
    policy_id: str
    label: str
    detail: str
    severity: str


def aggregate_hero_policy_audit(
    episodes: Sequence[EnemyDecisionEpisode],
    *,
    policy_id: str,
) -> HeroPolicyAuditReport:
    metrics = HeroPolicyAuditMetrics()
    threat_exposure = HeroThreatExposureMetrics()
    marked_damage_total = 0
    package_followup_kills = 0
    for episode in episodes:
        marked_damage_total += episode.metrics.mark_flow.total_damage_to_marked
        for action in episode.hero_actions:
            metrics = _accumulate_action(metrics, action)
        package_followup_kills += _package_followup_kills_for_episode(episode.hero_actions)
        threat_exposure = _accumulate_episode_threat_exposure(
            threat_exposure,
            episode,
        )

    if package_followup_kills:
        metrics = HeroPolicyAuditMetrics(
            **{
                **metrics.__dict__,
                "package_followup_kills": package_followup_kills,
            }
        )

    return HeroPolicyAuditReport(
        policy_id=policy_id,
        metrics=metrics,
        episode_count=len(episodes),
        marked_damage_total=marked_damage_total,
        threat_exposure=threat_exposure,
    )


def evaluate_hero_policy_audit(
    reports: Sequence[HeroPolicyAuditReport],
    *,
    completion_by_policy: Mapping[str, float] | None = None,
) -> tuple[HeroPolicyAuditFinding, ...]:
    completion_by_policy = completion_by_policy or {}
    mixed_report = _report_for(reports, "mixed")
    mixed_completion = completion_by_policy.get("mixed")
    mixed_marked_damage = mixed_report.marked_damage_total if mixed_report else None
    mixed_marked_dmg_per_action = (
        mixed_report.marked_dmg_per_hero_action
        if mixed_report and mixed_report.metrics.total_hero_actions > 0
        else None
    )
    mixed_marked_dmg_per_episode = (
        mixed_report.marked_dmg_per_episode if mixed_report else None
    )
    mixed_kill_conversion = (
        mixed_report.metrics.kill_conversion if mixed_report else None
    )

    findings: list[HeroPolicyAuditFinding] = []
    for report in reports:
        policy_completion = completion_by_policy.get(report.policy_id)
        findings.extend(
            _findings_for_report(
                report,
                policy_completion=policy_completion,
                mixed_completion=mixed_completion,
                mixed_marked_damage=mixed_marked_damage,
                mixed_marked_dmg_per_action=mixed_marked_dmg_per_action,
                mixed_marked_dmg_per_episode=mixed_marked_dmg_per_episode,
                mixed_kill_conversion=mixed_kill_conversion,
                mixed_report=mixed_report,
            )
        )
    return tuple(_sort_findings(findings))


def format_hero_policy_audit_section(
    reports: Sequence[HeroPolicyAuditReport],
    findings: Sequence[HeroPolicyAuditFinding],
) -> list[str]:
    if not reports:
        return []

    display_ids = _display_policy_ids(reports, findings)
    report_by_id = {report.policy_id: report for report in reports}

    lines = ["Hero policy audit:"]
    for policy_id in display_ids:
        report = report_by_id.get(policy_id)
        if report is None:
            continue
        metrics = report.metrics
        followup_suffix = ""
        if metrics.package_followup_kills:
            followup_suffix = f" pkg_followup={metrics.package_followup_kills}"
        lines.append(
            f"  {policy_id}: actions={metrics.total_hero_actions} "
            f"kills={metrics.kills} ignored_kill="
            f"{metrics.ignored_killable_opportunities}/{metrics.killable_opportunities} "
            f"effort_actions={metrics.effort_actions} "
            f"effort_kill_actions={metrics.effort_kill_actions} "
            f"pkg_nl={metrics.nonlethal_package_attacks} "
            f"marked_dmg={report.marked_damage_total} "
            f"marked_dmg/ep={report.marked_dmg_per_episode:.1f} "
            f"marked_dmg/act={report.marked_dmg_per_hero_action:.2f} "
            f"pkg_nl_low_hp={metrics.nonlethal_package_low_hp_created}{followup_suffix}"
        )

    threat_lines = _format_threat_exposure_section(reports, display_ids, report_by_id)
    if threat_lines:
        lines.extend(threat_lines)

    if findings:
        lines.append("Audit findings:")
        for finding in findings:
            lines.append(
                f"  {finding.policy_id}: {finding.label} {finding.severity} - "
                f"{finding.detail}"
            )
    return lines


def _format_threat_exposure_section(
    reports: Sequence[HeroPolicyAuditReport],
    display_ids: Sequence[str],
    report_by_id: Mapping[str, HeroPolicyAuditReport],
) -> list[str]:
    del reports
    lines: list[str] = []
    for policy_id in display_ids:
        report = report_by_id.get(policy_id)
        if report is None:
            continue
        if not lines:
            lines.append("Threat exposure:")
        exposure = report.threat_exposure
        lines.append(
            "  {policy}: setup_kills={setup_kills} payoff_kills={payoff_kills} "
            "generic_kills={generic_kills} setup_attacks={setup_attacks} "
            "payoff_attacks={payoff_attacks} generic_attacks={generic_attacks} "
            "payoff_actions/act={payoff_act:.2f} payoff_actions/ep={payoff_ep:.1f} "
            "stalker_cut={stalker_cut} stalker_hook={stalker_hook} "
            "generic_kill_while_payoff_alive={generic_while_payoff} "
            "generic_kill_while_setup_alive={generic_while_setup}".format(
                policy=policy_id,
                setup_kills=exposure.setup_kills,
                payoff_kills=exposure.payoff_kills,
                generic_kills=exposure.generic_kills,
                setup_attacks=exposure.setup_attacks,
                payoff_attacks=exposure.payoff_attacks,
                generic_attacks=exposure.generic_attacks,
                payoff_act=exposure.payoff_actions_per_hero_action(
                    report.metrics.total_hero_actions
                ),
                payoff_ep=exposure.payoff_actions_per_episode(report.episode_count),
                stalker_cut=exposure.stalker_cut_uses,
                stalker_hook=exposure.stalker_hook_uses,
                generic_while_payoff=exposure.generic_kill_while_payoff_alive,
                generic_while_setup=exposure.generic_kill_while_setup_alive,
            )
        )
    return lines


def _accumulate_action(
    metrics: HeroPolicyAuditMetrics,
    action: HeroPolicyActionRecord,
) -> HeroPolicyAuditMetrics:
    total_hero_actions = metrics.total_hero_actions + 1
    offensive_actions = metrics.offensive_actions
    healing_actions = metrics.healing_actions
    if action.is_heal:
        healing_actions += 1
    elif action.estimated_damage > 0:
        offensive_actions += 1

    kills = metrics.kills + int(action.produced_kill)
    killable_opportunities = metrics.killable_opportunities + action.killable_opportunities
    killable_opportunities_taken = metrics.killable_opportunities_taken + int(
        action.killable
    )
    ignored_killable_opportunities = metrics.ignored_killable_opportunities + int(
        action.ignored_killable_opportunity
    )
    effort_spent = metrics.effort_spent + action.effort_cost
    effort_actions = metrics.effort_actions + int(action.effort_cost > 0)
    effort_kill_actions = metrics.effort_kill_actions + int(
        action.effort_cost > 0 and action.produced_kill
    )
    nonlethal_effort_damage = metrics.nonlethal_effort_damage
    if (
        action.effort_cost > 0
        and not action.killable
        and not action.is_heal
        and action.estimated_damage > 0
    ):
        nonlethal_effort_damage += 1

    zero_effort_useful_attacks = metrics.zero_effort_useful_attacks
    if action.effort_cost == 0 and not action.is_heal and action.estimated_damage > 0:
        zero_effort_useful_attacks += 1

    package_target_attacks = metrics.package_target_attacks
    setup_enemy_attacks = metrics.setup_enemy_attacks
    payoff_enemy_attacks = metrics.payoff_enemy_attacks
    nonlethal_package_attacks = metrics.nonlethal_package_attacks
    nonlethal_package_low_hp_created = metrics.nonlethal_package_low_hp_created
    if action.package_target:
        package_target_attacks += 1
        if action.package_target == "setup":
            setup_enemy_attacks += 1
        elif action.package_target == "payoff":
            payoff_enemy_attacks += 1
        if (
            not action.killable
            and not action.is_heal
            and action.estimated_damage > 0
        ):
            nonlethal_package_attacks += 1
            if action.target_hp_remaining in (1, 2):
                nonlethal_package_low_hp_created += 1

    marked_hero_turns = metrics.marked_hero_turns + int(action.marked_hero_present)
    enemies_left_low_hp = metrics.enemies_left_low_hp
    if (
        not action.is_heal
        and action.estimated_damage > 0
        and action.target_hp_remaining in (1, 2)
    ):
        enemies_left_low_hp += 1

    return HeroPolicyAuditMetrics(
        total_hero_actions=total_hero_actions,
        offensive_actions=offensive_actions,
        healing_actions=healing_actions,
        kills=kills,
        killable_opportunities=killable_opportunities,
        killable_opportunities_taken=killable_opportunities_taken,
        ignored_killable_opportunities=ignored_killable_opportunities,
        effort_spent=effort_spent,
        effort_actions=effort_actions,
        effort_kill_actions=effort_kill_actions,
        nonlethal_effort_damage=nonlethal_effort_damage,
        zero_effort_useful_attacks=zero_effort_useful_attacks,
        package_target_attacks=package_target_attacks,
        setup_enemy_attacks=setup_enemy_attacks,
        payoff_enemy_attacks=payoff_enemy_attacks,
        nonlethal_package_attacks=nonlethal_package_attacks,
        nonlethal_package_low_hp_created=nonlethal_package_low_hp_created,
        package_followup_kills=metrics.package_followup_kills,
        marked_hero_turns=marked_hero_turns,
        enemies_left_low_hp=enemies_left_low_hp,
    )


def _package_followup_kills_for_episode(
    actions: Sequence[HeroPolicyActionRecord],
) -> int:
    primed: set[str] = set()
    followup_kills = 0
    for action in actions:
        if (
            action.package_target
            and not action.killable
            and not action.is_heal
            and action.estimated_damage > 0
            and action.target_hp_remaining in (1, 2)
        ):
            primed.add(action.target_id)
        if action.produced_kill and action.target_id in primed:
            followup_kills += 1
            primed.discard(action.target_id)
    return followup_kills


def _mark_damage_finding_detail(
    report: HeroPolicyAuditReport,
    *,
    mixed_report: HeroPolicyAuditReport | None,
) -> tuple[str, str, float | None, float | None]:
    policy_value: float
    mixed_value: float | None = None
    if report.metrics.total_hero_actions > 0 and mixed_report is not None:
        if mixed_report.metrics.total_hero_actions > 0:
            policy_value = report.marked_dmg_per_hero_action
            mixed_value = mixed_report.marked_dmg_per_hero_action
            return (
                f"marked_dmg/act={policy_value:.2f} vs mixed {mixed_value:.2f}",
                "action",
                policy_value,
                mixed_value,
            )
    if report.episode_count > 0 and mixed_report is not None:
        policy_value = report.marked_dmg_per_episode
        mixed_value = mixed_report.marked_dmg_per_episode
        return (
            f"marked_dmg/ep={policy_value:.1f} vs mixed {mixed_value:.1f}",
            "episode",
            policy_value,
            mixed_value,
        )
    return (
        f"marked_damage={report.marked_damage_total}",
        "total",
        float(report.marked_damage_total),
        float(mixed_report.marked_damage_total) if mixed_report else None,
    )


def _findings_for_report(
    report: HeroPolicyAuditReport,
    *,
    policy_completion: float | None,
    mixed_completion: float | None,
    mixed_marked_damage: int | None,
    mixed_marked_dmg_per_action: float | None,
    mixed_marked_dmg_per_episode: float | None,
    mixed_kill_conversion: float | None,
    mixed_report: HeroPolicyAuditReport | None = None,
) -> list[HeroPolicyAuditFinding]:
    findings: list[HeroPolicyAuditFinding] = []
    metrics = report.metrics
    policy_id = report.policy_id

    if metrics.ignored_killable_opportunities >= 3 and metrics.ignored_killable_rate >= 0.25:
        severity = GATE_WARN if metrics.ignored_killable_rate >= 0.40 else GATE_INFO
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_IGNORED_KILLABLES,
                detail=(
                    f"ignored {metrics.ignored_killable_opportunities}/"
                    f"{metrics.killable_opportunities} killable opportunities "
                    f"({metrics.ignored_killable_rate:.0%})"
                ),
                severity=severity,
            )
        )

    if (
        policy_id != "mixed"
        and mixed_kill_conversion is not None
        and mixed_kill_conversion > 0
        and metrics.killable_opportunities >= 5
        and metrics.kill_conversion < mixed_kill_conversion * 0.70
    ):
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_LOW_KILL_CONVERSION,
                detail=(
                    f"kill conversion {metrics.kill_conversion:.0%} vs mixed "
                    f"{mixed_kill_conversion:.0%}"
                ),
                severity=GATE_WARN,
            )
        )

    if (
        metrics.effort_actions >= 3
        and metrics.effort_kill_rate < 0.35
        and metrics.nonlethal_effort_damage >= 2
    ):
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_EFFORT_LOW_VALUE,
                detail=(
                    f"effort_kill_actions={metrics.effort_kill_actions}/"
                    f"{metrics.effort_actions}, "
                    f"nonlethal_effort_damage={metrics.nonlethal_effort_damage}"
                ),
                severity=GATE_WARN,
            )
        )

    if (
        policy_id != "mixed"
        and mixed_completion is not None
        and policy_completion is not None
        and metrics.nonlethal_package_attacks >= 3
        and metrics.package_fixation_rate >= 0.30
        and policy_completion < mixed_completion
    ):
        fixation_detail = (
            f"nonlethal_package_attacks={metrics.nonlethal_package_attacks}/"
            f"{metrics.package_target_attacks}, "
            f"pkg_nl_low_hp={metrics.nonlethal_package_low_hp_created}/"
            f"{metrics.nonlethal_package_attacks}, "
            f"completion {policy_completion:.0%} vs mixed {mixed_completion:.0%}"
        )
        low_hp_rate = (
            metrics.nonlethal_package_low_hp_created
            / max(1, metrics.nonlethal_package_attacks)
        )
        if low_hp_rate < 0.15:
            fixation_detail += ", low setup value"
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_PACKAGE_FIXATION,
                detail=fixation_detail,
                severity=GATE_WARN,
            )
        )

    findings.extend(
        _mark_outcome_findings(
            report,
            policy_id=policy_id,
            metrics=metrics,
            policy_completion=policy_completion,
            mixed_completion=mixed_completion,
            mixed_marked_damage=mixed_marked_damage,
            mixed_report=mixed_report,
        )
    )

    findings.extend(
        _threat_exposure_findings(
            report,
            policy_id=policy_id,
            policy_completion=policy_completion,
            mixed_completion=mixed_completion,
            mixed_report=mixed_report,
        )
    )

    return findings


def _accumulate_episode_threat_exposure(
    exposure: HeroThreatExposureMetrics,
    episode: EnemyDecisionEpisode,
) -> HeroThreatExposureMetrics:
    payoff_targets: set[str] = set()
    setup_targets: set[str] = set()
    killed_targets: set[str] = set()

    setup_kills = exposure.setup_kills
    payoff_kills = exposure.payoff_kills
    generic_kills = exposure.generic_kills
    setup_attacks = exposure.setup_attacks
    payoff_attacks = exposure.payoff_attacks
    generic_attacks = exposure.generic_attacks
    setup_killable_opportunities = exposure.setup_killable_opportunities
    payoff_killable_opportunities = exposure.payoff_killable_opportunities
    generic_killable_opportunities = exposure.generic_killable_opportunities
    payoff_enemies_left_alive_after_actions = exposure.payoff_enemies_left_alive_after_actions
    setup_enemies_left_alive_after_actions = exposure.setup_enemies_left_alive_after_actions
    generic_kill_while_payoff_alive = exposure.generic_kill_while_payoff_alive
    generic_kill_while_setup_alive = exposure.generic_kill_while_setup_alive

    for action in episode.hero_actions:
        if not action.is_heal and action.estimated_damage > 0:
            if action.package_target == "setup":
                setup_targets.add(action.target_id)
                setup_attacks += 1
            elif action.package_target == "payoff":
                payoff_targets.add(action.target_id)
                payoff_attacks += 1
            else:
                generic_attacks += 1

        if action.killable_opportunities > 0:
            if action.package_target == "setup":
                setup_killable_opportunities += action.killable_opportunities
            elif action.package_target == "payoff":
                payoff_killable_opportunities += action.killable_opportunities
            else:
                generic_killable_opportunities += action.killable_opportunities

        if action.produced_kill:
            if action.package_target == "payoff":
                payoff_kills += 1
            elif action.package_target == "setup":
                setup_kills += 1
            else:
                generic_kills += 1
                if payoff_targets - killed_targets:
                    generic_kill_while_payoff_alive += 1
                if setup_targets - killed_targets:
                    generic_kill_while_setup_alive += 1
            killed_targets.add(action.target_id)

        payoff_alive = len(payoff_targets - killed_targets)
        setup_alive = len(setup_targets - killed_targets)
        if payoff_alive > 0:
            payoff_enemies_left_alive_after_actions += payoff_alive
        if setup_alive > 0:
            setup_enemies_left_alive_after_actions += setup_alive

    stalker_cut_uses = exposure.stalker_cut_uses
    stalker_hook_uses = exposure.stalker_hook_uses
    enemy_payoff_skill_uses = exposure.enemy_payoff_skill_uses
    for record in episode.records:
        skill_id = record.chosen_skill_id
        if skill_id == "stalker_cut":
            stalker_cut_uses += 1
        if skill_id == "stalker_hook":
            stalker_hook_uses += 1
        if skill_id in PAYOFF_SKILL_IDS:
            enemy_payoff_skill_uses += 1

    return HeroThreatExposureMetrics(
        setup_kills=setup_kills,
        payoff_kills=payoff_kills,
        generic_kills=generic_kills,
        setup_attacks=setup_attacks,
        payoff_attacks=payoff_attacks,
        generic_attacks=generic_attacks,
        setup_killable_opportunities=setup_killable_opportunities,
        payoff_killable_opportunities=payoff_killable_opportunities,
        generic_killable_opportunities=generic_killable_opportunities,
        payoff_enemies_left_alive_after_actions=payoff_enemies_left_alive_after_actions,
        setup_enemies_left_alive_after_actions=setup_enemies_left_alive_after_actions,
        stalker_cut_uses=stalker_cut_uses,
        stalker_hook_uses=stalker_hook_uses,
        enemy_payoff_skill_uses=enemy_payoff_skill_uses,
        generic_kill_while_payoff_alive=generic_kill_while_payoff_alive,
        generic_kill_while_setup_alive=generic_kill_while_setup_alive,
    )


def _threat_exposure_findings(
    report: HeroPolicyAuditReport,
    *,
    policy_id: str,
    policy_completion: float | None,
    mixed_completion: float | None,
    mixed_report: HeroPolicyAuditReport | None,
) -> list[HeroPolicyAuditFinding]:
    if policy_id == "mixed" or mixed_report is None:
        return []

    exposure = report.threat_exposure
    mixed_exposure = mixed_report.threat_exposure
    total_actions = report.metrics.total_hero_actions
    if total_actions <= 0 or mixed_report.metrics.total_hero_actions <= 0:
        return []

    policy_payoff_rate = exposure.payoff_actions_per_hero_action(total_actions)
    mixed_payoff_rate = mixed_exposure.payoff_actions_per_hero_action(
        mixed_report.metrics.total_hero_actions
    )
    if mixed_payoff_rate <= 0:
        return []

    findings: list[HeroPolicyAuditFinding] = []

    if policy_payoff_rate >= mixed_payoff_rate * _PAYOFF_EXPOSURE_MATERIALITY_RATIO:
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_PAYOFF_EXPOSURE_HIGH,
                detail=(
                    f"payoff_actions/act={policy_payoff_rate:.2f} vs mixed "
                    f"{mixed_payoff_rate:.2f}, stalker_cut={exposure.stalker_cut_uses}"
                ),
                severity=GATE_WARN,
            )
        )

    if (
        exposure.generic_kill_while_payoff_alive >= _GENERIC_KILL_WHILE_PAYOFF_ALIVE_MIN
        and policy_payoff_rate >= mixed_payoff_rate
        and exposure.generic_kills > exposure.payoff_kills
    ):
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_GENERIC_KILLS_OVER_PAYOFF_PRESSURE,
                detail=(
                    f"generic_kills={exposure.generic_kills} payoff_kills="
                    f"{exposure.payoff_kills}, generic_kill_while_payoff_alive="
                    f"{exposure.generic_kill_while_payoff_alive}, "
                    f"payoff_actions/act={policy_payoff_rate:.2f}"
                ),
                severity=GATE_WARN,
            )
        )

    setup_alive_rate = exposure.setup_enemies_left_alive_after_actions / total_actions
    if (
        setup_alive_rate >= _SETUP_ALIVE_ACTION_RATE_THRESHOLD
        and policy_payoff_rate >= mixed_payoff_rate
        and report.marked_dmg_per_hero_action >= mixed_report.marked_dmg_per_hero_action
    ):
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_SETUP_PRESSURE_UNANSWERED,
                detail=(
                    f"setup_alive_after={exposure.setup_enemies_left_alive_after_actions}/"
                    f"{total_actions} actions, marked_dmg/act="
                    f"{report.marked_dmg_per_hero_action:.2f} vs mixed "
                    f"{mixed_report.marked_dmg_per_hero_action:.2f}"
                ),
                severity=GATE_WARN,
            )
        )

    if (
        policy_completion is not None
        and mixed_completion is not None
        and policy_payoff_rate <= mixed_payoff_rate * 1.10
        and policy_completion >= mixed_completion - _COMPLETION_MATERIALITY_THRESHOLD
        and exposure.payoff_kills + exposure.setup_kills >= 1
    ):
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_HEALTHY_THREAT_REMOVAL,
                detail=(
                    f"payoff_actions/act={policy_payoff_rate:.2f} vs mixed "
                    f"{mixed_payoff_rate:.2f}, completion {policy_completion:.0%} vs "
                    f"mixed {mixed_completion:.0%}"
                ),
                severity=GATE_INFO,
            )
        )

    return findings


def _mark_damage_better_than_mixed(
    policy_mark_value: float,
    mixed_mark_value: float,
    mark_mode: str,
    *,
    report: HeroPolicyAuditReport,
    mixed_marked_damage: int | None,
    threshold: float = 0.85,
) -> bool:
    if mark_mode == "action" or mark_mode == "episode":
        return policy_mark_value < mixed_mark_value * threshold
    if (
        mark_mode == "total"
        and mixed_marked_damage is not None
        and mixed_marked_damage > 0
    ):
        return report.marked_damage_total < mixed_marked_damage * threshold
    return False


def _mark_damage_worse_than_mixed(
    policy_mark_value: float,
    mixed_mark_value: float,
    mark_mode: str,
    *,
    report: HeroPolicyAuditReport,
    mixed_marked_damage: int | None,
) -> bool:
    if mark_mode == "action" or mark_mode == "episode":
        return policy_mark_value >= mixed_mark_value
    if mark_mode == "total" and mixed_marked_damage is not None:
        return report.marked_damage_total >= mixed_marked_damage
    return False


def _mark_outcome_findings(
    report: HeroPolicyAuditReport,
    *,
    policy_id: str,
    metrics: HeroPolicyAuditMetrics,
    policy_completion: float | None,
    mixed_completion: float | None,
    mixed_marked_damage: int | None,
    mixed_report: HeroPolicyAuditReport | None,
) -> list[HeroPolicyAuditFinding]:
    if policy_id == "mixed" or metrics.package_target_attacks < 2:
        return []

    mark_detail, mark_mode, policy_mark_value, mixed_mark_value = _mark_damage_finding_detail(
        report,
        mixed_report=mixed_report,
    )
    if mixed_mark_value is None or policy_mark_value is None:
        return []

    findings: list[HeroPolicyAuditFinding] = []

    if _mark_damage_better_than_mixed(
        policy_mark_value,
        mixed_mark_value,
        mark_mode,
        report=report,
        mixed_marked_damage=mixed_marked_damage,
    ) and (
        policy_completion is None
        or mixed_completion is None
        or policy_completion >= mixed_completion
    ):
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_HEALTHY_PACKAGE_RESPONSE,
                detail=mark_detail,
                severity=GATE_INFO,
            )
        )
        return findings

    if not _mark_damage_worse_than_mixed(
        policy_mark_value,
        mixed_mark_value,
        mark_mode,
        report=report,
        mixed_marked_damage=mixed_marked_damage,
    ):
        return findings

    if mixed_completion is None or policy_completion is None:
        detail = f"{mark_detail} (outcome context unavailable)"
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_MARK_RESPONSE_NOT_WORKING,
                detail=detail,
                severity=GATE_WARN,
            )
        )
        return findings

    if policy_completion >= mixed_completion + _COMPLETION_MATERIALITY_THRESHOLD:
        findings.append(
            HeroPolicyAuditFinding(
                policy_id=policy_id,
                label=LABEL_MARK_PRESSURE_PERSISTS,
                detail=(
                    f"{mark_detail}, completion {policy_completion:.0%} vs "
                    f"mixed {mixed_completion:.0%}"
                ),
                severity=GATE_INFO,
            )
        )
        return findings

    findings.append(
        HeroPolicyAuditFinding(
            policy_id=policy_id,
            label=LABEL_MARK_RESPONSE_NOT_WORKING,
            detail=mark_detail,
            severity=GATE_WARN,
        )
    )
    return findings


def _report_for(
    reports: Sequence[HeroPolicyAuditReport],
    policy_id: str,
) -> HeroPolicyAuditReport | None:
    for report in reports:
        if report.policy_id == policy_id:
            return report
    return None


def _display_policy_ids(
    reports: Sequence[HeroPolicyAuditReport],
    findings: Sequence[HeroPolicyAuditFinding],
) -> list[str]:
    report_ids = {report.policy_id for report in reports}
    finding_ids = {finding.policy_id for finding in findings}
    display_ids: list[str] = []
    for policy_id in _DEFAULT_DISPLAY_POLICIES:
        if policy_id in report_ids:
            display_ids.append(policy_id)
    for report in reports:
        if report.policy_id not in display_ids and report.policy_id in finding_ids:
            display_ids.append(report.policy_id)
    return display_ids


def _sort_findings(
    findings: Sequence[HeroPolicyAuditFinding],
) -> list[HeroPolicyAuditFinding]:
    policy_order = {
        policy_id: index for index, policy_id in enumerate(_DEFAULT_DISPLAY_POLICIES)
    }

    def sort_key(finding: HeroPolicyAuditFinding) -> tuple[int, int, str]:
        return (
            _SEVERITY_RANK.get(finding.severity, 99),
            policy_order.get(finding.policy_id, 99),
            finding.label,
        )

    return sorted(findings, key=sort_key)
