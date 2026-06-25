"""Dev-only score-only heuristic oracle and counterplay metrics for the AI lab."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    choose_heuristic_enemy_candidate,
    score_heuristic_enemy_features,
)
from game.combat.enemy_learning import EnemyDecisionRecord
from game.core.events import DamageEvent, DownedEvent
from game.dev.ai_packages import DEFAULT_PACKAGE_CONTRACTS
from game.dev.maze_mark_guard_package import SETUP_SKILL_IDS, WARD_SKILL_IDS
from game.dev.route_lab import RouteEnvelopeScore, pre_boss_failure_count, score_route_envelope
from game.dev.train_enemy_ai import TrainingRunSummary

ORACLE_MODE_LABEL = "score-only heuristic (depth-0)"

FINDING_PASS = "PASS"
FINDING_WARN = "WARN"
FINDING_FAIL = "FAIL"

ORACLE_MISS_RATE_WARN = 0.15
DOMINANT_SKILL_RATE_WARN = 0.35
DOWNED_DAMAGE_RATE_WARN = 0.05
EARLY_DOWNED_ROUND_WARN = 3.0
MORTAL_WOUND_PER_RUN_WARN = 0.5

_SETUP_ACTION_TAGS = frozenset(
    {
        "guard",
        "mark_target",
        "drag_forward",
        "setup",
        "formation",
    }
)


@dataclass(frozen=True)
class OracleReportConfig:
    miss_threshold: int = 1
    max_example_misses: int = 3


@dataclass(frozen=True)
class OracleDecisionRecord:
    encounter_id: str
    package_id: str
    round_number: int
    actor_id: str
    chosen_skill_id: str
    chosen_target_id: str
    oracle_skill_id: str
    oracle_target_id: str
    chosen_score: int
    oracle_score: int
    delta: int
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterplayMetrics:
    total_enemy_actions: int = 0
    oracle_miss_count: int = 0
    oracle_miss_rate: float = 0.0
    average_miss_delta: float = 0.0
    largest_miss: tuple[str, str, str, int] | None = None
    dominant_skill_id: str = ""
    dominant_skill_rate: float = 0.0
    same_skill_spam_rate: float = 0.0
    damage_to_downed_count: int | None = None
    damage_to_downed_rate: float | None = None
    downed_created_count: int = 0
    mortal_wounds_created_count: int = 0
    deaths_created_count: int = 0
    first_downed_round_avg: float | None = None
    noncompletion_count: int | None = None
    noncompletion_rate: float | None = None
    pre_boss_failure_count: int | None = None
    boss_entry_hp: float | None = None
    final_party_hp: float | None = None
    final_effort: float | None = None
    average_rounds: float = 0.0
    setup_or_special_action_count: int = 0
    run_count: int = 1


@dataclass(frozen=True)
class CounterplayFinding:
    finding_id: str
    status: str
    detail: str


@dataclass(frozen=True)
class OracleReport:
    route_id: str
    seed_count: int
    hero_policy_id: str
    preset_id: str
    oracle_mode: str = ORACLE_MODE_LABEL
    miss_records: tuple[OracleDecisionRecord, ...] = ()
    metrics: CounterplayMetrics = field(default_factory=CounterplayMetrics)
    findings: tuple[CounterplayFinding, ...] = ()
    envelope_score: RouteEnvelopeScore | None = None


def analyze_oracle_record(
    record: EnemyDecisionRecord,
    *,
    encounter_id: str,
    package_id: str,
    miss_threshold: int,
) -> OracleDecisionRecord | None:
    candidates = record.trace.candidates
    if not candidates:
        return None

    oracle = choose_heuristic_enemy_candidate(candidates)
    if oracle is None:
        return None

    chosen = _matching_candidate(candidates, record.chosen_skill_id, record.chosen_target_id)
    chosen_score = chosen.score if chosen is not None else score_heuristic_enemy_features(
        record.chosen_features
    )
    oracle_score = oracle.score
    delta = oracle_score - chosen_score

    same_action = (
        record.chosen_skill_id == oracle.skill_id
        and record.chosen_target_id == oracle.target_id
    )
    if same_action or delta < miss_threshold:
        return None

    reason_codes: list[str] = []
    if record.chosen_skill_id != oracle.skill_id:
        reason_codes.append("skill_mismatch")
    if record.chosen_target_id != oracle.target_id:
        reason_codes.append("target_mismatch")

    return OracleDecisionRecord(
        encounter_id=encounter_id,
        package_id=package_id,
        round_number=record.round_number,
        actor_id=record.enemy_id,
        chosen_skill_id=record.chosen_skill_id,
        chosen_target_id=record.chosen_target_id,
        oracle_skill_id=oracle.skill_id,
        oracle_target_id=oracle.target_id,
        chosen_score=chosen_score,
        oracle_score=oracle_score,
        delta=delta,
        reason_codes=tuple(reason_codes),
    )


def build_oracle_report(
    summary: TrainingRunSummary,
    config: OracleReportConfig | None = None,
) -> OracleReport:
    config = config or OracleReportConfig()
    miss_records: list[OracleDecisionRecord] = []
    total_actions = 0

    for episode in summary.learned_episodes:
        for record in episode.records:
            total_actions += 1
            package_id = _package_id_for_enemy(record.enemy_class_id)
            miss = analyze_oracle_record(
                record,
                encounter_id=episode.encounter_id,
                package_id=package_id,
                miss_threshold=config.miss_threshold,
            )
            if miss is not None:
                miss_records.append(miss)

    metrics = aggregate_counterplay_metrics(summary, miss_records=tuple(miss_records))
    envelope_score = _route_envelope_score(summary)
    findings = evaluate_counterplay_findings(metrics, envelope_score)

    return OracleReport(
        route_id=summary.route_id,
        seed_count=summary.seed_count,
        hero_policy_id=summary.hero_policy_id,
        preset_id=summary.preset_id,
        miss_records=tuple(
            sorted(
                miss_records,
                key=lambda item: (-item.delta, item.actor_id, item.round_number),
            )
        ),
        metrics=metrics,
        findings=findings,
        envelope_score=envelope_score,
    )


def aggregate_counterplay_metrics(
    summary: TrainingRunSummary,
    *,
    miss_records: tuple[OracleDecisionRecord, ...] = (),
) -> CounterplayMetrics:
    episodes = summary.learned_episodes
    total_actions = sum(len(episode.records) for episode in episodes)
    miss_count = len(miss_records)
    miss_rate = miss_count / total_actions if total_actions else 0.0
    average_miss_delta = (
        sum(record.delta for record in miss_records) / miss_count if miss_count else 0.0
    )
    largest_miss = _largest_miss_tuple(miss_records)

    skill_counter: Counter[str] = Counter()
    setup_count = 0
    consecutive_same = 0
    skill_transitions = 0
    downed_created = 0
    mortal_wounds = 0
    deaths = 0
    rounds_total = 0
    first_downed_rounds: list[int] = []
    damage_to_downed_count = 0
    damage_events_with_hp_before = 0

    episode_count = len(episodes)
    run_count = summary.seed_count if summary.route_id else max(1, episode_count)

    for episode in episodes:
        rounds_total += episode.metrics.rounds_elapsed
        downed_created += episode.metrics.hero_downs
        mortal_wounds += episode.metrics.mortal_wounds
        deaths += episode.metrics.hero_deaths

        episode_first_downed: int | None = None
        previous_skill = ""
        for record in episode.records:
            skill_counter[record.chosen_skill_id] += 1
            if _is_setup_or_special_action(record):
                setup_count += 1
            if previous_skill and record.chosen_skill_id == previous_skill:
                consecutive_same += 1
            if previous_skill:
                skill_transitions += 1
            previous_skill = record.chosen_skill_id

            for event in record.events:
                if isinstance(event, DownedEvent):
                    if episode_first_downed is None:
                        episode_first_downed = record.round_number
                if isinstance(event, DamageEvent):
                    if event.hp_before is not None:
                        damage_events_with_hp_before += 1
                        if event.hp_before == 0 and event.amount > 0:
                            damage_to_downed_count += 1

        if episode_first_downed is not None:
            first_downed_rounds.append(episode_first_downed)

    dominant_skill_id, dominant_skill_rate = _dominant_skill(skill_counter, total_actions)
    same_skill_spam_rate = (
        consecutive_same / skill_transitions if skill_transitions else 0.0
    )
    average_rounds = rounds_total / len(episodes) if episodes else 0.0
    first_downed_round_avg = (
        sum(first_downed_rounds) / len(first_downed_rounds)
        if first_downed_rounds
        else None
    )

    damage_to_downed: int | None
    damage_to_downed_rate: float | None
    if damage_events_with_hp_before == 0:
        damage_to_downed = None
        damage_to_downed_rate = None
    else:
        damage_to_downed = damage_to_downed_count
        damage_to_downed_rate = (
            damage_to_downed_count / total_actions if total_actions else 0.0
        )

    noncompletion_count: int | None = None
    noncompletion_rate: float | None = None
    pre_boss_failure_count_value: int | None = None
    boss_entry_hp: float | None = None
    final_party_hp: float | None = None
    final_effort: float | None = None
    if summary.route_id:
        route_summary = summary.learned_route_summary
        routes = route_summary.route_count
        completed = route_summary.completed_count
        noncompletion_count = routes - completed
        noncompletion_rate = noncompletion_count / routes if routes else 0.0
        pre_boss_failure_count_value = pre_boss_failure_count(route_summary.failed_at_counts)
        boss_entry_hp = route_summary.average_hp_entering_cave_mini_boss
        final_party_hp = route_summary.average_final_hero_hp
        final_effort = route_summary.average_final_hero_effort

    return CounterplayMetrics(
        total_enemy_actions=total_actions,
        oracle_miss_count=miss_count,
        oracle_miss_rate=miss_rate,
        average_miss_delta=average_miss_delta,
        largest_miss=largest_miss,
        dominant_skill_id=dominant_skill_id,
        dominant_skill_rate=dominant_skill_rate,
        same_skill_spam_rate=same_skill_spam_rate,
        damage_to_downed_count=damage_to_downed,
        damage_to_downed_rate=damage_to_downed_rate,
        downed_created_count=downed_created,
        mortal_wounds_created_count=mortal_wounds,
        deaths_created_count=deaths,
        first_downed_round_avg=first_downed_round_avg,
        noncompletion_count=noncompletion_count,
        noncompletion_rate=noncompletion_rate,
        pre_boss_failure_count=pre_boss_failure_count_value,
        boss_entry_hp=boss_entry_hp,
        final_party_hp=final_party_hp,
        final_effort=final_effort,
        average_rounds=average_rounds,
        setup_or_special_action_count=setup_count,
        run_count=run_count,
    )


def evaluate_counterplay_findings(
    metrics: CounterplayMetrics,
    envelope_score: RouteEnvelopeScore | None,
) -> tuple[CounterplayFinding, ...]:
    findings: list[CounterplayFinding] = []

    if metrics.oracle_miss_rate > ORACLE_MISS_RATE_WARN:
        findings.append(
            CounterplayFinding(
                finding_id="oracle_misses_high",
                status=FINDING_WARN,
                detail=(
                    f"score-only heuristic oracle misses on "
                    f"{metrics.oracle_miss_rate:.1%} of enemy actions."
                ),
            )
        )
    else:
        findings.append(
            CounterplayFinding(
                finding_id="oracle_misses_high",
                status=FINDING_PASS,
                detail="oracle miss rate within target band.",
            )
        )

    if metrics.dominant_skill_rate > DOMINANT_SKILL_RATE_WARN:
        findings.append(
            CounterplayFinding(
                finding_id="dominant_skill_high",
                status=FINDING_WARN,
                detail=(
                    f"{metrics.dominant_skill_id} accounts for "
                    f"{metrics.dominant_skill_rate:.0%} of enemy actions."
                ),
            )
        )
    else:
        findings.append(
            CounterplayFinding(
                finding_id="dominant_skill_high",
                status=FINDING_PASS,
                detail="dominant skill usage within target band.",
            )
        )

    if metrics.damage_to_downed_rate is not None:
        if metrics.damage_to_downed_rate > DOWNED_DAMAGE_RATE_WARN:
            findings.append(
                CounterplayFinding(
                    finding_id="downed_pressure_high",
                    status=FINDING_WARN,
                    detail=(
                        f"damage to already-Downed heroes on "
                        f"{metrics.damage_to_downed_rate:.1%} of enemy actions."
                    ),
                )
            )
        else:
            label = "low" if metrics.damage_to_downed_count == 0 else "within band"
            findings.append(
                CounterplayFinding(
                    finding_id="downed_pressure_high",
                    status=FINDING_PASS,
                    detail=f"damage to downed: {label}.",
                )
            )

    if metrics.first_downed_round_avg is not None and metrics.downed_created_count > 0:
        if metrics.first_downed_round_avg < EARLY_DOWNED_ROUND_WARN:
            findings.append(
                CounterplayFinding(
                    finding_id="early_downed_high",
                    status=FINDING_WARN,
                    detail=(
                        f"first Downed averages round {metrics.first_downed_round_avg:.1f}."
                    ),
                )
            )
        else:
            findings.append(
                CounterplayFinding(
                    finding_id="early_downed_high",
                    status=FINDING_PASS,
                    detail="first Downed timing within target band.",
                )
            )

    episode_count = max(1, metrics.run_count)
    mortal_per_run = metrics.mortal_wounds_created_count / episode_count
    if mortal_per_run > MORTAL_WOUND_PER_RUN_WARN:
        findings.append(
            CounterplayFinding(
                finding_id="mortal_wound_pressure_high",
                status=FINDING_WARN,
                detail=f"mortal wounds/run: {mortal_per_run:.2f}.",
            )
        )
    else:
        findings.append(
            CounterplayFinding(
                finding_id="mortal_wound_pressure_high",
                status=FINDING_PASS,
                detail="mortal wound pressure within target band.",
            )
        )

    if envelope_score is not None:
        if envelope_score.status == "PASS" and not envelope_score.warnings:
            findings.append(
                CounterplayFinding(
                    finding_id="route_too_clean",
                    status=FINDING_PASS,
                    detail=f"envelope {envelope_score.envelope_id}: {envelope_score.status}.",
                )
            )
        elif envelope_score.status == "PASS":
            findings.append(
                CounterplayFinding(
                    finding_id="route_too_clean",
                    status=FINDING_WARN,
                    detail=(
                        f"envelope {envelope_score.envelope_id}: "
                        f"{'; '.join(envelope_score.warnings)}"
                    ),
                )
            )
        else:
            findings.append(
                CounterplayFinding(
                    finding_id="route_too_clean",
                    status=FINDING_PASS,
                    detail=(
                        f"route not too clean; envelope {envelope_score.envelope_id} "
                        f"status {envelope_score.status}."
                    ),
                )
            )

        if envelope_score.status in {"FAIL", "WARN"}:
            bimodal_warning = _bimodal_collapse_warning(envelope_score)
            if bimodal_warning is not None:
                findings.append(
                    CounterplayFinding(
                        finding_id="route_bimodal_collapse",
                        status=(
                            FINDING_FAIL
                            if envelope_score.status == "FAIL"
                            else FINDING_WARN
                        ),
                        detail=bimodal_warning.removeprefix("route_bimodal_collapse: "),
                    )
                )
            warning_text = (
                bimodal_warning
                if bimodal_warning is not None
                else (
                    "; ".join(envelope_score.warnings)
                    if envelope_score.warnings
                    else envelope_score.status.lower()
                )
            )
            findings.append(
                CounterplayFinding(
                    finding_id="route_too_punishing",
                    status=FINDING_WARN if envelope_score.status == "WARN" else FINDING_FAIL,
                    detail=(
                        f"envelope {envelope_score.envelope_id}: {warning_text}"
                    ),
                )
            )
        else:
            findings.append(
                CounterplayFinding(
                    finding_id="route_too_punishing",
                    status=FINDING_PASS,
                    detail=f"envelope {envelope_score.envelope_id}: in band.",
                )
            )

    return tuple(sorted(findings, key=lambda item: item.finding_id))


def format_oracle_report(report: OracleReport) -> str:
    metrics = report.metrics
    lines = [
        "AI Lab Oracle Report",
        f"Route: {report.route_id or 'isolated encounters'}",
        f"Seeds: {report.seed_count}",
        f"Hero policy: {report.hero_policy_id}",
        f"Preset: {report.preset_id}",
        f"Oracle mode: {report.oracle_mode}",
        "",
        "Oracle:",
        f"- enemy actions checked: {metrics.total_enemy_actions}",
        (
            f"- oracle misses: {metrics.oracle_miss_count} "
            f"({metrics.oracle_miss_rate:.1%})"
        ),
        f"- average miss delta: {metrics.average_miss_delta:.2f}",
    ]
    if metrics.largest_miss is not None:
        actor_id, chosen_skill, oracle_skill, delta = metrics.largest_miss
        lines.append(
            f"- largest miss: {actor_id} chose {chosen_skill} -> "
            f"oracle preferred {oracle_skill} (+{delta})"
        )

    lines.extend(
        [
            "",
            "Counterplay:",
            f"- first downed round avg: {_format_optional_float(metrics.first_downed_round_avg)}",
            (
                f"- mortal wounds/run: "
                f"{_mortal_wounds_per_run(metrics):.2f}"
            ),
            f"- deaths/run: {_deaths_per_run(metrics):.2f}",
            (
                f"- dominant skill: {metrics.dominant_skill_id or 'n/a'} "
                f"{metrics.dominant_skill_rate:.0%}"
            ),
            f"- damage to downed: {_format_damage_to_downed(metrics)}",
            f"- setup/special actions: {metrics.setup_or_special_action_count}",
            f"- average rounds: {metrics.average_rounds:.1f}",
            f"- noncompletion: {_format_noncompletion(metrics)}",
            (
                f"- pre-boss failures: "
                f"{_format_optional_int(metrics.pre_boss_failure_count)}"
            ),
            f"- boss entry hp: {_format_optional_float(metrics.boss_entry_hp)}",
            f"- final party hp: {_format_optional_float(metrics.final_party_hp)}",
            f"- final effort: {_format_optional_float(metrics.final_effort)}",
            "",
            "Findings:",
        ]
    )
    if not report.findings:
        lines.append("- none")
    else:
        for finding in report.findings:
            lines.append(f"- {finding.status} {finding.finding_id}: {finding.detail}")

    return "\n".join(lines)


def _matching_candidate(
    candidates: Sequence[EnemyDecisionCandidate],
    skill_id: str,
    target_id: str,
) -> EnemyDecisionCandidate | None:
    for candidate in candidates:
        if candidate.skill_id == skill_id and candidate.target_id == target_id:
            return candidate
    return None


def _package_id_for_enemy(enemy_class_id: str) -> str:
    for contract in DEFAULT_PACKAGE_CONTRACTS:
        if enemy_class_id in contract.enemy_ids:
            return contract.package_id
    return ""


def _is_setup_or_special_action(record: EnemyDecisionRecord) -> bool:
    if record.chosen_skill_id in SETUP_SKILL_IDS or record.chosen_skill_id in WARD_SKILL_IDS:
        return True
    candidate = _matching_candidate(
        record.trace.candidates,
        record.chosen_skill_id,
        record.chosen_target_id,
    )
    if candidate is None:
        return False
    return bool(candidate.skill_tags & _SETUP_ACTION_TAGS)


def _dominant_skill(
    skill_counter: Counter[str],
    total_actions: int,
) -> tuple[str, float]:
    if not skill_counter or total_actions <= 0:
        return "", 0.0
    skill_id, count = skill_counter.most_common(1)[0]
    return skill_id, count / total_actions


def _largest_miss_tuple(
    miss_records: Sequence[OracleDecisionRecord],
) -> tuple[str, str, str, int] | None:
    if not miss_records:
        return None
    best = max(
        miss_records,
        key=lambda item: (item.delta, item.actor_id, item.round_number),
    )
    return (best.actor_id, best.chosen_skill_id, best.oracle_skill_id, best.delta)


def _route_envelope_score(summary: TrainingRunSummary) -> RouteEnvelopeScore | None:
    if not summary.route_id:
        return None
    envelope_id = _default_envelope_for_route(summary.route_id)
    return score_route_envelope(summary.learned_route_summary, envelope_id=envelope_id)


def _default_envelope_for_route(route_id: str) -> str:
    if route_id == "opening_pressure_path":
        return "optional_pressure_path"
    return "critical_path"


def _bimodal_collapse_warning(envelope_score: RouteEnvelopeScore) -> str | None:
    for warning in envelope_score.warnings:
        if warning.startswith("route_bimodal_collapse:"):
            return warning
    return None


def _format_noncompletion(metrics: CounterplayMetrics) -> str:
    if metrics.noncompletion_count is None:
        return "n/a"
    if metrics.noncompletion_rate is None:
        return str(metrics.noncompletion_count)
    return f"{metrics.noncompletion_count} ({metrics.noncompletion_rate:.0%})"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _format_damage_to_downed(metrics: CounterplayMetrics) -> str:
    if metrics.damage_to_downed_rate is None:
        return "n/a"
    if metrics.damage_to_downed_count == 0:
        return "low"
    return f"{metrics.damage_to_downed_rate:.1%}"


def _mortal_wounds_per_run(metrics: CounterplayMetrics) -> float:
    if metrics.run_count <= 0:
        return 0.0
    return metrics.mortal_wounds_created_count / metrics.run_count


def _deaths_per_run(metrics: CounterplayMetrics) -> float:
    if metrics.run_count <= 0:
        return 0.0
    return metrics.deaths_created_count / metrics.run_count
