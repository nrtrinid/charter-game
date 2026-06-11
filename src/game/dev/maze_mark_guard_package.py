"""Dev-only Maze mark/guard/stalker package contract metrics for policy-band reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from game.combat.enemy_learning import EnemyDecisionEpisode
from game.dev.ai_decisions import GATE_FAIL, GATE_PASS, GATE_WARN

GATE_INFO = "INFO"

MAZE_MARK_GUARD_PACKAGE_ID = "maze_mark_guard"

MAZE_PACKAGE_ENCOUNTER_IDS = (
    "generated_maze_probe",
    "generated_maze_pattern_cell",
    "generated_maze_glass_pack",
    "generated_maze_stalker",
    "maze_depth_1",
)

WARD_ROLE_ENCOUNTER_IDS = (
    "generated_maze_pattern_cell",
    "generated_maze_stalker",
)

SETUP_SKILL_IDS = frozenset({"splinter_mark", "mark_the_pattern"})
PAYOFF_SKILL_IDS = frozenset({"stalker_cut", "stalker_hook"})
WARD_SKILL_IDS = frozenset({"ward_pattern"})

DIAG_PACKAGE_NEVER_SETS_UP = "package_never_sets_up"
DIAG_PAYOFF_NEVER_FIRES = "payoff_never_fires"
DIAG_SMART_POLICY_DELETES_PACKAGE = "smart_policy_deletes_package"
DIAG_SMART_POLICY_ANSWERS_PACKAGE = "smart_policy_answers_package"
DIAG_NAIVE_GETS_PUNISHED = "naive_gets_punished"
DIAG_WARD_ROLE_INVISIBLE = "ward_role_invisible"

_SCORING_STATUS_RANK = {GATE_PASS: 0, GATE_WARN: 1, GATE_FAIL: 2}


@dataclass(frozen=True)
class MazePackageThresholds:
    min_package_episodes: int = 2
    smart_beats_mixed_warn: int = 25
    naive_punished_spread_warn: int = 20


@dataclass(frozen=True)
class MazeMarkGuardMetrics:
    maze_episodes: int = 0
    episodes_with_setup: int = 0
    mark_setup_attempts: int = 0
    marks_applied: int = 0
    ward_actions: int = 0
    guard_uses: int = 0
    guard_damage_blocked: int = 0
    stalker_cut_uses: int = 0
    stalker_hook_uses: int = 0
    package_payoff_episodes: int = 0
    exploited_by_enemy_hit: int = 0
    vulnerable_payoffs: int = 0
    marked_damage: int = 0
    forced_movement_after_setup: int = 0
    ignored_marked_legal_attacks: int = 0
    setup_without_payoff_episodes: int = 0

    @property
    def setup_rate(self) -> float:
        if self.maze_episodes == 0:
            return 0.0
        return self.episodes_with_setup / self.maze_episodes

    @property
    def payoff_rate(self) -> float:
        if self.episodes_with_setup == 0:
            return 0.0
        return self.package_payoff_episodes / self.episodes_with_setup


@dataclass(frozen=True)
class PolicyPackageInput:
    hero_policy_id: str
    completion_rate: float
    maze_package: MazeMarkGuardMetrics
    ward_role_package: MazeMarkGuardMetrics | None = None


@dataclass(frozen=True)
class PackageContractGateResult:
    gate_id: str
    status: str
    detail: str
    diagnostic_label: str = ""
    diagnostic_only: bool = False


def episode_has_setup(episode: EnemyDecisionEpisode) -> bool:
    mark_setup_attempts = sum(
        1 for record in episode.records if record.chosen_skill_id in SETUP_SKILL_IDS
    )
    return mark_setup_attempts > 0 or episode.metrics.mark_flow.marks_applied > 0


def episode_has_true_payoff(episode: EnemyDecisionEpisode) -> bool:
    if not episode_has_setup(episode):
        return False
    mark_flow = episode.metrics.mark_flow
    if mark_flow.exploited_by_enemy_hit > 0:
        return True
    if mark_flow.vulnerable_payoffs > 0:
        return True
    if mark_flow.total_damage_to_marked > 0:
        return True
    if episode.metrics.forced_movement > 0:
        return True
    return False


def aggregate_maze_mark_guard_metrics(
    episodes: Sequence[EnemyDecisionEpisode],
) -> MazeMarkGuardMetrics:
    return _aggregate_maze_episodes(_maze_episodes(episodes))


def aggregate_maze_mark_guard_by_encounter(
    episodes: Sequence[EnemyDecisionEpisode],
    encounter_ids: Sequence[str],
) -> MazeMarkGuardMetrics:
    allowed = frozenset(encounter_ids)
    filtered = tuple(
        episode for episode in _maze_episodes(episodes) if episode.encounter_id in allowed
    )
    return _aggregate_maze_episodes(filtered)


def mixed_maze_episode_count(inputs: Sequence[PolicyPackageInput]) -> int:
    for item in inputs:
        if item.hero_policy_id == "mixed":
            return item.maze_package.maze_episodes
    return 0


def evaluate_maze_package_gates(
    inputs: Sequence[PolicyPackageInput],
    *,
    thresholds: MazePackageThresholds | None = None,
) -> tuple[PackageContractGateResult, ...]:
    resolved = thresholds or MazePackageThresholds()
    mixed = _input_for(inputs, "mixed")
    if mixed is None:
        return ()

    if mixed.maze_package.maze_episodes < resolved.min_package_episodes:
        return ()

    gates: list[PackageContractGateResult] = []
    gates.extend(_scoring_gates(inputs, resolved))
    gates.extend(_informational_gates(inputs, resolved))
    return tuple(gates)


def package_overall_status(gates: Sequence[PackageContractGateResult]) -> str:
    scoring = [
        gate.status
        for gate in gates
        if not gate.diagnostic_only and gate.status != GATE_INFO
    ]
    if not scoring:
        return GATE_PASS
    return max(scoring, key=lambda status: _SCORING_STATUS_RANK.get(status, 0))


def format_maze_package_section(
    inputs: Sequence[PolicyPackageInput],
    gates: Sequence[PackageContractGateResult],
    *,
    insufficient_samples: bool,
    thresholds: MazePackageThresholds | None = None,
) -> list[str]:
    resolved = thresholds or MazePackageThresholds()
    lines = [f"Maze package ({MAZE_MARK_GUARD_PACKAGE_ID}):"]
    if insufficient_samples:
        mixed_episodes = mixed_maze_episode_count(inputs)
        lines.append(
            f"  insufficient samples (maze_episodes={mixed_episodes}, "
            f"need {resolved.min_package_episodes})"
        )
    for item in inputs:
        package = item.maze_package
        if package.maze_episodes == 0 and not insufficient_samples:
            continue
        lines.append(
            f"  {item.hero_policy_id}: "
            f"setup={package.episodes_with_setup}/{package.maze_episodes} "
            f"true_payoff={package.package_payoff_episodes}/"
            f"{package.episodes_with_setup} "
            f"ward_actions={package.ward_actions} guard_uses={package.guard_uses} "
            f"stalker_cut={package.stalker_cut_uses} hook={package.stalker_hook_uses} "
            f"exploited={package.exploited_by_enemy_hit} "
            f"vulnerable={package.vulnerable_payoffs} "
            f"marked_dmg={package.marked_damage} "
            f"forced={package.forced_movement_after_setup}"
        )

    scoring_gates = [
        gate for gate in gates if not gate.diagnostic_only and gate.status != GATE_INFO
    ]
    note_gates = [gate for gate in gates if gate.diagnostic_only or gate.status == GATE_INFO]

    lines.append("Package gates:")
    if scoring_gates:
        for gate in scoring_gates:
            label = f" {gate.diagnostic_label}" if gate.diagnostic_label else ""
            lines.append(f"  {gate.gate_id}: {gate.status} ({gate.detail}){label}")
    elif insufficient_samples:
        lines.append("  skipped (insufficient samples)")
    else:
        lines.append("  none")

    if note_gates:
        lines.append("Package notes:")
        for gate in note_gates:
            label = gate.diagnostic_label or gate.gate_id
            lines.append(f"  {gate.gate_id}: {gate.detail} {label}")

    lines.append(f"Package overall: {package_overall_status(gates)}")
    return lines


def _maze_episodes(episodes: Sequence[EnemyDecisionEpisode]) -> tuple[EnemyDecisionEpisode, ...]:
    return tuple(
        episode
        for episode in episodes
        if episode.encounter_id in MAZE_PACKAGE_ENCOUNTER_IDS
    )


def _aggregate_maze_episodes(
    maze_episodes: Sequence[EnemyDecisionEpisode],
) -> MazeMarkGuardMetrics:
    if not maze_episodes:
        return MazeMarkGuardMetrics()

    mark_setup_attempts = 0
    marks_applied = 0
    ward_actions = 0
    guard_uses = 0
    guard_damage_blocked = 0
    stalker_cut_uses = 0
    stalker_hook_uses = 0
    package_payoff_episodes = 0
    exploited_by_enemy_hit = 0
    vulnerable_payoffs = 0
    marked_damage = 0
    forced_movement_after_setup = 0
    ignored_marked_legal_attacks = 0
    episodes_with_setup = 0
    setup_without_payoff_episodes = 0

    for episode in maze_episodes:
        for record in episode.records:
            skill_id = record.chosen_skill_id
            if skill_id in SETUP_SKILL_IDS:
                mark_setup_attempts += 1
            if skill_id in WARD_SKILL_IDS:
                ward_actions += 1
            if skill_id == "stalker_cut":
                stalker_cut_uses += 1
            if skill_id == "stalker_hook":
                stalker_hook_uses += 1

        mark_flow = episode.metrics.mark_flow
        guard_flow = episode.metrics.guard_flow
        marks_applied += int(mark_flow.marks_applied)
        guard_uses += int(guard_flow.guard_uses)
        guard_damage_blocked += int(guard_flow.guard_damage_blocked)
        exploited_by_enemy_hit += int(mark_flow.exploited_by_enemy_hit)
        vulnerable_payoffs += int(mark_flow.vulnerable_payoffs)
        marked_damage += int(mark_flow.total_damage_to_marked)
        ignored_marked_legal_attacks += int(mark_flow.ignored_marked_legal_attacks)

        has_setup = episode_has_setup(episode)
        if has_setup:
            episodes_with_setup += 1
            if episode.metrics.forced_movement > 0:
                forced_movement_after_setup += int(episode.metrics.forced_movement)
            if episode_has_true_payoff(episode):
                package_payoff_episodes += 1
            else:
                setup_without_payoff_episodes += 1

    return MazeMarkGuardMetrics(
        maze_episodes=len(maze_episodes),
        episodes_with_setup=episodes_with_setup,
        mark_setup_attempts=mark_setup_attempts,
        marks_applied=marks_applied,
        ward_actions=ward_actions,
        guard_uses=guard_uses,
        guard_damage_blocked=guard_damage_blocked,
        stalker_cut_uses=stalker_cut_uses,
        stalker_hook_uses=stalker_hook_uses,
        package_payoff_episodes=package_payoff_episodes,
        exploited_by_enemy_hit=exploited_by_enemy_hit,
        vulnerable_payoffs=vulnerable_payoffs,
        marked_damage=marked_damage,
        forced_movement_after_setup=forced_movement_after_setup,
        ignored_marked_legal_attacks=ignored_marked_legal_attacks,
        setup_without_payoff_episodes=setup_without_payoff_episodes,
    )


def _scoring_gates(
    inputs: Sequence[PolicyPackageInput],
    thresholds: MazePackageThresholds,
) -> list[PackageContractGateResult]:
    gates: list[PackageContractGateResult] = []
    mixed = _input_for(inputs, "mixed")
    if mixed is None:
        return gates

    if _all_policies_missing_setup(inputs):
        gates.append(
            PackageContractGateResult(
                "package_never_sets_up",
                GATE_FAIL,
                "no maze mark/guard setup observed",
                DIAG_PACKAGE_NEVER_SETS_UP,
            )
        )
        return gates

    if (
        mixed.maze_package.episodes_with_setup > 0
        and mixed.maze_package.package_payoff_episodes == 0
    ):
        gates.append(
            PackageContractGateResult(
                "payoff_never_fires",
                GATE_WARN,
                (
                    f"mixed setup={mixed.maze_package.episodes_with_setup} "
                    f"true_payoff=0"
                ),
                DIAG_PAYOFF_NEVER_FIRES,
            )
        )

    naive = _input_for(inputs, "naive")
    anti_mark = _input_for(inputs, "anti_mark")
    if naive is not None and anti_mark is not None:
        if (
            naive.maze_package.package_payoff_episodes > 0
            and anti_mark.maze_package.package_payoff_episodes == 0
            and (anti_mark.completion_rate - mixed.completion_rate) * 100
            >= thresholds.smart_beats_mixed_warn
        ):
            gates.append(
                PackageContractGateResult(
                    "smart_policy_deletes_package",
                    GATE_WARN,
                    (
                        "anti_mark "
                        f"+{(anti_mark.completion_rate - mixed.completion_rate) * 100:.0f}pt "
                        f"vs mixed; naive true_payoff={naive.maze_package.package_payoff_episodes}"
                    ),
                    DIAG_SMART_POLICY_DELETES_PACKAGE,
                )
            )

    if naive is not None:
        smart_best = max(
            _completion_for(inputs, "anti_mark"),
            _completion_for(inputs, "conservative"),
        )
        naive_has_pressure = (
            naive.maze_package.episodes_with_setup > 0
            or naive.maze_package.package_payoff_episodes > 0
        )
        if (
            naive_has_pressure
            and (smart_best - naive.completion_rate) * 100
            >= thresholds.naive_punished_spread_warn
        ):
            gates.append(
                PackageContractGateResult(
                    "naive_gets_punished",
                    GATE_WARN,
                    f"naive {naive.completion_rate * 100:.0f}% vs smart {smart_best * 100:.0f}%",
                    DIAG_NAIVE_GETS_PUNISHED,
                )
            )

    ward_subset = mixed.ward_role_package or MazeMarkGuardMetrics()
    if ward_subset.maze_episodes > 0 and ward_subset.ward_actions == 0:
        gates.append(
            PackageContractGateResult(
                "ward_role_invisible",
                GATE_WARN,
                f"ward_actions=0 across {ward_subset.maze_episodes} ward-role episodes",
                DIAG_WARD_ROLE_INVISIBLE,
            )
        )

    return gates


def _informational_gates(
    inputs: Sequence[PolicyPackageInput],
    thresholds: MazePackageThresholds,
) -> list[PackageContractGateResult]:
    mixed = _input_for(inputs, "mixed")
    anti_mark = _input_for(inputs, "anti_mark")
    if mixed is None or anti_mark is None:
        return []

    naive = _input_for(inputs, "naive")
    naive_has_payoff = (
        naive is not None and naive.maze_package.package_payoff_episodes > 0
    )
    anti_suppressed = anti_mark.maze_package.package_payoff_episodes == 0
    anti_beats_mixed = (
        (anti_mark.completion_rate - mixed.completion_rate) * 100
        >= thresholds.smart_beats_mixed_warn
    )
    if anti_suppressed and not anti_beats_mixed and (
        naive_has_payoff or mixed.maze_package.package_payoff_episodes > 0
    ):
        return [
            PackageContractGateResult(
                "smart_policy_answers_package",
                GATE_INFO,
                "anti_mark suppressed payoff without trivializing route completion",
                DIAG_SMART_POLICY_ANSWERS_PACKAGE,
                diagnostic_only=True,
            )
        ]
    return []


def _all_policies_missing_setup(inputs: Sequence[PolicyPackageInput]) -> bool:
    return all(
        item.maze_package.episodes_with_setup == 0
        and item.maze_package.mark_setup_attempts == 0
        and item.maze_package.marks_applied == 0
        for item in inputs
    )


def _input_for(
    inputs: Sequence[PolicyPackageInput],
    hero_policy_id: str,
) -> PolicyPackageInput | None:
    for item in inputs:
        if item.hero_policy_id == hero_policy_id:
            return item
    return None


def _completion_for(inputs: Sequence[PolicyPackageInput], hero_policy_id: str) -> float:
    item = _input_for(inputs, hero_policy_id)
    return item.completion_rate if item is not None else 0.0
