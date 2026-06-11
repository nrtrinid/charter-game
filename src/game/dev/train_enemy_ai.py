"""Dev-only offline training harness for enemy decision weights."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from game.campaign.company import CompanyState, HeroState, create_new_company
from game.campaign.roster import sync_company_from_combat
from game.combat.combat_state import CombatState, LifeState
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionPolicy,
    EnemyDecisionRuntimeContext,
)
from game.combat.enemy_learning import (
    SUPPORTED_ENEMY_MOVEMENT_MODES,
    SUPPORTED_ENEMY_WAIT_MODES,
    SUPPORTED_HERO_POLICY_IDS,
    BossSequenceMetrics,
    BossTargetingMetrics,
    EnemyDecisionEpisode,
    EnemyDecisionRecord,
    EnemyPressureMetrics,
    GuardFlowMetrics,
    LinearEnemyDecisionPolicy,
    MarkFlowMetrics,
    create_hero_policy,
    learn_linear_enemy_weights,
    run_enemy_learning_episode,
)
from game.combat.formation import FormationSlot
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.expedition.cave import create_encounter_combat

SUPPORTED_PRESET_IDS = ("fresh", "lightly_wounded", "attrition", "low_effort", "bad_formation")
SUPPORTED_ROUTE_IDS = ("opening_critical_path", "opening_pressure_path")
SUPPORTED_POLICY_SCOPE_IDS = ("global", "per_encounter", "per_role", "boss")
_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()
_ROUTE_ENCOUNTERS: dict[str, tuple[str, ...]] = {
    "opening_critical_path": ("shallow_cave", "cave_mini_boss"),
    "opening_pressure_path": (
        "road_bandits",
        "wolf_pack",
        "shallow_cave",
        "cave_mini_boss",
        "maze_depth_1",
    ),
}


@dataclass(frozen=True)
class TrainingRunConfig:
    encounter_ids: tuple[str, ...] = ()
    seeds: int = 50
    max_rounds: int = 20
    hero_policy_id: str = "naive"
    evaluation_hero_policy_ids: tuple[str, ...] = ()
    policy_scope_ids: tuple[str, ...] = ("global",)
    preset_id: str = "fresh"
    route_id: str = ""
    enemy_wait_mode: str = "none"
    enemy_movement_mode: str = "recovery_only"
    definitions: GameDefinitions | None = None
    learned_weight_overrides: Mapping[str, int] | None = None


@dataclass(frozen=True)
class PolicyPressureSummary:
    episode_count: int
    average_reward: float
    victor_counts: dict[str, int]
    average_rounds: float
    average_enemy_decisions: float
    average_hero_hp_remaining: float
    average_lowest_hero_hp_reached: float
    total_hero_damage: int
    hero_downs: int
    hero_deaths: int
    mortal_wounds: int
    average_effort_remaining: float
    healing_actions: int
    marks_applied: int
    marks_exploited: int
    guard_actions: int
    forced_movement: int
    enemy_recovery_moves: int
    enemy_proactive_moves: int
    enemy_waits: int
    waited_then_attacked_next_activation: int
    waited_then_marked_hit: int
    waited_then_payoff: int
    waited_then_down: int
    waited_then_no_payoff: int
    waited_then_no_attack: int
    waits_without_payoff: int
    move_then_attack_next_activation: int
    move_then_marked_hit: int
    move_then_payoff: int
    move_then_down: int
    move_unlocks_future_skill: int
    move_into_marked_lane: int
    move_wasted_no_followup: int
    moves_without_payoff: int
    recovery_move_then_attack: int
    recovery_move_then_payoff: int
    recovery_move_wasted: int
    recovery_moves_without_followup: int
    all_enemy_wait_rounds: int
    end_round_enemy_burst_damage: int
    end_round_enemy_burst_downs: int
    boss_sequence: BossSequenceMetrics
    boss_targeting: BossTargetingMetrics
    mark_flow: MarkFlowMetrics
    guard_flow: GuardFlowMetrics


@dataclass(frozen=True)
class EncounterTrainingBreakdown:
    encounter_id: str
    heuristic: PolicyPressureSummary
    learned: PolicyPressureSummary


@dataclass(frozen=True)
class RouteEncounterResult:
    encounter_id: str
    hp_entering: int
    effort_entering: int
    hp_leaving: int
    effort_leaving: int
    reward: int
    final_victor: str
    downs: int
    deaths: int
    mortal_wounds: int


@dataclass(frozen=True)
class RouteTrainingResult:
    route_id: str
    seed: int
    episodes: tuple[EnemyDecisionEpisode, ...]
    encounters: tuple[RouteEncounterResult, ...]
    completed: bool
    failed_at_encounter_id: str | None
    final_hero_hp_total: int
    final_hero_effort_total: int

    @property
    def total_reward(self) -> int:
        return sum(episode.total_reward for episode in self.episodes)

    @property
    def total_downs(self) -> int:
        return sum(episode.metrics.hero_downs for episode in self.episodes)

    @property
    def total_deaths(self) -> int:
        return sum(episode.metrics.hero_deaths for episode in self.episodes)

    @property
    def total_mortal_wounds(self) -> int:
        return sum(episode.metrics.mortal_wounds for episode in self.episodes)


@dataclass(frozen=True)
class RoutePressureSummary:
    route_count: int
    completed_count: int
    failed_at_counts: dict[str, int]
    average_reward: float
    average_final_hero_hp: float
    average_final_hero_effort: float
    average_downs: float
    average_deaths: float
    average_mortal_wounds: float
    average_hp_entering_cave_mini_boss: float
    average_hp_leaving_cave_mini_boss: float


@dataclass(frozen=True)
class LearnedEnemyPolicyModel:
    scope_id: str
    definitions: GameDefinitions
    global_weights: dict[str, int]
    per_encounter_weights: dict[str, dict[str, int]]
    per_role_weights: dict[str, dict[str, int]]
    boss_weights: dict[str, int]

    @property
    def display_weights(self) -> dict[str, int]:
        if self.scope_id == "boss":
            return self.boss_weights or self.global_weights
        return self.global_weights

    def policy_for(self, encounter_id: str) -> ScopedLinearEnemyDecisionPolicy:
        return ScopedLinearEnemyDecisionPolicy(self, encounter_id)


@dataclass(frozen=True)
class ScopedLinearEnemyDecisionPolicy:
    model: LearnedEnemyPolicyModel
    encounter_id: str

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        return LinearEnemyDecisionPolicy(
            self._weights_for(state.actor(enemy_id).class_id)
        ).choose(state, definitions, enemy_id, runtime_context)

    def _weights_for(self, enemy_class_id: str) -> dict[str, int]:
        if self.model.scope_id == "per_encounter":
            return self.model.per_encounter_weights.get(
                self.encounter_id,
                self.model.global_weights,
            )
        if self.model.scope_id == "per_role":
            role_id = _enemy_role_id(self.model.definitions, enemy_class_id)
            return self.model.per_role_weights.get(role_id, self.model.global_weights)
        if self.model.scope_id == "boss":
            tags = _enemy_tags(self.model.definitions, enemy_class_id)
            if "boss" in tags:
                return self.model.boss_weights or self.model.global_weights
        return self.model.global_weights


@dataclass(frozen=True)
class TrainingPolicyEvaluation:
    policy_scope_id: str
    evaluation_hero_policy_id: str
    episodes: tuple[EnemyDecisionEpisode, ...]
    route_results: tuple[RouteTrainingResult, ...]
    learned_weights: dict[str, int]
    encounter_breakdowns: tuple[EncounterTrainingBreakdown, ...]

    @property
    def total_reward(self) -> int:
        return sum(episode.total_reward for episode in self.episodes)

    @property
    def average_reward(self) -> float:
        return _average_reward(self.episodes)

    @property
    def victor_counts(self) -> dict[str, int]:
        return _victor_counts(self.episodes)

    @property
    def average_records(self) -> float:
        return _average_records(self.episodes)

    @property
    def route_summary(self) -> RoutePressureSummary:
        return _route_pressure_summary(self.route_results)


@dataclass(frozen=True)
class TrainingRunSummary:
    encounter_ids: tuple[str, ...]
    seed_count: int
    hero_policy_id: str
    evaluation_hero_policy_ids: tuple[str, ...]
    policy_scope_ids: tuple[str, ...]
    preset_id: str
    route_id: str
    enemy_wait_mode: str
    enemy_movement_mode: str
    heuristic_episodes: tuple[EnemyDecisionEpisode, ...]
    learned_episodes: tuple[EnemyDecisionEpisode, ...]
    heuristic_route_results: tuple[RouteTrainingResult, ...]
    learned_route_results: tuple[RouteTrainingResult, ...]
    learned_weights: dict[str, int]
    encounter_breakdowns: tuple[EncounterTrainingBreakdown, ...]
    policy_evaluations: tuple[TrainingPolicyEvaluation, ...]

    @property
    def heuristic_total_reward(self) -> int:
        return sum(episode.total_reward for episode in self.heuristic_episodes)

    @property
    def learned_total_reward(self) -> int:
        return sum(episode.total_reward for episode in self.learned_episodes)

    @property
    def heuristic_average_reward(self) -> float:
        return _average_reward(self.heuristic_episodes)

    @property
    def learned_average_reward(self) -> float:
        return _average_reward(self.learned_episodes)

    @property
    def heuristic_victors(self) -> dict[str, int]:
        return _victor_counts(self.heuristic_episodes)

    @property
    def learned_victors(self) -> dict[str, int]:
        return _victor_counts(self.learned_episodes)

    @property
    def heuristic_average_records(self) -> float:
        return _average_records(self.heuristic_episodes)

    @property
    def learned_average_records(self) -> float:
        return _average_records(self.learned_episodes)

    @property
    def top_weights(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(
                self.learned_weights.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )

    @property
    def heuristic_route_summary(self) -> RoutePressureSummary:
        return _route_pressure_summary(self.heuristic_route_results)

    @property
    def learned_route_summary(self) -> RoutePressureSummary:
        return _route_pressure_summary(self.learned_route_results)


def run_training_harness(config: TrainingRunConfig) -> TrainingRunSummary:
    definitions = config.definitions or load_game_definitions()
    preset_id = _validate_preset_id(config.preset_id)
    route_id = _validate_route_id(config.route_id)
    policy_scope_ids = _validate_policy_scope_ids(config.policy_scope_ids)
    enemy_wait_mode = _validate_enemy_wait_mode(config.enemy_wait_mode)
    enemy_movement_mode = _validate_enemy_movement_mode(config.enemy_movement_mode)
    encounter_ids = (
        _route_encounter_ids(route_id)
        if route_id
        else _selected_encounter_ids(definitions, config.encounter_ids)
    )
    hero_policy_id = create_hero_policy(config.hero_policy_id).policy_id
    evaluation_hero_policy_ids = _evaluation_hero_policy_ids(
        config.evaluation_hero_policy_ids,
        hero_policy_id,
    )
    seed_values = tuple(range(1, max(0, config.seeds) + 1))

    if route_id:
        heuristic_route_results = tuple(
            _run_sequence(
                definitions,
                encounter_ids,
                seed,
                route_id=route_id,
                max_rounds=config.max_rounds,
                hero_policy_id=hero_policy_id,
                preset_id=preset_id,
                enemy_wait_mode=enemy_wait_mode,
                enemy_movement_mode=enemy_movement_mode,
            )
            for seed in seed_values
        )
        heuristic_episodes = _flatten_route_episodes(heuristic_route_results)
    else:
        heuristic_route_results = ()
        heuristic_episodes = tuple(
            _run_episode(
                definitions,
                encounter_id,
                seed,
                max_rounds=config.max_rounds,
                hero_policy_id=hero_policy_id,
                preset_id=preset_id,
                enemy_wait_mode=enemy_wait_mode,
                enemy_movement_mode=enemy_movement_mode,
            )
            for encounter_id in encounter_ids
            for seed in seed_values
        )

    policy_models = {
        scope_id: _learn_policy_model(definitions, heuristic_episodes, scope_id)
        for scope_id in policy_scope_ids
    }
    if config.learned_weight_overrides:
        policy_models = {
            scope_id: _apply_learned_weight_overrides(
                model,
                config.learned_weight_overrides,
            )
            for scope_id, model in policy_models.items()
        }
    policy_evaluations = tuple(
        _evaluate_policy_model(
            definitions,
            encounter_ids,
            seed_values,
            route_id=route_id,
            max_rounds=config.max_rounds,
            hero_policy_id=evaluation_hero_policy_id,
            preset_id=preset_id,
            enemy_wait_mode=enemy_wait_mode,
            enemy_movement_mode=enemy_movement_mode,
            model=policy_models[scope_id],
        )
        for scope_id in policy_scope_ids
        for evaluation_hero_policy_id in evaluation_hero_policy_ids
    )
    primary_evaluation = _primary_evaluation(
        policy_evaluations,
        scope_id="global",
        hero_policy_id=hero_policy_id,
    )

    return TrainingRunSummary(
        encounter_ids=encounter_ids,
        seed_count=len(seed_values),
        hero_policy_id=hero_policy_id,
        evaluation_hero_policy_ids=evaluation_hero_policy_ids,
        policy_scope_ids=policy_scope_ids,
        preset_id=preset_id,
        route_id=route_id,
        enemy_wait_mode=enemy_wait_mode,
        enemy_movement_mode=enemy_movement_mode,
        heuristic_episodes=heuristic_episodes,
        learned_episodes=primary_evaluation.episodes,
        heuristic_route_results=heuristic_route_results,
        learned_route_results=primary_evaluation.route_results,
        learned_weights=primary_evaluation.learned_weights,
        encounter_breakdowns=_encounter_breakdowns(
            encounter_ids,
            heuristic_episodes,
            primary_evaluation.episodes,
        ),
        policy_evaluations=policy_evaluations,
    )


def _evaluate_policy_model(
    definitions: GameDefinitions,
    encounter_ids: tuple[str, ...],
    seed_values: tuple[int, ...],
    *,
    route_id: str,
    max_rounds: int,
    hero_policy_id: str,
    preset_id: str,
    enemy_wait_mode: str,
    enemy_movement_mode: str,
    model: LearnedEnemyPolicyModel,
) -> TrainingPolicyEvaluation:
    if route_id:
        route_results = tuple(
            _run_sequence(
                definitions,
                encounter_ids,
                seed,
                route_id=route_id,
                max_rounds=max_rounds,
                policy_model=model,
                hero_policy_id=hero_policy_id,
                preset_id=preset_id,
                enemy_wait_mode=enemy_wait_mode,
                enemy_movement_mode=enemy_movement_mode,
            )
            for seed in seed_values
        )
        episodes = _flatten_route_episodes(route_results)
    else:
        route_results = ()
        episodes = tuple(
            _run_episode(
                definitions,
                encounter_id,
                seed,
                max_rounds=max_rounds,
                policy=model.policy_for(encounter_id),
                hero_policy_id=hero_policy_id,
                preset_id=preset_id,
                enemy_wait_mode=enemy_wait_mode,
                enemy_movement_mode=enemy_movement_mode,
            )
            for encounter_id in encounter_ids
            for seed in seed_values
        )
    return TrainingPolicyEvaluation(
        policy_scope_id=model.scope_id,
        evaluation_hero_policy_id=hero_policy_id,
        episodes=episodes,
        route_results=route_results,
        learned_weights=model.display_weights,
        encounter_breakdowns=_encounter_breakdowns(encounter_ids, (), episodes),
    )


def format_training_summary(summary: TrainingRunSummary) -> str:
    lines = [
        "Enemy AI Training Harness",
        f"Encounters: {', '.join(summary.encounter_ids)}",
        f"Seeds: {summary.seed_count}",
        f"Training opponent: {summary.hero_policy_id} scripted hero policy",
        f"Evaluation opponents: {', '.join(summary.evaluation_hero_policy_ids)}",
        f"Policy scopes: {', '.join(summary.policy_scope_ids)}",
        f"Preset: {summary.preset_id}",
        f"Route: {summary.route_id or 'isolated encounters'}",
        f"Enemy wait mode: {summary.enemy_wait_mode}",
        f"Enemy movement mode: {summary.enemy_movement_mode}",
        (
            "Heuristic reward: "
            f"{summary.heuristic_total_reward} total, "
            f"{summary.heuristic_average_reward:.1f} average"
        ),
        (
            "Learned reward: "
            f"{summary.learned_total_reward} total, "
            f"{summary.learned_average_reward:.1f} average"
        ),
        f"Heuristic victors: {_format_counts(summary.heuristic_victors)}",
        f"Learned victors: {_format_counts(summary.learned_victors)}",
        (
            "Enemy decisions per episode: "
            f"{summary.heuristic_average_records:.1f} heuristic, "
            f"{summary.learned_average_records:.1f} learned"
        ),
        "Top learned feature weights:",
    ]
    top_weights = summary.top_weights[:10]
    if top_weights:
        lines.extend(f"  {name}: {weight}" for name, weight in top_weights)
    else:
        lines.append("  none")
    if summary.route_id:
        heuristic_route = summary.heuristic_route_summary
        learned_route = summary.learned_route_summary
        lines.extend(
            [
                "Route Results:",
                (
                    "  heuristic completed "
                    f"{heuristic_route.completed_count}/{heuristic_route.route_count}; "
                    f"failed {_format_counts(heuristic_route.failed_at_counts)}; "
                    f"final HP {heuristic_route.average_final_hero_hp:.1f}; "
                    f"final Effort {heuristic_route.average_final_hero_effort:.1f}"
                ),
                (
                    "  learned completed "
                    f"{learned_route.completed_count}/{learned_route.route_count}; "
                    f"failed {_format_counts(learned_route.failed_at_counts)}; "
                    f"final HP {learned_route.average_final_hero_hp:.1f}; "
                    f"final Effort {learned_route.average_final_hero_effort:.1f}"
                ),
                (
                    "  cave_mini_boss HP in/out: "
                    f"{heuristic_route.average_hp_entering_cave_mini_boss:.1f}/"
                    f"{heuristic_route.average_hp_leaving_cave_mini_boss:.1f} -> "
                    f"{learned_route.average_hp_entering_cave_mini_boss:.1f}/"
                    f"{learned_route.average_hp_leaving_cave_mini_boss:.1f}"
                ),
            ]
        )
    lines.append("Per Encounter:")
    for breakdown in summary.encounter_breakdowns:
        lines.append(
            "  "
            f"{breakdown.encounter_id}: "
            f"reward {breakdown.heuristic.average_reward:.1f} -> "
            f"{breakdown.learned.average_reward:.1f}; "
            f"damage {breakdown.heuristic.total_hero_damage} -> "
            f"{breakdown.learned.total_hero_damage}; "
            f"downs {breakdown.heuristic.hero_downs} -> {breakdown.learned.hero_downs}; "
            f"marks {breakdown.heuristic.marks_applied}/"
            f"{breakdown.heuristic.marks_exploited} -> "
            f"{breakdown.learned.marks_applied}/{breakdown.learned.marks_exploited}"
        )
    boss_lines = _boss_sequence_lines(summary.encounter_breakdowns)
    if boss_lines:
        lines.append("Boss Sequence:")
        lines.extend(boss_lines)
    boss_targeting_lines = _boss_targeting_lines(summary.encounter_breakdowns)
    if boss_targeting_lines:
        lines.append("Boss Targeting:")
        lines.extend(boss_targeting_lines)
    mark_lines = _mark_flow_lines(summary.encounter_breakdowns)
    if mark_lines:
        lines.append("Mark Flow:")
        lines.extend(mark_lines)
    guard_lines = _guard_flow_lines(summary.encounter_breakdowns)
    if guard_lines:
        lines.append("Guard Flow:")
        lines.extend(guard_lines)
    timing_lines = _enemy_timing_lines(summary.encounter_breakdowns)
    if timing_lines:
        lines.append("Enemy Timing:")
        lines.extend(timing_lines)
    movement_lines = _movement_quality_lines(summary.encounter_breakdowns)
    if movement_lines:
        lines.append("Movement Quality:")
        lines.extend(movement_lines)
    if len(summary.policy_evaluations) > 1:
        lines.append("Policy Evaluations:")
        for evaluation in summary.policy_evaluations:
            route_summary = evaluation.route_summary
            route_text = (
                f"; routes {route_summary.completed_count}/{route_summary.route_count}"
                if summary.route_id
                else ""
            )
            lines.append(
                "  "
                f"{evaluation.policy_scope_id} vs {evaluation.evaluation_hero_policy_id}: "
                f"reward {evaluation.total_reward} total, "
                f"{evaluation.average_reward:.1f} average; "
                f"victors {_format_counts(evaluation.victor_counts)}"
                f"{route_text}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train dev-only enemy AI feature weights.")
    parser.add_argument(
        "--encounter",
        action="append",
        default=[],
        help="authored encounter id to include; may be supplied multiple times",
    )
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds per encounter")
    parser.add_argument("--max-rounds", type=int, default=20, help="round cap per episode")
    parser.add_argument(
        "--preset",
        choices=SUPPORTED_PRESET_IDS,
        default="fresh",
        help="deterministic starting attrition preset",
    )
    parser.add_argument(
        "--route",
        choices=SUPPORTED_ROUTE_IDS,
        default="",
        help="first-quest combat sequence to evaluate instead of isolated encounters",
    )
    parser.add_argument(
        "--hero-policy",
        choices=SUPPORTED_HERO_POLICY_IDS,
        default="naive",
        help="scripted hero policy used as the training opponent",
    )
    parser.add_argument(
        "--eval-hero-policy",
        action="append",
        default=[],
        choices=SUPPORTED_HERO_POLICY_IDS,
        help="scripted hero policy to evaluate against; may be supplied multiple times",
    )
    parser.add_argument(
        "--cross-policy",
        action="store_true",
        help="evaluate learned weights against every supported hero policy",
    )
    parser.add_argument(
        "--policy-scope",
        action="append",
        default=[],
        choices=SUPPORTED_POLICY_SCOPE_IDS,
        help="learned policy scope to evaluate; may be supplied multiple times",
    )
    parser.add_argument(
        "--enemy-wait-mode",
        choices=SUPPORTED_ENEMY_WAIT_MODES,
        default="none",
        help="dev-only enemy wait candidate mode",
    )
    parser.add_argument(
        "--enemy-move-mode",
        choices=SUPPORTED_ENEMY_MOVEMENT_MODES,
        default="recovery_only",
        help="dev-only enemy movement candidate mode",
    )
    args = parser.parse_args(argv)
    evaluation_hero_policy_ids = (
        SUPPORTED_HERO_POLICY_IDS if args.cross_policy else tuple(args.eval_hero_policy)
    )
    policy_scope_ids = tuple(args.policy_scope) or ("global",)

    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=tuple(args.encounter),
            seeds=args.seeds,
            max_rounds=args.max_rounds,
            hero_policy_id=args.hero_policy,
            evaluation_hero_policy_ids=evaluation_hero_policy_ids,
            policy_scope_ids=policy_scope_ids,
            preset_id=args.preset,
            route_id=args.route,
            enemy_wait_mode=args.enemy_wait_mode,
            enemy_movement_mode=args.enemy_move_mode,
        )
    )
    print(format_training_summary(summary))
    return 0


def _boss_sequence_lines(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> list[str]:
    lines: list[str] = []
    for breakdown in breakdowns:
        heuristic = breakdown.heuristic.boss_sequence
        learned = breakdown.learned.boss_sequence
        if not _has_boss_sequence_activity(heuristic) and not _has_boss_sequence_activity(learned):
            continue
        parts = [
            f"grabs {heuristic.grab_uses}->{learned.grab_uses}",
            f"bites {heuristic.bite_uses}->{learned.bite_uses}",
        ]
        if heuristic.grab_to_bite_same_target or learned.grab_to_bite_same_target:
            parts.append(
                "grab->bite same "
                f"{heuristic.grab_to_bite_same_target}->{learned.grab_to_bite_same_target}"
            )
        if (
            heuristic.grabbed_target_escaped_before_bite
            or learned.grabbed_target_escaped_before_bite
        ):
            parts.append(
                "escaped "
                f"{heuristic.grabbed_target_escaped_before_bite}->"
                f"{learned.grabbed_target_escaped_before_bite}"
            )
        if heuristic.boss_killed_before_first_bite or learned.boss_killed_before_first_bite:
            parts.append(
                "boss killed pre-bite "
                f"{heuristic.boss_killed_before_first_bite}->"
                f"{learned.boss_killed_before_first_bite}"
            )
        if (
            heuristic.boss_killed_after_grab_before_bite
            or learned.boss_killed_after_grab_before_bite
        ):
            parts.append(
                "boss killed post-grab "
                f"{heuristic.boss_killed_after_grab_before_bite}->"
                f"{learned.boss_killed_after_grab_before_bite}"
            )
        if heuristic.bone_soldier_guarded_boss or learned.bone_soldier_guarded_boss:
            parts.append(
                "guarded "
                f"{heuristic.bone_soldier_guarded_boss}->"
                f"{learned.bone_soldier_guarded_boss}"
            )
        lines.append(f"  {breakdown.encounter_id}: " + "; ".join(parts))
    return lines


def _boss_targeting_lines(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> list[str]:
    lines: list[str] = []
    for breakdown in breakdowns:
        heuristic = breakdown.heuristic.boss_targeting
        learned = breakdown.learned.boss_targeting
        if not _has_boss_targeting_activity(heuristic) and not _has_boss_targeting_activity(
            learned
        ):
            continue
        parts = [
            "grab targets "
            f"{_format_count_mapping(heuristic.grab_target_classes)} -> "
            f"{_format_count_mapping(learned.grab_target_classes)}",
            "bite targets "
            f"{_format_count_mapping(heuristic.bite_target_classes)} -> "
            f"{_format_count_mapping(learned.bite_target_classes)}",
            f"support grabs {heuristic.support_grabs}->{learned.support_grabs}",
        ]
        if heuristic.support_grabs_with_effort or learned.support_grabs_with_effort:
            parts.append(
                "support+Effort "
                f"{heuristic.support_grabs_with_effort}->"
                f"{learned.support_grabs_with_effort}"
            )
        if heuristic.support_grabs_not_acted or learned.support_grabs_not_acted:
            parts.append(
                "support not-acted "
                f"{heuristic.support_grabs_not_acted}->"
                f"{learned.support_grabs_not_acted}"
            )
        if (
            heuristic.support_grab_to_bite_same_target
            or learned.support_grab_to_bite_same_target
        ):
            parts.append(
                "support grab->bite "
                f"{heuristic.support_grab_to_bite_same_target}->"
                f"{learned.support_grab_to_bite_same_target}"
            )
        if heuristic.support_grab_downs or learned.support_grab_downs:
            parts.append(
                f"support downs {heuristic.support_grab_downs}->{learned.support_grab_downs}"
            )
        if heuristic.direct_front_bites or learned.direct_front_bites:
            parts.append(
                f"direct front bites {heuristic.direct_front_bites}->{learned.direct_front_bites}"
            )
        lines.append(f"  {breakdown.encounter_id}: " + "; ".join(parts))
    return lines


def _mark_flow_lines(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> list[str]:
    lines: list[str] = []
    for breakdown in breakdowns:
        heuristic = breakdown.heuristic.mark_flow
        learned = breakdown.learned.mark_flow
        if not _has_mark_flow_activity(heuristic) and not _has_mark_flow_activity(learned):
            continue
        parts = [
            f"marks {heuristic.marks_applied}->{learned.marks_applied}",
            f"refreshed {heuristic.marks_refreshed}->{learned.marks_refreshed}",
            f"hits {heuristic.exploited_by_enemy_hit}->{learned.exploited_by_enemy_hit}",
        ]
        if heuristic.multi_hit_focus or learned.multi_hit_focus:
            parts.append(f"multi-hit {heuristic.multi_hit_focus}->{learned.multi_hit_focus}")
        if heuristic.vulnerable_payoffs or learned.vulnerable_payoffs:
            parts.append(
                f"payoffs {heuristic.vulnerable_payoffs}->{learned.vulnerable_payoffs}"
            )
        if heuristic.marked_downs or learned.marked_downs:
            parts.append(f"marked downs {heuristic.marked_downs}->{learned.marked_downs}")
        if heuristic.ignored_marked_legal_attacks or learned.ignored_marked_legal_attacks:
            parts.append(
                "ignored legal "
                f"{heuristic.ignored_marked_legal_attacks}->"
                f"{learned.ignored_marked_legal_attacks}"
            )
        if heuristic.mark_ally_reach_count or learned.mark_ally_reach_count:
            parts.append(
                "avg ally reach "
                f"{heuristic.average_ally_reach:.1f}->{learned.average_ally_reach:.1f}"
            )
        lines.append(f"  {breakdown.encounter_id}: " + "; ".join(parts))
    return lines


def _has_boss_sequence_activity(metrics: BossSequenceMetrics) -> bool:
    return any(
        (
            metrics.grab_uses,
            metrics.bite_uses,
            metrics.boss_killed_before_first_bite,
            metrics.boss_killed_after_grab_before_bite,
            metrics.bone_soldier_guarded_boss,
        )
    )


def _has_boss_targeting_activity(metrics: BossTargetingMetrics) -> bool:
    return any(
        (
            metrics.grab_target_classes,
            metrics.bite_target_classes,
            metrics.support_grabs,
            metrics.direct_front_bites,
        )
    )


def _has_mark_flow_activity(metrics: MarkFlowMetrics) -> bool:
    return any(
        (
            metrics.marks_applied,
            metrics.marks_refreshed,
            metrics.exploited_by_enemy_hit,
            metrics.vulnerable_payoffs,
            metrics.marked_downs,
            metrics.ignored_marked_legal_attacks,
        )
    )


def _guard_flow_lines(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> list[str]:
    lines: list[str] = []
    for breakdown in breakdowns:
        heuristic = breakdown.heuristic.guard_flow
        learned = breakdown.learned.guard_flow
        if not _has_guard_flow_activity(heuristic) and not _has_guard_flow_activity(learned):
            continue
        parts = [
            f"guard uses {heuristic.guard_uses}->{learned.guard_uses}",
            f"dead guard {heuristic.dead_guard_uses}->{learned.dead_guard_uses}",
        ]
        if heuristic.guard_damage_blocked or learned.guard_damage_blocked:
            parts.append(
                "damage blocked "
                f"{heuristic.guard_damage_blocked}->{learned.guard_damage_blocked}"
            )
        if heuristic.guarded_ally_acted_after_guard or learned.guarded_ally_acted_after_guard:
            parts.append(
                "guarded acted "
                f"{heuristic.guarded_ally_acted_after_guard}->"
                f"{learned.guarded_ally_acted_after_guard}"
            )
        if (
            heuristic.guarded_ally_used_payoff_after_guard
            or learned.guarded_ally_used_payoff_after_guard
        ):
            parts.append(
                "guarded payoff "
                f"{heuristic.guarded_ally_used_payoff_after_guard}->"
                f"{learned.guarded_ally_used_payoff_after_guard}"
            )
        if heuristic.guarded_ally_downs_after_guard or learned.guarded_ally_downs_after_guard:
            parts.append(
                "guarded downs "
                f"{heuristic.guarded_ally_downs_after_guard}->"
                f"{learned.guarded_ally_downs_after_guard}"
            )
        if heuristic.guard_wasted_no_followup or learned.guard_wasted_no_followup:
            parts.append(
                "guard wasted "
                f"{heuristic.guard_wasted_no_followup}->{learned.guard_wasted_no_followup}"
            )
        if heuristic.guard_targets or learned.guard_targets:
            parts.append(
                "targets "
                f"{_format_count_mapping(heuristic.guard_targets)} -> "
                f"{_format_count_mapping(learned.guard_targets)}"
            )
        lines.append(f"  {breakdown.encounter_id}: " + "; ".join(parts))
    return lines


def _has_guard_flow_activity(metrics: GuardFlowMetrics) -> bool:
    return any(
        (
            metrics.guard_uses,
            metrics.dead_guard_uses,
            metrics.guard_damage_blocked,
            metrics.guarded_ally_acted_after_guard,
            metrics.guarded_ally_used_payoff_after_guard,
            metrics.guarded_ally_downs_after_guard,
            metrics.guard_wasted_no_followup,
            metrics.guard_targets,
        )
    )


def _enemy_timing_lines(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> list[str]:
    lines: list[str] = []
    for breakdown in breakdowns:
        heuristic = breakdown.heuristic
        learned = breakdown.learned
        if not _has_enemy_timing_activity(heuristic) and not _has_enemy_timing_activity(learned):
            continue
        parts = [
            f"waits {heuristic.enemy_waits}->{learned.enemy_waits}",
            (
                "waited attacks "
                f"{heuristic.waited_then_attacked_next_activation}->"
                f"{learned.waited_then_attacked_next_activation}"
            ),
            f"waited hits {heuristic.waited_then_marked_hit}->{learned.waited_then_marked_hit}",
            f"waited payoffs {heuristic.waited_then_payoff}->{learned.waited_then_payoff}",
            f"waited downs {heuristic.waited_then_down}->{learned.waited_then_down}",
            f"waits no payoff {heuristic.waits_without_payoff}->{learned.waits_without_payoff}",
            (
                "recovery "
                f"{heuristic.enemy_recovery_moves}->{learned.enemy_recovery_moves}"
            ),
            (
                "recovery->attack "
                f"{heuristic.recovery_move_then_attack}->{learned.recovery_move_then_attack}"
            ),
            (
                "recovery wasted "
                f"{heuristic.recovery_move_wasted}->{learned.recovery_move_wasted}"
            ),
        ]
        if heuristic.end_round_enemy_burst_damage or learned.end_round_enemy_burst_damage:
            parts.append(
                "burst damage "
                f"{heuristic.end_round_enemy_burst_damage}->"
                f"{learned.end_round_enemy_burst_damage}"
            )
        if heuristic.end_round_enemy_burst_downs or learned.end_round_enemy_burst_downs:
            parts.append(
                "burst downs "
                f"{heuristic.end_round_enemy_burst_downs}->"
                f"{learned.end_round_enemy_burst_downs}"
            )
        learned_label = _wait_quality_label(learned)
        if learned_label:
            parts.append(f"wait quality: {learned_label}")
        lines.append(f"  {breakdown.encounter_id}: " + "; ".join(parts))
    return lines


def _wait_quality_label(summary: PolicyPressureSummary) -> str:
    if summary.enemy_waits == 0:
        return ""
    payoff_rate = summary.waited_then_payoff / summary.enemy_waits
    if payoff_rate >= 0.4:
        return "GOOD"
    if payoff_rate >= 0.15:
        return "MIXED"
    return "POOR"


def _movement_quality_lines(
    breakdowns: tuple[EncounterTrainingBreakdown, ...],
) -> list[str]:
    lines: list[str] = []
    for breakdown in breakdowns:
        heuristic = breakdown.heuristic
        learned = breakdown.learned
        if not _has_movement_quality_activity(heuristic) and not _has_movement_quality_activity(
            learned
        ):
            continue
        parts = [
            f"proactive {heuristic.enemy_proactive_moves}->{learned.enemy_proactive_moves}",
            (
                "move->attack "
                f"{heuristic.move_then_attack_next_activation}->"
                f"{learned.move_then_attack_next_activation}"
            ),
            f"move hits {heuristic.move_then_marked_hit}->{learned.move_then_marked_hit}",
            f"move payoffs {heuristic.move_then_payoff}->{learned.move_then_payoff}",
            f"move downs {heuristic.move_then_down}->{learned.move_then_down}",
            (
                "unlock "
                f"{heuristic.move_unlocks_future_skill}->"
                f"{learned.move_unlocks_future_skill}"
            ),
            (
                "marked lane "
                f"{heuristic.move_into_marked_lane}->{learned.move_into_marked_lane}"
            ),
            (
                "wasted "
                f"{heuristic.move_wasted_no_followup}->{learned.move_wasted_no_followup}"
            ),
        ]
        move_label = _move_quality_label(learned)
        if move_label:
            parts.append(f"move quality: {move_label}")
        lines.append(f"  {breakdown.encounter_id}: " + "; ".join(parts))
    return lines


def _move_quality_label(summary: PolicyPressureSummary) -> str:
    if summary.enemy_proactive_moves == 0:
        return ""
    payoff_rate = summary.move_then_payoff / summary.enemy_proactive_moves
    waste_rate = summary.move_wasted_no_followup / summary.enemy_proactive_moves
    if payoff_rate >= 0.4 and waste_rate <= 0.3:
        return "GOOD"
    if payoff_rate >= 0.15 or waste_rate <= 0.6:
        return "MIXED"
    return "POOR"


def _has_enemy_timing_activity(summary: PolicyPressureSummary) -> bool:
    return any(
        (
            summary.enemy_recovery_moves,
            summary.enemy_waits,
            summary.waited_then_attacked_next_activation,
            summary.waited_then_marked_hit,
            summary.waited_then_payoff,
            summary.waited_then_down,
            summary.waited_then_no_payoff,
            summary.waited_then_no_attack,
            summary.waits_without_payoff,
            summary.recovery_move_then_attack,
            summary.recovery_move_then_payoff,
            summary.recovery_move_wasted,
            summary.recovery_moves_without_followup,
            summary.all_enemy_wait_rounds,
            summary.end_round_enemy_burst_damage,
            summary.end_round_enemy_burst_downs,
        )
    )


def _has_movement_quality_activity(summary: PolicyPressureSummary) -> bool:
    return any(
        (
            summary.enemy_proactive_moves,
            summary.move_then_attack_next_activation,
            summary.move_then_marked_hit,
            summary.move_then_payoff,
            summary.move_then_down,
            summary.move_unlocks_future_skill,
            summary.move_into_marked_lane,
            summary.move_wasted_no_followup,
            summary.moves_without_payoff,
        )
    )


def _selected_encounter_ids(
    definitions: GameDefinitions,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    if not requested:
        return tuple(sorted(definitions.encounters))
    missing = tuple(
        encounter_id for encounter_id in requested if encounter_id not in definitions.encounters
    )
    if missing:
        raise ValueError(f"Unknown encounter id: {', '.join(missing)}")
    return requested


def _validate_preset_id(preset_id: str) -> str:
    if preset_id in SUPPORTED_PRESET_IDS:
        return preset_id
    raise ValueError(f"Unknown preset id: {preset_id}")


def _validate_route_id(route_id: str) -> str:
    if not route_id or route_id in SUPPORTED_ROUTE_IDS:
        return route_id
    raise ValueError(f"Unknown route id: {route_id}")


def _validate_policy_scope_ids(scope_ids: tuple[str, ...]) -> tuple[str, ...]:
    selected = scope_ids or ("global",)
    missing = tuple(scope_id for scope_id in selected if scope_id not in SUPPORTED_POLICY_SCOPE_IDS)
    if missing:
        raise ValueError(f"Unknown policy scope: {', '.join(missing)}")
    if "global" not in selected:
        return ("global", *selected)
    return selected


def _validate_enemy_wait_mode(mode: str) -> str:
    if mode in SUPPORTED_ENEMY_WAIT_MODES:
        return mode
    raise ValueError(f"Unknown enemy wait mode: {mode}")


def _validate_enemy_movement_mode(mode: str) -> str:
    if mode in SUPPORTED_ENEMY_MOVEMENT_MODES:
        return mode
    raise ValueError(f"Unknown enemy movement mode: {mode}")


def _evaluation_hero_policy_ids(
    requested: tuple[str, ...],
    default_policy_id: str,
) -> tuple[str, ...]:
    selected = requested or (default_policy_id,)
    return tuple(create_hero_policy(policy_id).policy_id for policy_id in selected)


def _route_encounter_ids(route_id: str) -> tuple[str, ...]:
    return _ROUTE_ENCOUNTERS[route_id]


def _primary_evaluation(
    evaluations: tuple[TrainingPolicyEvaluation, ...],
    *,
    scope_id: str,
    hero_policy_id: str,
) -> TrainingPolicyEvaluation:
    for evaluation in evaluations:
        if (
            evaluation.policy_scope_id == scope_id
            and evaluation.evaluation_hero_policy_id == hero_policy_id
        ):
            return evaluation
    for evaluation in evaluations:
        if evaluation.policy_scope_id == scope_id:
            return evaluation
    raise ValueError("Primary learned policy evaluation was not produced.")


def _learn_policy_model(
    definitions: GameDefinitions,
    episodes: tuple[EnemyDecisionEpisode, ...],
    scope_id: str,
) -> LearnedEnemyPolicyModel:
    global_weights = learn_linear_enemy_weights(episodes)
    return LearnedEnemyPolicyModel(
        scope_id=scope_id,
        definitions=definitions,
        global_weights=global_weights,
        per_encounter_weights={
            encounter_id: _learn_weights_from_records(
                record
                for episode in episodes
                if episode.encounter_id == encounter_id
                for record in episode.records
            )
            for encounter_id in sorted({episode.encounter_id for episode in episodes})
        },
        per_role_weights={
            role_id: _learn_weights_from_records(
                record
                for episode in episodes
                for record in episode.records
                if _record_role_id(definitions, record) == role_id
            )
            for role_id in sorted(
                {
                    _record_role_id(definitions, record)
                    for episode in episodes
                    for record in episode.records
                }
            )
        },
        boss_weights=_learn_weights_from_records(
            record
            for episode in episodes
            for record in episode.records
            if "boss" in _record_enemy_tags(definitions, record)
        ),
    )


def _apply_learned_weight_overrides(
    model: LearnedEnemyPolicyModel,
    overrides: Mapping[str, int],
) -> LearnedEnemyPolicyModel:
    return LearnedEnemyPolicyModel(
        scope_id=model.scope_id,
        definitions=model.definitions,
        global_weights=_merge_weights(model.global_weights, overrides),
        per_encounter_weights={
            encounter_id: _merge_weights(weights, overrides)
            for encounter_id, weights in model.per_encounter_weights.items()
        },
        per_role_weights={
            role_id: _merge_weights(weights, overrides)
            for role_id, weights in model.per_role_weights.items()
        },
        boss_weights=_merge_weights(model.boss_weights, overrides),
    )


def _merge_weights(
    base: Mapping[str, int],
    overrides: Mapping[str, int],
) -> dict[str, int]:
    merged = dict(base)
    merged.update({name: int(weight) for name, weight in overrides.items()})
    return merged


def _learn_weights_from_records(records: Iterable[EnemyDecisionRecord]) -> dict[str, int]:
    learned: dict[str, int] = {}
    for record in records:
        if record.action_reward <= 0:
            continue
        for feature_name, feature_value in record.chosen_features.items():
            if feature_value == 0:
                continue
            learned[feature_name] = learned.get(feature_name, 0) + (
                feature_value * record.action_reward
            )
    return learned


def _run_episode(
    definitions: GameDefinitions,
    encounter_id: str,
    seed: int,
    *,
    max_rounds: int,
    policy: EnemyDecisionPolicy | None = None,
    hero_policy_id: str = "naive",
    preset_id: str = "fresh",
    enemy_wait_mode: str = "none",
    enemy_movement_mode: str = "recovery_only",
) -> EnemyDecisionEpisode:
    company = create_new_company(definitions)
    _apply_preset(company, preset_id)
    combat = create_encounter_combat(company, definitions, encounter_id)
    hero_policy = create_hero_policy(hero_policy_id, encounter_id=encounter_id, seed=seed)
    return run_enemy_learning_episode(
        combat,
        definitions,
        GameRng(seed),
        policy=policy,
        max_rounds=max_rounds,
        encounter_id=encounter_id,
        encounter_name=encounter_id.replace("_", " ").title(),
        hero_policy=hero_policy,
        enemy_wait_mode=enemy_wait_mode,
        enemy_movement_mode=enemy_movement_mode,
    )


def _run_sequence(
    definitions: GameDefinitions,
    encounter_ids: tuple[str, ...],
    seed: int,
    *,
    route_id: str,
    max_rounds: int,
    policy_model: LearnedEnemyPolicyModel | None = None,
    hero_policy_id: str = "naive",
    preset_id: str = "fresh",
    enemy_wait_mode: str = "none",
    enemy_movement_mode: str = "recovery_only",
) -> RouteTrainingResult:
    company = create_new_company(definitions)
    _apply_preset(company, preset_id)
    episodes: list[EnemyDecisionEpisode] = []
    encounter_results: list[RouteEncounterResult] = []
    rng = GameRng(seed)
    failed_at_encounter_id: str | None = None
    for encounter_id in encounter_ids:
        hp_entering = _party_hp_total(company)
        effort_entering = _party_effort_total(company)
        combat = create_encounter_combat(company, definitions, encounter_id)
        hero_policy = create_hero_policy(hero_policy_id, encounter_id=encounter_id, seed=seed)
        episode = run_enemy_learning_episode(
            combat,
            definitions,
            rng,
            policy=policy_model.policy_for(encounter_id) if policy_model is not None else None,
            max_rounds=max_rounds,
            encounter_id=encounter_id,
            encounter_name=encounter_id.replace("_", " ").title(),
            hero_policy=hero_policy,
            enemy_wait_mode=enemy_wait_mode,
            enemy_movement_mode=enemy_movement_mode,
        )
        episodes.append(episode)
        sync_company_from_combat(company, combat.heroes, combat.party_formation)
        hp_leaving = _party_hp_total(company)
        effort_leaving = _party_effort_total(company)
        encounter_results.append(
            RouteEncounterResult(
                encounter_id=encounter_id,
                hp_entering=hp_entering,
                effort_entering=effort_entering,
                hp_leaving=hp_leaving,
                effort_leaving=effort_leaving,
                reward=episode.total_reward,
                final_victor=episode.final_victor,
                downs=episode.metrics.hero_downs,
                deaths=episode.metrics.hero_deaths,
                mortal_wounds=episode.metrics.mortal_wounds,
            )
        )
        if combat.is_defeat():
            failed_at_encounter_id = encounter_id
            break
    return RouteTrainingResult(
        route_id=route_id,
        seed=seed,
        episodes=tuple(episodes),
        encounters=tuple(encounter_results),
        completed=failed_at_encounter_id is None and len(episodes) == len(encounter_ids),
        failed_at_encounter_id=failed_at_encounter_id,
        final_hero_hp_total=_party_hp_total(company),
        final_hero_effort_total=_party_effort_total(company),
    )


def _apply_preset(company: CompanyState, preset_id: str) -> None:
    if preset_id == "fresh":
        return
    active = _active_heroes(company)
    if preset_id == "lightly_wounded":
        for hero in active:
            hero.hp = max(1, (hero.max_hp * 3) // 4)
    elif preset_id == "attrition":
        for hero in active:
            hero.hp = max(1, hero.max_hp // 2)
            hero.effort = max(0, hero.effort - 2)
    elif preset_id == "low_effort":
        for hero in active:
            hero.effort = min(hero.effort, 1)
    elif preset_id == "bad_formation":
        _apply_bad_formation(company)
    else:
        raise ValueError(f"Unknown preset id: {preset_id}")


def _active_heroes(company: CompanyState) -> list[HeroState]:
    by_id = {hero.hero_id: hero for hero in company.roster if hero.life_state != LifeState.DEAD}
    return [
        by_id[hero_id]
        for hero_id in company.active_party_slots.values()
        if hero_id is not None and hero_id in by_id
    ]


def _party_hp_total(company: CompanyState) -> int:
    return sum(hero.hp for hero in _active_heroes(company))


def _party_effort_total(company: CompanyState) -> int:
    return sum(hero.effort for hero in _active_heroes(company))


def _apply_bad_formation(company: CompanyState) -> None:
    swaps = (
        (FormationSlot.FRONT_LEFT, FormationSlot.BACK_LEFT),
        (FormationSlot.FRONT_RIGHT, FormationSlot.BACK_RIGHT),
    )
    by_id = {hero.hero_id: hero for hero in company.roster}
    for first, second in swaps:
        first_id = company.active_party_slots.get(first)
        second_id = company.active_party_slots.get(second)
        company.active_party_slots[first] = second_id
        company.active_party_slots[second] = first_id
        if first_id is not None and first_id in by_id:
            by_id[first_id].formation_slot = second
        if second_id is not None and second_id in by_id:
            by_id[second_id].formation_slot = first


def _encounter_breakdowns(
    encounter_ids: tuple[str, ...],
    heuristic_episodes: tuple[EnemyDecisionEpisode, ...],
    learned_episodes: tuple[EnemyDecisionEpisode, ...],
) -> tuple[EncounterTrainingBreakdown, ...]:
    return tuple(
        EncounterTrainingBreakdown(
            encounter_id=encounter_id,
            heuristic=_pressure_summary(
                tuple(
                    episode
                    for episode in heuristic_episodes
                    if episode.encounter_id == encounter_id
                )
            ),
            learned=_pressure_summary(
                tuple(
                    episode
                    for episode in learned_episodes
                    if episode.encounter_id == encounter_id
                )
            ),
        )
        for encounter_id in encounter_ids
    )


def _pressure_summary(
    episodes: tuple[EnemyDecisionEpisode, ...],
) -> PolicyPressureSummary:
    metrics = tuple(episode.metrics for episode in episodes)
    return PolicyPressureSummary(
        episode_count=len(episodes),
        average_reward=_average_reward(episodes),
        victor_counts=_victor_counts(episodes),
        average_rounds=_average_metric(metrics, "rounds_elapsed"),
        average_enemy_decisions=_average_metric(metrics, "enemy_decisions"),
        average_hero_hp_remaining=_average_metric(metrics, "final_hero_hp_total"),
        average_lowest_hero_hp_reached=_average_metric(metrics, "lowest_hero_hp_reached"),
        total_hero_damage=_sum_metric(metrics, "total_hero_damage"),
        hero_downs=_sum_metric(metrics, "hero_downs"),
        hero_deaths=_sum_metric(metrics, "hero_deaths"),
        mortal_wounds=_sum_metric(metrics, "mortal_wounds"),
        average_effort_remaining=_average_metric(metrics, "final_hero_effort_total"),
        healing_actions=_sum_metric(metrics, "healing_actions"),
        marks_applied=_sum_metric(metrics, "marks_applied"),
        marks_exploited=_sum_metric(metrics, "marks_exploited"),
        guard_actions=_sum_metric(metrics, "guard_actions"),
        forced_movement=_sum_metric(metrics, "forced_movement"),
        enemy_recovery_moves=_sum_metric(metrics, "enemy_recovery_moves"),
        enemy_proactive_moves=_sum_metric(metrics, "enemy_proactive_moves"),
        enemy_waits=_sum_metric(metrics, "enemy_waits"),
        waited_then_attacked_next_activation=_sum_metric(
            metrics,
            "waited_then_attacked_next_activation",
        ),
        waited_then_marked_hit=_sum_metric(metrics, "waited_then_marked_hit"),
        waited_then_payoff=_sum_metric(metrics, "waited_then_payoff"),
        waited_then_down=_sum_metric(metrics, "waited_then_down"),
        waited_then_no_payoff=_sum_metric(metrics, "waited_then_no_payoff"),
        waited_then_no_attack=_sum_metric(metrics, "waited_then_no_attack"),
        waits_without_payoff=_sum_metric(metrics, "waits_without_payoff"),
        move_then_attack_next_activation=_sum_metric(
            metrics,
            "move_then_attack_next_activation",
        ),
        move_then_marked_hit=_sum_metric(metrics, "move_then_marked_hit"),
        move_then_payoff=_sum_metric(metrics, "move_then_payoff"),
        move_then_down=_sum_metric(metrics, "move_then_down"),
        move_unlocks_future_skill=_sum_metric(metrics, "move_unlocks_future_skill"),
        move_into_marked_lane=_sum_metric(metrics, "move_into_marked_lane"),
        move_wasted_no_followup=_sum_metric(metrics, "move_wasted_no_followup"),
        moves_without_payoff=_sum_metric(metrics, "moves_without_payoff"),
        recovery_move_then_attack=_sum_metric(metrics, "recovery_move_then_attack"),
        recovery_move_then_payoff=_sum_metric(metrics, "recovery_move_then_payoff"),
        recovery_move_wasted=_sum_metric(metrics, "recovery_move_wasted"),
        recovery_moves_without_followup=_sum_metric(
            metrics,
            "recovery_moves_without_followup",
        ),
        all_enemy_wait_rounds=_sum_metric(metrics, "all_enemy_wait_rounds"),
        end_round_enemy_burst_damage=_sum_metric(metrics, "end_round_enemy_burst_damage"),
        end_round_enemy_burst_downs=_sum_metric(metrics, "end_round_enemy_burst_downs"),
        boss_sequence=_sum_boss_sequence(metrics),
        boss_targeting=_sum_boss_targeting(metrics),
        mark_flow=_sum_mark_flow(metrics),
        guard_flow=_sum_guard_flow(metrics),
    )


def _sum_boss_sequence(metrics: tuple[EnemyPressureMetrics, ...]) -> BossSequenceMetrics:
    return BossSequenceMetrics(
        grab_uses=sum(metric.boss_sequence.grab_uses for metric in metrics),
        bite_uses=sum(metric.boss_sequence.bite_uses for metric in metrics),
        grab_damage=sum(metric.boss_sequence.grab_damage for metric in metrics),
        bite_damage=sum(metric.boss_sequence.bite_damage for metric in metrics),
        grab_to_bite_same_target=sum(
            metric.boss_sequence.grab_to_bite_same_target for metric in metrics
        ),
        grab_to_bite_any_target=sum(
            metric.boss_sequence.grab_to_bite_any_target for metric in metrics
        ),
        grabbed_target_escaped_before_bite=sum(
            metric.boss_sequence.grabbed_target_escaped_before_bite for metric in metrics
        ),
        grabbed_target_remained_in_bite_range=sum(
            metric.boss_sequence.grabbed_target_remained_in_bite_range for metric in metrics
        ),
        bite_hit_dragged_target=sum(
            metric.boss_sequence.bite_hit_dragged_target for metric in metrics
        ),
        bite_hit_frontliner=sum(metric.boss_sequence.bite_hit_frontliner for metric in metrics),
        bite_downs=sum(metric.boss_sequence.bite_downs for metric in metrics),
        boss_killed_before_first_bite=sum(
            metric.boss_sequence.boss_killed_before_first_bite for metric in metrics
        ),
        boss_killed_after_grab_before_bite=sum(
            metric.boss_sequence.boss_killed_after_grab_before_bite for metric in metrics
        ),
        boss_actions_before_death=sum(
            metric.boss_sequence.boss_actions_before_death for metric in metrics
        ),
        boss_rounds_survived=sum(metric.boss_sequence.boss_rounds_survived for metric in metrics),
        bone_soldier_guarded_boss=sum(
            metric.boss_sequence.bone_soldier_guarded_boss for metric in metrics
        ),
    )


def _sum_boss_targeting(metrics: tuple[EnemyPressureMetrics, ...]) -> BossTargetingMetrics:
    return BossTargetingMetrics(
        grab_target_classes=_sum_count_mappings(
            metric.boss_targeting.grab_target_classes for metric in metrics
        ),
        bite_target_classes=_sum_count_mappings(
            metric.boss_targeting.bite_target_classes for metric in metrics
        ),
        support_grabs=sum(metric.boss_targeting.support_grabs for metric in metrics),
        support_grabs_with_effort=sum(
            metric.boss_targeting.support_grabs_with_effort for metric in metrics
        ),
        support_grabs_not_acted=sum(
            metric.boss_targeting.support_grabs_not_acted for metric in metrics
        ),
        support_grab_to_bite_same_target=sum(
            metric.boss_targeting.support_grab_to_bite_same_target for metric in metrics
        ),
        support_grab_downs=sum(metric.boss_targeting.support_grab_downs for metric in metrics),
        direct_front_bites=sum(metric.boss_targeting.direct_front_bites for metric in metrics),
    )


def _sum_mark_flow(metrics: tuple[EnemyPressureMetrics, ...]) -> MarkFlowMetrics:
    return MarkFlowMetrics(
        marks_applied=sum(metric.mark_flow.marks_applied for metric in metrics),
        marks_refreshed=sum(metric.mark_flow.marks_refreshed for metric in metrics),
        marks_applied_to_already_marked=sum(
            metric.mark_flow.marks_applied_to_already_marked for metric in metrics
        ),
        exploited_by_enemy_hit=sum(metric.mark_flow.exploited_by_enemy_hit for metric in metrics),
        multi_hit_focus=sum(metric.mark_flow.multi_hit_focus for metric in metrics),
        vulnerable_payoffs=sum(metric.mark_flow.vulnerable_payoffs for metric in metrics),
        total_damage_to_marked=sum(metric.mark_flow.total_damage_to_marked for metric in metrics),
        marked_downs=sum(metric.mark_flow.marked_downs for metric in metrics),
        marked_deaths=sum(metric.mark_flow.marked_deaths for metric in metrics),
        marked_mortal_wounds=sum(metric.mark_flow.marked_mortal_wounds for metric in metrics),
        mark_ally_reach_total=sum(metric.mark_flow.mark_ally_reach_total for metric in metrics),
        mark_ally_reach_count=sum(metric.mark_flow.mark_ally_reach_count for metric in metrics),
        best_focus_marks=sum(metric.mark_flow.best_focus_marks for metric in metrics),
        attacks_against_marked_when_legal=sum(
            metric.mark_flow.attacks_against_marked_when_legal for metric in metrics
        ),
        ignored_marked_legal_attacks=sum(
            metric.mark_flow.ignored_marked_legal_attacks for metric in metrics
        ),
    )


def _sum_guard_flow(metrics: tuple[EnemyPressureMetrics, ...]) -> GuardFlowMetrics:
    return GuardFlowMetrics(
        guard_uses=sum(metric.guard_flow.guard_uses for metric in metrics),
        dead_guard_uses=sum(metric.guard_flow.dead_guard_uses for metric in metrics),
        guard_targets=_sum_count_mappings(metric.guard_flow.guard_targets for metric in metrics),
        guard_targets_by_enemy_id=_sum_count_mappings(
            metric.guard_flow.guard_targets_by_enemy_id for metric in metrics
        ),
        guard_damage_blocked=sum(
            metric.guard_flow.guard_damage_blocked for metric in metrics
        ),
        guarded_ally_survived_to_next_activation=sum(
            metric.guard_flow.guarded_ally_survived_to_next_activation
            for metric in metrics
        ),
        guarded_ally_acted_after_guard=sum(
            metric.guard_flow.guarded_ally_acted_after_guard for metric in metrics
        ),
        guarded_ally_used_payoff_after_guard=sum(
            metric.guard_flow.guarded_ally_used_payoff_after_guard for metric in metrics
        ),
        guarded_ally_downs_after_guard=sum(
            metric.guard_flow.guarded_ally_downs_after_guard for metric in metrics
        ),
        guard_expired_or_consumed_without_payoff=sum(
            metric.guard_flow.guard_expired_or_consumed_without_payoff
            for metric in metrics
        ),
        guard_wasted_no_followup=sum(
            metric.guard_flow.guard_wasted_no_followup for metric in metrics
        ),
    )


def _flatten_route_episodes(
    route_results: tuple[RouteTrainingResult, ...],
) -> tuple[EnemyDecisionEpisode, ...]:
    return tuple(episode for result in route_results for episode in result.episodes)


def _route_pressure_summary(
    route_results: tuple[RouteTrainingResult, ...],
) -> RoutePressureSummary:
    failed_at_counts = Counter(
        result.failed_at_encounter_id
        for result in route_results
        if result.failed_at_encounter_id is not None
    )
    cave_entries = tuple(
        encounter
        for result in route_results
        for encounter in result.encounters
        if encounter.encounter_id == "cave_mini_boss"
    )
    route_count = len(route_results)
    return RoutePressureSummary(
        route_count=route_count,
        completed_count=sum(1 for result in route_results if result.completed),
        failed_at_counts=dict(failed_at_counts),
        average_reward=_average_route_value(route_results, "total_reward"),
        average_final_hero_hp=_average_route_value(route_results, "final_hero_hp_total"),
        average_final_hero_effort=_average_route_value(
            route_results,
            "final_hero_effort_total",
        ),
        average_downs=_average_route_value(route_results, "total_downs"),
        average_deaths=_average_route_value(route_results, "total_deaths"),
        average_mortal_wounds=_average_route_value(route_results, "total_mortal_wounds"),
        average_hp_entering_cave_mini_boss=_average_encounter_value(
            cave_entries,
            "hp_entering",
        ),
        average_hp_leaving_cave_mini_boss=_average_encounter_value(
            cave_entries,
            "hp_leaving",
        ),
    )


def _average_route_value(
    route_results: tuple[RouteTrainingResult, ...],
    field_name: str,
) -> float:
    if not route_results:
        return 0.0
    return sum(int(getattr(result, field_name)) for result in route_results) / len(
        route_results
    )


def _average_encounter_value(
    encounters: tuple[RouteEncounterResult, ...],
    field_name: str,
) -> float:
    if not encounters:
        return 0.0
    return sum(int(getattr(encounter, field_name)) for encounter in encounters) / len(
        encounters
    )


def _record_role_id(definitions: GameDefinitions, record: EnemyDecisionRecord) -> str:
    enemy_class_id = getattr(record, "enemy_class_id", "")
    return _enemy_role_id(definitions, enemy_class_id)


def _record_enemy_tags(
    definitions: GameDefinitions,
    record: EnemyDecisionRecord,
) -> frozenset[str]:
    enemy_tags = getattr(record, "enemy_tags", ())
    if enemy_tags:
        return frozenset(enemy_tags)
    enemy_class_id = getattr(record, "enemy_class_id", "")
    return _enemy_tags(definitions, enemy_class_id)


def _enemy_role_id(definitions: GameDefinitions, enemy_class_id: str) -> str:
    tags = _enemy_tags(definitions, enemy_class_id)
    for role_id in ("boss", "bandit", "wolf", "horror", "maze", "beast", "undead", "human"):
        if role_id in tags:
            return role_id
    return enemy_class_id or "unknown"


def _enemy_tags(definitions: GameDefinitions, enemy_class_id: str) -> frozenset[str]:
    enemy_definition = definitions.enemies.get(enemy_class_id)
    if enemy_definition is None:
        return frozenset()
    return frozenset(enemy_definition.tags)


def _average_metric(metrics: tuple[EnemyPressureMetrics, ...], field_name: str) -> float:
    if not metrics:
        return 0.0
    return sum(int(getattr(metric, field_name)) for metric in metrics) / len(metrics)


def _sum_metric(metrics: tuple[EnemyPressureMetrics, ...], field_name: str) -> int:
    return sum(int(getattr(metric, field_name)) for metric in metrics)


def _sum_count_mappings(mappings: Iterable[Mapping[str, int]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for mapping in mappings:
        counts.update(mapping)
    return dict(counts)


def _average_reward(episodes: tuple[EnemyDecisionEpisode, ...]) -> float:
    if not episodes:
        return 0.0
    return sum(episode.total_reward for episode in episodes) / len(episodes)


def _average_records(episodes: tuple[EnemyDecisionEpisode, ...]) -> float:
    if not episodes:
        return 0.0
    return sum(len(episode.records) for episode in episodes) / len(episodes)


def _victor_counts(episodes: tuple[EnemyDecisionEpisode, ...]) -> dict[str, int]:
    return dict(Counter(episode.final_victor for episode in episodes))


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _format_count_mapping(counts: Mapping[str, int], *, limit: int = 4) -> str:
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shown = ordered[:limit]
    text = ", ".join(f"{key}={value}" for key, value in shown)
    remaining = sum(value for _, value in ordered[limit:])
    if remaining:
        text = f"{text}, other={remaining}"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
