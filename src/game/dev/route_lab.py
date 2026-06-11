"""Dev-only route and generated Maze evaluation helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from game.campaign.company import (
    CompanyState,
    GeneratedDungeonState,
    MazeRecipe,
    create_new_company,
)
from game.campaign.roster import sync_company_from_combat
from game.combat.enemy_learning import (
    EnemyDecisionEpisode,
    create_hero_policy,
    run_enemy_learning_episode,
)
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.dev.train_enemy_ai import (
    SUPPORTED_PRESET_IDS,
    SUPPORTED_ROUTE_IDS,
    RoutePressureSummary,
    TrainingRunConfig,
    TrainingRunSummary,
    _apply_preset,
    _party_effort_total,
    _party_hp_total,
    run_training_harness,
)
from game.expedition.cave import create_encounter_combat
from game.expedition.generated_maze import generate_maze_breach_route
from game.expedition.maze_director import (
    MAZE_PRESSURE_PROFILES,
    choose_maze_recipe,
    recipe_from_profile,
)

SUPPORTED_ROUTE_ENVELOPE_IDS = (
    "critical_path",
    "optional_pressure_path",
    "generated_maze_scout",
    "generated_maze_hunt",
)
SUPPORTED_GENERATED_ROUTE_STRATEGIES = ("mainline", "take_reward_branch", "take_all_optional")


@dataclass(frozen=True)
class RouteEnvelope:
    envelope_id: str
    completion_min: float
    completion_max: float
    final_hp_min: float = 0.0
    final_hp_max: float = 999.0
    boss_entry_hp_min: float = 0.0
    boss_entry_hp_max: float = 999.0
    boss_exit_hp_min: float = 0.0
    boss_exit_hp_max: float = 999.0
    allow_early_failures: bool = False


@dataclass(frozen=True)
class RouteEnvelopeScore:
    envelope_id: str
    status: str
    score: int
    warnings: tuple[str, ...] = ()
    metric_values: Mapping[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteLabConfig:
    route_id: str
    seeds: int = 50
    max_rounds: int = 20
    hero_policy_id: str = "mixed"
    preset_id: str = "fresh"
    envelope_id: str = ""
    enemy_wait_mode: str = "none"
    enemy_movement_mode: str = "recovery_only"
    definitions: GameDefinitions | None = None


@dataclass(frozen=True)
class RouteLabSummary:
    route_id: str
    seed_count: int
    hero_policy_id: str
    preset_id: str
    training_summary: TrainingRunSummary
    envelope_score: RouteEnvelopeScore


@dataclass(frozen=True)
class GeneratedRouteEncounterResult:
    encounter_id: str
    hp_entering: int
    effort_entering: int
    hp_leaving: int
    effort_leaving: int
    final_victor: str
    reward: int


@dataclass(frozen=True)
class GeneratedRouteRunResult:
    seed: int
    route: GeneratedDungeonState
    strategy_id: str
    episodes: tuple[EnemyDecisionEpisode, ...]
    encounters: tuple[GeneratedRouteEncounterResult, ...]
    completed: bool
    failed_at_encounter_id: str | None
    final_hero_hp_total: int
    final_hero_effort_total: int


@dataclass(frozen=True)
class GeneratedRouteLabConfig:
    breach_id: str = "shallow_cave_breach"
    seeds: int = 50
    max_rounds: int = 20
    hero_policy_id: str = "mixed"
    preset_id: str = "fresh"
    strategy_id: str = "mainline"
    pressure_profile_id: str = ""
    envelope_id: str = ""
    definitions: GameDefinitions | None = None


@dataclass(frozen=True)
class GeneratedRouteLabSummary:
    breach_id: str
    seed_count: int
    hero_policy_id: str
    preset_id: str
    strategy_id: str
    pressure_profile_id: str
    runs: tuple[GeneratedRouteRunResult, ...]
    envelope_score: RouteEnvelopeScore


ROUTE_ENVELOPES: Mapping[str, RouteEnvelope] = {
    "critical_path": RouteEnvelope(
        envelope_id="critical_path",
        completion_min=0.55,
        completion_max=0.98,
        final_hp_min=4,
        boss_entry_hp_min=8,
        boss_entry_hp_max=30,
        boss_exit_hp_max=28,
        allow_early_failures=False,
    ),
    "optional_pressure_path": RouteEnvelope(
        envelope_id="optional_pressure_path",
        completion_min=0.05,
        completion_max=0.70,
        final_hp_min=0,
        boss_entry_hp_min=1,
        boss_entry_hp_max=26,
        allow_early_failures=True,
    ),
    "generated_maze_scout": RouteEnvelope(
        envelope_id="generated_maze_scout",
        completion_min=0.40,
        completion_max=0.95,
        final_hp_min=2,
        allow_early_failures=False,
    ),
    "generated_maze_hunt": RouteEnvelope(
        envelope_id="generated_maze_hunt",
        completion_min=0.15,
        completion_max=0.80,
        final_hp_min=0,
        allow_early_failures=True,
    ),
}


def run_route_lab(config: RouteLabConfig) -> RouteLabSummary:
    definitions = config.definitions or load_game_definitions()
    route_id = _validate_supported(config.route_id, SUPPORTED_ROUTE_IDS, "route")
    preset_id = _validate_supported(config.preset_id, SUPPORTED_PRESET_IDS, "preset")
    envelope_id = config.envelope_id or _default_envelope_for_route(route_id)
    training_summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            route_id=route_id,
            seeds=config.seeds,
            max_rounds=config.max_rounds,
            hero_policy_id=config.hero_policy_id,
            preset_id=preset_id,
            enemy_wait_mode=config.enemy_wait_mode,
            enemy_movement_mode=config.enemy_movement_mode,
        )
    )
    return RouteLabSummary(
        route_id=route_id,
        seed_count=config.seeds,
        hero_policy_id=training_summary.hero_policy_id,
        preset_id=preset_id,
        training_summary=training_summary,
        envelope_score=score_route_envelope(
            training_summary.learned_route_summary,
            envelope_id=envelope_id,
        ),
    )


def run_generated_route_lab(config: GeneratedRouteLabConfig) -> GeneratedRouteLabSummary:
    definitions = config.definitions or load_game_definitions()
    preset_id = _validate_supported(config.preset_id, SUPPORTED_PRESET_IDS, "preset")
    strategy_id = _validate_supported(
        config.strategy_id,
        SUPPORTED_GENERATED_ROUTE_STRATEGIES,
        "generated route strategy",
    )
    envelope_id = config.envelope_id or _default_generated_envelope(
        config.pressure_profile_id
    )
    runs = tuple(
        _run_generated_route(
            definitions,
            seed,
            breach_id=config.breach_id,
            max_rounds=config.max_rounds,
            hero_policy_id=config.hero_policy_id,
            preset_id=preset_id,
            strategy_id=strategy_id,
            pressure_profile_id=config.pressure_profile_id,
        )
        for seed in range(1, max(0, config.seeds) + 1)
    )
    return GeneratedRouteLabSummary(
        breach_id=config.breach_id,
        seed_count=len(runs),
        hero_policy_id=config.hero_policy_id,
        preset_id=preset_id,
        strategy_id=strategy_id,
        pressure_profile_id=config.pressure_profile_id,
        runs=runs,
        envelope_score=score_generated_route_envelope(runs, envelope_id=envelope_id),
    )


def score_route_envelope(
    route_summary: RoutePressureSummary,
    *,
    envelope_id: str,
) -> RouteEnvelopeScore:
    envelope = _envelope(envelope_id)
    completion_rate = (
        route_summary.completed_count / route_summary.route_count
        if route_summary.route_count
        else 0.0
    )
    metrics: dict[str, float | int | str] = {
        "completion_rate": completion_rate,
        "routes": route_summary.route_count,
        "completed": route_summary.completed_count,
        "final_hp": route_summary.average_final_hero_hp,
        "final_effort": route_summary.average_final_hero_effort,
        "boss_entry_hp": route_summary.average_hp_entering_cave_mini_boss,
        "boss_exit_hp": route_summary.average_hp_leaving_cave_mini_boss,
    }
    return _score_envelope_values(
        envelope,
        metrics,
        cast(Mapping[str | None, int], route_summary.failed_at_counts),
    )


def score_generated_route_envelope(
    runs: Sequence[GeneratedRouteRunResult],
    *,
    envelope_id: str,
) -> RouteEnvelopeScore:
    envelope = _envelope(envelope_id)
    route_count = len(runs)
    completion_rate = sum(1 for run in runs if run.completed) / route_count if route_count else 0.0
    final_hp = _average(run.final_hero_hp_total for run in runs)
    final_effort = _average(run.final_hero_effort_total for run in runs)
    failed_at_counts = Counter(
        run.failed_at_encounter_id for run in runs if run.failed_at_encounter_id is not None
    )
    metrics: dict[str, float | int | str] = {
        "completion_rate": completion_rate,
        "routes": route_count,
        "completed": sum(1 for run in runs if run.completed),
        "final_hp": final_hp,
        "final_effort": final_effort,
        "average_route_length": _average(
            len(_main_route_nodes(run.route)) for run in runs
        ),
        "average_combat_count": _average(len(run.encounters) for run in runs),
        "repeated_encounter_patterns": _repeated_encounter_patterns(runs),
    }
    return _score_envelope_values(
        envelope,
        metrics,
        cast(Mapping[str | None, int], dict(failed_at_counts)),
    )


def format_route_lab_summary(summary: RouteLabSummary) -> str:
    route = summary.training_summary.learned_route_summary
    lines = [
        "Route Lab",
        f"Route: {summary.route_id}",
        f"Seeds: {summary.seed_count}",
        f"Hero policy: {summary.hero_policy_id}",
        f"Preset: {summary.preset_id}",
        (
            "Completion: "
            f"{route.completed_count}/{route.route_count}; "
            f"failed {_format_counts(cast(Mapping[str | None, int], route.failed_at_counts))}"
        ),
        (
            "Final condition: "
            f"HP {route.average_final_hero_hp:.1f}; "
            f"Effort {route.average_final_hero_effort:.1f}; "
            f"downs {route.average_downs:.1f}; "
            f"deaths {route.average_deaths:.1f}"
        ),
        _format_envelope_score(summary.envelope_score),
    ]
    return "\n".join(lines)


def format_generated_route_lab_summary(summary: GeneratedRouteLabSummary) -> str:
    completed = sum(1 for run in summary.runs if run.completed)
    failed = Counter(
        run.failed_at_encounter_id for run in summary.runs if run.failed_at_encounter_id
    )
    route_lengths = [len(_main_route_nodes(run.route)) for run in summary.runs]
    combat_counts = [len(run.encounters) for run in summary.runs]
    lines = [
        "Generated Route Lab",
        f"Breach: {summary.breach_id}",
        f"Seeds: {summary.seed_count}",
        f"Hero policy: {summary.hero_policy_id}",
        f"Preset: {summary.preset_id}",
        f"Strategy: {summary.strategy_id}",
        f"Pressure profile: {summary.pressure_profile_id or 'director'}",
        "Completion: "
        f"{completed}/{len(summary.runs)}; "
        f"failed {_format_counts(cast(Mapping[str | None, int], dict(failed)))}",
        (
            "Route shape: "
            f"rooms {_average(route_lengths):.1f}; "
            f"combats {_average(combat_counts):.1f}"
        ),
        _format_envelope_score(summary.envelope_score),
    ]
    return "\n".join(lines)


def _run_generated_route(
    definitions: GameDefinitions,
    seed: int,
    *,
    breach_id: str,
    max_rounds: int,
    hero_policy_id: str,
    preset_id: str,
    strategy_id: str,
    pressure_profile_id: str,
) -> GeneratedRouteRunResult:
    company = create_new_company(definitions)
    _apply_preset(company, preset_id)
    rng = GameRng(seed)
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id=breach_id,
        return_node_id=breach_id,
        rng=rng,
        seed=seed,
        recipe=_generated_recipe(company, breach_id, seed, pressure_profile_id),
    )
    episodes: list[EnemyDecisionEpisode] = []
    encounters: list[GeneratedRouteEncounterResult] = []
    failed_at: str | None = None
    for encounter_id in _encounter_sequence_for_strategy(route, strategy_id):
        hp_entering = _party_hp_total(company)
        effort_entering = _party_effort_total(company)
        combat = create_encounter_combat(company, definitions, encounter_id)
        episode = run_enemy_learning_episode(
            combat,
            definitions,
            rng,
            max_rounds=max_rounds,
            encounter_id=encounter_id,
            encounter_name=encounter_id.replace("_", " ").title(),
            hero_policy=create_hero_policy(hero_policy_id, encounter_id=encounter_id, seed=seed),
        )
        episodes.append(episode)
        sync_company_from_combat(company, combat.heroes, combat.party_formation)
        encounters.append(
            GeneratedRouteEncounterResult(
                encounter_id=encounter_id,
                hp_entering=hp_entering,
                effort_entering=effort_entering,
                hp_leaving=_party_hp_total(company),
                effort_leaving=_party_effort_total(company),
                final_victor=episode.final_victor,
                reward=episode.total_reward,
            )
        )
        if combat.is_defeat():
            failed_at = encounter_id
            break
    return GeneratedRouteRunResult(
        seed=seed,
        route=route,
        strategy_id=strategy_id,
        episodes=tuple(episodes),
        encounters=tuple(encounters),
        completed=failed_at is None,
        failed_at_encounter_id=failed_at,
        final_hero_hp_total=_party_hp_total(company),
        final_hero_effort_total=_party_effort_total(company),
    )


def _generated_recipe(
    company: CompanyState,
    breach_id: str,
    seed: int,
    pressure_profile_id: str,
) -> MazeRecipe:
    if pressure_profile_id:
        for profile in MAZE_PRESSURE_PROFILES:
            if profile.pressure_id == pressure_profile_id:
                return recipe_from_profile(profile, GameRng(seed))
        raise ValueError(f"Unknown Maze pressure profile: {pressure_profile_id}")
    return choose_maze_recipe(
        company,
        source_node_id=breach_id,
        run_number=1,
        rng=GameRng(seed),
    )


def _encounter_sequence_for_strategy(
    route: GeneratedDungeonState,
    strategy_id: str,
) -> tuple[str, ...]:
    nodes = _main_route_nodes(route)
    selected = [node.encounter for node in nodes if node.encounter]
    if strategy_id == "take_all_optional":
        selected.extend(
            node.encounter
            for node in sorted(route.nodes, key=lambda node: node.id)
            if node.encounter and node not in nodes
        )
    return tuple(encounter_id for encounter_id in selected if encounter_id)


def _main_route_nodes(route: GeneratedDungeonState):
    prefix = f"{route.run_id}_room_"
    return tuple(
        sorted(
            (node for node in route.nodes if node.id.startswith(prefix)),
            key=lambda node: int(node.id.removeprefix(prefix)),
        )
    )


def _score_envelope_values(
    envelope: RouteEnvelope,
    metrics: Mapping[str, float | int | str],
    failed_at_counts: Mapping[str | None, int],
) -> RouteEnvelopeScore:
    score = 100
    warnings: list[str] = []
    severe = False
    completion_rate = float(metrics.get("completion_rate", 0.0))
    if completion_rate < envelope.completion_min:
        score -= 25
        warnings.append("completion below target band")
        if completion_rate < envelope.completion_min * 0.5:
            severe = True
    elif completion_rate > envelope.completion_max:
        score -= 15
        warnings.append("completion above target band")
    final_hp = float(metrics.get("final_hp", 0.0))
    if final_hp < envelope.final_hp_min:
        score -= 15
        warnings.append("final HP below target band")
    elif final_hp > envelope.final_hp_max:
        score -= 10
        warnings.append("final HP above target band")
    boss_entry = float(metrics.get("boss_entry_hp", 0.0))
    if boss_entry:
        if boss_entry < envelope.boss_entry_hp_min:
            score -= 15
            warnings.append("party reaches boss already doomed")
            if boss_entry < envelope.boss_entry_hp_min * 0.5:
                severe = True
        elif boss_entry > envelope.boss_entry_hp_max:
            score -= 10
            warnings.append("party reaches boss too healthy")
    if not envelope.allow_early_failures:
        early_failures = sum(
            count
            for encounter_id, count in failed_at_counts.items()
            if encounter_id not in {None, "cave_mini_boss", "generated_maze_hunt"}
        )
        if early_failures:
            score -= 20
            warnings.append("early failures observed")
            if early_failures >= max(1, int(float(metrics.get("routes", 0)) * 0.25)):
                severe = True
    if severe or score < 55:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"
    return RouteEnvelopeScore(
        envelope_id=envelope.envelope_id,
        status=status,
        score=max(0, score),
        warnings=tuple(warnings),
        metric_values=dict(metrics),
    )


def _envelope(envelope_id: str) -> RouteEnvelope:
    if envelope_id in ROUTE_ENVELOPES:
        return ROUTE_ENVELOPES[envelope_id]
    raise ValueError(f"Unknown route envelope: {envelope_id}")


def _default_envelope_for_route(route_id: str) -> str:
    if route_id == "opening_pressure_path":
        return "optional_pressure_path"
    return "critical_path"


def _default_generated_envelope(pressure_profile_id: str) -> str:
    if pressure_profile_id == "marked_hunt":
        return "generated_maze_hunt"
    return "generated_maze_scout"


def _validate_supported(value: str, supported: Sequence[str], label: str) -> str:
    if value in supported:
        return value
    raise ValueError(f"Unknown {label}: {value}")


def _average(values) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _repeated_encounter_patterns(runs: Sequence[GeneratedRouteRunResult]) -> int:
    patterns = Counter(tuple(result.encounter_id for result in run.encounters) for run in runs)
    return sum(count - 1 for count in patterns.values() if count > 1)


def _format_envelope_score(score: RouteEnvelopeScore) -> str:
    warnings = "; ".join(score.warnings) if score.warnings else "in envelope"
    return f"Envelope {score.envelope_id}: {score.status} score={score.score}; {warnings}"


def _format_counts(counts: Mapping[str | None, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


__all__ = [
    "GeneratedRouteLabConfig",
    "GeneratedRouteLabSummary",
    "RouteEnvelope",
    "RouteEnvelopeScore",
    "RouteLabConfig",
    "RouteLabSummary",
    "SUPPORTED_GENERATED_ROUTE_STRATEGIES",
    "SUPPORTED_ROUTE_ENVELOPE_IDS",
    "format_generated_route_lab_summary",
    "format_route_lab_summary",
    "run_generated_route_lab",
    "run_route_lab",
    "score_generated_route_envelope",
    "score_route_envelope",
]
