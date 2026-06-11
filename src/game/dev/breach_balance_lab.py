"""Dev-only automatic balancing for generated Maze breach fights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from game.combat.formation import FormationSlot
from game.content.definitions import GameDefinitions
from game.data.loaders import default_data_dir, load_game_definitions, validate_references
from game.data.schemas import (
    EncounterDefinition,
    EncounterEnemyDefinition,
    EnemyDefinition,
    SkillDefinition,
)
from game.dev.route_lab import (
    GeneratedRouteLabConfig,
    GeneratedRouteLabSummary,
    run_generated_route_lab,
)

SCOUT_COMPLETION_MIN = 0.30
SCOUT_COMPLETION_MAX = 0.80
HUNT_COMPLETION_MIN = 0.10
HUNT_COMPLETION_MAX = 0.60
MIN_TACTICAL_PRESSURE = 1
MIN_SCOUT_PATTERNS = 2
MIN_HUNT_COMBATS = 2
BALANCED_ENCOUNTER_IDS = ("maze_depth_1", "generated_maze_hunt")


@dataclass(frozen=True)
class EncounterEdit:
    encounter_id: str
    enemies: tuple[EncounterEnemyDefinition, ...]


@dataclass(frozen=True)
class EnemyStatEdit:
    enemy_id: str
    updates: Mapping[str, int]


@dataclass(frozen=True)
class SkillEdit:
    skill_id: str
    definition: SkillDefinition


@dataclass(frozen=True)
class EnemyEdit:
    enemy_id: str
    definition: EnemyDefinition


@dataclass(frozen=True)
class BreachFightCandidate:
    candidate_id: str
    description: str
    skill_edits: tuple[SkillEdit, ...] = ()
    enemy_edits: tuple[EnemyEdit, ...] = ()
    encounter_edits: tuple[EncounterEdit, ...] = ()
    enemy_stat_edits: tuple[EnemyStatEdit, ...] = ()


@dataclass(frozen=True)
class BreachFightPromotionGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class BreachFightCandidateResult:
    candidate: BreachFightCandidate
    scout_summary: GeneratedRouteLabSummary
    hunt_summary: GeneratedRouteLabSummary
    gates: tuple[BreachFightPromotionGate, ...]
    score: int

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


@dataclass(frozen=True)
class BreachFightBalanceConfig:
    seeds: int = 50
    max_rounds: int = 20
    hero_policy_id: str = "mixed"
    preset_id: str = "attrition"
    strategy_id: str = "take_all_optional"
    dry_run: bool = False
    apply_best: bool = True
    data_dir: Path | None = None
    definitions: GameDefinitions | None = None
    candidates: tuple[BreachFightCandidate, ...] = ()


@dataclass(frozen=True)
class BreachFightBalanceResult:
    baseline_scout: GeneratedRouteLabSummary
    baseline_hunt: GeneratedRouteLabSummary
    results: tuple[BreachFightCandidateResult, ...]
    selected: BreachFightCandidateResult | None
    applied: bool
    files_written: tuple[Path, ...] = ()


def run_breach_fight_balance(
    config: BreachFightBalanceConfig,
) -> BreachFightBalanceResult:
    definitions = config.definitions or load_game_definitions(config.data_dir)
    baseline_scout = _run_generated_summary(
        definitions,
        config,
        pressure_profile_id="breach_probe",
    )
    baseline_hunt = _run_generated_summary(
        definitions,
        config,
        pressure_profile_id="marked_hunt",
    )
    results = tuple(
        sorted(
            (
                _evaluate_candidate(definitions, config, candidate)
                for candidate in (config.candidates or _default_candidates())
            ),
            key=lambda result: (
                not result.passed,
                -result.score,
                result.candidate.candidate_id,
            ),
        )
    )
    selected = next((result for result in results if result.passed), None)
    applied = False
    files_written: tuple[Path, ...] = ()
    if selected is not None and config.apply_best and not config.dry_run:
        files_written = _apply_candidate_to_yaml(
            selected.candidate,
            data_dir=config.data_dir or default_data_dir(),
        )
        validate_references(load_game_definitions(config.data_dir))
        applied = True
    return BreachFightBalanceResult(
        baseline_scout=baseline_scout,
        baseline_hunt=baseline_hunt,
        results=results,
        selected=selected,
        applied=applied,
        files_written=files_written,
    )


def format_breach_fight_balance_result(result: BreachFightBalanceResult) -> str:
    lines = [
        "Breach Fight Balance Lab",
        f"Baseline scout: {_summary_line(result.baseline_scout)}",
        f"Baseline hunt: {_summary_line(result.baseline_hunt)}",
        "Ranked candidates:",
    ]
    for index, candidate_result in enumerate(result.results[:8], start=1):
        status = "PASS" if candidate_result.passed else "FAIL"
        gate_text = "; ".join(
            f"{gate.name}={'ok' if gate.passed else 'no'}" for gate in candidate_result.gates
        )
        lines.append(
            f"  {index}. {candidate_result.candidate.candidate_id}: "
            f"{status} score={candidate_result.score}; {gate_text}"
        )
        lines.append(f"     scout: {_summary_line(candidate_result.scout_summary)}")
        lines.append(f"     hunt: {_summary_line(candidate_result.hunt_summary)}")
    if result.selected is None:
        lines.append("Selected: none - no candidate passed hard gates.")
    else:
        lines.append(f"Selected: {result.selected.candidate.candidate_id}")
        lines.append(f"Applied: {'yes' if result.applied else 'no'}")
        if result.files_written:
            lines.append(
                "Files written: "
                + ", ".join(str(path) for path in result.files_written)
            )
    return "\n".join(lines)


def _evaluate_candidate(
    definitions: GameDefinitions,
    config: BreachFightBalanceConfig,
    candidate: BreachFightCandidate,
) -> BreachFightCandidateResult:
    candidate_definitions = _definitions_for_candidate(definitions, candidate)
    validate_references(candidate_definitions)
    scout = _run_generated_summary(
        candidate_definitions,
        config,
        pressure_profile_id="breach_probe",
    )
    hunt = _run_generated_summary(
        candidate_definitions,
        config,
        pressure_profile_id="marked_hunt",
    )
    gates = _promotion_gates(scout, hunt)
    score = _candidate_score(scout, hunt, gates, candidate)
    return BreachFightCandidateResult(
        candidate=candidate,
        scout_summary=scout,
        hunt_summary=hunt,
        gates=gates,
        score=score,
    )


def _definitions_for_candidate(
    definitions: GameDefinitions,
    candidate: BreachFightCandidate,
) -> GameDefinitions:
    encounters = dict(definitions.encounters)
    skills = dict(definitions.skills)
    for skill_edit in candidate.skill_edits:
        if skill_edit.skill_id != skill_edit.definition.id:
            raise ValueError(
                f"Skill edit key does not match definition: {skill_edit.skill_id}"
            )
        skills[skill_edit.skill_id] = skill_edit.definition
    enemies = dict(definitions.enemies)
    for enemy_edit in candidate.enemy_edits:
        if enemy_edit.enemy_id != enemy_edit.definition.id:
            raise ValueError(
                f"Enemy edit key does not match definition: {enemy_edit.enemy_id}"
            )
        enemies[enemy_edit.enemy_id] = enemy_edit.definition
    for stat_edit in candidate.enemy_stat_edits:
        if stat_edit.enemy_id not in enemies:
            raise ValueError(f"Unknown enemy stat edit id: {stat_edit.enemy_id}")
        enemies[stat_edit.enemy_id] = enemies[stat_edit.enemy_id].model_copy(
            update=dict(stat_edit.updates)
        )
    for encounter_edit in candidate.encounter_edits:
        encounters[encounter_edit.encounter_id] = EncounterDefinition(
            id=encounter_edit.encounter_id,
            enemies=list(encounter_edit.enemies),
        )
    return replace(
        definitions,
        skills_file=definitions.skills_file.model_copy(update={"skills": skills}),
        enemies_file=definitions.enemies_file.model_copy(update={"enemies": enemies}),
        expeditions_file=definitions.expeditions_file.model_copy(
            update={"encounters": encounters}
        ),
    )


def _promotion_gates(
    scout: GeneratedRouteLabSummary,
    hunt: GeneratedRouteLabSummary,
) -> tuple[BreachFightPromotionGate, ...]:
    scout_completion = _completion_rate(scout)
    hunt_completion = _completion_rate(hunt)
    scout_patterns = _distinct_patterns(scout)
    hunt_combats = _average_combat_count(hunt)
    tactical_pressure = _tactical_pressure(scout) + _tactical_pressure(hunt)
    return (
        BreachFightPromotionGate(
            "scout_completion",
            SCOUT_COMPLETION_MIN <= scout_completion <= SCOUT_COMPLETION_MAX,
            f"{scout_completion:.2f}",
        ),
        BreachFightPromotionGate(
            "hunt_completion",
            HUNT_COMPLETION_MIN <= hunt_completion <= HUNT_COMPLETION_MAX,
            f"{hunt_completion:.2f}",
        ),
        BreachFightPromotionGate(
            "scout_variety",
            scout_patterns >= MIN_SCOUT_PATTERNS,
            str(scout_patterns),
        ),
        BreachFightPromotionGate(
            "hunt_combats",
            hunt_combats >= MIN_HUNT_COMBATS,
            f"{hunt_combats:.1f}",
        ),
        BreachFightPromotionGate(
            "tactical_pressure",
            tactical_pressure >= MIN_TACTICAL_PRESSURE,
            str(tactical_pressure),
        ),
    )


def _candidate_score(
    scout: GeneratedRouteLabSummary,
    hunt: GeneratedRouteLabSummary,
    gates: Sequence[BreachFightPromotionGate],
    candidate: BreachFightCandidate,
) -> int:
    gate_score = sum(25 for gate in gates if gate.passed)
    pressure = min(30, _tactical_pressure(scout) + _tactical_pressure(hunt))
    variety = min(20, _distinct_patterns(scout) * 5)
    churn_penalty = len(candidate.enemy_stat_edits) + len(candidate.encounter_edits)
    return gate_score + pressure + variety - churn_penalty


def _run_generated_summary(
    definitions: GameDefinitions,
    config: BreachFightBalanceConfig,
    *,
    pressure_profile_id: str,
) -> GeneratedRouteLabSummary:
    return run_generated_route_lab(
        GeneratedRouteLabConfig(
            breach_id="shallow_cave_breach",
            seeds=config.seeds,
            max_rounds=config.max_rounds,
            hero_policy_id=config.hero_policy_id,
            preset_id=config.preset_id,
            strategy_id=config.strategy_id,
            pressure_profile_id=pressure_profile_id,
            definitions=definitions,
        )
    )


def _completion_rate(summary: GeneratedRouteLabSummary) -> float:
    if not summary.runs:
        return 0.0
    return sum(1 for run in summary.runs if run.completed) / len(summary.runs)


def _average_combat_count(summary: GeneratedRouteLabSummary) -> float:
    if not summary.runs:
        return 0.0
    return sum(len(run.encounters) for run in summary.runs) / len(summary.runs)


def _distinct_patterns(summary: GeneratedRouteLabSummary) -> int:
    return len(
        {
            tuple(encounter.encounter_id for encounter in run.encounters)
            for run in summary.runs
        }
    )


def _tactical_pressure(summary: GeneratedRouteLabSummary) -> int:
    pressure = 0
    for run in summary.runs:
        for episode in run.episodes:
            metrics = episode.metrics
            pressure += int(metrics.marks_applied > 0)
            pressure += int(metrics.mark_flow.exploited_by_enemy_hit > 0)
            pressure += int(metrics.mark_flow.vulnerable_payoffs > 0)
            pressure += int(metrics.guard_actions > 0)
            pressure += int(metrics.forced_movement > 0)
            pressure += int(metrics.hero_downs > 0)
    return pressure


def _apply_candidate_to_yaml(
    candidate: BreachFightCandidate,
    *,
    data_dir: Path,
) -> tuple[Path, ...]:
    written: list[Path] = []
    if candidate.skill_edits:
        skills_path = data_dir / "skills.yaml"
        skills_data = _load_yaml(skills_path)
        skills = skills_data.setdefault("skills", {})
        for skill_edit in candidate.skill_edits:
            skills[skill_edit.skill_id] = skill_edit.definition.model_dump(mode="json")
        _write_yaml(skills_path, skills_data)
        written.append(skills_path)
    if candidate.enemy_edits or candidate.enemy_stat_edits:
        enemies_path = data_dir / "enemies.yaml"
        enemies_data = _load_yaml(enemies_path)
        enemies = enemies_data.setdefault("enemies", {})
        for enemy_edit in candidate.enemy_edits:
            enemies[enemy_edit.enemy_id] = enemy_edit.definition.model_dump(mode="json")
        for stat_edit in candidate.enemy_stat_edits:
            if stat_edit.enemy_id not in enemies:
                raise ValueError(f"Unknown enemy in YAML: {stat_edit.enemy_id}")
            enemies[stat_edit.enemy_id].update(dict(stat_edit.updates))
        _write_yaml(enemies_path, enemies_data)
        written.append(enemies_path)
    if candidate.encounter_edits:
        expeditions_path = data_dir / "expeditions.yaml"
        expeditions_data = _load_yaml(expeditions_path)
        encounters = expeditions_data.setdefault("encounters", {})
        for encounter_edit in candidate.encounter_edits:
            encounters[encounter_edit.encounter_id] = {
                "id": encounter_edit.encounter_id,
                "enemies": [
                    {
                        "enemy_id": enemy.enemy_id,
                        "actor_id": enemy.actor_id,
                        "formation_slot": enemy.formation_slot.value,
                    }
                    for enemy in encounter_edit.enemies
                ],
            }
        _write_yaml(expeditions_path, expeditions_data)
        written.append(expeditions_path)
    return tuple(written)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return data


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def _summary_line(summary: GeneratedRouteLabSummary) -> str:
    return (
        f"{sum(1 for run in summary.runs if run.completed)}/{len(summary.runs)} "
        f"complete; patterns={_distinct_patterns(summary)}; "
        f"combats={_average_combat_count(summary):.1f}; "
        f"pressure={_tactical_pressure(summary)}"
    )


def _default_candidates() -> tuple[BreachFightCandidate, ...]:
    return tuple(
        sorted(
            (
                _baseline_candidate(),
                _authored_maze_roles_candidate(),
                _balanced_maze_roles_candidate(),
                _guarded_hunt_candidate(),
                _attrition_pack_candidate(),
                _brutal_glass_candidate(),
            ),
            key=lambda candidate: candidate.candidate_id,
        )
    )


def _baseline_candidate() -> BreachFightCandidate:
    return BreachFightCandidate(
        candidate_id="baseline",
        description="Current authored breach fights.",
    )


def _guarded_hunt_candidate() -> BreachFightCandidate:
    return BreachFightCandidate(
        candidate_id="guarded_hunt",
        description="Acolyte/leech scouting pressure with a guarded Maw hunt.",
        encounter_edits=(
            EncounterEdit(
                "maze_depth_1",
                (
                    _enemy("maze_acolyte", "maze_acolyte_1", FormationSlot.FRONT_LEFT),
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.BACK_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_hunt",
                (
                    _enemy("cave_maw_brute", "breach_maw_1", FormationSlot.FRONT_LEFT),
                    _enemy("bone_soldier", "bone_guard_1", FormationSlot.FRONT_RIGHT),
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.BACK_RIGHT),
                ),
            ),
        ),
    )


def _authored_maze_roles_candidate() -> BreachFightCandidate:
    return BreachFightCandidate(
        candidate_id="authored_maze_roles",
        description="Full Maze role pack: splinters, wards, stalkers, and a guarded Maw hunt.",
        skill_edits=(
            SkillEdit(
                "splinter_mark",
                _skill(
                    "splinter_mark",
                    "Splinter Mark",
                    category="special",
                    effort_cost=1,
                    attack_type="magic",
                    accuracy=92,
                    damage=1,
                    damage_min=1,
                    damage_max=2,
                    tags=("enemy", "maze", "glass", "mark"),
                    description=(
                        "A bright shard points through a hero and leaves the Maze looking."
                    ),
                    reaction_window=True,
                    intent_label="Splinter Mark",
                    threat_level="medium",
                    obvious_effect="Marking glass sting",
                ),
            ),
            SkillEdit(
                "ward_pattern",
                _skill(
                    "ward_pattern",
                    "Ward Pattern",
                    category="support",
                    effort_cost=1,
                    attack_type="magic",
                    accuracy=100,
                    damage=0,
                    tags=("enemy", "maze", "support", "guard"),
                    description="The Ward folds a black pattern over an ally.",
                    intent_label="Ward Pattern",
                    threat_level="medium",
                    obvious_effect="Guarding support",
                ),
            ),
            SkillEdit(
                "stalker_cut",
                _skill(
                    "stalker_cut",
                    "Stalker Cut",
                    category="special",
                    effort_cost=0,
                    attack_type="reach",
                    usable_from="front_only",
                    accuracy=72,
                    damage=2,
                    damage_min=2,
                    damage_max=3,
                    tags=("enemy", "maze", "hunter", "vulnerable_bonus"),
                    description="A long cut that finds heroes already marked by the breach.",
                    reaction_window=True,
                    intent_label="Stalker Cut",
                    threat_level="high",
                    obvious_effect="Mark payoff",
                ),
            ),
            SkillEdit(
                "stalker_hook",
                _skill(
                    "stalker_hook",
                    "Stalker Hook",
                    category="special",
                    effort_cost=1,
                    attack_type="ranged",
                    accuracy=78,
                    damage=1,
                    damage_min=1,
                    damage_max=2,
                    tags=("enemy", "maze", "hunter", "formation", "drag_forward"),
                    description=(
                        "A hook of black glass drags a safer hero into the wrong line."
                    ),
                    reaction_window=True,
                    intent_label="Stalker Hook",
                    threat_level="medium",
                    obvious_effect="Formation disruption",
                ),
            ),
        ),
        enemy_edits=(
            EnemyEdit(
                "glass_splinter",
                _enemy_definition(
                    "glass_splinter",
                    "Glass Splinter",
                    max_hp=4,
                    speed=7,
                    accuracy=5,
                    defense=0,
                    damage=1,
                    max_effort=2,
                    skills=("glass_bite", "splinter_mark"),
                    formation_slot=FormationSlot.BACK_LEFT,
                    tags=("maze", "glass", "beast"),
                ),
            ),
            EnemyEdit(
                "pattern_ward",
                _enemy_definition(
                    "pattern_ward",
                    "Pattern Ward",
                    max_hp=10,
                    speed=2,
                    accuracy=5,
                    defense=1,
                    damage=1,
                    max_effort=2,
                    skills=("ward_pattern", "black_pulse"),
                    formation_slot=FormationSlot.FRONT_RIGHT,
                    tags=("maze", "horror", "support"),
                ),
            ),
            EnemyEdit(
                "breach_stalker",
                _enemy_definition(
                    "breach_stalker",
                    "Breach Stalker",
                    max_hp=13,
                    speed=4,
                    accuracy=6,
                    defense=1,
                    damage=2,
                    max_effort=3,
                    skills=("stalker_cut", "stalker_hook"),
                    formation_slot=FormationSlot.FRONT_LEFT,
                    tags=("maze", "hunter"),
                ),
            ),
        ),
        encounter_edits=(
            EncounterEdit(
                "maze_depth_1",
                (
                    _enemy("maze_acolyte", "maze_acolyte_1", FormationSlot.FRONT_LEFT),
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.BACK_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_probe",
                (
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.BACK_RIGHT),
                    _enemy("glass_splinter", "glass_splinter_1", FormationSlot.BACK_LEFT),
                ),
            ),
            EncounterEdit(
                "generated_maze_pattern_cell",
                (
                    _enemy("maze_acolyte", "maze_acolyte_1", FormationSlot.FRONT_LEFT),
                    _enemy("pattern_ward", "pattern_ward_1", FormationSlot.FRONT_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_glass_pack",
                (
                    _enemy("glass_splinter", "glass_splinter_1", FormationSlot.BACK_LEFT),
                    _enemy("glass_splinter", "glass_splinter_2", FormationSlot.BACK_RIGHT),
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.FRONT_LEFT),
                ),
            ),
            EncounterEdit(
                "generated_maze_stalker",
                (
                    _enemy("breach_stalker", "breach_stalker_1", FormationSlot.FRONT_LEFT),
                    _enemy("pattern_ward", "pattern_ward_1", FormationSlot.FRONT_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_hunt_guarded",
                _guarded_hunt_enemies(),
            ),
            EncounterEdit(
                "generated_maze_hunt",
                _guarded_hunt_enemies(),
            ),
        ),
    )


def _balanced_maze_roles_candidate() -> BreachFightCandidate:
    return replace(
        _authored_maze_roles_candidate(),
        candidate_id="balanced_maze_roles",
        description=(
            "Harder Maze role pack tuned by the route lab for punishing scout and hunt bands."
        ),
        encounter_edits=(
            EncounterEdit(
                "maze_depth_1",
                (
                    _enemy("maze_acolyte", "maze_acolyte_1", FormationSlot.FRONT_LEFT),
                    _enemy(
                        "breach_stalker",
                        "breach_stalker_1",
                        FormationSlot.FRONT_RIGHT,
                    ),
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.BACK_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_probe",
                (
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.FRONT_LEFT),
                    _enemy("glass_splinter", "glass_splinter_1", FormationSlot.BACK_LEFT),
                    _enemy("glass_splinter", "glass_splinter_2", FormationSlot.BACK_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_pattern_cell",
                (
                    _enemy("maze_acolyte", "maze_acolyte_1", FormationSlot.FRONT_LEFT),
                    _enemy("pattern_ward", "pattern_ward_1", FormationSlot.FRONT_RIGHT),
                    _enemy("glass_splinter", "glass_splinter_1", FormationSlot.BACK_LEFT),
                ),
            ),
            EncounterEdit(
                "generated_maze_stalker",
                (
                    _enemy("breach_stalker", "breach_stalker_1", FormationSlot.FRONT_LEFT),
                    _enemy("pattern_ward", "pattern_ward_1", FormationSlot.FRONT_RIGHT),
                    _enemy("glass_splinter", "glass_splinter_1", FormationSlot.BACK_LEFT),
                ),
            ),
            EncounterEdit(
                "generated_maze_hunt_guarded",
                (
                    _enemy("cave_maw_brute", "breach_maw_1", FormationSlot.FRONT_LEFT),
                    _enemy("pattern_ward", "pattern_ward_1", FormationSlot.FRONT_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_hunt",
                (
                    _enemy("cave_maw_brute", "breach_maw_1", FormationSlot.FRONT_LEFT),
                    _enemy("pattern_ward", "pattern_ward_1", FormationSlot.FRONT_RIGHT),
                ),
            ),
        ),
    )


def _attrition_pack_candidate() -> BreachFightCandidate:
    return BreachFightCandidate(
        candidate_id="attrition_pack",
        description="More leech drain and tougher acolyte pressure.",
        encounter_edits=(
            EncounterEdit(
                "maze_depth_1",
                (
                    _enemy("maze_acolyte", "maze_acolyte_1", FormationSlot.FRONT_LEFT),
                    _enemy("maze_leech", "maze_leech_1", FormationSlot.BACK_LEFT),
                    _enemy("maze_leech", "maze_leech_2", FormationSlot.BACK_RIGHT),
                ),
            ),
            EncounterEdit(
                "generated_maze_hunt",
                (
                    _enemy("cave_maw_brute", "breach_maw_1", FormationSlot.FRONT_LEFT),
                    _enemy("maze_acolyte", "maze_acolyte_1", FormationSlot.BACK_LEFT),
                ),
            ),
        ),
        enemy_stat_edits=(
            EnemyStatEdit("maze_acolyte", {"max_hp": 13}),
            EnemyStatEdit("maze_leech", {"accuracy": 6}),
        ),
    )


def _brutal_glass_candidate() -> BreachFightCandidate:
    return BreachFightCandidate(
        candidate_id="brutal_glass",
        description="Higher glass damage with guarded hunt pressure.",
        encounter_edits=_guarded_hunt_candidate().encounter_edits,
        enemy_stat_edits=(
            EnemyStatEdit("maze_leech", {"max_hp": 6, "accuracy": 6}),
            EnemyStatEdit("cave_maw_brute", {"accuracy": 7}),
        ),
    )


def _enemy(
    enemy_id: str,
    actor_id: str,
    formation_slot: FormationSlot,
) -> EncounterEnemyDefinition:
    return EncounterEnemyDefinition(
        enemy_id=enemy_id,
        actor_id=actor_id,
        formation_slot=formation_slot,
    )


def _guarded_hunt_enemies() -> tuple[EncounterEnemyDefinition, ...]:
    return (
        _enemy("cave_maw_brute", "breach_maw_1", FormationSlot.FRONT_LEFT),
        _enemy("pattern_ward", "pattern_ward_1", FormationSlot.FRONT_RIGHT),
        _enemy("glass_splinter", "glass_splinter_1", FormationSlot.BACK_LEFT),
    )


def _skill(
    skill_id: str,
    name: str,
    *,
    category: str,
    effort_cost: int,
    attack_type: str,
    accuracy: int,
    damage: int,
    tags: tuple[str, ...],
    description: str,
    usable_from: str = "any_position",
    damage_min: int | None = None,
    damage_max: int | None = None,
    reaction_window: bool = False,
    intent_label: str | None = None,
    threat_level: str = "normal",
    obvious_effect: str = "",
) -> SkillDefinition:
    return SkillDefinition.model_validate(
        {
            "id": skill_id,
            "name": name,
            "category": category,
            "effort_cost": effort_cost,
            "attack_type": attack_type,
            "usable_from": usable_from,
            "accuracy": accuracy,
            "damage": damage,
            "damage_min": damage_min,
            "damage_max": damage_max,
            "tags": list(tags),
            "description": description,
            "reaction_window": reaction_window,
            "intent_label": intent_label,
            "threat_level": threat_level,
            "obvious_effect": obvious_effect,
        }
    )


def _enemy_definition(
    enemy_id: str,
    name: str,
    *,
    max_hp: int,
    speed: int,
    accuracy: int,
    defense: int,
    damage: int,
    max_effort: int,
    skills: tuple[str, ...],
    formation_slot: FormationSlot,
    tags: tuple[str, ...],
) -> EnemyDefinition:
    return EnemyDefinition(
        id=enemy_id,
        name=name,
        max_hp=max_hp,
        speed=speed,
        accuracy=accuracy,
        defense=defense,
        damage=damage,
        max_effort=max_effort,
        skills=list(skills),
        formation_slot=formation_slot,
        tags=list(tags),
    )


__all__ = [
    "BreachFightBalanceConfig",
    "BreachFightBalanceResult",
    "BreachFightCandidate",
    "BreachFightCandidateResult",
    "BreachFightPromotionGate",
    "EnemyEdit",
    "EncounterEdit",
    "EnemyStatEdit",
    "SkillEdit",
    "format_breach_fight_balance_result",
    "run_breach_fight_balance",
]
