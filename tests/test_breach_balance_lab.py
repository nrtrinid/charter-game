from __future__ import annotations

import pytest

from game.combat.formation import FormationSlot
from game.data.schemas import EncounterEnemyDefinition, EnemyDefinition, SkillDefinition
from game.dev.breach_balance_lab import (
    BreachFightBalanceConfig,
    BreachFightCandidate,
    EncounterEdit,
    EnemyEdit,
    SkillEdit,
    format_breach_fight_balance_result,
    run_breach_fight_balance,
)
from tests.conftest import get_definitions

pytestmark = pytest.mark.slow


def test_breach_balance_lab_is_seed_deterministic_in_dry_run() -> None:
    definitions = get_definitions()
    config = BreachFightBalanceConfig(
        definitions=definitions,
        seeds=1,
        max_rounds=1,
        dry_run=True,
    )

    first = run_breach_fight_balance(config)
    second = run_breach_fight_balance(config)

    assert [result.candidate.candidate_id for result in first.results] == [
        result.candidate.candidate_id for result in second.results
    ]
    assert [result.score for result in first.results] == [
        result.score for result in second.results
    ]
    assert first.applied is False
    assert first.files_written == ()


def test_breach_balance_lab_does_not_mutate_original_definitions() -> None:
    definitions = get_definitions()
    encounter_ids = tuple(sorted(definitions.encounters))
    original_maze_depth = definitions.encounters["maze_depth_1"]
    candidate = BreachFightCandidate(
        candidate_id="test_empty_maze_depth",
        description="A deliberately invalid-feeling but schema-valid test variant.",
        encounter_edits=(EncounterEdit("maze_depth_1", ()),),
    )

    run_breach_fight_balance(
        BreachFightBalanceConfig(
            definitions=definitions,
            seeds=1,
            max_rounds=1,
            dry_run=True,
            candidates=(candidate,),
        )
    )

    assert tuple(sorted(definitions.encounters)) == encounter_ids
    assert definitions.encounters["maze_depth_1"] == original_maze_depth


def test_breach_balance_lab_can_evaluate_candidate_only_content() -> None:
    definitions = get_definitions()
    candidate = BreachFightCandidate(
        candidate_id="candidate_only_content",
        description="Content that exists only in the candidate clone.",
        skill_edits=(
            SkillEdit(
                "test_maze_mark",
                SkillDefinition.model_validate(
                    {
                        "id": "test_maze_mark",
                        "name": "Test Maze Mark",
                        "category": "special",
                        "effort_cost": 0,
                        "attack_type": "magic",
                        "accuracy": 100,
                        "damage": 0,
                        "tags": ["enemy", "mark"],
                    }
                ),
            ),
        ),
        enemy_edits=(
            EnemyEdit(
                "test_maze_enemy",
                EnemyDefinition(
                    id="test_maze_enemy",
                    name="Test Maze Enemy",
                    max_hp=1,
                    speed=1,
                    accuracy=1,
                    defense=0,
                    damage=0,
                    max_effort=0,
                    skills=["test_maze_mark"],
                    formation_slot=FormationSlot.FRONT_LEFT,
                    tags=["maze"],
                ),
            ),
        ),
        encounter_edits=(
            EncounterEdit(
                "maze_depth_1",
                (
                    EncounterEnemyDefinition(
                        enemy_id="test_maze_enemy",
                        actor_id="test_maze_enemy_1",
                        formation_slot=FormationSlot.FRONT_LEFT,
                    ),
                ),
            ),
        ),
    )

    result = run_breach_fight_balance(
        BreachFightBalanceConfig(
            definitions=definitions,
            seeds=1,
            max_rounds=1,
            dry_run=True,
            candidates=(candidate,),
        )
    )

    assert result.results[0].candidate.candidate_id == "candidate_only_content"
    assert "test_maze_mark" not in definitions.skills
    assert "test_maze_enemy" not in definitions.enemies


def test_breach_balance_lab_fails_closed_without_passing_candidate() -> None:
    definitions = get_definitions()
    candidate = BreachFightCandidate(
        candidate_id="baseline_only",
        description="Current authored breach fights only.",
    )

    result = run_breach_fight_balance(
        BreachFightBalanceConfig(
            definitions=definitions,
            seeds=1,
            max_rounds=1,
            dry_run=False,
            candidates=(candidate,),
        )
    )
    text = format_breach_fight_balance_result(result)

    assert result.selected is None
    assert result.applied is False
    assert result.files_written == ()
    assert "Selected: none" in text
