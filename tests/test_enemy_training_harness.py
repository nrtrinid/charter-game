from __future__ import annotations

import pytest

from game.campaign.company import create_new_company
from game.combat.enemy_learning import SUPPORTED_HERO_POLICY_IDS
from game.dev.train_enemy_ai import (
    SUPPORTED_POLICY_SCOPE_IDS,
    SUPPORTED_PRESET_IDS,
    TrainingRunConfig,
    format_training_summary,
    run_training_harness,
)
from tests.conftest import get_definitions

pytestmark = pytest.mark.slow


def test_default_config_uses_naive_hero_policy() -> None:
    assert TrainingRunConfig().hero_policy_id == "naive"
    assert TrainingRunConfig().evaluation_hero_policy_ids == ()
    assert TrainingRunConfig().policy_scope_ids == ("global",)
    assert TrainingRunConfig().preset_id == "fresh"
    assert TrainingRunConfig().route_id == ""
    assert TrainingRunConfig().enemy_wait_mode == "none"
    assert TrainingRunConfig().enemy_movement_mode == "recovery_only"


def test_default_config_selects_all_authored_encounters() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(definitions=definitions, seeds=1, max_rounds=1)
    )

    assert summary.encounter_ids == tuple(sorted(definitions.encounters))
    assert summary.hero_policy_id == "naive"
    assert summary.preset_id == "fresh"
    assert summary.route_id == ""
    assert summary.enemy_wait_mode == "none"
    assert summary.enemy_movement_mode == "recovery_only"
    assert len(summary.heuristic_episodes) == len(definitions.encounters)
    assert len(summary.learned_episodes) == len(definitions.encounters)
    assert {breakdown.encounter_id for breakdown in summary.encounter_breakdowns} == set(
        definitions.encounters
    )


def test_explicit_encounter_filter_runs_only_requested_encounters() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits", "wolf_pack"),
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.encounter_ids == ("road_bandits", "wolf_pack")
    assert {episode.encounter_id for episode in summary.heuristic_episodes} == {
        "road_bandits",
        "wolf_pack",
    }


def test_invalid_encounter_id_fails_clearly() -> None:
    definitions = get_definitions()

    with pytest.raises(ValueError, match="Unknown encounter id: missing"):
        run_training_harness(
            TrainingRunConfig(
                definitions=definitions,
                encounter_ids=("missing",),
                seeds=1,
            )
        )


def test_invalid_hero_policy_id_fails_clearly() -> None:
    definitions = get_definitions()

    with pytest.raises(ValueError, match="Unknown hero policy: missing"):
        run_training_harness(
            TrainingRunConfig(
                definitions=definitions,
                encounter_ids=("road_bandits",),
                hero_policy_id="missing",
                seeds=1,
            )
        )


def test_invalid_preset_id_fails_clearly() -> None:
    definitions = get_definitions()

    with pytest.raises(ValueError, match="Unknown preset id: missing"):
        run_training_harness(
            TrainingRunConfig(
                definitions=definitions,
                encounter_ids=("road_bandits",),
                preset_id="missing",
                seeds=1,
            )
        )


def test_invalid_route_id_fails_clearly() -> None:
    definitions = get_definitions()

    with pytest.raises(ValueError, match="Unknown route id: missing"):
        run_training_harness(
            TrainingRunConfig(
                definitions=definitions,
                route_id="missing",
                seeds=1,
            )
        )


def test_invalid_policy_scope_id_fails_clearly() -> None:
    definitions = get_definitions()

    with pytest.raises(ValueError, match="Unknown policy scope: missing"):
        run_training_harness(
            TrainingRunConfig(
                definitions=definitions,
                encounter_ids=("road_bandits",),
                policy_scope_ids=("missing",),
                seeds=1,
            )
        )


def test_invalid_enemy_timing_modes_fail_clearly() -> None:
    definitions = get_definitions()

    with pytest.raises(ValueError, match="Unknown enemy wait mode: tactical"):
        run_training_harness(
            TrainingRunConfig(
                definitions=definitions,
                encounter_ids=("road_bandits",),
                enemy_wait_mode="tactical",
                seeds=1,
            )
        )
    with pytest.raises(ValueError, match="Unknown enemy movement mode: tactical"):
        run_training_harness(
            TrainingRunConfig(
                definitions=definitions,
                encounter_ids=("road_bandits",),
                enemy_movement_mode="tactical",
                seeds=1,
            )
        )


def test_training_harness_accepts_enemy_wait_and_move_modes() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=1,
            enemy_wait_mode="package_only",
            enemy_movement_mode="package_only",
        )
    )
    text = format_training_summary(summary)

    assert summary.enemy_wait_mode == "package_only"
    assert summary.enemy_movement_mode == "package_only"
    assert "Enemy wait mode: package_only" in text
    assert "Enemy movement mode: package_only" in text
    if "Enemy Timing:" in text:
        assert "waited attacks" in text
        assert "waits no payoff" in text
        if "wait quality:" in text:
            assert any(label in text for label in ("GOOD", "MIXED", "POOR"))
    if "Movement Quality:" in text:
        assert "move->attack" in text
        if "move quality:" in text:
            assert any(label in text for label in ("GOOD", "MIXED", "POOR"))


@pytest.mark.parametrize("hero_policy_id", SUPPORTED_HERO_POLICY_IDS)
def test_training_harness_accepts_supported_hero_policy_ids(hero_policy_id: str) -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            hero_policy_id=hero_policy_id,
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.hero_policy_id == hero_policy_id
    assert summary.heuristic_episodes
    assert summary.learned_episodes


@pytest.mark.parametrize("scope_id", SUPPORTED_POLICY_SCOPE_IDS)
def test_training_harness_accepts_supported_policy_scopes(scope_id: str) -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            policy_scope_ids=(scope_id,),
            seeds=1,
            max_rounds=1,
        )
    )

    assert scope_id in summary.policy_scope_ids
    assert {evaluation.policy_scope_id for evaluation in summary.policy_evaluations} >= {
        "global",
        scope_id,
    }


@pytest.mark.parametrize("preset_id", SUPPORTED_PRESET_IDS)
def test_training_harness_accepts_supported_presets(preset_id: str) -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            preset_id=preset_id,
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.preset_id == preset_id
    assert summary.heuristic_episodes
    assert summary.learned_episodes


def test_summary_includes_heuristic_and_learned_metrics() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=2,
        )
    )

    assert summary.heuristic_episodes
    assert summary.learned_episodes
    assert summary.hero_policy_id == "naive"
    assert summary.heuristic_total_reward >= 0
    assert summary.learned_total_reward >= 0
    assert summary.heuristic_victors
    assert summary.learned_victors
    assert summary.heuristic_average_records > 0
    assert summary.evaluation_hero_policy_ids == ("naive",)
    assert summary.policy_scope_ids == ("global",)
    assert len(summary.policy_evaluations) == 1
    breakdown = summary.encounter_breakdowns[0]
    assert breakdown.encounter_id == "road_bandits"
    assert breakdown.heuristic.episode_count == 1
    assert breakdown.learned.episode_count == 1
    assert breakdown.heuristic.average_enemy_decisions > 0


def test_learned_weights_are_non_empty_when_rewarded_episodes_exist() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=2,
        )
    )

    assert summary.learned_weights
    assert summary.top_weights


def test_formatted_summary_contains_expected_sections() -> None:
    definitions = get_definitions()
    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=1,
        )
    )

    text = format_training_summary(summary)

    assert "Enemy AI Training Harness" in text
    assert "Encounters: road_bandits" in text
    assert "Seeds: 1" in text
    assert "Training opponent: naive scripted hero policy" in text
    assert "Evaluation opponents: naive" in text
    assert "Policy scopes: global" in text
    assert "Heuristic reward:" in text
    assert "Learned reward:" in text
    assert "Heuristic victors:" in text
    assert "Learned victors:" in text
    assert "Top learned feature weights:" in text
    assert "Preset: fresh" in text
    assert "Route: isolated encounters" in text
    assert "Per Encounter:" in text
    assert "damage" in text
    assert "marks" in text


def test_formatted_summary_includes_mark_flow_when_mark_activity_exists() -> None:
    definitions = get_definitions()
    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=2,
        )
    )

    text = format_training_summary(summary)

    assert "Mark Flow:" in text
    assert "road_bandits:" in text
    assert "avg ally reach" in text


def test_formatted_summary_includes_boss_sequence_when_boss_activity_exists() -> None:
    definitions = get_definitions()
    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("cave_mini_boss",),
            seeds=1,
            max_rounds=2,
        )
    )

    text = format_training_summary(summary)

    assert "Boss Sequence:" in text
    assert "cave_mini_boss:" in text
    assert "grabs" in text
    assert "bites" in text
    assert "Boss Targeting:" in text
    assert "grab targets" in text
    assert "bite targets" in text


def test_formatted_summary_includes_guard_flow_when_guard_activity_exists() -> None:
    definitions = get_definitions()
    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("shallow_cave",),
            seeds=1,
            max_rounds=2,
        )
    )

    text = format_training_summary(summary)

    assert "Guard Flow:" in text
    assert "shallow_cave:" in text
    assert "dead guard" in text
    assert "guard wasted" in text


def test_cross_policy_evaluation_runs_all_requested_opponents() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            hero_policy_id="mixed",
            evaluation_hero_policy_ids=("naive", "survival", "conservative"),
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.evaluation_hero_policy_ids == ("naive", "survival", "conservative")
    assert {
        evaluation.evaluation_hero_policy_id for evaluation in summary.policy_evaluations
    } == {"naive", "survival", "conservative"}


def test_policy_scope_evaluation_compares_global_and_local_weights() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits", "cave_mini_boss"),
            policy_scope_ids=("global", "per_encounter", "per_role", "boss"),
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.policy_scope_ids == ("global", "per_encounter", "per_role", "boss")
    assert {evaluation.policy_scope_id for evaluation in summary.policy_evaluations} == {
        "global",
        "per_encounter",
        "per_role",
        "boss",
    }
    assert all(evaluation.learned_weights for evaluation in summary.policy_evaluations)


def test_opening_critical_path_route_runs_sequence_and_breakdowns() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            route_id="opening_critical_path",
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.route_id == "opening_critical_path"
    assert summary.encounter_ids == ("shallow_cave", "cave_mini_boss")
    assert {episode.encounter_id for episode in summary.heuristic_episodes}.issubset(
        set(summary.encounter_ids)
    )
    assert summary.encounter_breakdowns[0].encounter_id == "shallow_cave"
    assert len(summary.heuristic_route_results) == 1
    assert len(summary.learned_route_results) == 1
    assert summary.heuristic_route_summary.route_count == 1
    assert summary.learned_route_summary.route_count == 1
    assert summary.heuristic_route_results[0].encounters[0].encounter_id == "shallow_cave"
    assert summary.heuristic_route_results[0].encounters[0].hp_entering >= (
        summary.heuristic_route_results[0].encounters[0].hp_leaving
    )


def test_route_level_metrics_track_cave_mini_boss_entry_and_exit() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            route_id="opening_critical_path",
            preset_id="attrition",
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.heuristic_route_summary.average_hp_entering_cave_mini_boss > 0
    assert summary.heuristic_route_summary.average_hp_leaving_cave_mini_boss >= 0


def test_formatted_summary_includes_route_and_policy_evaluation_sections() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            route_id="opening_critical_path",
            hero_policy_id="mixed",
            evaluation_hero_policy_ids=("naive", "mixed"),
            policy_scope_ids=("global", "per_encounter"),
            seeds=1,
            max_rounds=1,
        )
    )

    text = format_training_summary(summary)

    assert "Route Results:" in text
    assert "completed" in text
    assert "cave_mini_boss HP in/out:" in text
    assert "Policy Evaluations:" in text
    assert "global vs naive:" in text
    assert "per_encounter vs mixed:" in text


def test_opening_pressure_path_includes_first_quest_pressure_encounters() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            route_id="opening_pressure_path",
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.route_id == "opening_pressure_path"
    assert summary.encounter_ids == (
        "road_bandits",
        "wolf_pack",
        "shallow_cave",
        "cave_mini_boss",
        "maze_depth_1",
    )


def test_conservative_policy_is_accepted_by_harness() -> None:
    definitions = get_definitions()

    summary = run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            hero_policy_id="conservative",
            seeds=1,
            max_rounds=1,
        )
    )

    assert summary.hero_policy_id == "conservative"


def test_training_harness_does_not_mutate_existing_campaign_memory() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    dungeon_memory = dict(company.dungeon_memory)
    world_memory = dict(company.world_memory)
    breach_memory = dict(company.breach_memory)

    run_training_harness(
        TrainingRunConfig(
            definitions=definitions,
            encounter_ids=("road_bandits",),
            seeds=1,
            max_rounds=1,
        )
    )

    assert company.dungeon_memory == dungeon_memory
    assert company.world_memory == world_memory
    assert company.breach_memory == breach_memory
