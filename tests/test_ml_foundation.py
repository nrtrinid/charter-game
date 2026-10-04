from __future__ import annotations

import json

import pytest

from game.campaign.company import create_new_company
from game.combat.enemy_learning import (
    EnemyDecisionEpisode,
    create_hero_policy,
    run_enemy_learning_episode,
)
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.expedition.cave import create_encounter_combat
from game.ml.artifacts import ResearchArtifactManifest, extraction_manifest_pairs
from game.ml.dataset import DatasetExtractionConfig, extract_enemy_decision_dataset
from game.ml.features import ENEMY_DECISION_FEATURE_NAMES, ENEMY_DECISION_FEATURE_SCHEMA
from game.ml.splits import DEFAULT_SEED_SPLIT_SPEC, DatasetSplit
from tests.conftest import get_definitions


def test_feature_schema_is_explicit_versioned_and_ordered() -> None:
    schema = ENEMY_DECISION_FEATURE_SCHEMA

    assert schema.schema_id == "enemy-decision-candidate-v1"
    assert schema.version == 1
    assert len(schema.names) == 39
    assert schema.names == ENEMY_DECISION_FEATURE_NAMES
    assert schema.names[:4] == (
        "formation",
        "marked_focus",
        "damage_pressure",
        "boss_pressure",
    )
    assert schema.names[-4:] == ("effort_drain", "mark", "drag_forward", "role_flavor")


def test_feature_schema_encoding_does_not_depend_on_mapping_order() -> None:
    schema = ENEMY_DECISION_FEATURE_SCHEMA
    forward = {name: index for index, name in enumerate(schema.names)}
    reverse = dict(reversed(tuple(forward.items())))

    assert schema.encode(forward) == schema.encode(reverse)
    assert schema.decode(schema.encode(reverse)) == forward


def test_feature_schema_rejects_missing_and_unexpected_features() -> None:
    schema = ENEMY_DECISION_FEATURE_SCHEMA
    valid = dict.fromkeys(schema.names, 0)
    missing = dict(valid)
    missing.pop(schema.names[0])
    unexpected = dict(valid)
    unexpected["future_outcome"] = 1

    with pytest.raises(ValueError, match="missing"):
        schema.encode(missing)
    with pytest.raises(ValueError, match="unexpected"):
        schema.encode(unexpected)


def test_observation_schema_excludes_future_and_identity_metadata() -> None:
    forbidden = {
        "final_victor",
        "eventual_episode_reward",
        "post_action_damage",
        "post_action_hp",
        "future_events",
        "seed",
        "hero_policy_id",
        "actor_instance_id",
        "target_instance_id",
        "encounter_name",
    }

    assert forbidden.isdisjoint(ENEMY_DECISION_FEATURE_NAMES)


def test_seed_splits_are_deterministic_disjoint_and_human_inspectable() -> None:
    spec = DEFAULT_SEED_SPLIT_SPEC
    train = set(spec.seeds_for(DatasetSplit.TRAIN))
    validation = set(spec.seeds_for(DatasetSplit.VALIDATION))
    test = set(spec.seeds_for(DatasetSplit.TEST))

    assert (min(train), max(train)) == (1, 256)
    assert (min(validation), max(validation)) == (10001, 10064)
    assert (min(test), max(test)) == (20001, 20128)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert spec.split_for_seed(1) is DatasetSplit.TRAIN
    assert spec.split_for_seed(10001) is DatasetSplit.VALIDATION
    assert spec.split_for_seed(20001) is DatasetSplit.TEST


def test_seed_split_rejects_out_of_partition_and_duplicate_extraction_seeds() -> None:
    spec = DEFAULT_SEED_SPLIT_SPEC

    with pytest.raises(ValueError, match="expected exactly one"):
        spec.split_for_seed(9999)
    with pytest.raises(ValueError, match="unique"):
        spec.validate_seeds(DatasetSplit.TRAIN, (1, 1))
    with pytest.raises(ValueError, match="do not belong"):
        spec.validate_seeds(DatasetSplit.TRAIN, (1, 10001))


def test_grouped_extraction_preserves_simulator_candidates_and_selected_label() -> None:
    definitions = get_definitions()
    config = DatasetExtractionConfig(
        split=DatasetSplit.TRAIN,
        encounter_ids=("road_bandits",),
        hero_policy_ids=("naive",),
        seeds=(1,),
        max_rounds=1,
    )

    dataset = extract_enemy_decision_dataset(config, definitions=definitions)
    episode = _run_reference_episode(definitions, encounter_id="road_bandits", seed=1)

    assert dataset.records
    assert episode.records
    first = dataset.records[0]
    source = episode.records[0]
    extracted_identities = tuple(
        (candidate.skill_id, candidate.target_id) for candidate in first.candidates
    )
    source_identities = tuple(
        (candidate.skill_id, candidate.target_id) for candidate in source.trace.candidates
    )
    assert first.simulation_seed == 1
    assert first.split is DatasetSplit.TRAIN
    assert len(first.candidates) == len(source.trace.candidates)
    assert extracted_identities == source_identities
    assert first.chosen_candidate.skill_id == source.chosen_skill_id
    assert first.chosen_candidate.target_id == source.chosen_target_id
    assert 0 <= first.chosen_index < len(first.candidates)
    assert all(
        len(candidate.feature_values) == len(ENEMY_DECISION_FEATURE_NAMES)
        for candidate in first.candidates
    )


def test_records_from_one_seed_never_cross_splits() -> None:
    definitions = get_definitions()
    dataset = extract_enemy_decision_dataset(
        DatasetExtractionConfig(
            split=DatasetSplit.VALIDATION,
            encounter_ids=("road_bandits",),
            hero_policy_ids=("naive", "damage_race"),
            seeds=(10001,),
            max_rounds=1,
        ),
        definitions=definitions,
    )

    assert dataset.records
    assert {record.simulation_seed for record in dataset.records} == {10001}
    assert {record.split for record in dataset.records} == {DatasetSplit.VALIDATION}


def test_repeated_small_extraction_is_exactly_deterministic() -> None:
    definitions = get_definitions()
    config = DatasetExtractionConfig(
        split=DatasetSplit.TRAIN,
        encounter_ids=("road_bandits",),
        hero_policy_ids=("naive", "survival"),
        seeds=(1, 2),
        max_rounds=1,
    )

    first = extract_enemy_decision_dataset(config, definitions=definitions)
    second = extract_enemy_decision_dataset(config, definitions=definitions)

    assert first == second


def test_outcomes_are_metadata_not_candidate_observations() -> None:
    definitions = get_definitions()
    dataset = extract_enemy_decision_dataset(
        DatasetExtractionConfig(
            split=DatasetSplit.TRAIN,
            encounter_ids=("road_bandits",),
            hero_policy_ids=("naive",),
            seeds=(1,),
            max_rounds=1,
        ),
        definitions=definitions,
    )

    first = dataset.records[0]
    assert isinstance(first.outcome.action_reward, int)
    assert first.outcome.episode_final_victor
    assert len(first.chosen_candidate.feature_values) == 39
    assert not hasattr(first.chosen_candidate, "action_reward")
    assert not hasattr(first.chosen_candidate, "episode_final_victor")


def test_artifact_manifest_serialization_is_deterministic_and_records_provenance() -> None:
    config = DatasetExtractionConfig(
        split=DatasetSplit.TRAIN,
        encounter_ids=("road_bandits", "wolf_pack"),
        hero_policy_ids=("naive", "damage_race"),
        seeds=(1, 2),
        max_rounds=3,
    )
    manifest = ResearchArtifactManifest(
        artifact_kind="enemy_decision_dataset",
        repository_commit="129f7ee",
        content_hashes=(("data/enemies.yaml", "abc"), ("data/skills.yaml", "def")),
        split_name=config.split.value,
        encounter_ids=config.encounter_ids,
        training_hero_policy_ids=config.hero_policy_ids,
        extraction_config=extraction_manifest_pairs(config),
        runtime_versions=(("python", "3.13"),),
    )

    first = manifest.to_json()
    second = manifest.to_json()
    decoded = json.loads(first)

    assert first == second
    assert decoded["format_version"] == 1
    assert decoded["feature_schema_version"] == 1
    assert decoded["repository_commit"] == "129f7ee"
    assert decoded["split_name"] == "train"
    assert decoded["encounter_ids"] == ["road_bandits", "wolf_pack"]
    assert decoded["training_hero_policy_ids"] == ["naive", "damage_race"]
    assert decoded["extraction_config"]["seeds"] == "1,2"


def _run_reference_episode(
    definitions: GameDefinitions,
    *,
    encounter_id: str,
    seed: int,
) -> EnemyDecisionEpisode:
    company = create_new_company(definitions)
    combat = create_encounter_combat(company, definitions, encounter_id)
    return run_enemy_learning_episode(
        combat,
        definitions,
        GameRng(seed),
        max_rounds=1,
        encounter_id=encounter_id,
        encounter_name=encounter_id.replace("_", " ").title(),
        hero_policy=create_hero_policy("naive", encounter_id=encounter_id, seed=seed),
    )
