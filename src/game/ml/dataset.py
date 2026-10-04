"""Grouped deterministic enemy-decision datasets built from simulator traces."""

from __future__ import annotations

from dataclasses import dataclass

from game.campaign.company import create_new_company
from game.combat.enemy_learning import (
    EnemyDecisionEpisode,
    create_hero_policy,
    run_enemy_learning_episode,
)
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.expedition.cave import create_encounter_combat
from game.ml.features import ENEMY_DECISION_FEATURE_SCHEMA, EnemyDecisionFeatureSchema
from game.ml.splits import DEFAULT_SEED_SPLIT_SPEC, DatasetSplit, SeedSplitSpec

DATASET_FORMAT_ID = "enemy-decision-dataset-v1"
DATASET_FORMAT_VERSION = 1


@dataclass(frozen=True)
class EnemyDecisionCandidateRecord:
    skill_id: str
    target_id: str
    skill_order: int
    target_order: int
    skill_tags: tuple[str, ...]
    feature_values: tuple[int, ...]


@dataclass(frozen=True)
class EnemyDecisionOutcomeMetadata:
    action_reward: int
    episode_final_victor: str
    episode_total_reward: int


@dataclass(frozen=True)
class EnemyDecisionDatasetRecord:
    feature_schema_id: str
    feature_schema_version: int
    split: DatasetSplit
    encounter_id: str
    encounter_name: str
    simulation_seed: int
    hero_policy_id: str
    enemy_id: str
    enemy_class_id: str
    enemy_tags: tuple[str, ...]
    round_number: int
    action_index: int
    candidates: tuple[EnemyDecisionCandidateRecord, ...]
    chosen_index: int
    outcome: EnemyDecisionOutcomeMetadata

    @property
    def chosen_candidate(self) -> EnemyDecisionCandidateRecord:
        return self.candidates[self.chosen_index]


@dataclass(frozen=True)
class DatasetExtractionConfig:
    split: DatasetSplit
    encounter_ids: tuple[str, ...]
    hero_policy_ids: tuple[str, ...]
    seeds: tuple[int, ...] = ()
    max_rounds: int = 20
    enemy_wait_mode: str = "none"
    enemy_movement_mode: str = "recovery_only"


@dataclass(frozen=True)
class EnemyDecisionDataset:
    format_id: str
    format_version: int
    feature_schema_id: str
    feature_schema_version: int
    split_spec_id: str
    split_spec_version: int
    config: DatasetExtractionConfig
    records: tuple[EnemyDecisionDatasetRecord, ...]


def extract_enemy_decision_dataset(
    config: DatasetExtractionConfig,
    *,
    definitions: GameDefinitions | None = None,
    feature_schema: EnemyDecisionFeatureSchema = ENEMY_DECISION_FEATURE_SCHEMA,
    split_spec: SeedSplitSpec = DEFAULT_SEED_SPLIT_SPEC,
) -> EnemyDecisionDataset:
    """Run deterministic simulator episodes and group every recorded enemy decision."""
    game_definitions = definitions or load_game_definitions()
    _validate_extraction_config(config, game_definitions)
    seeds = config.seeds or split_spec.seeds_for(config.split)
    split_spec.validate_seeds(config.split, seeds)

    records: list[EnemyDecisionDatasetRecord] = []
    for hero_policy_id in config.hero_policy_ids:
        for encounter_id in config.encounter_ids:
            for seed in seeds:
                company = create_new_company(game_definitions)
                combat = create_encounter_combat(company, game_definitions, encounter_id)
                hero_policy = create_hero_policy(
                    hero_policy_id,
                    encounter_id=encounter_id,
                    seed=seed,
                )
                episode = run_enemy_learning_episode(
                    combat,
                    game_definitions,
                    GameRng(seed),
                    max_rounds=config.max_rounds,
                    encounter_id=encounter_id,
                    encounter_name=encounter_id.replace("_", " ").title(),
                    hero_policy=hero_policy,
                    enemy_wait_mode=config.enemy_wait_mode,
                    enemy_movement_mode=config.enemy_movement_mode,
                )
                records.extend(
                    records_from_episode(
                        episode,
                        hero_policy_id=hero_policy_id,
                        split=config.split,
                        feature_schema=feature_schema,
                    )
                )

    return EnemyDecisionDataset(
        format_id=DATASET_FORMAT_ID,
        format_version=DATASET_FORMAT_VERSION,
        feature_schema_id=feature_schema.schema_id,
        feature_schema_version=feature_schema.version,
        split_spec_id=split_spec.spec_id,
        split_spec_version=split_spec.version,
        config=config,
        records=tuple(records),
    )


def records_from_episode(
    episode: EnemyDecisionEpisode,
    *,
    hero_policy_id: str,
    split: DatasetSplit,
    feature_schema: EnemyDecisionFeatureSchema = ENEMY_DECISION_FEATURE_SCHEMA,
) -> tuple[EnemyDecisionDatasetRecord, ...]:
    if episode.seed is None:
        raise ValueError("Enemy decision datasets require an explicit simulation seed.")

    records: list[EnemyDecisionDatasetRecord] = []
    for source in episode.records:
        chosen = source.trace.chosen
        if chosen is None:
            raise ValueError("Recorded enemy decision is missing its selected candidate.")
        candidate_records = tuple(
            EnemyDecisionCandidateRecord(
                skill_id=candidate.skill_id,
                target_id=candidate.target_id,
                skill_order=candidate.skill_order,
                target_order=candidate.target_order,
                skill_tags=tuple(sorted(candidate.skill_tags)),
                feature_values=feature_schema.encode(candidate.features),
            )
            for candidate in source.trace.candidates
        )
        chosen_matches = tuple(
            index
            for index, candidate in enumerate(source.trace.candidates)
            if candidate == chosen
        )
        if len(chosen_matches) != 1:
            raise ValueError(
                "Recorded enemy decision must select exactly one member of its legal candidate set."
            )
        records.append(
            EnemyDecisionDatasetRecord(
                feature_schema_id=feature_schema.schema_id,
                feature_schema_version=feature_schema.version,
                split=split,
                encounter_id=episode.encounter_id,
                encounter_name=episode.encounter_name,
                simulation_seed=episode.seed,
                hero_policy_id=hero_policy_id,
                enemy_id=source.enemy_id,
                enemy_class_id=source.enemy_class_id,
                enemy_tags=source.enemy_tags,
                round_number=source.round_number,
                action_index=source.action_index,
                candidates=candidate_records,
                chosen_index=chosen_matches[0],
                outcome=EnemyDecisionOutcomeMetadata(
                    action_reward=source.action_reward,
                    episode_final_victor=episode.final_victor,
                    episode_total_reward=episode.total_reward,
                ),
            )
        )
    return tuple(records)


def _validate_extraction_config(
    config: DatasetExtractionConfig,
    definitions: GameDefinitions,
) -> None:
    if not config.encounter_ids:
        raise ValueError("Dataset extraction requires at least one explicit encounter id.")
    if not config.hero_policy_ids:
        raise ValueError("Dataset extraction requires at least one explicit hero policy id.")
    if config.max_rounds <= 0:
        raise ValueError("Dataset extraction max_rounds must be positive.")

    missing_encounters = tuple(
        encounter_id
        for encounter_id in config.encounter_ids
        if encounter_id not in definitions.encounters
    )
    if missing_encounters:
        raise ValueError(f"Unknown encounter id: {', '.join(missing_encounters)}")
    for hero_policy_id in config.hero_policy_ids:
        create_hero_policy(hero_policy_id)
