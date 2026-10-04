"""Dev-only ML research foundations built on the deterministic simulator."""

from game.ml.artifacts import ResearchArtifactManifest, extraction_manifest_pairs, sha256_file
from game.ml.dataset import (
    DATASET_FORMAT_ID,
    DATASET_FORMAT_VERSION,
    DatasetExtractionConfig,
    EnemyDecisionCandidateRecord,
    EnemyDecisionDataset,
    EnemyDecisionDatasetRecord,
    EnemyDecisionOutcomeMetadata,
    extract_enemy_decision_dataset,
    records_from_episode,
)
from game.ml.features import (
    ENEMY_DECISION_FEATURE_NAMES,
    ENEMY_DECISION_FEATURE_SCHEMA,
    ENEMY_DECISION_FEATURE_SCHEMA_ID,
    ENEMY_DECISION_FEATURE_SCHEMA_VERSION,
    EnemyDecisionFeatureSchema,
)
from game.ml.splits import (
    DEFAULT_SEED_SPLIT_SPEC,
    SEED_SPLIT_SPEC_ID,
    SEED_SPLIT_SPEC_VERSION,
    DatasetSplit,
    SeedPartition,
    SeedSplitSpec,
)

__all__ = [
    "DATASET_FORMAT_ID",
    "DATASET_FORMAT_VERSION",
    "DEFAULT_SEED_SPLIT_SPEC",
    "ENEMY_DECISION_FEATURE_NAMES",
    "ENEMY_DECISION_FEATURE_SCHEMA",
    "ENEMY_DECISION_FEATURE_SCHEMA_ID",
    "ENEMY_DECISION_FEATURE_SCHEMA_VERSION",
    "SEED_SPLIT_SPEC_ID",
    "SEED_SPLIT_SPEC_VERSION",
    "DatasetExtractionConfig",
    "DatasetSplit",
    "EnemyDecisionCandidateRecord",
    "EnemyDecisionDataset",
    "EnemyDecisionDatasetRecord",
    "EnemyDecisionFeatureSchema",
    "EnemyDecisionOutcomeMetadata",
    "ResearchArtifactManifest",
    "SeedPartition",
    "SeedSplitSpec",
    "extract_enemy_decision_dataset",
    "extraction_manifest_pairs",
    "records_from_episode",
    "sha256_file",
]
