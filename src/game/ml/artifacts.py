"""Small deterministic provenance manifests for ML research artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from game.ml.dataset import DatasetExtractionConfig
from game.ml.features import ENEMY_DECISION_FEATURE_SCHEMA
from game.ml.splits import DEFAULT_SEED_SPLIT_SPEC

ARTIFACT_MANIFEST_FORMAT_ID = "charter-ml-artifact-manifest-v1"
ARTIFACT_MANIFEST_FORMAT_VERSION = 1
ManifestScalar = str | int | float | bool | None
ManifestPairs = tuple[tuple[str, ManifestScalar], ...]


@dataclass(frozen=True)
class ResearchArtifactManifest:
    artifact_kind: str
    repository_commit: str
    split_name: str
    encounter_ids: tuple[str, ...]
    training_hero_policy_ids: tuple[str, ...]
    evaluation_hero_policy_ids: tuple[str, ...] = ()
    content_hashes: tuple[tuple[str, str], ...] = ()
    extraction_config: ManifestPairs = ()
    model_type: str = ""
    model_config: ManifestPairs = ()
    normalization_provenance: ManifestPairs = ()
    optimizer_config: ManifestPairs = ()
    runtime_versions: tuple[tuple[str, str], ...] = ()
    model_checksum: str = ""
    format_id: str = ARTIFACT_MANIFEST_FORMAT_ID
    format_version: int = ARTIFACT_MANIFEST_FORMAT_VERSION
    feature_schema_id: str = ENEMY_DECISION_FEATURE_SCHEMA.schema_id
    feature_schema_version: int = ENEMY_DECISION_FEATURE_SCHEMA.version
    split_spec_id: str = DEFAULT_SEED_SPLIT_SPEC.spec_id
    split_spec_version: int = DEFAULT_SEED_SPLIT_SPEC.version

    def to_dict(self) -> dict[str, object]:
        return {
            "format_id": self.format_id,
            "format_version": self.format_version,
            "artifact_kind": self.artifact_kind,
            "feature_schema_id": self.feature_schema_id,
            "feature_schema_version": self.feature_schema_version,
            "repository_commit": self.repository_commit,
            "content_hashes": dict(sorted(self.content_hashes)),
            "split_spec_id": self.split_spec_id,
            "split_spec_version": self.split_spec_version,
            "split_name": self.split_name,
            "encounter_ids": list(self.encounter_ids),
            "training_hero_policy_ids": list(self.training_hero_policy_ids),
            "evaluation_hero_policy_ids": list(self.evaluation_hero_policy_ids),
            "extraction_config": dict(sorted(self.extraction_config)),
            "model_type": self.model_type,
            "model_config": dict(sorted(self.model_config)),
            "normalization_provenance": dict(sorted(self.normalization_provenance)),
            "optimizer_config": dict(sorted(self.optimizer_config)),
            "runtime_versions": dict(sorted(self.runtime_versions)),
            "model_checksum": self.model_checksum,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def extraction_manifest_pairs(config: DatasetExtractionConfig) -> ManifestPairs:
    seeds = ",".join(str(seed) for seed in config.seeds) if config.seeds else "split-default"
    return (
        ("enemy_movement_mode", config.enemy_movement_mode),
        ("enemy_wait_mode", config.enemy_wait_mode),
        ("max_rounds", config.max_rounds),
        ("seeds", seeds),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
