"""Deterministic supervised training utilities for grouped enemy decisions."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from game.ml.artifacts import sha256_file
from game.ml.dataset import EnemyDecisionDataset
from game.ml.features import ENEMY_DECISION_FEATURE_SCHEMA
from game.ml.models import CandidateScorerSpec, build_candidate_scorer
from game.ml.splits import DatasetSplit

NORMALIZATION_FORMAT_ID = "enemy-decision-normalization-v1"
NORMALIZATION_FORMAT_VERSION = 1
MODEL_ARTIFACT_FORMAT_ID = "enemy-decision-candidate-scorer-v1"
MODEL_ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FeatureNormalization:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    zero_variance: tuple[bool, ...]
    feature_schema_id: str = ENEMY_DECISION_FEATURE_SCHEMA.schema_id
    feature_schema_version: int = ENEMY_DECISION_FEATURE_SCHEMA.version
    format_id: str = NORMALIZATION_FORMAT_ID
    format_version: int = NORMALIZATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        expected = len(ENEMY_DECISION_FEATURE_SCHEMA.names)
        if len(self.means) != expected or len(self.scales) != expected:
            raise ValueError(f"Normalization expected {expected} statistics per field.")
        if len(self.zero_variance) != expected:
            raise ValueError(f"Normalization expected {expected} zero-variance flags.")
        if any(scale <= 0.0 for scale in self.scales):
            raise ValueError("Normalization scales must be positive.")
        _validate_schema_identity(self.feature_schema_id, self.feature_schema_version)

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != len(self.means):
            raise ValueError(
                f"Normalization expected trailing dimension {len(self.means)}, "
                f"got {values.shape[-1]}"
            )
        means = torch.tensor(self.means, dtype=values.dtype, device=values.device)
        scales = torch.tensor(self.scales, dtype=values.dtype, device=values.device)
        return (values - means) / scales

    def to_dict(self) -> dict[str, object]:
        return {
            "format_id": self.format_id,
            "format_version": self.format_version,
            "feature_schema_id": self.feature_schema_id,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(ENEMY_DECISION_FEATURE_SCHEMA.names),
            "means": list(self.means),
            "scales": list(self.scales),
            "zero_variance": list(self.zero_variance),
            "zero_variance_scale": 1.0,
            "fit_split": DatasetSplit.TRAIN.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureNormalization:
        if payload.get("format_id") != NORMALIZATION_FORMAT_ID:
            raise ValueError("Unsupported normalization format id.")
        if payload.get("format_version") != NORMALIZATION_FORMAT_VERSION:
            raise ValueError("Unsupported normalization format version.")
        names = tuple(str(name) for name in payload.get("feature_names", ()))
        if names != ENEMY_DECISION_FEATURE_SCHEMA.names:
            raise ValueError("Normalization feature ordering is incompatible with current schema.")
        return cls(
            means=tuple(float(value) for value in payload["means"]),
            scales=tuple(float(value) for value in payload["scales"]),
            zero_variance=tuple(bool(value) for value in payload["zero_variance"]),
            feature_schema_id=str(payload["feature_schema_id"]),
            feature_schema_version=int(payload["feature_schema_version"]),
        )


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01
    batch_size: int = 512
    max_epochs: int = 50
    early_stopping_patience: int = 6
    min_delta: float = 1e-5
    optimizer: str = "adam"

    def to_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
            "optimizer": self.optimizer,
        }


_DEFAULT_TRAINING_CONFIG = TrainingConfig()


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_cross_entropy: float
    validation_cross_entropy: float


@dataclass(frozen=True)
class TrainingResult:
    training_seed: int
    selected_epoch: int
    epochs_ran: int
    best_validation_cross_entropy: float
    history: tuple[EpochResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "training_seed": self.training_seed,
            "selected_epoch": self.selected_epoch,
            "epochs_ran": self.epochs_ran,
            "best_validation_cross_entropy": self.best_validation_cross_entropy,
            "history": [
                {
                    "epoch": item.epoch,
                    "train_cross_entropy": item.train_cross_entropy,
                    "validation_cross_entropy": item.validation_cross_entropy,
                }
                for item in self.history
            ],
        }


@dataclass
class TrainedCandidateScorer:
    spec: CandidateScorerSpec
    model: nn.Module
    training_config: TrainingConfig
    result: TrainingResult


@dataclass(frozen=True)
class PreparedDecisionGroup:
    candidate_count: int
    record_indices: tuple[int, ...]
    features: torch.Tensor
    chosen_indices: torch.Tensor


@dataclass(frozen=True)
class PreparedDecisionGroups:
    feature_names: tuple[str, ...]
    groups: tuple[PreparedDecisionGroup, ...]
    decision_count: int


def fit_train_normalization(dataset: EnemyDecisionDataset) -> FeatureNormalization:
    if dataset.config.split is not DatasetSplit.TRAIN:
        raise ValueError("Normalization may only be fitted from the TRAIN split.")
    vectors = [
        candidate.feature_values
        for record in dataset.records
        for candidate in record.candidates
    ]
    if not vectors:
        raise ValueError("Cannot fit normalization from an empty training dataset.")
    values = torch.tensor(vectors, dtype=torch.float64)
    means = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    zero_variance = std == 0.0
    scales = torch.where(zero_variance, torch.ones_like(std), std)
    return FeatureNormalization(
        means=tuple(float(value) for value in means.tolist()),
        scales=tuple(float(value) for value in scales.tolist()),
        zero_variance=tuple(bool(value) for value in zero_variance.tolist()),
    )


def save_normalization(normalization: FeatureNormalization, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalization.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def load_normalization(path: Path) -> FeatureNormalization:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Normalization artifact must contain a JSON object.")
    return FeatureNormalization.from_dict(payload)


def prepare_decision_groups(
    dataset: EnemyDecisionDataset,
    normalization: FeatureNormalization,
    feature_names: tuple[str, ...],
) -> PreparedDecisionGroups:
    feature_indices = ENEMY_DECISION_FEATURE_SCHEMA.indices_for(feature_names)
    grouped_indices: dict[int, list[int]] = {}
    for record_index, record in enumerate(dataset.records):
        candidate_count = len(record.candidates)
        if candidate_count <= 0:
            raise ValueError("Every training decision must contain at least one legal candidate.")
        grouped_indices.setdefault(candidate_count, []).append(record_index)

    groups: list[PreparedDecisionGroup] = []
    for candidate_count in sorted(grouped_indices):
        record_indices = tuple(grouped_indices[candidate_count])
        raw = torch.tensor(
            [
                [
                    dataset.records[index].candidates[candidate].feature_values
                    for candidate in range(candidate_count)
                ]
                for index in record_indices
            ],
            dtype=torch.float32,
        )
        normalized = normalization.normalize_tensor(raw)
        selected = normalized[..., list(feature_indices)]
        targets = torch.tensor(
            [dataset.records[index].chosen_index for index in record_indices],
            dtype=torch.long,
        )
        groups.append(
            PreparedDecisionGroup(
                candidate_count=candidate_count,
                record_indices=record_indices,
                features=selected,
                chosen_indices=targets,
            )
        )
    return PreparedDecisionGroups(
        feature_names=feature_names,
        groups=tuple(groups),
        decision_count=len(dataset.records),
    )


def candidate_set_logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("Candidate-set features must be [decisions, candidates, features].")
    decisions, candidates, width = features.shape
    logits = model(features.reshape(decisions * candidates, width))
    return logits.reshape(decisions, candidates)


def candidate_set_cross_entropy(
    model: nn.Module,
    features: torch.Tensor,
    chosen_indices: torch.Tensor,
) -> torch.Tensor:
    logits = candidate_set_logits(model, features)
    if chosen_indices.shape != (features.shape[0],):
        raise ValueError("Chosen-index tensor must contain one label per decision.")
    return F.cross_entropy(logits, chosen_indices, reduction="mean")


def train_candidate_scorer(
    spec: CandidateScorerSpec,
    train_groups: PreparedDecisionGroups,
    validation_groups: PreparedDecisionGroups,
    *,
    training_seed: int,
    config: TrainingConfig = _DEFAULT_TRAINING_CONFIG,
) -> TrainedCandidateScorer:
    if train_groups.feature_names != spec.feature_names:
        raise ValueError("Training feature selection does not match scorer specification.")
    if validation_groups.feature_names != spec.feature_names:
        raise ValueError("Validation feature selection does not match scorer specification.")
    if config.optimizer != "adam":
        raise ValueError("Checkpoint 2 predeclares Adam as the only optimizer.")
    if config.batch_size <= 0 or config.max_epochs <= 0:
        raise ValueError("Training batch size and max epochs must be positive.")

    torch.manual_seed(training_seed)
    torch.use_deterministic_algorithms(True)
    model = build_candidate_scorer(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(training_seed)

    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[EpochResult] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_decisions = 0
        for group in train_groups.groups:
            permutation = torch.randperm(group.features.shape[0], generator=generator)
            for start in range(0, len(permutation), config.batch_size):
                indices = permutation[start : start + config.batch_size]
                batch_features = group.features[indices]
                batch_targets = group.chosen_indices[indices]
                optimizer.zero_grad(set_to_none=True)
                loss = candidate_set_cross_entropy(model, batch_features, batch_targets)
                loss.backward()
                optimizer.step()
                count = int(batch_features.shape[0])
                train_loss_total += float(loss.detach()) * count
                train_decisions += count

        validation_ce = prepared_cross_entropy(model, validation_groups, config.batch_size)
        train_ce = train_loss_total / max(1, train_decisions)
        history.append(EpochResult(epoch, train_ce, validation_ce))
        if validation_ce < best_validation - config.min_delta:
            best_validation = validation_ce
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Training never produced a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return TrainedCandidateScorer(
        spec=spec,
        model=model,
        training_config=config,
        result=TrainingResult(
            training_seed=training_seed,
            selected_epoch=best_epoch,
            epochs_ran=len(history),
            best_validation_cross_entropy=best_validation,
            history=tuple(history),
        ),
    )


def prepared_cross_entropy(
    model: nn.Module,
    groups: PreparedDecisionGroups,
    batch_size: int,
) -> float:
    model.eval()
    loss_total = 0.0
    decision_total = 0
    with torch.no_grad():
        for group in groups.groups:
            for start in range(0, group.features.shape[0], batch_size):
                features = group.features[start : start + batch_size]
                targets = group.chosen_indices[start : start + batch_size]
                loss = candidate_set_cross_entropy(model, features, targets)
                count = int(features.shape[0])
                loss_total += float(loss) * count
                decision_total += count
    if decision_total == 0:
        raise ValueError("Cannot evaluate cross-entropy on an empty decision set.")
    return loss_total / decision_total


def save_model_artifact(
    trained: TrainedCandidateScorer,
    metadata_path: Path,
    *,
    normalization_checksum: str,
) -> dict[str, Any]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = metadata_path.with_suffix(".pt")
    torch.save(trained.model.state_dict(), model_path)
    model_checksum = sha256_file(model_path)
    payload: dict[str, Any] = {
        "format_id": MODEL_ARTIFACT_FORMAT_ID,
        "format_version": MODEL_ARTIFACT_FORMAT_VERSION,
        "feature_schema_id": ENEMY_DECISION_FEATURE_SCHEMA.schema_id,
        "feature_schema_version": ENEMY_DECISION_FEATURE_SCHEMA.version,
        "model_file": model_path.name,
        "model_checksum": model_checksum,
        "normalization_checksum": normalization_checksum,
        "scorer_spec": trained.spec.to_dict(),
        "training_config": trained.training_config.to_dict(),
        "training_result": trained.result.to_dict(),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
    }
    metadata_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


@dataclass
class LoadedCandidateScorer:
    spec: CandidateScorerSpec
    model: nn.Module
    metadata: dict[str, object]


def load_model_artifact(metadata_path: Path) -> LoadedCandidateScorer:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model artifact metadata must contain a JSON object.")
    if payload.get("format_id") != MODEL_ARTIFACT_FORMAT_ID:
        raise ValueError("Unsupported model artifact format id.")
    if payload.get("format_version") != MODEL_ARTIFACT_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format version.")
    _validate_schema_identity(
        str(payload.get("feature_schema_id")),
        int(payload.get("feature_schema_version", -1)),
    )
    spec_payload = payload.get("scorer_spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("Model artifact is missing scorer specification.")
    model_type = str(spec_payload.get("model_type"))
    if model_type not in {"linear", "mlp"}:
        raise ValueError(f"Unknown model artifact type: {model_type}")
    spec = CandidateScorerSpec(
        model_type=model_type,  # type: ignore[arg-type]
        feature_names=tuple(str(name) for name in spec_payload.get("feature_names", ())),
        hidden_units=tuple(int(value) for value in spec_payload.get("hidden_units", ())),
        activation=str(spec_payload.get("activation")),
    )
    ENEMY_DECISION_FEATURE_SCHEMA.indices_for(spec.feature_names)
    model_path = metadata_path.with_name(str(payload["model_file"]))
    actual_checksum = sha256_file(model_path)
    if actual_checksum != payload.get("model_checksum"):
        raise ValueError("Model artifact checksum mismatch.")
    model = build_candidate_scorer(spec)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedCandidateScorer(spec=spec, model=model, metadata=payload)


def _validate_schema_identity(schema_id: str, schema_version: int) -> None:
    if schema_id != ENEMY_DECISION_FEATURE_SCHEMA.schema_id:
        raise ValueError(
            f"Incompatible feature schema id: {schema_id}; "
            f"expected {ENEMY_DECISION_FEATURE_SCHEMA.schema_id}."
        )
    if schema_version != ENEMY_DECISION_FEATURE_SCHEMA.version:
        raise ValueError(
            f"Incompatible feature schema version: {schema_version}; "
            f"expected {ENEMY_DECISION_FEATURE_SCHEMA.version}."
        )


