"""Pairwise outcome-ranking models and metrics for CP3."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from game.ml.models import CandidateScorerSpec, build_candidate_scorer
from game.ml.outcome_dataset import (
    CounterfactualDataset,
    PreferencePair,
    decisive_preference_pairs,
)
from game.ml.outcome_features import CP3_FEATURE_SCHEMA
from game.ml.splits import DatasetSplit

OUTCOME_NORMALIZATION_FORMAT_ID = "enemy-outcome-normalization-v1"
OUTCOME_MODEL_FORMAT_ID = "enemy-outcome-ranker-v1"


@dataclass(frozen=True)
class OutcomeNormalization:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    zero_variance: tuple[bool, ...]
    schema_id: str = CP3_FEATURE_SCHEMA.schema_id
    schema_version: int = CP3_FEATURE_SCHEMA.version

    def __post_init__(self) -> None:
        expected = len(self.feature_names)
        if self.feature_names != CP3_FEATURE_SCHEMA.names:
            raise ValueError("Outcome normalization feature order does not match CP3 schema.")
        if len(self.means) != expected or len(self.scales) != expected:
            raise ValueError("Outcome normalization statistic length mismatch.")
        if len(self.zero_variance) != expected:
            raise ValueError("Outcome normalization zero-variance length mismatch.")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("Outcome normalization scales must be positive.")

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        means = torch.tensor(self.means, dtype=values.dtype, device=values.device)
        scales = torch.tensor(self.scales, dtype=values.dtype, device=values.device)
        return (values - means) / scales

    def to_dict(self) -> dict[str, object]:
        return {
            "format_id": OUTCOME_NORMALIZATION_FORMAT_ID,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "zero_variance": list(self.zero_variance),
            "fit_split": "train",
        }


@dataclass(frozen=True)
class PairwiseTrainingConfig:
    learning_rate: float = 0.005
    batch_size: int = 1024
    max_epochs: int = 60
    early_stopping_patience: int = 8
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


@dataclass(frozen=True)
class PairwiseEpochResult:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class PairwiseTrainingResult:
    training_seed: int
    selected_epoch: int
    epochs_ran: int
    best_validation_loss: float
    history: tuple[PairwiseEpochResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "training_seed": self.training_seed,
            "selected_epoch": self.selected_epoch,
            "epochs_ran": self.epochs_ran,
            "best_validation_loss": self.best_validation_loss,
            "history": [
                {
                    "epoch": item.epoch,
                    "train_loss": item.train_loss,
                    "validation_loss": item.validation_loss,
                    "validation_accuracy": item.validation_accuracy,
                }
                for item in self.history
            ],
        }


@dataclass
class TrainedOutcomeRanker:
    spec: CandidateScorerSpec
    model: nn.Module
    config: PairwiseTrainingConfig
    result: PairwiseTrainingResult


@dataclass(frozen=True)
class LoadedOutcomeRanker:
    spec: CandidateScorerSpec
    model: nn.Module
    metadata: dict[str, Any]


def fit_outcome_normalization(dataset: CounterfactualDataset) -> OutcomeNormalization:
    if dataset.split is not DatasetSplit.TRAIN:
        raise ValueError("Outcome normalization may only be fitted on TRAIN.")
    vectors = [
        candidate.observation for record in dataset.records for candidate in record.candidates
    ]
    if not vectors:
        raise ValueError("Cannot normalize an empty counterfactual dataset.")
    values = torch.tensor(vectors, dtype=torch.float64)
    means = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    zero = std == 0
    scales = torch.where(zero, torch.ones_like(std), std)
    return OutcomeNormalization(
        feature_names=CP3_FEATURE_SCHEMA.names,
        means=tuple(float(item) for item in means.tolist()),
        scales=tuple(float(item) for item in scales.tolist()),
        zero_variance=tuple(bool(item) for item in zero.tolist()),
    )


def save_outcome_normalization(normalization: OutcomeNormalization, path: Path) -> str:
    text = json.dumps(normalization.to_dict(), sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _sha256_file(path)


def load_outcome_normalization(path: Path) -> OutcomeNormalization:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_id") != OUTCOME_NORMALIZATION_FORMAT_ID:
        raise ValueError("Unsupported outcome normalization format.")
    names = tuple(str(name) for name in payload["feature_names"])
    return OutcomeNormalization(
        feature_names=names,
        means=tuple(float(item) for item in payload["means"]),
        scales=tuple(float(item) for item in payload["scales"]),
        zero_variance=tuple(bool(item) for item in payload["zero_variance"]),
        schema_id=str(payload["schema_id"]),
        schema_version=int(payload["schema_version"]),
    )


def train_pairwise_ranker(
    spec: CandidateScorerSpec,
    train_dataset: CounterfactualDataset,
    validation_dataset: CounterfactualDataset,
    normalization: OutcomeNormalization,
    *,
    training_seed: int,
    config: PairwiseTrainingConfig,
) -> TrainedOutcomeRanker:
    if spec.feature_names != CP3_FEATURE_SCHEMA.names:
        raise ValueError("Outcome ranker must use the frozen CP3 observation schema.")
    train_pairs = decisive_preference_pairs(train_dataset)
    validation_pairs = decisive_preference_pairs(validation_dataset)
    if not train_pairs or not validation_pairs:
        raise ValueError("Pairwise training requires decisive TRAIN and VALIDATION pairs.")
    torch.manual_seed(training_seed)
    torch.use_deterministic_algorithms(True)
    model = build_candidate_scorer(spec)
    if config.optimizer != "adam":
        raise ValueError("CP3 v1 supports only Adam.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[PairwiseEpochResult] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(training_seed + epoch)
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        batch_losses: list[tuple[float, int]] = []
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            preferred, other = _pair_tensors(train_dataset, train_pairs, indices, normalization)
            optimizer.zero_grad(set_to_none=True)
            difference = model(preferred) - model(other)
            loss = F.softplus(-difference).mean()
            loss.backward()
            optimizer.step()
            batch_losses.append((float(loss.detach()), len(indices)))
        train_loss = _weighted_mean(batch_losses)
        validation_loss, validation_accuracy = evaluate_pairwise(
            model,
            validation_dataset,
            normalization,
        )
        history.append(PairwiseEpochResult(epoch, train_loss, validation_loss, validation_accuracy))
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("Pairwise training failed to select a model state.")
    model.load_state_dict(best_state)
    model.eval()
    return TrainedOutcomeRanker(
        spec=spec,
        model=model,
        config=config,
        result=PairwiseTrainingResult(
            training_seed=training_seed,
            selected_epoch=best_epoch,
            epochs_ran=len(history),
            best_validation_loss=best_loss,
            history=tuple(history),
        ),
    )


def evaluate_pairwise(
    model: nn.Module,
    dataset: CounterfactualDataset,
    normalization: OutcomeNormalization,
) -> tuple[float, float]:
    pairs = decisive_preference_pairs(dataset)
    if not pairs:
        return 0.0, 0.0
    preferred, other = _pair_tensors(dataset, pairs, list(range(len(pairs))), normalization)
    model.eval()
    with torch.no_grad():
        difference = model(preferred) - model(other)
        loss = float(F.softplus(-difference).mean())
        accuracy = float((difference > 0).to(torch.float64).mean())
    return loss, accuracy


def grouped_pairwise_accuracy(
    model: nn.Module,
    dataset: CounterfactualDataset,
    normalization: OutcomeNormalization,
    *,
    group_by: str,
) -> dict[str, dict[str, float | int]]:
    pairs = decisive_preference_pairs(dataset)
    grouped: dict[str, list[PreferencePair]] = {}
    for pair in pairs:
        record = dataset.records[pair.record_index]
        if group_by == "encounter":
            key = record.encounter_id
        elif group_by == "hero_policy":
            key = record.hero_policy_id
        elif group_by == "enemy_class":
            key = record.enemy_class_id
        else:
            raise ValueError(f"Unknown pairwise grouping: {group_by}")
        grouped.setdefault(key, []).append(pair)
    output: dict[str, dict[str, float | int]] = {}
    for key, items in sorted(grouped.items()):
        preferred, other = _pair_tensors(
            dataset, tuple(items), list(range(len(items))), normalization
        )
        with torch.no_grad():
            difference = model(preferred) - model(other)
            output[key] = {
                "pair_count": len(items),
                "accuracy": float((difference > 0).to(torch.float64).mean()),
                "loss": float(F.softplus(-difference).mean()),
            }
    return output


def choose_candidate_index(
    model: nn.Module,
    record_index: int,
    dataset: CounterfactualDataset,
    normalization: OutcomeNormalization,
) -> int:
    record = dataset.records[record_index]
    raw = torch.tensor(
        [candidate.observation for candidate in record.candidates], dtype=torch.float32
    )
    values = normalization.normalize(raw)
    model.eval()
    with torch.no_grad():
        scores = model(values)
    return int(torch.argmax(scores).item())


def save_outcome_ranker(
    trained: TrainedOutcomeRanker,
    metadata_path: Path,
    *,
    normalization_checksum: str,
) -> dict[str, Any]:
    model_path = metadata_path.with_suffix(".pt")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained.model.state_dict(), model_path)
    model_checksum = _sha256_file(model_path)
    payload: dict[str, Any] = {
        "format_id": OUTCOME_MODEL_FORMAT_ID,
        "observation_schema_id": CP3_FEATURE_SCHEMA.schema_id,
        "observation_schema_version": CP3_FEATURE_SCHEMA.version,
        "scorer_spec": trained.spec.to_dict(),
        "training_config": trained.config.to_dict(),
        "training_result": trained.result.to_dict(),
        "normalization_checksum": normalization_checksum,
        "model_file": model_path.name,
        "model_checksum": model_checksum,
        "runtime_versions": {"python": sys.version.split()[0], "torch": torch.__version__},
    }
    metadata_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def load_outcome_ranker(metadata_path: Path) -> LoadedOutcomeRanker:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("format_id") != OUTCOME_MODEL_FORMAT_ID:
        raise ValueError("Unsupported outcome ranker artifact format.")
    if payload.get("observation_schema_id") != CP3_FEATURE_SCHEMA.schema_id:
        raise ValueError("Outcome ranker observation schema mismatch.")
    spec_payload = payload["scorer_spec"]
    spec = CandidateScorerSpec(
        model_type=spec_payload["model_type"],
        feature_names=tuple(spec_payload["feature_names"]),
        hidden_units=tuple(spec_payload["hidden_units"]),
        activation=spec_payload["activation"],
    )
    model_path = metadata_path.with_name(str(payload["model_file"]))
    if _sha256_file(model_path) != payload["model_checksum"]:
        raise ValueError("Outcome ranker model checksum mismatch.")
    model = build_candidate_scorer(spec)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return LoadedOutcomeRanker(spec=spec, model=model, metadata=payload)


def _pair_tensors(
    dataset: CounterfactualDataset,
    pairs: tuple[PreferencePair, ...],
    indices: list[int],
    normalization: OutcomeNormalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    preferred_vectors: list[tuple[float, ...]] = []
    other_vectors: list[tuple[float, ...]] = []
    for index in indices:
        pair = pairs[index]
        record = dataset.records[pair.record_index]
        preferred_vectors.append(record.candidates[pair.preferred_index].observation)
        other_vectors.append(record.candidates[pair.other_index].observation)
    preferred = normalization.normalize(torch.tensor(preferred_vectors, dtype=torch.float32))
    other = normalization.normalize(torch.tensor(other_vectors, dtype=torch.float32))
    return preferred, other


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    total_count = sum(count for _value, count in values)
    return sum(value * count for value, count in values) / total_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
