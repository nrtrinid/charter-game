"""Research-only candidate scorers for supervised enemy-policy imitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

ModelType = Literal["linear", "mlp"]


@dataclass(frozen=True)
class CandidateScorerSpec:
    model_type: ModelType
    feature_names: tuple[str, ...]
    hidden_units: tuple[int, ...] = ()
    activation: str = "none"

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "feature_names": list(self.feature_names),
            "input_dim": self.input_dim,
            "hidden_units": list(self.hidden_units),
            "activation": self.activation,
        }


class LinearCandidateScorer(nn.Module):
    """One normalized candidate feature vector -> one scalar logit."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class MLPCandidateScorer(nn.Module):
    """Small 64/32 ReLU MLP used only by the research harness."""

    def __init__(self, input_dim: int, hidden_units: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        first, second = hidden_units
        self.network = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def linear_scorer_spec(feature_names: tuple[str, ...]) -> CandidateScorerSpec:
    return CandidateScorerSpec(
        model_type="linear",
        feature_names=feature_names,
        hidden_units=(),
        activation="none",
    )


def mlp_scorer_spec(feature_names: tuple[str, ...]) -> CandidateScorerSpec:
    return CandidateScorerSpec(
        model_type="mlp",
        feature_names=feature_names,
        hidden_units=(64, 32),
        activation="relu",
    )


def build_candidate_scorer(spec: CandidateScorerSpec) -> nn.Module:
    if spec.input_dim <= 0:
        raise ValueError("Candidate scorer requires at least one input feature.")
    if spec.model_type == "linear":
        if spec.hidden_units or spec.activation != "none":
            raise ValueError("Linear scorer spec must not declare hidden layers or activation.")
        return LinearCandidateScorer(spec.input_dim)
    if spec.model_type == "mlp":
        if spec.hidden_units != (64, 32) or spec.activation != "relu":
            raise ValueError("MLP v1 must use the predeclared 64/32 ReLU architecture.")
        return MLPCandidateScorer(spec.input_dim, hidden_units=(64, 32))
    raise ValueError(f"Unknown candidate scorer type: {spec.model_type}")
