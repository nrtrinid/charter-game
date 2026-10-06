# ruff: noqa: E501
"""CP10 conservative route-policy research helpers.

Research only. Production combat owns legality, resolution, persistence, and termination.
This module owns transparent support diagnostics and conservative policy control.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from game.combat.combat_state import CombatState
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionPolicy,
    EnemyDecisionRuntimeContext,
    explain_enemy_decision,
)
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.ml.counterfactual import CandidateKey, decision_state_digest, rng_state_digest
from game.ml.outcome_features import CP3_FEATURE_SCHEMA
from game.ml.route_policy_learning import (
    PRIMARY_FEATURE_NAMES,
    FrozenRouteRankerPolicy,
    RouteNormalization,
    RouteSourceDecision,
    candidate_key,
    encode_primary,
)

_DEFAULT_RUNTIME = EnemyDecisionRuntimeContext()

WELL_SUPPORTED = "WELL_SUPPORTED"
EDGE_OF_SUPPORT = "EDGE_OF_SUPPORT"
POORLY_SUPPORTED = "POORLY_SUPPORTED"


@dataclass(frozen=True)
class SupportThresholds:
    well_distance_max: float
    edge_distance_max: float
    well_range_exceedance_max: int = 0
    edge_range_exceedance_max: int = 2


@dataclass(frozen=True)
class SupportEvidence:
    category: str
    nearest_neighbor_distance: float
    feature_range_exceedances: int


@dataclass(frozen=True)
class SupportProfile:
    feature_names: tuple[str, ...]
    train_vectors: tuple[tuple[float, ...], ...]
    normalized_min: tuple[float, ...]
    normalized_max: tuple[float, ...]
    thresholds: SupportThresholds

    def classify_normalized(self, vector: torch.Tensor) -> SupportEvidence:
        if tuple(self.feature_names) != PRIMARY_FEATURE_NAMES:
            raise ValueError("Support profile schema does not match CP9 primary route schema.")
        train = torch.tensor(self.train_vectors, dtype=torch.float32)
        distance = torch.linalg.vector_norm(train - vector, dim=1) / math.sqrt(vector.numel())
        nearest = float(distance.min().item())
        lo = torch.tensor(self.normalized_min, dtype=torch.float32)
        hi = torch.tensor(self.normalized_max, dtype=torch.float32)
        exceedances = int(torch.logical_or(vector < lo, vector > hi).sum().item())
        t = self.thresholds
        if nearest <= t.well_distance_max and exceedances <= t.well_range_exceedance_max:
            category = WELL_SUPPORTED
        elif nearest <= t.edge_distance_max and exceedances <= t.edge_range_exceedance_max:
            category = EDGE_OF_SUPPORT
        else:
            category = POORLY_SUPPORTED
        return SupportEvidence(category, nearest, exceedances)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute support threshold from an empty training set.")
    point = (len(ordered) - 1) * probability
    low = int(point)
    high = min(len(ordered) - 1, low + 1)
    fraction = point - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def fit_support_profile(
    record_vectors: list[list[tuple[float, ...]]], normalization: RouteNormalization
) -> SupportProfile:
    if not record_vectors:
        raise ValueError("Support profile requires CP9 TRAIN records.")
    normalized_records: list[torch.Tensor] = []
    for vectors in record_vectors:
        raw = torch.tensor(vectors, dtype=torch.float32)
        normalized_records.append(normalization.normalize(raw))
    nearest_distances: list[float] = []
    for index, normalized_vectors in enumerate(normalized_records):
        others = torch.cat(
            [item for other_index, item in enumerate(normalized_records) if other_index != index],
            dim=0,
        )
        for candidate_vector in normalized_vectors:
            distances = torch.linalg.vector_norm(others - candidate_vector, dim=1) / math.sqrt(
                candidate_vector.numel()
            )
            nearest_distances.append(float(distances.min().item()))
    all_vectors = torch.cat(normalized_records, dim=0)
    thresholds = SupportThresholds(
        well_distance_max=_quantile(nearest_distances, 0.75),
        edge_distance_max=_quantile(nearest_distances, 0.95),
    )
    return SupportProfile(
        feature_names=tuple(normalization.feature_names),
        train_vectors=tuple(tuple(float(value) for value in row.tolist()) for row in all_vectors),
        normalized_min=tuple(float(value) for value in all_vectors.min(dim=0).values.tolist()),
        normalized_max=tuple(float(value) for value in all_vectors.max(dim=0).values.tolist()),
        thresholds=thresholds,
    )


def support_profile_payload(profile: SupportProfile) -> dict[str, Any]:
    return {
        "feature_names": list(profile.feature_names),
        "training_candidate_count": len(profile.train_vectors),
        "normalized_min": list(profile.normalized_min),
        "normalized_max": list(profile.normalized_max),
        "thresholds": asdict(profile.thresholds),
        "criteria": {
            WELL_SUPPORTED: "distance<=q75 AND feature_range_exceedances<=0",
            EDGE_OF_SUPPORT: "distance<=q95 AND feature_range_exceedances<=2",
            POORLY_SUPPORTED: "otherwise",
        },
        "distance_definition": "L2 distance in CP9 TRAIN-normalized feature space divided by sqrt(feature_count)",
        "threshold_source": "CP9 TRAIN candidates only; cross-source-record nearest neighbors",
    }


def route_scores(
    policy: FrozenRouteRankerPolicy,
    state: CombatState,
    definitions: GameDefinitions,
    candidates: tuple[EnemyDecisionCandidate, ...],
    runtime_context: EnemyDecisionRuntimeContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.tensor(
        [encode_primary(state, definitions, item, runtime_context) for item in candidates],
        dtype=torch.float32,
    )
    normalized = policy.normalization.normalize(raw)
    policy.model.eval()
    with torch.no_grad():
        scores = policy.model(normalized).reshape(-1)
    return normalized, scores


def pi2_scores(
    pi2_policy: EnemyDecisionPolicy,
    state: CombatState,
    definitions: GameDefinitions,
    candidates: tuple[EnemyDecisionCandidate, ...],
    runtime_context: EnemyDecisionRuntimeContext,
) -> torch.Tensor | None:
    model = getattr(pi2_policy, "model", None)
    normalization = getattr(pi2_policy, "normalization", None)
    if model is None or normalization is None:
        return None
    raw = torch.tensor(
        [
            CP3_FEATURE_SCHEMA.encode(state, definitions, item, runtime_context)
            for item in candidates
        ],
        dtype=torch.float32,
    )
    normalized = normalization.normalize(raw)
    model.eval()
    with torch.no_grad():
        return model(normalized).reshape(-1)


def candidate_family(observation: tuple[float, ...]) -> tuple[str, ...]:
    values = dict(zip(PRIMARY_FEATURE_NAMES, observation, strict=True))
    families: list[str] = []
    if values["skill_tag_mark"] > 0.5:
        families.append("MARK")
    if values["skill_tag_drag_forward"] > 0.5:
        families.append("GRAB")
    if values["skill_tag_guard"] > 0.5:
        families.append("GUARD")
    if values["skill_tag_effort_drain"] > 0.5:
        families.append("EFFORT_DRAIN")
    if values["projected_damage_max"] > 0.0:
        families.append("DIRECT_PRESSURE")
    return tuple(families)


class DiagnosticRoutePolicy:
    """Execute one policy while auditing π2/route choices, scores, and TRAIN support."""

    def __init__(
        self,
        *,
        executed_name: str,
        executed_policy: EnemyDecisionPolicy,
        anchor_policy: EnemyDecisionPolicy,
        route_policy: FrozenRouteRankerPolicy,
        support_profile: SupportProfile,
    ) -> None:
        self.executed_name = executed_name
        self.executed_policy = executed_policy
        self.anchor_policy = anchor_policy
        self.route_policy = route_policy
        self.support_profile = support_profile
        self.decisions: list[dict[str, Any]] = []

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            return None
        candidates = tuple(trace.candidates)
        anchor = self.anchor_policy.choose(state, definitions, enemy_id, runtime_context)
        route_choice = self.route_policy.choose(state, definitions, enemy_id, runtime_context)
        executed = self.executed_policy.choose(state, definitions, enemy_id, runtime_context)
        if anchor is None or route_choice is None or executed is None:
            return executed
        keys = tuple(candidate_key(item) for item in candidates)
        anchor_key = candidate_key(anchor)
        route_key = candidate_key(route_choice)
        executed_key = candidate_key(executed)
        normalized, scores = route_scores(
            self.route_policy, state, definitions, candidates, runtime_context
        )
        anchor_index = keys.index(anchor_key)
        route_index = keys.index(route_key)
        executed_index = keys.index(executed_key)
        order = torch.argsort(scores, descending=True)
        runner_index = int(order[1].item()) if len(order) > 1 else int(order[0].item())
        support = self.support_profile.classify_normalized(normalized[route_index])
        pi2 = pi2_scores(self.anchor_policy, state, definitions, candidates, runtime_context)
        observations = tuple(
            encode_primary(state, definitions, item, runtime_context) for item in candidates
        )
        target_hp_index = PRIMARY_FEATURE_NAMES.index("target_hp")
        target_hero_index = PRIMARY_FEATURE_NAMES.index("target_is_hero")
        hero_target_hps = [
            obs[target_hp_index] for obs in observations if obs[target_hero_index] > 0.5
        ]
        route_obs = observations[route_index]
        self.decisions.append(
            {
                "action_index": len(self.decisions),
                "round": state.round_number,
                "state_digest": decision_state_digest(state, runtime_context),
                "enemy_id": enemy_id,
                "enemy_class": state.actor(enemy_id).class_id,
                "candidate_keys": [asdict(item) for item in keys],
                "candidate_families": [list(candidate_family(obs)) for obs in observations],
                "anchor_key": asdict(anchor_key),
                "route_key": asdict(route_key),
                "executed_key": asdict(executed_key),
                "anchor_index": anchor_index,
                "route_index": route_index,
                "executed_index": executed_index,
                "route_scores": [float(value) for value in scores.tolist()],
                "pi2_scores": None if pi2 is None else [float(value) for value in pi2.tolist()],
                "top_vs_anchor_margin": float(scores[route_index] - scores[anchor_index]),
                "top_vs_runner_margin": float(scores[route_index] - scores[runner_index]),
                "support": asdict(support),
                "agreement": anchor_key == route_key,
                "party_hp_total": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("party_hp_total")]
                ),
                "party_effort_total": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("party_effort_total")]
                ),
                "marked_hero_count": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("marked_hero_count")]
                ),
                "enemy_hp_total": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("enemy_hp_total")]
                ),
                "enemy_effort_total": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("enemy_effort_total")]
                ),
                "actor_hp": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("actor_hp")]
                ),
                "actor_effort": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("actor_effort")]
                ),
                "route_choice_families": list(candidate_family(observations[route_index])),
                "anchor_choice_families": list(candidate_family(observations[anchor_index])),
                "route_target_hp": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("target_hp")]
                ),
                "route_target_is_lowest_hp": bool(
                    route_obs[target_hero_index] > 0.5
                    and hero_target_hps
                    and route_obs[target_hp_index] <= min(hero_target_hps)
                ),
                "route_target_is_downed": bool(
                    route_obs[PRIMARY_FEATURE_NAMES.index("target_is_downed")] > 0.5
                ),
                "route_target_support_capable": bool(
                    route_obs[PRIMARY_FEATURE_NAMES.index("target_heal_skill_count")]
                    or route_obs[PRIMARY_FEATURE_NAMES.index("target_support_skill_count")]
                    or route_obs[PRIMARY_FEATURE_NAMES.index("target_recovery_skill_count")]
                ),
                "route_target_protected": bool(
                    route_obs[PRIMARY_FEATURE_NAMES.index("target_is_protected")] > 0.5
                ),
                "anchor_target_hp": float(
                    observations[anchor_index][PRIMARY_FEATURE_NAMES.index("target_hp")]
                ),
                "route_projected_damage_max": float(
                    observations[route_index][PRIMARY_FEATURE_NAMES.index("projected_damage_max")]
                ),
                "anchor_projected_damage_max": float(
                    observations[anchor_index][PRIMARY_FEATURE_NAMES.index("projected_damage_max")]
                ),
            }
        )
        return executed


class ConservativeRoutePolicy:
    """Use route-ranker deviations only when a frozen evidence gate accepts them."""

    def __init__(
        self,
        *,
        anchor_policy: EnemyDecisionPolicy,
        route_policy: FrozenRouteRankerPolicy,
        support_profile: SupportProfile,
        gate_mode: str,
        margin_threshold: float,
        policy_id: str,
    ) -> None:
        allowed = {
            "PURE_PI2",
            "PURE_ROUTE_RANKER",
            "SUPPORT_GATED",
            "MARGIN_GATED",
            "SUPPORT_AND_MARGIN_GATED",
        }
        if gate_mode not in allowed:
            raise ValueError(f"Unknown conservative gate mode: {gate_mode}")
        self.anchor_policy = anchor_policy
        self.route_policy = route_policy
        self.support_profile = support_profile
        self.gate_mode = gate_mode
        self.margin_threshold = float(margin_threshold)
        self.policy_id = policy_id
        self.audit: list[dict[str, Any]] = []

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            return None
        candidates = tuple(trace.candidates)
        anchor = self.anchor_policy.choose(state, definitions, enemy_id, runtime_context)
        proposed = self.route_policy.choose(state, definitions, enemy_id, runtime_context)
        if anchor is None or proposed is None:
            return anchor or proposed
        keys = tuple(candidate_key(item) for item in candidates)
        anchor_key = candidate_key(anchor)
        proposed_key = candidate_key(proposed)
        anchor_index = keys.index(anchor_key)
        proposed_index = keys.index(proposed_key)
        normalized, scores = route_scores(
            self.route_policy, state, definitions, candidates, runtime_context
        )
        support = self.support_profile.classify_normalized(normalized[proposed_index])
        margin = float(scores[proposed_index] - scores[anchor_index])
        differs = proposed_index != anchor_index
        support_ok = support.category == WELL_SUPPORTED
        margin_ok = margin >= self.margin_threshold
        if self.gate_mode == "PURE_PI2":
            accepted = False
        elif self.gate_mode == "PURE_ROUTE_RANKER":
            accepted = differs
        elif self.gate_mode == "SUPPORT_GATED":
            accepted = differs and support_ok
        elif self.gate_mode == "MARGIN_GATED":
            accepted = differs and margin_ok
        else:
            accepted = differs and support_ok and margin_ok
        chosen = proposed if accepted else anchor
        observations = tuple(
            encode_primary(state, definitions, item, runtime_context) for item in candidates
        )
        self.audit.append(
            {
                "action_index": len(self.audit),
                "round": state.round_number,
                "state_digest": decision_state_digest(state, runtime_context),
                "enemy_id": enemy_id,
                "anchor_action": asdict(anchor_key),
                "proposed_route_action": asdict(proposed_key),
                "executed_action": asdict(candidate_key(chosen)),
                "differed": differs,
                "gate_accepted": accepted,
                "gate_mode": self.gate_mode,
                "support": asdict(support),
                "top_vs_anchor_margin": margin,
                "margin_threshold": self.margin_threshold,
                "route_action_families": list(candidate_family(observations[proposed_index])),
                "anchor_action_families": list(candidate_family(observations[anchor_index])),
                "executed_action_families": list(
                    candidate_family(observations[keys.index(candidate_key(chosen))])
                ),
                "candidate_families": [list(candidate_family(obs)) for obs in observations],
            }
        )
        return chosen


class LimitedDeviationPolicy:
    def __init__(
        self,
        *,
        anchor_policy: EnemyDecisionPolicy,
        route_policy: EnemyDecisionPolicy,
        deviation_limit: int,
    ) -> None:
        self.anchor_policy = anchor_policy
        self.route_policy = route_policy
        self.deviation_limit = deviation_limit
        self.deviations_used = 0
        self.audit: list[dict[str, Any]] = []

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME,
    ) -> EnemyDecisionCandidate | None:
        anchor = self.anchor_policy.choose(state, definitions, enemy_id, runtime_context)
        route = self.route_policy.choose(state, definitions, enemy_id, runtime_context)
        if anchor is None or route is None:
            return anchor or route
        differs = candidate_key(anchor) != candidate_key(route)
        use_route = differs and self.deviations_used < self.deviation_limit
        if use_route:
            self.deviations_used += 1
        chosen = route if use_route else anchor
        self.audit.append(
            {
                "action_index": len(self.audit),
                "state_digest": decision_state_digest(state, runtime_context),
                "anchor_action": asdict(candidate_key(anchor)),
                "route_action": asdict(candidate_key(route)),
                "executed_action": asdict(candidate_key(chosen)),
                "differed": differs,
                "used_route_deviation": use_route,
                "deviations_used": self.deviations_used,
            }
        )
        return chosen


class BranchContinuationPolicy:
    def __init__(
        self,
        *,
        prefix_policy: EnemyDecisionPolicy,
        continuation_policy: EnemyDecisionPolicy,
        source: RouteSourceDecision,
        candidate_override: CandidateKey,
    ) -> None:
        self.prefix_policy = prefix_policy
        self.continuation_policy = continuation_policy
        self.source = source
        self.candidate_override = candidate_override
        self._global_index = 0
        self._rng: GameRng | None = None
        self._encounter_id = ""
        self._encounter_index = -1
        self.reached_target = False
        self.replay_state_digest = ""
        self.replay_rng_digest = ""
        self.replay_candidates: tuple[CandidateKey, ...] = ()

    def bind_encounter(self, rng: GameRng, *, context: Any, encounter_id: str) -> None:
        self._rng = rng
        self._encounter_id = encounter_id
        self._encounter_index = int(context.encounter_index)

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME,
    ) -> EnemyDecisionCandidate | None:
        if self._rng is None:
            raise RuntimeError("Branch policy missing encounter RNG.")
        current = self._global_index
        self._global_index += 1
        target = self.source.capture.global_action_index
        if current < target:
            return self.prefix_policy.choose(state, definitions, enemy_id, runtime_context)
        if current > target:
            return self.continuation_policy.choose(state, definitions, enemy_id, runtime_context)
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            raise RuntimeError("Branch target has no legal candidates.")
        keys = tuple(candidate_key(item) for item in trace.candidates)
        expected = self.source.capture
        self.reached_target = True
        self.replay_state_digest = decision_state_digest(state, runtime_context)
        self.replay_rng_digest = rng_state_digest(self._rng)
        self.replay_candidates = keys
        mismatches: list[str] = []
        if self._encounter_index != expected.encounter_index:
            mismatches.append("encounter_index")
        if self._encounter_id != expected.encounter_id:
            mismatches.append("encounter_id")
        if enemy_id != expected.enemy_id:
            mismatches.append("enemy_id")
        if state.round_number != expected.round_number:
            mismatches.append("round_number")
        if self.replay_state_digest != expected.state_digest:
            mismatches.append("state_digest")
        if self.replay_rng_digest != expected.rng_digest:
            mismatches.append("rng_digest")
        if keys != expected.candidates:
            mismatches.append("candidate_collection")
        if mismatches:
            raise RuntimeError("Branch prefix mismatch: " + ", ".join(mismatches))
        for candidate in trace.candidates:
            if candidate_key(candidate) == self.candidate_override:
                return candidate
        raise RuntimeError("Selected branch candidate is no longer legal.")
