"""Offline imitation metrics and held-out simulator rollout evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from game.campaign.company import create_new_company
from game.combat.combat_state import CombatState
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionPolicy,
    EnemyDecisionRuntimeContext,
    explain_enemy_decision,
)
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.expedition.cave import create_encounter_combat
from game.ml.dataset import EnemyDecisionDataset
from game.ml.features import ENEMY_DECISION_FEATURE_SCHEMA
from game.ml.training import FeatureNormalization, prepare_decision_groups

_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()


@dataclass(frozen=True)
class DecisionEvaluation:
    cross_entropy: float
    correct: bool
    reciprocal_rank: float
    candidate_count: int


@dataclass(frozen=True)
class ResearchScorerPolicy:
    model: nn.Module
    normalization: FeatureNormalization
    feature_names: tuple[str, ...]

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            return None
        feature_indices = ENEMY_DECISION_FEATURE_SCHEMA.indices_for(self.feature_names)
        raw = torch.tensor(
            [
                ENEMY_DECISION_FEATURE_SCHEMA.encode(candidate.features)
                for candidate in trace.candidates
            ],
            dtype=torch.float32,
        )
        normalized = self.normalization.normalize_tensor(raw)
        selected = normalized[:, list(feature_indices)]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(selected)
        selected_index = int(torch.argmax(logits).item())
        return trace.candidates[selected_index]


def evaluate_imitation(
    model: nn.Module,
    dataset: EnemyDecisionDataset,
    normalization: FeatureNormalization,
    feature_names: tuple[str, ...],
    *,
    batch_size: int = 2048,
) -> dict[str, object]:
    groups = prepare_decision_groups(dataset, normalization, feature_names)
    evaluations: list[DecisionEvaluation | None] = [None] * len(dataset.records)
    model.eval()
    with torch.no_grad():
        for group in groups.groups:
            for start in range(0, group.features.shape[0], batch_size):
                stop = start + batch_size
                features = group.features[start:stop]
                targets = group.chosen_indices[start:stop]
                decisions, candidates, width = features.shape
                logits = model(features.reshape(decisions * candidates, width)).reshape(
                    decisions,
                    candidates,
                )
                cross_entropy = F.cross_entropy(logits, targets, reduction="none")
                predictions = torch.argmax(logits, dim=1)
                chosen_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                ranks = (logits > chosen_logits.unsqueeze(1)).sum(dim=1) + 1
                record_indices = group.record_indices[start:stop]
                for row, record_index in enumerate(record_indices):
                    evaluations[record_index] = DecisionEvaluation(
                        cross_entropy=float(cross_entropy[row]),
                        correct=bool(predictions[row] == targets[row]),
                        reciprocal_rank=1.0 / int(ranks[row]),
                        candidate_count=group.candidate_count,
                    )

    if any(item is None for item in evaluations):
        raise RuntimeError("Offline evaluation did not score every dataset decision.")
    scored = tuple(item for item in evaluations if item is not None)
    return {
        "overall": _summarize_imitation(scored),
        "by_encounter": _grouped_imitation_summary(
            dataset,
            scored,
            lambda index: dataset.records[index].encounter_id,
        ),
        "by_enemy_class": _grouped_imitation_summary(
            dataset,
            scored,
            lambda index: dataset.records[index].enemy_class_id or "unknown",
        ),
        "by_hero_policy": _grouped_imitation_summary(
            dataset,
            scored,
            lambda index: dataset.records[index].hero_policy_id,
        ),
    }


def _grouped_imitation_summary(
    dataset: EnemyDecisionDataset,
    evaluations: tuple[DecisionEvaluation, ...],
    key_for_index: Any,
) -> dict[str, object]:
    grouped: dict[str, list[DecisionEvaluation]] = {}
    for index, evaluation in enumerate(evaluations):
        grouped.setdefault(str(key_for_index(index)), []).append(evaluation)
    return {
        key: _summarize_imitation(tuple(grouped[key]))
        for key in sorted(grouped)
    }


def _summarize_imitation(evaluations: tuple[DecisionEvaluation, ...]) -> dict[str, object]:
    if not evaluations:
        return {
            "decision_count": 0,
            "cross_entropy": 0.0,
            "top1_agreement": 0.0,
            "mean_reciprocal_rank": 0.0,
            "candidate_count": {"min": 0, "max": 0, "mean": 0.0, "histogram": {}},
        }
    candidate_counts = [item.candidate_count for item in evaluations]
    histogram = Counter(candidate_counts)
    return {
        "decision_count": len(evaluations),
        "cross_entropy": mean(item.cross_entropy for item in evaluations),
        "top1_agreement": mean(float(item.correct) for item in evaluations),
        "mean_reciprocal_rank": mean(item.reciprocal_rank for item in evaluations),
        "candidate_count": {
            "min": min(candidate_counts),
            "max": max(candidate_counts),
            "mean": mean(candidate_counts),
            "histogram": {str(key): histogram[key] for key in sorted(histogram)},
        },
    }


@dataclass(frozen=True)
class RolloutPolicySpec:
    policy_name: str
    policy: EnemyDecisionPolicy | None
    enemy_wait_mode: str = "none"
    enemy_movement_mode: str = "recovery_only"


@dataclass(frozen=True)
class RolloutCaseResult:
    policy_name: str
    encounter_id: str
    hero_policy_id: str
    seed: int
    enemy_win: int
    hero_win: int
    final_hero_hp: int
    final_hero_effort: int
    hero_downs: int
    hero_deaths: int
    mortal_wounds: int
    rounds: int
    legacy_shaped_reward: int
    marks_applied: int
    marks_exploited: int
    guard_actions: int
    boss_grab_to_bite_same_target: int

    @property
    def case_key(self) -> tuple[str, str, int]:
        return self.encounter_id, self.hero_policy_id, self.seed


def run_rollout_comparison(
    definitions: GameDefinitions,
    *,
    encounter_ids: tuple[str, ...],
    hero_policy_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    policies: tuple[RolloutPolicySpec, ...],
    max_rounds: int = 20,
    bootstrap_seed: int = 20261004,
    bootstrap_samples: int = 1000,
) -> dict[str, object]:
    results: list[RolloutCaseResult] = []
    for hero_policy_id in hero_policy_ids:
        for encounter_id in encounter_ids:
            for seed in seeds:
                for policy_spec in policies:
                    results.append(
                        _run_rollout_case(
                            definitions,
                            encounter_id=encounter_id,
                            hero_policy_id=hero_policy_id,
                            seed=seed,
                            policy_spec=policy_spec,
                            max_rounds=max_rounds,
                        )
                    )
    by_policy = {
        policy.policy_name: _summarize_rollout(
            tuple(result for result in results if result.policy_name == policy.policy_name)
        )
        for policy in policies
    }
    baseline_name = policies[0].policy_name
    paired = {
        policy.policy_name: _paired_rollout_summary(
            tuple(result for result in results if result.policy_name == baseline_name),
            tuple(result for result in results if result.policy_name == policy.policy_name),
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        for policy in policies[1:]
    }
    return {
        "case_count_per_policy": len(encounter_ids) * len(hero_policy_ids) * len(seeds),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "baseline_policy": baseline_name,
        "by_policy": by_policy,
        "paired_vs_heuristic": paired,
    }


def _run_rollout_case(
    definitions: GameDefinitions,
    *,
    encounter_id: str,
    hero_policy_id: str,
    seed: int,
    policy_spec: RolloutPolicySpec,
    max_rounds: int,
) -> RolloutCaseResult:
    company = create_new_company(definitions)
    combat = create_encounter_combat(company, definitions, encounter_id)
    hero_policy = create_hero_policy(hero_policy_id, encounter_id=encounter_id, seed=seed)
    episode = run_enemy_learning_episode(
        combat,
        definitions,
        GameRng(seed),
        policy=policy_spec.policy,
        max_rounds=max_rounds,
        encounter_id=encounter_id,
        encounter_name=encounter_id.replace("_", " ").title(),
        hero_policy=hero_policy,
        enemy_wait_mode=policy_spec.enemy_wait_mode,
        enemy_movement_mode=policy_spec.enemy_movement_mode,
    )
    return RolloutCaseResult(
        policy_name=policy_spec.policy_name,
        encounter_id=encounter_id,
        hero_policy_id=hero_policy_id,
        seed=seed,
        enemy_win=int(episode.final_victor == "enemies"),
        hero_win=int(episode.final_victor == "heroes"),
        final_hero_hp=episode.metrics.final_hero_hp_total,
        final_hero_effort=episode.metrics.final_hero_effort_total,
        hero_downs=episode.metrics.hero_downs,
        hero_deaths=episode.metrics.hero_deaths,
        mortal_wounds=episode.metrics.mortal_wounds,
        rounds=episode.metrics.rounds_elapsed,
        legacy_shaped_reward=episode.total_reward,
        marks_applied=episode.metrics.marks_applied,
        marks_exploited=episode.metrics.marks_exploited,
        guard_actions=episode.metrics.guard_actions,
        boss_grab_to_bite_same_target=episode.metrics.boss_sequence.grab_to_bite_same_target,
    )


def _summarize_rollout(results: tuple[RolloutCaseResult, ...]) -> dict[str, object]:
    if not results:
        return {"case_count": 0}
    metric_names = _paired_metric_names()
    summary: dict[str, object] = {"case_count": len(results)}
    for name in metric_names:
        values = [float(getattr(result, name)) for result in results]
        summary[name] = {"mean": mean(values), "median": median(values)}
    return summary


def _paired_rollout_summary(
    baseline: tuple[RolloutCaseResult, ...],
    comparison: tuple[RolloutCaseResult, ...],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    baseline_by_key = {result.case_key: result for result in baseline}
    comparison_by_key = {result.case_key: result for result in comparison}
    if baseline_by_key.keys() != comparison_by_key.keys():
        raise ValueError("Paired rollout policies must cover exactly the same deterministic cases.")
    output: dict[str, object] = {"paired_case_count": len(baseline_by_key)}
    for offset, metric_name in enumerate(_paired_metric_names()):
        differences = tuple(
            float(getattr(comparison_by_key[key], metric_name))
            - float(getattr(baseline_by_key[key], metric_name))
            for key in sorted(baseline_by_key)
        )
        lower, upper = _bootstrap_mean_interval(
            differences,
            seed=bootstrap_seed + offset,
            samples=bootstrap_samples,
        )
        output[metric_name] = {
            "mean_difference": mean(differences) if differences else 0.0,
            "median_difference": median(differences) if differences else 0.0,
            "bootstrap_95_ci": [lower, upper],
        }
    return output


def _paired_metric_names() -> tuple[str, ...]:
    return (
        "enemy_win",
        "hero_win",
        "final_hero_hp",
        "final_hero_effort",
        "hero_downs",
        "hero_deaths",
        "mortal_wounds",
        "rounds",
        "legacy_shaped_reward",
        "marks_applied",
        "marks_exploited",
        "guard_actions",
        "boss_grab_to_bite_same_target",
    )


def _bootstrap_mean_interval(
    values: tuple[float, ...],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = GameRng(seed)
    means: list[float] = []
    for _ in range(samples):
        resample = [rng.choice(values) for _ in values]
        means.append(mean(resample))
    means.sort()
    lower_index = max(0, int(0.025 * len(means)))
    upper_index = min(len(means) - 1, int(0.975 * len(means)))
    return means[lower_index], means[upper_index]
