from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from typing import Any

from game.combat.enemy_decision import HeuristicEnemyDecisionPolicy
from game.content.definitions import GameDefinitions
from game.ml.outcome_dataset import CounterfactualDataset
from game.ml.outcome_features import CP3_FEATURE_NAMES
from game.ml.policy_iteration import (
    PolicyCounterfactualDataset,
    PolicyStateSummary,
    collect_policy_source_decisions,
)


@dataclass(frozen=True)
class DistributionRecord:
    encounter_id: str
    hero_policy_id: str
    simulation_seed: int
    action_index: int
    enemy_class_id: str
    state_digest: str
    rng_digest: str
    candidate_count: int
    selected_skill_id: str
    state_summary: PolicyStateSummary


def reconstruct_cp3_distribution_records(
    definitions: GameDefinitions,
    dataset: CounterfactualDataset,
) -> tuple[DistributionRecord, ...]:
    grouped: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for record in dataset.records:
        grouped[(record.encounter_id, record.hero_policy_id, record.simulation_seed)].append(record)
    output: list[DistributionRecord] = []
    heuristic = HeuristicEnemyDecisionPolicy()
    for (encounter_id, hero_policy_id, seed), records in sorted(grouped.items()):
        replayed = collect_policy_source_decisions(
            definitions,
            source_policy=heuristic,
            source_policy_id="heuristic",
            split=dataset.split,
            encounter_id=encounter_id,
            hero_policy_id=hero_policy_id,
            seed=seed,
        )
        by_index = {source.action_index: source for source in replayed}
        for record in records:
            source = by_index.get(record.action_index)
            if source is None:
                raise ValueError("CP3 distribution replay did not reach a sampled action index.")
            if source.state_digest != record.state_digest:
                raise ValueError("CP3 distribution replay state digest mismatch.")
            if source.rng_digest != record.rng_digest:
                raise ValueError("CP3 distribution replay RNG digest mismatch.")
            selected = record.candidates[record.heuristic_index]
            output.append(
                DistributionRecord(
                    encounter_id=record.encounter_id,
                    hero_policy_id=record.hero_policy_id,
                    simulation_seed=record.simulation_seed,
                    action_index=record.action_index,
                    enemy_class_id=record.enemy_class_id,
                    state_digest=record.state_digest,
                    rng_digest=record.rng_digest,
                    candidate_count=len(record.candidates),
                    selected_skill_id=selected.key.skill_id,
                    state_summary=source.state_summary,
                )
            )
    output.sort(
        key=lambda item: (
            item.encounter_id,
            item.hero_policy_id,
            item.simulation_seed,
            item.action_index,
        )
    )
    return tuple(output)


def cp4_distribution_records(
    dataset: PolicyCounterfactualDataset,
) -> tuple[DistributionRecord, ...]:
    return tuple(
        DistributionRecord(
            encounter_id=record.encounter_id,
            hero_policy_id=record.hero_policy_id,
            simulation_seed=record.simulation_seed,
            action_index=record.action_index,
            enemy_class_id=record.enemy_class_id,
            state_digest=record.state_digest,
            rng_digest=record.rng_digest,
            candidate_count=len(record.candidates),
            selected_skill_id=record.candidates[record.source_policy_index].key.skill_id,
            state_summary=record.state_summary,
        )
        for record in dataset.records
    )


def build_distribution_audit(
    cp3_dataset: CounterfactualDataset,
    cp3_records: tuple[DistributionRecord, ...],
    cp4_dataset: PolicyCounterfactualDataset,
) -> dict[str, Any]:
    cp4_records = cp4_distribution_records(cp4_dataset)
    return {
        "method_id": "cp3-to-cp4-distribution-audit-v1",
        "cp3_source_policy": "heuristic",
        "cp4_source_policy": cp4_dataset.source_policy_id,
        "state_distribution": {
            "cp3": summarize_distribution(cp3_records),
            "cp4": summarize_distribution(cp4_records),
        },
        "feature_distribution": compare_feature_distributions(cp3_dataset, cp4_dataset),
    }


def summarize_distribution(records: tuple[DistributionRecord, ...]) -> dict[str, Any]:
    if not records:
        return {"decision_count": 0}
    round_values = [record.state_summary.round_number for record in records]
    candidate_counts = [record.candidate_count for record in records]
    hero_hp = [record.state_summary.party_hp_total for record in records]
    party_effort = [record.state_summary.party_effort_total for record in records]
    enemy_hp = [record.state_summary.enemy_hp_total for record in records]
    enemy_effort = [record.state_summary.enemy_effort_total for record in records]
    return {
        "decision_count": len(records),
        "rounds": _numeric_summary(round_values),
        "round_histogram": _counter(round_values),
        "candidate_counts": _numeric_summary(candidate_counts),
        "candidate_count_histogram": _counter(candidate_counts),
        "enemy_class_counts": _counter(record.enemy_class_id for record in records),
        "hero_life_state_per_decision": {
            "alive": _numeric_summary([record.state_summary.hero_alive for record in records]),
            "downed": _numeric_summary([record.state_summary.hero_downed for record in records]),
            "dead": _numeric_summary([record.state_summary.hero_dead for record in records]),
        },
        "enemy_life_state_per_decision": {
            "alive": _numeric_summary([record.state_summary.enemy_alive for record in records]),
            "downed": _numeric_summary([record.state_summary.enemy_downed for record in records]),
            "dead": _numeric_summary([record.state_summary.enemy_dead for record in records]),
        },
        "party_hp_total": _numeric_summary(hero_hp),
        "party_effort_total": _numeric_summary(party_effort),
        "enemy_hp_total": _numeric_summary(enemy_hp),
        "enemy_effort_total": _numeric_summary(enemy_effort),
        "hero_status_prevalence": _status_prevalence(records, team="hero"),
        "enemy_status_prevalence": _status_prevalence(records, team="enemy"),
        "selected_skill_counts": _counter(record.selected_skill_id for record in records),
    }


def compare_feature_distributions(
    cp3_dataset: CounterfactualDataset,
    cp4_dataset: PolicyCounterfactualDataset,
) -> dict[str, Any]:
    cp3_vectors = [
        candidate.observation for record in cp3_dataset.records for candidate in record.candidates
    ]
    cp4_vectors = [
        candidate.observation for record in cp4_dataset.records for candidate in record.candidates
    ]
    if not cp3_vectors or not cp4_vectors:
        raise ValueError("Feature distribution audit requires non-empty datasets.")
    cp3_exact = set(cp3_vectors)
    unseen = sum(vector not in cp3_exact for vector in cp4_vectors)
    features: list[dict[str, Any]] = []
    for index, name in enumerate(CP3_FEATURE_NAMES):
        before = [vector[index] for vector in cp3_vectors]
        after = [vector[index] for vector in cp4_vectors]
        before_mean = mean(before)
        after_mean = mean(after)
        before_std = pstdev(before)
        after_std = pstdev(after)
        before_min = min(before)
        before_max = max(before)
        outside = sum(value < before_min or value > before_max for value in after)
        standardized_mean_shift = (
            (after_mean - before_mean) / before_std if before_std > 0 else None
        )
        scale_ratio = after_std / before_std if before_std > 0 else None
        features.append(
            {
                "name": name,
                "cp3_mean": before_mean,
                "cp4_mean": after_mean,
                "cp3_std": before_std,
                "cp4_std": after_std,
                "standardized_mean_shift": standardized_mean_shift,
                "scale_ratio": scale_ratio,
                "cp3_min": before_min,
                "cp3_max": before_max,
                "cp4_min": min(after),
                "cp4_max": max(after),
                "cp4_outside_cp3_range_count": outside,
                "cp4_outside_cp3_range_rate": outside / len(after),
                "cp3_zero_variance": before_std == 0,
                "cp4_zero_variance": after_std == 0,
            }
        )
    ranked = sorted(
        (item for item in features if item["standardized_mean_shift"] is not None),
        key=lambda item: abs(float(item["standardized_mean_shift"])),
        reverse=True,
    )
    return {
        "cp3_candidate_observations": len(cp3_vectors),
        "cp4_candidate_observations": len(cp4_vectors),
        "cp4_exact_observation_unseen_count": unseen,
        "cp4_exact_observation_unseen_rate": unseen / len(cp4_vectors),
        "zero_variance_became_nonzero": [
            item["name"]
            for item in features
            if item["cp3_zero_variance"] and not item["cp4_zero_variance"]
        ],
        "nonzero_became_zero_variance": [
            item["name"]
            for item in features
            if not item["cp3_zero_variance"] and item["cp4_zero_variance"]
        ],
        "top_standardized_mean_shifts": ranked[:20],
        "features": features,
    }


def _status_prevalence(
    records: tuple[DistributionRecord, ...],
    *,
    team: str,
) -> dict[str, dict[str, float | int]]:
    names: set[str] = set()
    per_record: list[dict[str, int]] = []
    for record in records:
        values = dict(
            record.state_summary.hero_status_counts
            if team == "hero"
            else record.state_summary.enemy_status_counts
        )
        per_record.append(values)
        names.update(values)
    return {
        name: {
            "states_present": sum(values.get(name, 0) > 0 for values in per_record),
            "state_prevalence": sum(values.get(name, 0) > 0 for values in per_record)
            / len(records),
            "mean_actor_count": mean(values.get(name, 0) for values in per_record),
        }
        for name in sorted(names)
    }


def _numeric_summary(values: list[int] | list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "median": median(ordered),
        "min": ordered[0],
        "p10": _quantile(ordered, 0.10),
        "p90": _quantile(ordered, 0.90),
        "max": ordered[-1],
    }


def _quantile(values: list[int] | list[float], quantile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = quantile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower] * (1 - fraction) + values[upper] * fraction)


def _counter(values: Any) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))
    }


def audit_payload(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
