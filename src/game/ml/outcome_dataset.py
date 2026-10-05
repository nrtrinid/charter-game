"""Deterministic sampled counterfactual outcome datasets for CP3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from game.content.definitions import GameDefinitions
from game.ml.counterfactual import (
    BRANCH_METHOD_ID,
    OUTCOME_SCHEMA_ID,
    CandidateKey,
    CounterfactualOutcome,
    OutcomePreference,
    SourceDecision,
    collect_heuristic_source_decisions,
    compare_outcomes,
    run_counterfactual_branch,
)
from game.ml.outcome_features import CP3_FEATURE_SCHEMA
from game.ml.splits import DatasetSplit

COUNTERFACTUAL_DATASET_FORMAT_ID = "enemy-counterfactual-decision-v1"
COUNTERFACTUAL_DATASET_FORMAT_VERSION = 1
SAMPLING_METHOD_ID = "lowest-stable-provenance-sha256-per-stratum-v1"
PREFERENCE_RULE_ID = "win-then-loss-pareto-v1"


@dataclass(frozen=True)
class SamplingSpec:
    states_per_encounter_policy: int
    method_id: str = SAMPLING_METHOD_ID

    def __post_init__(self) -> None:
        if self.states_per_encounter_policy <= 0:
            raise ValueError("Sampling budget must be positive.")


@dataclass(frozen=True)
class CounterfactualCandidateRecord:
    key: CandidateKey
    observation: tuple[float, ...]
    outcome: CounterfactualOutcome


@dataclass(frozen=True)
class CounterfactualDecisionRecord:
    split: DatasetSplit
    encounter_id: str
    hero_policy_id: str
    simulation_seed: int
    action_index: int
    round_number: int
    enemy_class_id: str
    state_digest: str
    rng_digest: str
    candidates: tuple[CounterfactualCandidateRecord, ...]
    heuristic_index: int


@dataclass(frozen=True)
class CounterfactualDataset:
    split: DatasetSplit
    sampling_spec: SamplingSpec
    encounter_ids: tuple[str, ...]
    hero_policy_ids: tuple[str, ...]
    seed_count: int
    available_source_states: int
    records: tuple[CounterfactualDecisionRecord, ...]
    format_id: str = COUNTERFACTUAL_DATASET_FORMAT_ID
    format_version: int = COUNTERFACTUAL_DATASET_FORMAT_VERSION
    observation_schema_id: str = CP3_FEATURE_SCHEMA.schema_id
    observation_schema_version: int = CP3_FEATURE_SCHEMA.version
    outcome_schema_id: str = OUTCOME_SCHEMA_ID
    preference_rule_id: str = PREFERENCE_RULE_ID
    branch_method_id: str = BRANCH_METHOD_ID


@dataclass(frozen=True)
class PreferencePair:
    record_index: int
    preferred_index: int
    other_index: int


@dataclass(frozen=True)
class DatasetAudit:
    source_states: int
    candidate_branches: int
    enemy_win_branches: int
    states_with_win_loss_variation: int
    heuristic_dominated_states: int
    heuristic_loss_to_win_opportunities: int
    decisive_pairs: int
    pareto_loss_pairs: int
    ties: int
    incomparable_pairs: int


def sample_source_decisions(
    definitions: GameDefinitions,
    *,
    split: DatasetSplit,
    encounter_ids: tuple[str, ...],
    hero_policy_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    sampling_spec: SamplingSpec,
    max_rounds: int = 20,
) -> tuple[tuple[SourceDecision, ...], int]:
    selected: list[SourceDecision] = []
    available = 0
    for encounter_id in encounter_ids:
        for hero_policy_id in hero_policy_ids:
            ranked: list[tuple[str, SourceDecision]] = []
            for seed in seeds:
                decisions = collect_heuristic_source_decisions(
                    definitions,
                    split=split,
                    encounter_id=encounter_id,
                    hero_policy_id=hero_policy_id,
                    seed=seed,
                    max_rounds=max_rounds,
                )
                available += len(decisions)
                ranked.extend((_sample_key(decision), decision) for decision in decisions)
            ranked.sort(key=lambda item: item[0])
            selected.extend(
                decision for _key, decision in ranked[: sampling_spec.states_per_encounter_policy]
            )
    selected.sort(key=_source_sort_key)
    return tuple(selected), available


def build_counterfactual_dataset(
    definitions: GameDefinitions,
    *,
    split: DatasetSplit,
    encounter_ids: tuple[str, ...],
    hero_policy_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    sampling_spec: SamplingSpec,
    max_rounds: int = 20,
) -> CounterfactualDataset:
    sources, available = sample_source_decisions(
        definitions,
        split=split,
        encounter_ids=encounter_ids,
        hero_policy_ids=hero_policy_ids,
        seeds=seeds,
        sampling_spec=sampling_spec,
        max_rounds=max_rounds,
    )
    records: list[CounterfactualDecisionRecord] = []
    for source in sources:
        branches: list[CounterfactualCandidateRecord] = []
        heuristic_index = -1
        for index, (candidate, observation) in enumerate(
            zip(source.candidates, source.candidate_observations, strict=True)
        ):
            result = run_counterfactual_branch(
                definitions,
                source,
                candidate,
                max_rounds=max_rounds,
            )
            if candidate == source.heuristic_choice:
                heuristic_index = index
                if result.outcome != source.baseline_outcome:
                    raise RuntimeError("Heuristic control branch changed terminal outcome.")
                if result.final_state_digest != source.baseline_final_state_digest:
                    raise RuntimeError("Heuristic control branch changed final state.")
                if result.episode_digest != source.baseline_episode_digest:
                    raise RuntimeError("Heuristic control branch changed episode trajectory.")
            branches.append(
                CounterfactualCandidateRecord(
                    key=candidate,
                    observation=observation,
                    outcome=result.outcome,
                )
            )
        if heuristic_index < 0:
            raise RuntimeError("Sampled decision has no heuristic candidate index.")
        records.append(
            CounterfactualDecisionRecord(
                split=split,
                encounter_id=source.encounter_id,
                hero_policy_id=source.hero_policy_id,
                simulation_seed=source.simulation_seed,
                action_index=source.action_index,
                round_number=source.round_number,
                enemy_class_id=source.enemy_class_id,
                state_digest=source.state_digest,
                rng_digest=source.rng_digest,
                candidates=tuple(branches),
                heuristic_index=heuristic_index,
            )
        )
    return CounterfactualDataset(
        split=split,
        sampling_spec=sampling_spec,
        encounter_ids=encounter_ids,
        hero_policy_ids=hero_policy_ids,
        seed_count=len(seeds),
        available_source_states=available,
        records=tuple(records),
    )


def decisive_preference_pairs(dataset: CounterfactualDataset) -> tuple[PreferencePair, ...]:
    pairs: list[PreferencePair] = []
    for record_index, record in enumerate(dataset.records):
        for first in range(len(record.candidates)):
            for second in range(first + 1, len(record.candidates)):
                preference = compare_outcomes(
                    record.candidates[first].outcome,
                    record.candidates[second].outcome,
                )
                if preference is OutcomePreference.FIRST:
                    pairs.append(PreferencePair(record_index, first, second))
                elif preference is OutcomePreference.SECOND:
                    pairs.append(PreferencePair(record_index, second, first))
    return tuple(pairs)


def audit_counterfactual_dataset(dataset: CounterfactualDataset) -> DatasetAudit:
    branch_count = 0
    win_branches = 0
    win_variation = 0
    heuristic_dominated = 0
    heuristic_loss_to_win = 0
    decisive = 0
    pareto_losses = 0
    ties = 0
    incomparable = 0
    for record in dataset.records:
        outcomes = tuple(candidate.outcome for candidate in record.candidates)
        branch_count += len(outcomes)
        win_branches += sum(outcome.enemy_win for outcome in outcomes)
        if any(outcome.enemy_win for outcome in outcomes) and not all(
            outcome.enemy_win for outcome in outcomes
        ):
            win_variation += 1
        heuristic = outcomes[record.heuristic_index]
        dominated = False
        loss_to_win = False
        for index, outcome in enumerate(outcomes):
            if index == record.heuristic_index:
                continue
            preference = compare_outcomes(outcome, heuristic)
            if preference is OutcomePreference.FIRST:
                dominated = True
                if not heuristic.enemy_win and outcome.enemy_win:
                    loss_to_win = True
        heuristic_dominated += int(dominated)
        heuristic_loss_to_win += int(loss_to_win)
        for first in range(len(outcomes)):
            for second in range(first + 1, len(outcomes)):
                preference = compare_outcomes(outcomes[first], outcomes[second])
                if preference in {OutcomePreference.FIRST, OutcomePreference.SECOND}:
                    decisive += 1
                    if not outcomes[first].enemy_win and not outcomes[second].enemy_win:
                        pareto_losses += 1
                elif preference is OutcomePreference.TIE:
                    ties += 1
                else:
                    incomparable += 1
    return DatasetAudit(
        source_states=len(dataset.records),
        candidate_branches=branch_count,
        enemy_win_branches=win_branches,
        states_with_win_loss_variation=win_variation,
        heuristic_dominated_states=heuristic_dominated,
        heuristic_loss_to_win_opportunities=heuristic_loss_to_win,
        decisive_pairs=decisive,
        pareto_loss_pairs=pareto_losses,
        ties=ties,
        incomparable_pairs=incomparable,
    )


def save_counterfactual_dataset(dataset: CounterfactualDataset, path: Path) -> str:
    payload = _dataset_to_dict(dataset)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _sha256_file(path)


def load_counterfactual_dataset(path: Path) -> CounterfactualDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_id") != COUNTERFACTUAL_DATASET_FORMAT_ID:
        raise ValueError("Unsupported counterfactual dataset format id.")
    if int(payload.get("format_version", 0)) != COUNTERFACTUAL_DATASET_FORMAT_VERSION:
        raise ValueError("Unsupported counterfactual dataset format version.")
    if payload.get("observation_schema_id") != CP3_FEATURE_SCHEMA.schema_id:
        raise ValueError("Counterfactual dataset observation schema mismatch.")
    records = tuple(_record_from_dict(item) for item in payload["records"])
    return CounterfactualDataset(
        split=DatasetSplit(payload["split"]),
        sampling_spec=SamplingSpec(**payload["sampling_spec"]),
        encounter_ids=tuple(payload["encounter_ids"]),
        hero_policy_ids=tuple(payload["hero_policy_ids"]),
        seed_count=int(payload["seed_count"]),
        available_source_states=int(payload["available_source_states"]),
        records=records,
        observation_schema_id=str(payload["observation_schema_id"]),
        observation_schema_version=int(payload["observation_schema_version"]),
        outcome_schema_id=str(payload["outcome_schema_id"]),
        preference_rule_id=str(payload["preference_rule_id"]),
        branch_method_id=str(payload["branch_method_id"]),
    )


def _sample_key(source: SourceDecision) -> str:
    text = "|".join(
        (
            source.split.value,
            source.encounter_id,
            source.hero_policy_id,
            str(source.simulation_seed),
            str(source.action_index),
            str(source.round_number),
            source.enemy_class_id,
        )
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_sort_key(source: SourceDecision) -> tuple[str, str, int, int]:
    return (
        source.encounter_id,
        source.hero_policy_id,
        source.simulation_seed,
        source.action_index,
    )


def _dataset_to_dict(dataset: CounterfactualDataset) -> dict[str, Any]:
    return {
        "format_id": dataset.format_id,
        "format_version": dataset.format_version,
        "split": dataset.split.value,
        "sampling_spec": asdict(dataset.sampling_spec),
        "encounter_ids": list(dataset.encounter_ids),
        "hero_policy_ids": list(dataset.hero_policy_ids),
        "seed_count": dataset.seed_count,
        "available_source_states": dataset.available_source_states,
        "observation_schema_id": dataset.observation_schema_id,
        "observation_schema_version": dataset.observation_schema_version,
        "outcome_schema_id": dataset.outcome_schema_id,
        "preference_rule_id": dataset.preference_rule_id,
        "branch_method_id": dataset.branch_method_id,
        "records": [_record_to_dict(record) for record in dataset.records],
    }


def _record_to_dict(record: CounterfactualDecisionRecord) -> dict[str, Any]:
    return {
        "split": record.split.value,
        "encounter_id": record.encounter_id,
        "hero_policy_id": record.hero_policy_id,
        "simulation_seed": record.simulation_seed,
        "action_index": record.action_index,
        "round_number": record.round_number,
        "enemy_class_id": record.enemy_class_id,
        "state_digest": record.state_digest,
        "rng_digest": record.rng_digest,
        "heuristic_index": record.heuristic_index,
        "candidates": [
            {
                "key": asdict(candidate.key),
                "observation": list(candidate.observation),
                "outcome": asdict(candidate.outcome),
            }
            for candidate in record.candidates
        ],
    }


def _record_from_dict(payload: dict[str, Any]) -> CounterfactualDecisionRecord:
    return CounterfactualDecisionRecord(
        split=DatasetSplit(payload["split"]),
        encounter_id=str(payload["encounter_id"]),
        hero_policy_id=str(payload["hero_policy_id"]),
        simulation_seed=int(payload["simulation_seed"]),
        action_index=int(payload["action_index"]),
        round_number=int(payload["round_number"]),
        enemy_class_id=str(payload["enemy_class_id"]),
        state_digest=str(payload["state_digest"]),
        rng_digest=str(payload["rng_digest"]),
        heuristic_index=int(payload["heuristic_index"]),
        candidates=tuple(
            CounterfactualCandidateRecord(
                key=CandidateKey(**item["key"]),
                observation=tuple(float(value) for value in item["observation"]),
                outcome=CounterfactualOutcome(**item["outcome"]),
            )
            for item in payload["candidates"]
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
