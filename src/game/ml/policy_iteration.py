from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from game.campaign.company import create_new_company
from game.combat.combat_state import CombatState, LifeState, Tag
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
from game.ml.counterfactual import (
    BRANCH_METHOD_ID,
    OUTCOME_SCHEMA_ID,
    CandidateKey,
    CounterfactualOutcome,
    CounterfactualReplayMismatch,
    OutcomePreference,
    combat_state_digest,
    compare_outcomes,
    decision_state_digest,
    outcome_from_episode,
    rng_state_digest,
    stable_digest,
)
from game.ml.outcome_dataset import (
    PREFERENCE_RULE_ID,
    CounterfactualCandidateRecord,
    CounterfactualDataset,
    CounterfactualDecisionRecord,
    SamplingSpec,
)
from game.ml.outcome_features import CP3_FEATURE_SCHEMA
from game.ml.splits import DatasetSplit

POLICY_DATASET_FORMAT_ID = "enemy-policy-counterfactual-decision-v1"
POLICY_DATASET_FORMAT_VERSION = 1
_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()


@dataclass(frozen=True)
class PolicyStateSummary:
    round_number: int
    hero_alive: int
    hero_downed: int
    hero_dead: int
    enemy_alive: int
    enemy_downed: int
    enemy_dead: int
    party_hp_total: int
    party_effort_total: int
    enemy_hp_total: int
    enemy_effort_total: int
    hero_status_counts: tuple[tuple[str, int], ...]
    enemy_status_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PolicySourceDecision:
    split: DatasetSplit
    source_policy_id: str
    encounter_id: str
    hero_policy_id: str
    simulation_seed: int
    action_index: int
    round_number: int
    enemy_id: str
    enemy_class_id: str
    runtime_context: EnemyDecisionRuntimeContext
    state_digest: str
    rng_digest: str
    candidates: tuple[CandidateKey, ...]
    candidate_observations: tuple[tuple[float, ...], ...]
    source_choice: CandidateKey
    state_summary: PolicyStateSummary
    baseline_outcome: CounterfactualOutcome
    baseline_final_state_digest: str
    baseline_episode_digest: str


@dataclass(frozen=True)
class PolicyCounterfactualBranchResult:
    source: PolicySourceDecision
    forced_candidate: CandidateKey
    outcome: CounterfactualOutcome
    final_state_digest: str
    episode_digest: str
    replay_state_digest: str
    replay_rng_digest: str
    replay_candidates: tuple[CandidateKey, ...]
    continuation_policy_id: str


@dataclass(frozen=True)
class PolicyCounterfactualCandidateRecord:
    key: CandidateKey
    observation: tuple[float, ...]
    outcome: CounterfactualOutcome


@dataclass(frozen=True)
class PolicyCounterfactualDecisionRecord:
    split: DatasetSplit
    source_policy_id: str
    encounter_id: str
    hero_policy_id: str
    simulation_seed: int
    action_index: int
    round_number: int
    enemy_class_id: str
    state_digest: str
    rng_digest: str
    state_summary: PolicyStateSummary
    candidates: tuple[PolicyCounterfactualCandidateRecord, ...]
    source_policy_index: int


@dataclass(frozen=True)
class PolicyCounterfactualDataset:
    split: DatasetSplit
    source_policy_id: str
    continuation_policy_id: str
    sampling_spec: SamplingSpec
    encounter_ids: tuple[str, ...]
    hero_policy_ids: tuple[str, ...]
    seed_count: int
    available_source_states: int
    records: tuple[PolicyCounterfactualDecisionRecord, ...]
    format_id: str = POLICY_DATASET_FORMAT_ID
    format_version: int = POLICY_DATASET_FORMAT_VERSION
    observation_schema_id: str = CP3_FEATURE_SCHEMA.schema_id
    observation_schema_version: int = CP3_FEATURE_SCHEMA.version
    outcome_schema_id: str = OUTCOME_SCHEMA_ID
    preference_rule_id: str = PREFERENCE_RULE_ID
    branch_method_id: str = BRANCH_METHOD_ID


@dataclass(frozen=True)
class PolicyDatasetAudit:
    source_states: int
    candidate_branches: int
    enemy_win_branches: int
    states_with_win_loss_variation: int
    source_policy_dominated_states: int
    source_policy_loss_to_win_opportunities: int
    source_policy_win_to_loss_risks: int
    decisive_pairs: int
    pareto_loss_pairs: int
    ties: int
    incomparable_pairs: int
    source_states_with_decisive_signal: int


class FrozenOutcomeRankerPolicy:
    """Stateless research policy backed by a frozen outcome-ranker model."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalization: Any,
        *,
        policy_id: str,
    ) -> None:
        self.model = model
        self.normalization = normalization
        self.policy_id = policy_id

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
        raw = torch.tensor(
            [
                CP3_FEATURE_SCHEMA.encode(state, definitions, candidate, runtime_context)
                for candidate in trace.candidates
            ],
            dtype=torch.float32,
        )
        values = self.normalization.normalize(raw)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(values)
        return trace.candidates[int(torch.argmax(scores).item())]


class _RecordingPolicy:
    def __init__(
        self,
        rng: GameRng,
        source_policy: EnemyDecisionPolicy,
        *,
        source_policy_id: str,
    ) -> None:
        self.rng = rng
        self.source_policy = source_policy
        self.source_policy_id = source_policy_id
        self.captures: list[
            tuple[
                int,
                str,
                int,
                EnemyDecisionRuntimeContext,
                str,
                str,
                tuple[CandidateKey, ...],
                tuple[tuple[float, ...], ...],
                CandidateKey,
                PolicyStateSummary,
            ]
        ] = []
        self.action_index = 0

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        chosen = self.source_policy.choose(state, definitions, enemy_id, runtime_context)
        if chosen is None:
            return None
        if trace is None:
            raise CounterfactualReplayMismatch(
                "Source policy chose an action without a decision trace."
            )
        candidates = tuple(_candidate_key(candidate) for candidate in trace.candidates)
        chosen_key = _candidate_key(chosen)
        if chosen_key not in candidates:
            raise CounterfactualReplayMismatch(
                "Source policy returned a candidate outside the simulator legal set."
            )
        observations = tuple(
            CP3_FEATURE_SCHEMA.encode(state, definitions, candidate, runtime_context)
            for candidate in trace.candidates
        )
        self.captures.append(
            (
                self.action_index,
                enemy_id,
                state.round_number,
                runtime_context,
                decision_state_digest(state, runtime_context),
                rng_state_digest(self.rng),
                candidates,
                observations,
                chosen_key,
                state_summary(state),
            )
        )
        self.action_index += 1
        return chosen


class _OneStepPolicyReplay:
    def __init__(
        self,
        rng: GameRng,
        source: PolicySourceDecision,
        forced_candidate: CandidateKey,
        continuation_policy: EnemyDecisionPolicy,
        *,
        continuation_policy_id: str,
    ) -> None:
        self.rng = rng
        self.source = source
        self.forced_candidate = forced_candidate
        self.continuation_policy = continuation_policy
        self.continuation_policy_id = continuation_policy_id
        self.action_index = 0
        self.reached_target = False
        self.replay_state_digest = ""
        self.replay_rng_digest = ""
        self.replay_candidates: tuple[CandidateKey, ...] = ()

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        source_choice = self.continuation_policy.choose(
            state,
            definitions,
            enemy_id,
            runtime_context,
        )
        if source_choice is None:
            return None
        current_index = self.action_index
        self.action_index += 1
        if current_index != self.source.action_index:
            return source_choice

        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None:
            raise CounterfactualReplayMismatch("Target policy replay decision has no trace.")
        candidates = tuple(_candidate_key(candidate) for candidate in trace.candidates)
        state_digest = decision_state_digest(state, runtime_context)
        rng_digest = rng_state_digest(self.rng)
        self.replay_state_digest = state_digest
        self.replay_rng_digest = rng_digest
        self.replay_candidates = candidates
        self.reached_target = True
        mismatches: list[str] = []
        if enemy_id != self.source.enemy_id:
            mismatches.append("enemy_id")
        if state.round_number != self.source.round_number:
            mismatches.append("round_number")
        if runtime_context != self.source.runtime_context:
            mismatches.append("runtime_context")
        if state_digest != self.source.state_digest:
            mismatches.append("state_digest")
        if rng_digest != self.source.rng_digest:
            mismatches.append("rng_digest")
        if candidates != self.source.candidates:
            mismatches.append("candidate_collection")
        if mismatches:
            raise CounterfactualReplayMismatch(
                "Policy replay diverged before forced action: " + ", ".join(mismatches)
            )
        for candidate in trace.candidates:
            if _candidate_key(candidate) == self.forced_candidate:
                return candidate
        raise CounterfactualReplayMismatch("Forced candidate is no longer simulator-legal.")


def collect_policy_source_decisions(
    definitions: GameDefinitions,
    *,
    source_policy: EnemyDecisionPolicy,
    source_policy_id: str,
    split: DatasetSplit,
    encounter_id: str,
    hero_policy_id: str,
    seed: int,
    max_rounds: int = 20,
) -> tuple[PolicySourceDecision, ...]:
    state = _fresh_combat(definitions, encounter_id)
    rng = GameRng(seed)
    recorder = _RecordingPolicy(rng, source_policy, source_policy_id=source_policy_id)
    hero_policy = create_hero_policy(hero_policy_id, encounter_id=encounter_id, seed=seed)
    episode = run_enemy_learning_episode(
        state,
        definitions,
        rng,
        policy=recorder,
        max_rounds=max_rounds,
        encounter_id=encounter_id,
        encounter_name=encounter_id.replace("_", " ").title(),
        hero_policy=hero_policy,
        enemy_wait_mode="none",
        enemy_movement_mode="recovery_only",
    )
    if len(recorder.captures) != len(episode.records):
        raise CounterfactualReplayMismatch(
            f"Captured {len(recorder.captures)} decisions but episode recorded "
            f"{len(episode.records)}."
        )
    outcome = outcome_from_episode(state, episode)
    final_digest = combat_state_digest(state)
    episode_digest = stable_digest(episode)
    decisions: list[PolicySourceDecision] = []
    for record, capture in zip(episode.records, recorder.captures, strict=True):
        (
            action_index,
            enemy_id,
            round_number,
            runtime,
            state_digest,
            rng_digest,
            candidates,
            observations,
            chosen,
            summary,
        ) = capture
        if record.action_index != action_index:
            raise CounterfactualReplayMismatch(
                "Recorded action index differs from source-policy capture index."
            )
        record_choice = CandidateKey(
            record.chosen_skill_id,
            record.chosen_target_id,
            record.trace.chosen.skill_order if record.trace.chosen else -1,
            record.trace.chosen.target_order if record.trace.chosen else -1,
        )
        if record_choice != chosen:
            raise CounterfactualReplayMismatch(
                "Episode record differs from captured source-policy choice."
            )
        decisions.append(
            PolicySourceDecision(
                split=split,
                source_policy_id=source_policy_id,
                encounter_id=encounter_id,
                hero_policy_id=hero_policy_id,
                simulation_seed=seed,
                action_index=action_index,
                round_number=round_number,
                enemy_id=enemy_id,
                enemy_class_id=record.enemy_class_id,
                runtime_context=runtime,
                state_digest=state_digest,
                rng_digest=rng_digest,
                candidates=candidates,
                candidate_observations=observations,
                source_choice=chosen,
                state_summary=summary,
                baseline_outcome=outcome,
                baseline_final_state_digest=final_digest,
                baseline_episode_digest=episode_digest,
            )
        )
    return tuple(decisions)


def run_policy_counterfactual_branch(
    definitions: GameDefinitions,
    source: PolicySourceDecision,
    forced_candidate: CandidateKey,
    *,
    continuation_policy: EnemyDecisionPolicy,
    continuation_policy_id: str,
    max_rounds: int = 20,
) -> PolicyCounterfactualBranchResult:
    if forced_candidate not in source.candidates:
        raise ValueError("Forced candidate must belong to the sampled source decision.")
    state = _fresh_combat(definitions, source.encounter_id)
    rng = GameRng(source.simulation_seed)
    policy = _OneStepPolicyReplay(
        rng,
        source,
        forced_candidate,
        continuation_policy,
        continuation_policy_id=continuation_policy_id,
    )
    hero_policy = create_hero_policy(
        source.hero_policy_id,
        encounter_id=source.encounter_id,
        seed=source.simulation_seed,
    )
    episode = run_enemy_learning_episode(
        state,
        definitions,
        rng,
        policy=policy,
        max_rounds=max_rounds,
        encounter_id=source.encounter_id,
        encounter_name=source.encounter_id.replace("_", " ").title(),
        hero_policy=hero_policy,
        enemy_wait_mode="none",
        enemy_movement_mode="recovery_only",
    )
    if not policy.reached_target:
        raise CounterfactualReplayMismatch("Replay never reached the sampled source decision.")
    return PolicyCounterfactualBranchResult(
        source=source,
        forced_candidate=forced_candidate,
        outcome=outcome_from_episode(state, episode),
        final_state_digest=combat_state_digest(state),
        episode_digest=stable_digest(episode),
        replay_state_digest=policy.replay_state_digest,
        replay_rng_digest=policy.replay_rng_digest,
        replay_candidates=policy.replay_candidates,
        continuation_policy_id=continuation_policy_id,
    )


def verify_policy_control_branch(
    definitions: GameDefinitions,
    source: PolicySourceDecision,
    *,
    continuation_policy: EnemyDecisionPolicy,
    continuation_policy_id: str,
    max_rounds: int = 20,
) -> PolicyCounterfactualBranchResult:
    result = run_policy_counterfactual_branch(
        definitions,
        source,
        source.source_choice,
        continuation_policy=continuation_policy,
        continuation_policy_id=continuation_policy_id,
        max_rounds=max_rounds,
    )
    if result.outcome != source.baseline_outcome:
        raise CounterfactualReplayMismatch("Forced source-policy action changed terminal outcome.")
    if result.final_state_digest != source.baseline_final_state_digest:
        raise CounterfactualReplayMismatch("Forced source-policy action changed final state.")
    if result.episode_digest != source.baseline_episode_digest:
        raise CounterfactualReplayMismatch("Forced source-policy action changed trajectory.")
    return result


def sample_policy_source_decisions(
    definitions: GameDefinitions,
    *,
    source_policy: EnemyDecisionPolicy,
    source_policy_id: str,
    split: DatasetSplit,
    encounter_ids: tuple[str, ...],
    hero_policy_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    sampling_spec: SamplingSpec,
    max_rounds: int = 20,
) -> tuple[tuple[PolicySourceDecision, ...], int]:
    selected: list[PolicySourceDecision] = []
    available = 0
    for encounter_id in encounter_ids:
        for hero_policy_id in hero_policy_ids:
            ranked: list[tuple[str, PolicySourceDecision]] = []
            for seed in seeds:
                decisions = collect_policy_source_decisions(
                    definitions,
                    source_policy=source_policy,
                    source_policy_id=source_policy_id,
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


def build_policy_counterfactual_dataset(
    definitions: GameDefinitions,
    *,
    source_policy: EnemyDecisionPolicy,
    source_policy_id: str,
    continuation_policy: EnemyDecisionPolicy,
    continuation_policy_id: str,
    split: DatasetSplit,
    encounter_ids: tuple[str, ...],
    hero_policy_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    sampling_spec: SamplingSpec,
    max_rounds: int = 20,
) -> PolicyCounterfactualDataset:
    sources, available = sample_policy_source_decisions(
        definitions,
        source_policy=source_policy,
        source_policy_id=source_policy_id,
        split=split,
        encounter_ids=encounter_ids,
        hero_policy_ids=hero_policy_ids,
        seeds=seeds,
        sampling_spec=sampling_spec,
        max_rounds=max_rounds,
    )
    records: list[PolicyCounterfactualDecisionRecord] = []
    for source in sources:
        candidates: list[PolicyCounterfactualCandidateRecord] = []
        source_policy_index = -1
        for index, (candidate, observation) in enumerate(
            zip(source.candidates, source.candidate_observations, strict=True)
        ):
            result = run_policy_counterfactual_branch(
                definitions,
                source,
                candidate,
                continuation_policy=continuation_policy,
                continuation_policy_id=continuation_policy_id,
                max_rounds=max_rounds,
            )
            if candidate == source.source_choice:
                source_policy_index = index
                if result.outcome != source.baseline_outcome:
                    raise RuntimeError("Source-policy control branch changed terminal outcome.")
                if result.final_state_digest != source.baseline_final_state_digest:
                    raise RuntimeError("Source-policy control branch changed final state.")
                if result.episode_digest != source.baseline_episode_digest:
                    raise RuntimeError("Source-policy control branch changed episode trajectory.")
            candidates.append(
                PolicyCounterfactualCandidateRecord(
                    key=candidate,
                    observation=observation,
                    outcome=result.outcome,
                )
            )
        if source_policy_index < 0:
            raise RuntimeError("Sampled decision has no source-policy candidate index.")
        records.append(
            PolicyCounterfactualDecisionRecord(
                split=split,
                source_policy_id=source_policy_id,
                encounter_id=source.encounter_id,
                hero_policy_id=source.hero_policy_id,
                simulation_seed=source.simulation_seed,
                action_index=source.action_index,
                round_number=source.round_number,
                enemy_class_id=source.enemy_class_id,
                state_digest=source.state_digest,
                rng_digest=source.rng_digest,
                state_summary=source.state_summary,
                candidates=tuple(candidates),
                source_policy_index=source_policy_index,
            )
        )
    return PolicyCounterfactualDataset(
        split=split,
        source_policy_id=source_policy_id,
        continuation_policy_id=continuation_policy_id,
        sampling_spec=sampling_spec,
        encounter_ids=encounter_ids,
        hero_policy_ids=hero_policy_ids,
        seed_count=len(seeds),
        available_source_states=available,
        records=tuple(records),
    )


def to_ranking_dataset(dataset: PolicyCounterfactualDataset) -> CounterfactualDataset:
    records = tuple(
        CounterfactualDecisionRecord(
            split=record.split,
            encounter_id=record.encounter_id,
            hero_policy_id=record.hero_policy_id,
            simulation_seed=record.simulation_seed,
            action_index=record.action_index,
            round_number=record.round_number,
            enemy_class_id=record.enemy_class_id,
            state_digest=record.state_digest,
            rng_digest=record.rng_digest,
            candidates=tuple(
                CounterfactualCandidateRecord(
                    key=candidate.key,
                    observation=candidate.observation,
                    outcome=candidate.outcome,
                )
                for candidate in record.candidates
            ),
            heuristic_index=record.source_policy_index,
        )
        for record in dataset.records
    )
    return CounterfactualDataset(
        split=dataset.split,
        sampling_spec=dataset.sampling_spec,
        encounter_ids=dataset.encounter_ids,
        hero_policy_ids=dataset.hero_policy_ids,
        seed_count=dataset.seed_count,
        available_source_states=dataset.available_source_states,
        records=records,
    )


def audit_policy_dataset(dataset: PolicyCounterfactualDataset) -> PolicyDatasetAudit:
    branch_count = 0
    win_branches = 0
    win_variation = 0
    source_dominated = 0
    loss_to_win = 0
    win_to_loss_risks = 0
    decisive = 0
    pareto_losses = 0
    ties = 0
    incomparable = 0
    decisive_states = 0
    for record in dataset.records:
        outcomes = tuple(candidate.outcome for candidate in record.candidates)
        branch_count += len(outcomes)
        win_branches += sum(outcome.enemy_win for outcome in outcomes)
        if any(outcome.enemy_win for outcome in outcomes) and not all(
            outcome.enemy_win for outcome in outcomes
        ):
            win_variation += 1
        source_outcome = outcomes[record.source_policy_index]
        dominated = False
        missed_win = False
        risk_avoided = False
        state_decisive = False
        for index, outcome in enumerate(outcomes):
            if index == record.source_policy_index:
                continue
            preference = compare_outcomes(outcome, source_outcome)
            if preference is OutcomePreference.FIRST:
                dominated = True
                if not source_outcome.enemy_win and outcome.enemy_win:
                    missed_win = True
            if source_outcome.enemy_win and not outcome.enemy_win:
                risk_avoided = True
        source_dominated += int(dominated)
        loss_to_win += int(missed_win)
        win_to_loss_risks += int(risk_avoided)
        for first in range(len(outcomes)):
            for second in range(first + 1, len(outcomes)):
                preference = compare_outcomes(outcomes[first], outcomes[second])
                if preference in {OutcomePreference.FIRST, OutcomePreference.SECOND}:
                    decisive += 1
                    state_decisive = True
                    if not outcomes[first].enemy_win and not outcomes[second].enemy_win:
                        pareto_losses += 1
                elif preference is OutcomePreference.TIE:
                    ties += 1
                else:
                    incomparable += 1
        decisive_states += int(state_decisive)
    return PolicyDatasetAudit(
        source_states=len(dataset.records),
        candidate_branches=branch_count,
        enemy_win_branches=win_branches,
        states_with_win_loss_variation=win_variation,
        source_policy_dominated_states=source_dominated,
        source_policy_loss_to_win_opportunities=loss_to_win,
        source_policy_win_to_loss_risks=win_to_loss_risks,
        decisive_pairs=decisive,
        pareto_loss_pairs=pareto_losses,
        ties=ties,
        incomparable_pairs=incomparable,
        source_states_with_decisive_signal=decisive_states,
    )


def save_policy_dataset(dataset: PolicyCounterfactualDataset, path: Path) -> str:
    payload = {
        "format_id": dataset.format_id,
        "format_version": dataset.format_version,
        "split": dataset.split.value,
        "source_policy_id": dataset.source_policy_id,
        "continuation_policy_id": dataset.continuation_policy_id,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return _sha256_file(path)


def load_policy_dataset(path: Path) -> PolicyCounterfactualDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_id") != POLICY_DATASET_FORMAT_ID:
        raise ValueError("Unsupported policy counterfactual dataset format id.")
    if int(payload.get("format_version", 0)) != POLICY_DATASET_FORMAT_VERSION:
        raise ValueError("Unsupported policy counterfactual dataset format version.")
    if payload.get("observation_schema_id") != CP3_FEATURE_SCHEMA.schema_id:
        raise ValueError("Policy dataset observation schema mismatch.")
    return PolicyCounterfactualDataset(
        split=DatasetSplit(payload["split"]),
        source_policy_id=str(payload["source_policy_id"]),
        continuation_policy_id=str(payload["continuation_policy_id"]),
        sampling_spec=SamplingSpec(**payload["sampling_spec"]),
        encounter_ids=tuple(payload["encounter_ids"]),
        hero_policy_ids=tuple(payload["hero_policy_ids"]),
        seed_count=int(payload["seed_count"]),
        available_source_states=int(payload["available_source_states"]),
        records=tuple(_record_from_dict(item) for item in payload["records"]),
        observation_schema_id=str(payload["observation_schema_id"]),
        observation_schema_version=int(payload["observation_schema_version"]),
        outcome_schema_id=str(payload["outcome_schema_id"]),
        preference_rule_id=str(payload["preference_rule_id"]),
        branch_method_id=str(payload["branch_method_id"]),
    )


def state_summary(state: CombatState) -> PolicyStateSummary:
    heroes = tuple(state.heroes.values())
    enemies = tuple(state.enemies.values())
    return PolicyStateSummary(
        round_number=state.round_number,
        hero_alive=sum(actor.life_state == LifeState.ALIVE for actor in heroes),
        hero_downed=sum(actor.life_state == LifeState.DOWNED for actor in heroes),
        hero_dead=sum(actor.life_state == LifeState.DEAD for actor in heroes),
        enemy_alive=sum(actor.life_state == LifeState.ALIVE for actor in enemies),
        enemy_downed=sum(actor.life_state == LifeState.DOWNED for actor in enemies),
        enemy_dead=sum(actor.life_state == LifeState.DEAD for actor in enemies),
        party_hp_total=sum(max(0, actor.hp) for actor in heroes),
        party_effort_total=sum(max(0, actor.effort) for actor in heroes),
        enemy_hp_total=sum(max(0, actor.hp) for actor in enemies),
        enemy_effort_total=sum(max(0, actor.effort) for actor in enemies),
        hero_status_counts=_status_counts(heroes),
        enemy_status_counts=_status_counts(enemies),
    )


def _record_to_dict(record: PolicyCounterfactualDecisionRecord) -> dict[str, Any]:
    return {
        "split": record.split.value,
        "source_policy_id": record.source_policy_id,
        "encounter_id": record.encounter_id,
        "hero_policy_id": record.hero_policy_id,
        "simulation_seed": record.simulation_seed,
        "action_index": record.action_index,
        "round_number": record.round_number,
        "enemy_class_id": record.enemy_class_id,
        "state_digest": record.state_digest,
        "rng_digest": record.rng_digest,
        "state_summary": asdict(record.state_summary),
        "source_policy_index": record.source_policy_index,
        "candidates": [
            {
                "key": asdict(candidate.key),
                "observation": list(candidate.observation),
                "outcome": asdict(candidate.outcome),
            }
            for candidate in record.candidates
        ],
    }


def _record_from_dict(payload: dict[str, Any]) -> PolicyCounterfactualDecisionRecord:
    summary_payload = payload["state_summary"]
    summary = PolicyStateSummary(
        **{
            **summary_payload,
            "hero_status_counts": tuple(
                tuple(item) for item in summary_payload["hero_status_counts"]
            ),
            "enemy_status_counts": tuple(
                tuple(item) for item in summary_payload["enemy_status_counts"]
            ),
        }
    )
    return PolicyCounterfactualDecisionRecord(
        split=DatasetSplit(payload["split"]),
        source_policy_id=str(payload["source_policy_id"]),
        encounter_id=str(payload["encounter_id"]),
        hero_policy_id=str(payload["hero_policy_id"]),
        simulation_seed=int(payload["simulation_seed"]),
        action_index=int(payload["action_index"]),
        round_number=int(payload["round_number"]),
        enemy_class_id=str(payload["enemy_class_id"]),
        state_digest=str(payload["state_digest"]),
        rng_digest=str(payload["rng_digest"]),
        state_summary=summary,
        source_policy_index=int(payload["source_policy_index"]),
        candidates=tuple(
            PolicyCounterfactualCandidateRecord(
                key=CandidateKey(**item["key"]),
                observation=tuple(float(value) for value in item["observation"]),
                outcome=CounterfactualOutcome(**item["outcome"]),
            )
            for item in payload["candidates"]
        ),
    )


def _status_counts(actors: tuple[Any, ...]) -> tuple[tuple[str, int], ...]:
    return tuple((tag.name.lower(), sum(tag in actor.tags for actor in actors)) for tag in Tag)


def _candidate_key(candidate: EnemyDecisionCandidate) -> CandidateKey:
    return CandidateKey(
        candidate.skill_id,
        candidate.target_id,
        candidate.skill_order,
        candidate.target_order,
    )


def _fresh_combat(definitions: GameDefinitions, encounter_id: str) -> CombatState:
    return create_encounter_combat(create_new_company(definitions), definitions, encounter_id)


def _sample_key(source: PolicySourceDecision) -> str:
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


def _source_sort_key(source: PolicySourceDecision) -> tuple[str, str, int, int]:
    return (
        source.encounter_id,
        source.hero_policy_id,
        source.simulation_seed,
        source.action_index,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
