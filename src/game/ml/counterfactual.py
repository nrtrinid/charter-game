"""Deterministic one-step counterfactual replay for enemy-policy research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum

from game.campaign.company import create_new_company
from game.combat.combat_state import CombatState, LifeState
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionRuntimeContext,
    HeuristicEnemyDecisionPolicy,
    explain_enemy_decision,
)
from game.combat.enemy_learning import (
    EnemyDecisionEpisode,
    HeroDecisionPolicy,
    create_hero_policy,
    run_enemy_learning_episode,
)
from game.content.definitions import GameDefinitions
from game.core.events import CombatEffectEvent
from game.core.rng import GameRng
from game.expedition.cave import create_encounter_combat
from game.ml.outcome_features import CP3_FEATURE_SCHEMA
from game.ml.splits import DatasetSplit

BRANCH_METHOD_ID = "deterministic-prefix-replay-v1"
STATE_DIGEST_ID = "combat-decision-state-v1"
RNG_DIGEST_ID = "python-random-state-sha256-v1"
OUTCOME_SCHEMA_ID = "enemy-counterfactual-outcome-v1"
_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()


@dataclass(frozen=True, order=True)
class CandidateKey:
    skill_id: str
    target_id: str
    skill_order: int
    target_order: int


@dataclass(frozen=True)
class CounterfactualOutcome:
    enemy_win: bool
    final_hero_hp: int
    final_hero_effort: int
    terminal_hero_downs: int
    hero_deaths: int
    mortal_wounds: int
    enemy_caused_effort_drain: int
    rounds_elapsed: int


class OutcomePreference(StrEnum):
    FIRST = "first"
    SECOND = "second"
    TIE = "tie"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class SourceDecision:
    split: DatasetSplit
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
    heuristic_choice: CandidateKey
    baseline_outcome: CounterfactualOutcome
    baseline_final_state_digest: str
    baseline_episode_digest: str


@dataclass(frozen=True)
class CounterfactualBranchResult:
    source: SourceDecision
    forced_candidate: CandidateKey
    outcome: CounterfactualOutcome
    final_state_digest: str
    episode_digest: str
    replay_state_digest: str
    replay_rng_digest: str
    replay_candidates: tuple[CandidateKey, ...]
    continuation_policy_id: str = "heuristic"


class CounterfactualReplayMismatch(RuntimeError):
    """Raised when deterministic replay does not reproduce the sampled decision point."""


class _RecordingHeuristicPolicy:
    def __init__(self, rng: GameRng) -> None:
        self.rng = rng
        self.teacher = HeuristicEnemyDecisionPolicy()
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
        chosen = self.teacher.choose(state, definitions, enemy_id, runtime_context)
        if chosen is None:
            return None
        if trace is None:
            raise CounterfactualReplayMismatch(
                "Heuristic chose an action without a decision trace."
            )
        candidates = tuple(_candidate_key(candidate) for candidate in trace.candidates)
        chosen_key = _candidate_key(chosen)
        observations = tuple(
            CP3_FEATURE_SCHEMA.encode(state, definitions, candidate, runtime_context)
            for candidate in trace.candidates
        )
        if chosen_key not in candidates:
            raise CounterfactualReplayMismatch(
                "Heuristic choice was not in the simulator legal set."
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
            )
        )
        self.action_index += 1
        return chosen


class _OneStepReplayPolicy:
    def __init__(self, rng: GameRng, source: SourceDecision, forced: CandidateKey) -> None:
        self.rng = rng
        self.source = source
        self.forced = forced
        self.teacher = HeuristicEnemyDecisionPolicy()
        self.action_index = 0
        self.replay_state_digest = ""
        self.replay_rng_digest = ""
        self.replay_candidates: tuple[CandidateKey, ...] = ()
        self.reached_target = False

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        teacher_choice = self.teacher.choose(state, definitions, enemy_id, runtime_context)
        if teacher_choice is None:
            return None
        current_index = self.action_index
        self.action_index += 1
        if current_index != self.source.action_index:
            return teacher_choice

        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None:
            raise CounterfactualReplayMismatch("Target replay decision has no trace.")
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
                "Replay diverged before forced action: " + ", ".join(mismatches)
            )
        for candidate in trace.candidates:
            if _candidate_key(candidate) == self.forced:
                return candidate
        raise CounterfactualReplayMismatch("Forced candidate is no longer simulator-legal.")


def collect_heuristic_source_decisions(
    definitions: GameDefinitions,
    *,
    split: DatasetSplit,
    encounter_id: str,
    hero_policy_id: str,
    seed: int,
    max_rounds: int = 20,
) -> tuple[SourceDecision, ...]:
    state = _fresh_combat(definitions, encounter_id)
    rng = GameRng(seed)
    recorder = _RecordingHeuristicPolicy(rng)
    hero_policy = _hero_policy(hero_policy_id, encounter_id, seed)
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
    decisions: list[SourceDecision] = []
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
        ) = capture
        if record.action_index != action_index:
            raise CounterfactualReplayMismatch(
                "Recorded action index differs from replay capture index."
            )
        if (
            CandidateKey(
                record.chosen_skill_id,
                record.chosen_target_id,
                record.trace.chosen.skill_order if record.trace.chosen else -1,
                record.trace.chosen.target_order if record.trace.chosen else -1,
            )
            != chosen
        ):
            raise CounterfactualReplayMismatch(
                "Episode record differs from captured heuristic choice."
            )
        decisions.append(
            SourceDecision(
                split=split,
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
                heuristic_choice=chosen,
                baseline_outcome=outcome,
                baseline_final_state_digest=final_digest,
                baseline_episode_digest=episode_digest,
            )
        )
    return tuple(decisions)


def run_counterfactual_branch(
    definitions: GameDefinitions,
    source: SourceDecision,
    forced_candidate: CandidateKey,
    *,
    max_rounds: int = 20,
) -> CounterfactualBranchResult:
    if forced_candidate not in source.candidates:
        raise ValueError("Forced candidate must belong to the sampled source decision.")
    state = _fresh_combat(definitions, source.encounter_id)
    rng = GameRng(source.simulation_seed)
    policy = _OneStepReplayPolicy(rng, source, forced_candidate)
    hero_policy = _hero_policy(
        source.hero_policy_id,
        source.encounter_id,
        source.simulation_seed,
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
    return CounterfactualBranchResult(
        source=source,
        forced_candidate=forced_candidate,
        outcome=outcome_from_episode(state, episode),
        final_state_digest=combat_state_digest(state),
        episode_digest=stable_digest(episode),
        replay_state_digest=policy.replay_state_digest,
        replay_rng_digest=policy.replay_rng_digest,
        replay_candidates=policy.replay_candidates,
    )


def verify_heuristic_control_branch(
    definitions: GameDefinitions,
    source: SourceDecision,
    *,
    max_rounds: int = 20,
) -> CounterfactualBranchResult:
    result = run_counterfactual_branch(
        definitions,
        source,
        source.heuristic_choice,
        max_rounds=max_rounds,
    )
    if result.outcome != source.baseline_outcome:
        raise CounterfactualReplayMismatch("Forced heuristic action changed terminal outcome.")
    if result.final_state_digest != source.baseline_final_state_digest:
        raise CounterfactualReplayMismatch("Forced heuristic action changed final combat state.")
    if result.episode_digest != source.baseline_episode_digest:
        raise CounterfactualReplayMismatch("Forced heuristic action changed episode trajectory.")
    return result


def outcome_from_episode(
    state: CombatState, episode: EnemyDecisionEpisode
) -> CounterfactualOutcome:
    heroes = tuple(state.heroes.values())
    effort_drain = 0
    for record in episode.records:
        for event in record.events:
            if (
                isinstance(event, CombatEffectEvent)
                and event.resource == "effort"
                and event.delta < 0
            ):
                target_id = event.target_id or event.actor_id
                if target_id in state.heroes:
                    effort_drain += -event.delta
    return CounterfactualOutcome(
        enemy_win=episode.final_victor == "enemies",
        final_hero_hp=sum(max(0, hero.hp) for hero in heroes),
        final_hero_effort=sum(max(0, hero.effort) for hero in heroes),
        terminal_hero_downs=sum(hero.life_state == LifeState.DOWNED for hero in heroes),
        hero_deaths=sum(hero.life_state == LifeState.DEAD for hero in heroes),
        mortal_wounds=sum(hero.mortal_wounds for hero in heroes),
        enemy_caused_effort_drain=effort_drain,
        rounds_elapsed=episode.metrics.rounds_elapsed,
    )


def compare_outcomes(
    first: CounterfactualOutcome,
    second: CounterfactualOutcome,
) -> OutcomePreference:
    if first.enemy_win != second.enemy_win:
        return OutcomePreference.FIRST if first.enemy_win else OutcomePreference.SECOND
    if first.enemy_win and second.enemy_win:
        return OutcomePreference.TIE

    first_dimensions = _loss_deterioration_dimensions(first)
    second_dimensions = _loss_deterioration_dimensions(second)
    first_no_worse = all(a >= b for a, b in zip(first_dimensions, second_dimensions, strict=True))
    second_no_worse = all(b >= a for a, b in zip(first_dimensions, second_dimensions, strict=True))
    first_strict = any(a > b for a, b in zip(first_dimensions, second_dimensions, strict=True))
    second_strict = any(b > a for a, b in zip(first_dimensions, second_dimensions, strict=True))
    if first_no_worse and first_strict:
        return OutcomePreference.FIRST
    if second_no_worse and second_strict:
        return OutcomePreference.SECOND
    if not first_strict and not second_strict:
        return OutcomePreference.TIE
    return OutcomePreference.INCOMPARABLE


def _loss_deterioration_dimensions(outcome: CounterfactualOutcome) -> tuple[int, ...]:
    return (
        outcome.hero_deaths,
        outcome.terminal_hero_downs,
        outcome.mortal_wounds,
        -outcome.final_hero_hp,
        outcome.enemy_caused_effort_drain,
    )


def decision_state_digest(state: CombatState, runtime: EnemyDecisionRuntimeContext) -> str:
    return stable_digest({"state": state, "runtime_context": runtime, "format": STATE_DIGEST_ID})


def combat_state_digest(state: CombatState) -> str:
    return stable_digest({"state": state, "format": STATE_DIGEST_ID})


def rng_state_digest(rng: GameRng) -> str:
    random_engine = getattr(rng, "_random", None)
    if random_engine is None or not hasattr(random_engine, "getstate"):
        raise TypeError("GameRng does not expose a safely inspectable internal state.")
    payload = repr(random_engine.getstate()).encode("utf-8")
    return hashlib.sha256(RNG_DIGEST_ID.encode("utf-8") + b"\0" + payload).hexdigest()


def stable_digest(value: object) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True)
        )
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _candidate_key(candidate: EnemyDecisionCandidate) -> CandidateKey:
    return CandidateKey(
        skill_id=candidate.skill_id,
        target_id=candidate.target_id,
        skill_order=candidate.skill_order,
        target_order=candidate.target_order,
    )


def _fresh_combat(definitions: GameDefinitions, encounter_id: str) -> CombatState:
    company = create_new_company(definitions)
    return create_encounter_combat(company, definitions, encounter_id)


def _hero_policy(hero_policy_id: str, encounter_id: str, seed: int) -> HeroDecisionPolicy:
    return create_hero_policy(hero_policy_id, encounter_id=encounter_id, seed=seed)
