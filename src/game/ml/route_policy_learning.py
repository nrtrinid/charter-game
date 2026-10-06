"""Route-aware counterfactual learning over production-faithful Charter routes.

Research only. The simulator owns legality/resolution/persistence; this module owns
observation, deterministic replay, route-outcome preference, and small rankers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from game.campaign.company import create_new_company
from game.campaign.roster import sync_company_from_combat
from game.combat.combat_state import CombatState, LifeState, MoraleState
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionPolicy,
    EnemyDecisionRuntimeContext,
    explain_enemy_decision,
)
from game.combat.enemy_learning import create_hero_policy
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.expedition.cave import create_encounter_combat
from game.ml.confirmation_audit import mcnemar_exact, paired_difference_ci, wilson_interval
from game.ml.counterfactual import (
    CandidateKey,
    OutcomePreference,
    decision_state_digest,
    rng_state_digest,
    stable_digest,
)
from game.ml.models import CandidateScorerSpec, build_candidate_scorer
from game.ml.outcome_features import CP3_FEATURE_NAMES, CP3_FEATURE_SCHEMA
from game.ml.route_termination import (
    CombatTerminationReason,
    ResearchSafetyBudget,
    RouteTerminalStatus,
    behavior_counts,
    run_research_combat_with_termination,
)
from game.ml.stateful_route import ResearchRoute, derive_encounter_seed, snapshot_company

PRIMARY_SCHEMA_ID = "enemy-route-outcome-candidate-v1"
PRIMARY_SCHEMA_VERSION = 1
ORACLE_SCHEMA_ID = "enemy-route-context-oracle-candidate-v1"
ORACLE_SCHEMA_VERSION = 1
PERSISTENT_FEATURE_NAMES: tuple[str, ...] = (
    "party_max_hp_total",
    "party_max_effort_total",
    "party_downed_count",
    "party_mortal_wounds_total",
    "party_strain_total",
    "party_strain_marks_total",
    "party_morale_distress_count",
    "party_front_alive_count",
    "party_back_alive_count",
)
PRIMARY_FEATURE_NAMES: tuple[str, ...] = CP3_FEATURE_NAMES + PERSISTENT_FEATURE_NAMES
ROUTE_CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "route_encounter_index",
    "route_encounters_remaining",
    "route_progress_fraction",
)
ORACLE_FEATURE_NAMES: tuple[str, ...] = PRIMARY_FEATURE_NAMES + ROUTE_CONTEXT_FEATURE_NAMES
_DEFAULT_RUNTIME = EnemyDecisionRuntimeContext()


@dataclass(frozen=True)
class RouteContext:
    route_id: str
    encounter_index: int
    total_encounters: int

    @property
    def encounters_remaining(self) -> int:
        return max(0, self.total_encounters - self.encounter_index - 1)

    @property
    def progress_fraction(self) -> float:
        if self.total_encounters <= 1:
            return 1.0
        return self.encounter_index / (self.total_encounters - 1)


def _party_persistent_mapping(state: CombatState) -> dict[str, float]:
    heroes = tuple(state.heroes.values())
    return {
        "party_max_hp_total": float(sum(hero.max_hp for hero in heroes)),
        "party_max_effort_total": float(sum(hero.max_effort for hero in heroes)),
        "party_downed_count": float(sum(hero.life_state == LifeState.DOWNED for hero in heroes)),
        "party_mortal_wounds_total": float(sum(hero.mortal_wounds for hero in heroes)),
        "party_strain_total": float(sum(int(hero.strain.value) for hero in heroes)),
        "party_strain_marks_total": float(sum(len(hero.strain_marks) for hero in heroes)),
        "party_morale_distress_count": float(
            sum(hero.morale in {MoraleState.SHAKEN, MoraleState.BROKEN} for hero in heroes)
        ),
        "party_front_alive_count": float(
            sum(
                hero.is_alive() and hero.formation_slot.value.startswith("FRONT") for hero in heroes
            )
        ),
        "party_back_alive_count": float(
            sum(hero.is_alive() and hero.formation_slot.value.startswith("BACK") for hero in heroes)
        ),
    }


def encode_primary(
    state: CombatState,
    definitions: GameDefinitions,
    candidate: EnemyDecisionCandidate,
    runtime: EnemyDecisionRuntimeContext,
) -> tuple[float, ...]:
    base = CP3_FEATURE_SCHEMA.encode(state, definitions, candidate, runtime)
    persistent = _party_persistent_mapping(state)
    return base + tuple(float(persistent[name]) for name in PERSISTENT_FEATURE_NAMES)


def encode_oracle(
    state: CombatState,
    definitions: GameDefinitions,
    candidate: EnemyDecisionCandidate,
    runtime: EnemyDecisionRuntimeContext,
    context: RouteContext,
) -> tuple[float, ...]:
    return encode_primary(state, definitions, candidate, runtime) + (
        float(context.encounter_index),
        float(context.encounters_remaining),
        float(context.progress_fraction),
    )


def candidate_key(candidate: EnemyDecisionCandidate) -> CandidateKey:
    return CandidateKey(
        skill_id=candidate.skill_id,
        target_id=candidate.target_id,
        skill_order=candidate.skill_order,
        target_order=candidate.target_order,
    )


@dataclass(frozen=True)
class RouteOutcome:
    terminal_status: str
    enemy_victory: bool
    route_completed: bool
    final_hero_hp: int
    final_hero_effort: int
    final_downed: int
    hero_deaths: int
    mortal_wounds: int
    strain_marks: int
    final_company_digest: str

    @property
    def valid_terminal(self) -> bool:
        return self.terminal_status in {
            RouteTerminalStatus.ROUTE_COMPLETED.value,
            RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY.value,
            RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT.value,
        }


@dataclass(frozen=True)
class LocalOutcome:
    enemy_win: bool
    hero_deaths: int
    terminal_downs: int
    mortal_wounds: int
    final_hero_hp: int
    final_hero_effort: int


@dataclass(frozen=True)
class CapturedRouteDecision:
    global_action_index: int
    route_id: str
    encounter_index: int
    encounter_id: str
    total_encounters: int
    round_number: int
    enemy_id: str
    enemy_class_id: str
    runtime_context: EnemyDecisionRuntimeContext
    state_digest: str
    rng_digest: str
    candidates: tuple[CandidateKey, ...]
    primary_observations: tuple[tuple[float, ...], ...]
    oracle_observations: tuple[tuple[float, ...], ...]
    source_choice: CandidateKey
    party_hp_fraction: float
    party_effort_fraction: float


@dataclass(frozen=True)
class RouteSourceDecision:
    split: str
    hero_policy_id: str
    root_seed: int
    route: ResearchRoute
    capture: CapturedRouteDecision
    baseline_outcome: RouteOutcome
    baseline_route_digest: str


@dataclass(frozen=True)
class RouteBranchResult:
    source: RouteSourceDecision
    forced_candidate: CandidateKey
    route_outcome: RouteOutcome
    local_outcome: LocalOutcome | None
    route_digest: str
    replay_state_digest: str
    replay_rng_digest: str
    replay_candidates: tuple[CandidateKey, ...]
    reached_target: bool
    control_exact: bool


@dataclass(frozen=True)
class RouteCandidateRecord:
    key: CandidateKey
    primary_observation: tuple[float, ...]
    oracle_observation: tuple[float, ...]
    route_outcome: RouteOutcome
    local_outcome: LocalOutcome | None


@dataclass(frozen=True)
class RouteDecisionRecord:
    split: str
    route_id: str
    encounter_index: int
    encounter_id: str
    route_position: str
    hero_policy_id: str
    root_seed: int
    enemy_class_id: str
    round_number: int
    state_digest: str
    rng_digest: str
    candidates: tuple[RouteCandidateRecord, ...]
    source_policy_index: int
    control_exact: bool


@dataclass(frozen=True)
class RouteDataset:
    split: str
    records: tuple[RouteDecisionRecord, ...]


def route_position(index: int, total: int) -> str:
    if total <= 1:
        return "SINGLE"
    if index == 0:
        return "EARLY"
    if index == total - 1:
        return "LATE"
    return "MIDDLE"


class RecordingRoutePolicy:
    def __init__(self, source_policy: EnemyDecisionPolicy) -> None:
        self.source_policy = source_policy
        self.captures: list[CapturedRouteDecision] = []
        self._global_index = 0
        self._rng: GameRng | None = None
        self._context: RouteContext | None = None
        self._encounter_id = ""

    def bind_encounter(self, rng: GameRng, *, context: RouteContext, encounter_id: str) -> None:
        self._rng = rng
        self._context = context
        self._encounter_id = encounter_id

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME,
    ) -> EnemyDecisionCandidate | None:
        if self._rng is None or self._context is None:
            raise RuntimeError("Route recorder was not bound to an encounter RNG/context.")
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        chosen = self.source_policy.choose(state, definitions, enemy_id, runtime_context)
        if chosen is None:
            return None
        if trace is None or not trace.candidates:
            raise RuntimeError("Source policy chose without a legal decision trace.")
        keys = tuple(candidate_key(item) for item in trace.candidates)
        chosen_key = candidate_key(chosen)
        if chosen_key not in keys:
            raise RuntimeError("Source policy chose outside the legal candidate set.")
        primary = tuple(
            encode_primary(state, definitions, item, runtime_context) for item in trace.candidates
        )
        oracle = tuple(
            encode_oracle(state, definitions, item, runtime_context, self._context)
            for item in trace.candidates
        )
        max_hp = sum(hero.max_hp for hero in state.heroes.values())
        max_effort = sum(hero.max_effort for hero in state.heroes.values())
        hp = sum(max(0, hero.hp) for hero in state.heroes.values())
        effort = sum(max(0, hero.effort) for hero in state.heroes.values())
        self.captures.append(
            CapturedRouteDecision(
                global_action_index=self._global_index,
                route_id=self._context.route_id,
                encounter_index=self._context.encounter_index,
                encounter_id=self._encounter_id,
                total_encounters=self._context.total_encounters,
                round_number=state.round_number,
                enemy_id=enemy_id,
                enemy_class_id=state.actor(enemy_id).class_id,
                runtime_context=runtime_context,
                state_digest=decision_state_digest(state, runtime_context),
                rng_digest=rng_state_digest(self._rng),
                candidates=keys,
                primary_observations=primary,
                oracle_observations=oracle,
                source_choice=chosen_key,
                party_hp_fraction=hp / max_hp if max_hp else 0.0,
                party_effort_fraction=effort / max_effort if max_effort else 0.0,
            )
        )
        self._global_index += 1
        return chosen


class ForcingRoutePolicy:
    def __init__(
        self,
        continuation_policy: EnemyDecisionPolicy,
        source: RouteSourceDecision,
        forced_candidate: CandidateKey,
    ) -> None:
        self.continuation_policy = continuation_policy
        self.source = source
        self.forced_candidate = forced_candidate
        self._global_index = 0
        self._rng: GameRng | None = None
        self._context: RouteContext | None = None
        self._encounter_id = ""
        self.reached_target = False
        self.replay_state_digest = ""
        self.replay_rng_digest = ""
        self.replay_candidates: tuple[CandidateKey, ...] = ()

    def bind_encounter(self, rng: GameRng, *, context: RouteContext, encounter_id: str) -> None:
        self._rng = rng
        self._context = context
        self._encounter_id = encounter_id

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME,
    ) -> EnemyDecisionCandidate | None:
        if self._rng is None or self._context is None:
            raise RuntimeError("Route replay policy was not bound to encounter context.")
        continuation = self.continuation_policy.choose(
            state, definitions, enemy_id, runtime_context
        )
        if continuation is None:
            return None
        current = self._global_index
        self._global_index += 1
        if current != self.source.capture.global_action_index:
            return continuation
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            raise RuntimeError("Replay target has no decision trace.")
        keys = tuple(candidate_key(item) for item in trace.candidates)
        state_digest = decision_state_digest(state, runtime_context)
        rng_digest = rng_state_digest(self._rng)
        self.replay_state_digest = state_digest
        self.replay_rng_digest = rng_digest
        self.replay_candidates = keys
        self.reached_target = True
        expected = self.source.capture
        mismatches: list[str] = []
        if self._context.encounter_index != expected.encounter_index:
            mismatches.append("encounter_index")
        if self._encounter_id != expected.encounter_id:
            mismatches.append("encounter_id")
        if enemy_id != expected.enemy_id:
            mismatches.append("enemy_id")
        if state.round_number != expected.round_number:
            mismatches.append("round_number")
        if runtime_context != expected.runtime_context:
            mismatches.append("runtime_context")
        if state_digest != expected.state_digest:
            mismatches.append("state_digest")
        if rng_digest != expected.rng_digest:
            mismatches.append("rng_digest")
        if keys != expected.candidates:
            mismatches.append("candidate_collection")
        if mismatches:
            raise RuntimeError("Route prefix replay mismatch: " + ", ".join(mismatches))
        for item in trace.candidates:
            if candidate_key(item) == self.forced_candidate:
                return item
        raise RuntimeError("Forced route candidate is no longer legal.")


def _route_outcome(
    status: RouteTerminalStatus, company: Any, definitions: GameDefinitions
) -> RouteOutcome:
    final = snapshot_company(company, definitions)
    return RouteOutcome(
        terminal_status=status.value,
        enemy_victory=status == RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY,
        route_completed=status == RouteTerminalStatus.ROUTE_COMPLETED,
        final_hero_hp=int(final["hp_total"]),
        final_hero_effort=int(final["effort_total"]),
        final_downed=int(final["downed_count"]),
        hero_deaths=len(final["deceased_hero_ids"]),
        mortal_wounds=int(final["mortal_wounds_total"]),
        strain_marks=int(final["strain_marks_total"]),
        final_company_digest=stable_digest(final),
    )


def _local_outcome(state: CombatState, reason: CombatTerminationReason) -> LocalOutcome:
    heroes = tuple(state.heroes.values())
    return LocalOutcome(
        enemy_win=reason == CombatTerminationReason.ENEMY_VICTORY,
        hero_deaths=sum(hero.life_state == LifeState.DEAD for hero in heroes),
        terminal_downs=sum(hero.life_state == LifeState.DOWNED for hero in heroes),
        mortal_wounds=sum(hero.mortal_wounds for hero in heroes),
        final_hero_hp=sum(max(0, hero.hp) for hero in heroes),
        final_hero_effort=sum(max(0, hero.effort) for hero in heroes),
    )


def _route_digest(
    status: RouteTerminalStatus, final: Mapping[str, Any], encounters: Sequence[Mapping[str, Any]]
) -> str:
    return stable_digest(
        {
            "terminal_status": status.value,
            "final_company": final,
            "encounters": list(encounters),
            "format": "cp9-route-continuation-v1",
        }
    )


def run_route_trajectory(
    definitions: GameDefinitions,
    route: ResearchRoute,
    *,
    root_seed: int,
    enemy_policy_name: str,
    enemy_policy: EnemyDecisionPolicy,
    hero_policy_id: str,
    safety_budget: ResearchSafetyBudget,
    recording_policy: RecordingRoutePolicy | None = None,
    forcing_policy: ForcingRoutePolicy | None = None,
) -> tuple[RouteOutcome, str, list[dict[str, Any]], LocalOutcome | None]:
    if recording_policy is not None and forcing_policy is not None:
        raise ValueError(
            "A route run cannot record source decisions and force a branch simultaneously."
        )
    policy: EnemyDecisionPolicy = recording_policy or forcing_policy or enemy_policy
    company = create_new_company(definitions)
    encounters: list[dict[str, Any]] = []
    target_local: LocalOutcome | None = None
    terminal_status: RouteTerminalStatus | None = None
    for encounter_index, encounter_id in enumerate(route.encounter_ids):
        combat = create_encounter_combat(company, definitions, encounter_id)
        seed_namespace = route.seed_namespace or route.route_id
        seed_index = encounter_index
        if route.seed_by_encounter_identity:
            seed_index = sum(
                1 for prior in route.encounter_ids[:encounter_index] if prior == encounter_id
            )
        encounter_seed = derive_encounter_seed(
            root_seed,
            route_id=seed_namespace,
            encounter_index=seed_index,
            encounter_id=encounter_id,
            hero_policy_id=hero_policy_id,
        )
        rng = GameRng(encounter_seed)
        context = RouteContext(route.route_id, encounter_index, len(route.encounter_ids))
        if recording_policy is not None:
            recording_policy.bind_encounter(rng, context=context, encounter_id=encounter_id)
        if forcing_policy is not None:
            forcing_policy.bind_encounter(rng, context=context, encounter_id=encounter_id)
        result = run_research_combat_with_termination(
            combat,
            definitions,
            rng,
            enemy_policy_name=enemy_policy_name,
            enemy_policy=policy,
            hero_policy=create_hero_policy(
                hero_policy_id, encounter_id=encounter_id, seed=encounter_seed
            ),
            encounter_id=encounter_id,
            encounter_name=encounter_id.replace("_", " ").title(),
            safety_budget=safety_budget,
            enemy_wait_mode="none",
            enemy_movement_mode="recovery_only",
        )
        reason = result.termination_reason
        if forcing_policy is not None and forcing_policy.reached_target and target_local is None:
            if encounter_index == forcing_policy.source.capture.encounter_index:
                target_local = _local_outcome(combat, reason)
        encounter_payload = {
            "encounter_index": encounter_index,
            "encounter_id": encounter_id,
            "encounter_seed": encounter_seed,
            "termination_reason": reason.value,
            "rounds": result.rounds_elapsed,
            "transitions": result.transitions,
            "final_state_digest": result.final_state_digest,
            "final_rng_digest": result.final_rng_digest,
            "enemy_action_sequence": [list(item) for item in result.enemy_action_sequence],
            "hero_action_sequence": [list(item) for item in result.hero_action_sequence],
            "enemy_event_digest": result.enemy_event_digest,
            "behavior": behavior_counts(result, definitions),
        }
        if result.is_production_terminal:
            sync_company_from_combat(company, combat.heroes, combat.party_formation)
            encounter_payload["company_after_sync"] = snapshot_company(company, definitions)
        encounters.append(encounter_payload)
        if reason == CombatTerminationReason.HERO_VICTORY:
            continue
        if reason == CombatTerminationReason.ENEMY_VICTORY:
            terminal_status = RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY
        elif reason == CombatTerminationReason.PRODUCTION_RETREAT:
            terminal_status = RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT
        elif reason == CombatTerminationReason.DETECTED_CYCLE:
            terminal_status = RouteTerminalStatus.ROUTE_CENSORED_CYCLE
        elif reason == CombatTerminationReason.SAFETY_BOUND_EXCEEDED:
            terminal_status = RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND
        else:
            terminal_status = RouteTerminalStatus.ROUTE_INVALID
        break
    if terminal_status is None:
        terminal_status = RouteTerminalStatus.ROUTE_COMPLETED
    final = snapshot_company(company, definitions)
    outcome = _route_outcome(terminal_status, company, definitions)
    return outcome, _route_digest(terminal_status, final, encounters), encounters, target_local


def collect_route_source_decisions(
    definitions: GameDefinitions,
    route: ResearchRoute,
    *,
    root_seed: int,
    source_policy: EnemyDecisionPolicy,
    source_policy_name: str,
    hero_policy_id: str,
    split: str,
    safety_budget: ResearchSafetyBudget,
) -> tuple[RouteSourceDecision, ...]:
    recorder = RecordingRoutePolicy(source_policy)
    outcome, digest, _encounters, _ = run_route_trajectory(
        definitions,
        route,
        root_seed=root_seed,
        enemy_policy_name=source_policy_name,
        enemy_policy=source_policy,
        hero_policy_id=hero_policy_id,
        safety_budget=safety_budget,
        recording_policy=recorder,
    )
    return tuple(
        RouteSourceDecision(
            split=split,
            hero_policy_id=hero_policy_id,
            root_seed=root_seed,
            route=route,
            capture=capture,
            baseline_outcome=outcome,
            baseline_route_digest=digest,
        )
        for capture in recorder.captures
    )


def run_route_counterfactual_branch(
    definitions: GameDefinitions,
    source: RouteSourceDecision,
    forced_candidate: CandidateKey,
    *,
    continuation_policy: EnemyDecisionPolicy,
    continuation_policy_name: str,
    safety_budget: ResearchSafetyBudget,
) -> RouteBranchResult:
    if forced_candidate not in source.capture.candidates:
        raise ValueError("Forced candidate is not legal in the sampled source state.")
    forcing = ForcingRoutePolicy(continuation_policy, source, forced_candidate)
    outcome, digest, _encounters, local = run_route_trajectory(
        definitions,
        source.route,
        root_seed=source.root_seed,
        enemy_policy_name=continuation_policy_name,
        enemy_policy=continuation_policy,
        hero_policy_id=source.hero_policy_id,
        safety_budget=safety_budget,
        forcing_policy=forcing,
    )
    if not forcing.reached_target:
        raise RuntimeError("Route replay never reached the sampled source decision.")
    control = forced_candidate == source.capture.source_choice
    exact = (not control) or (
        outcome == source.baseline_outcome and digest == source.baseline_route_digest
    )
    return RouteBranchResult(
        source=source,
        forced_candidate=forced_candidate,
        route_outcome=outcome,
        local_outcome=local,
        route_digest=digest,
        replay_state_digest=forcing.replay_state_digest,
        replay_rng_digest=forcing.replay_rng_digest,
        replay_candidates=forcing.replay_candidates,
        reached_target=forcing.reached_target,
        control_exact=exact,
    )


def compare_route_outcomes(first: RouteOutcome, second: RouteOutcome) -> OutcomePreference:
    if not first.valid_terminal or not second.valid_terminal:
        return OutcomePreference.INCOMPARABLE
    if first.enemy_victory != second.enemy_victory:
        return OutcomePreference.FIRST if first.enemy_victory else OutcomePreference.SECOND
    if first.enemy_victory and second.enemy_victory:
        return OutcomePreference.TIE
    if not first.route_completed or not second.route_completed:
        # Real retreat is neither hero completion nor enemy victory; do not invent an ordering.
        return OutcomePreference.INCOMPARABLE
    first_dims = (
        first.hero_deaths,
        first.final_downed,
        first.mortal_wounds,
        -first.final_hero_hp,
        -first.final_hero_effort,
        first.strain_marks,
    )
    second_dims = (
        second.hero_deaths,
        second.final_downed,
        second.mortal_wounds,
        -second.final_hero_hp,
        -second.final_hero_effort,
        second.strain_marks,
    )
    first_no_worse = all(a >= b for a, b in zip(first_dims, second_dims, strict=True))
    second_no_worse = all(b >= a for a, b in zip(first_dims, second_dims, strict=True))
    first_strict = any(a > b for a, b in zip(first_dims, second_dims, strict=True))
    second_strict = any(b > a for a, b in zip(first_dims, second_dims, strict=True))
    if first_no_worse and first_strict:
        return OutcomePreference.FIRST
    if second_no_worse and second_strict:
        return OutcomePreference.SECOND
    if not first_strict and not second_strict:
        return OutcomePreference.TIE
    return OutcomePreference.INCOMPARABLE


def compare_local_outcomes(
    first: LocalOutcome | None, second: LocalOutcome | None
) -> OutcomePreference:
    if first is None or second is None:
        return OutcomePreference.INCOMPARABLE
    if first.enemy_win != second.enemy_win:
        return OutcomePreference.FIRST if first.enemy_win else OutcomePreference.SECOND
    if first.enemy_win and second.enemy_win:
        return OutcomePreference.TIE
    first_dims = (
        first.hero_deaths,
        first.terminal_downs,
        first.mortal_wounds,
        -first.final_hero_hp,
        -first.final_hero_effort,
    )
    second_dims = (
        second.hero_deaths,
        second.terminal_downs,
        second.mortal_wounds,
        -second.final_hero_hp,
        -second.final_hero_effort,
    )
    first_no_worse = all(a >= b for a, b in zip(first_dims, second_dims, strict=True))
    second_no_worse = all(b >= a for a, b in zip(first_dims, second_dims, strict=True))
    first_strict = any(a > b for a, b in zip(first_dims, second_dims, strict=True))
    second_strict = any(b > a for a, b in zip(first_dims, second_dims, strict=True))
    if first_no_worse and first_strict:
        return OutcomePreference.FIRST
    if second_no_worse and second_strict:
        return OutcomePreference.SECOND
    if not first_strict and not second_strict:
        return OutcomePreference.TIE
    return OutcomePreference.INCOMPARABLE


def build_route_decision_record(
    source: RouteSourceDecision,
    branches: Sequence[RouteBranchResult],
) -> RouteDecisionRecord:
    by_key = {branch.forced_candidate: branch for branch in branches}
    if set(by_key) != set(source.capture.candidates):
        raise ValueError("Route branch collection does not cover the exact legal candidate set.")
    candidates: list[RouteCandidateRecord] = []
    source_index = -1
    for index, (key, primary, oracle) in enumerate(
        zip(
            source.capture.candidates,
            source.capture.primary_observations,
            source.capture.oracle_observations,
            strict=True,
        )
    ):
        branch = by_key[key]
        candidates.append(
            RouteCandidateRecord(
                key=key,
                primary_observation=primary,
                oracle_observation=oracle,
                route_outcome=branch.route_outcome,
                local_outcome=branch.local_outcome,
            )
        )
        if key == source.capture.source_choice:
            source_index = index
    if source_index < 0:
        raise ValueError("Source choice missing from route dataset record.")
    control = by_key[source.capture.source_choice]
    return RouteDecisionRecord(
        split=source.split,
        route_id=source.route.route_id,
        encounter_index=source.capture.encounter_index,
        encounter_id=source.capture.encounter_id,
        route_position=route_position(
            source.capture.encounter_index, source.capture.total_encounters
        ),
        hero_policy_id=source.hero_policy_id,
        root_seed=source.root_seed,
        enemy_class_id=source.capture.enemy_class_id,
        round_number=source.capture.round_number,
        state_digest=source.capture.state_digest,
        rng_digest=source.capture.rng_digest,
        candidates=tuple(candidates),
        source_policy_index=source_index,
        control_exact=control.control_exact,
    )


@dataclass(frozen=True)
class PreferencePair:
    record_index: int
    preferred_index: int
    other_index: int
    preference_kind: str


def decisive_pairs(dataset: RouteDataset) -> tuple[PreferencePair, ...]:
    pairs: list[PreferencePair] = []
    for record_index, record in enumerate(dataset.records):
        for first_index in range(len(record.candidates)):
            for second_index in range(first_index + 1, len(record.candidates)):
                first = record.candidates[first_index].route_outcome
                second = record.candidates[second_index].route_outcome
                pref = compare_route_outcomes(first, second)
                if pref not in {OutcomePreference.FIRST, OutcomePreference.SECOND}:
                    continue
                preferred = first_index if pref == OutcomePreference.FIRST else second_index
                other = second_index if preferred == first_index else first_index
                kind = "win_loss" if first.enemy_victory != second.enemy_victory else "pareto_loss"
                pairs.append(PreferencePair(record_index, preferred, other, kind))
    return tuple(pairs)


def dataset_audit(dataset: RouteDataset) -> dict[str, Any]:
    decisive = 0
    win_loss = 0
    pareto = 0
    ties = 0
    incomparable = 0
    control_failures = 0
    censored_branches = 0
    states_with_decisive = 0
    states_with_enemy = 0
    states_with_completion = 0
    legal_candidates = 0
    for record in dataset.records:
        legal_candidates += len(record.candidates)
        control_failures += int(not record.control_exact)
        states_with_enemy += int(
            any(item.route_outcome.enemy_victory for item in record.candidates)
        )
        states_with_completion += int(
            any(item.route_outcome.route_completed for item in record.candidates)
        )
        censored_branches += sum(
            not item.route_outcome.valid_terminal for item in record.candidates
        )
        state_decisive = False
        for i in range(len(record.candidates)):
            for j in range(i + 1, len(record.candidates)):
                a = record.candidates[i].route_outcome
                b = record.candidates[j].route_outcome
                pref = compare_route_outcomes(a, b)
                if pref in {OutcomePreference.FIRST, OutcomePreference.SECOND}:
                    decisive += 1
                    state_decisive = True
                    if a.enemy_victory != b.enemy_victory:
                        win_loss += 1
                    else:
                        pareto += 1
                elif pref == OutcomePreference.TIE:
                    ties += 1
                else:
                    incomparable += 1
        states_with_decisive += int(state_decisive)
    return {
        "source_states": len(dataset.records),
        "legal_candidates": legal_candidates,
        "total_route_branches": legal_candidates,
        "decisive_pairs": decisive,
        "win_loss_decisive_pairs": win_loss,
        "loss_loss_pareto_decisive_pairs": pareto,
        "ties": ties,
        "incomparable_pairs": incomparable,
        "source_states_with_decisive_signal": states_with_decisive,
        "control_verification_failures": control_failures,
        "censored_branches": censored_branches,
        "states_with_at_least_one_enemy_victory_candidate": states_with_enemy,
        "states_with_at_least_one_route_completion_candidate": states_with_completion,
    }


def horizon_disagreement(dataset: RouteDataset) -> dict[str, Any]:
    comparable = 0
    same = 0
    reversed_count = 0
    local_only = route_only = 0
    motifs: dict[str, int] = {}
    positions: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for record in dataset.records:
        for i in range(len(record.candidates)):
            for j in range(i + 1, len(record.candidates)):
                route_pref = compare_route_outcomes(
                    record.candidates[i].route_outcome, record.candidates[j].route_outcome
                )
                local_pref = compare_local_outcomes(
                    record.candidates[i].local_outcome, record.candidates[j].local_outcome
                )
                route_dec = route_pref in {OutcomePreference.FIRST, OutcomePreference.SECOND}
                local_dec = local_pref in {OutcomePreference.FIRST, OutcomePreference.SECOND}
                if route_dec or local_dec:
                    comparable += 1
                if route_dec and local_dec:
                    if route_pref == local_pref:
                        same += 1
                    else:
                        reversed_count += 1
                        a = record.candidates[i].key.skill_id
                        b = record.candidates[j].key.skill_id
                        motif = "|".join(sorted((a, b)))
                        motifs[motif] = motifs.get(motif, 0) + 1
                        positions[record.route_position] = (
                            positions.get(record.route_position, 0) + 1
                        )
                        if len(examples) < 24:
                            examples.append(
                                {
                                    "route_id": record.route_id,
                                    "encounter_id": record.encounter_id,
                                    "route_position": record.route_position,
                                    "round": record.round_number,
                                    "first": asdict(record.candidates[i].key),
                                    "second": asdict(record.candidates[j].key),
                                    "local_preference": local_pref.value,
                                    "route_preference": route_pref.value,
                                }
                            )
                elif local_dec:
                    local_only += 1
                elif route_dec:
                    route_only += 1
    denom = same + reversed_count
    return {
        "comparable_pairs": comparable,
        "both_horizons_decisive": denom,
        "same_preference": same,
        "horizon_reversals": reversed_count,
        "horizon_reversal_rate": reversed_count / denom if denom else 0.0,
        "local_only_decisive": local_only,
        "route_only_decisive": route_only,
        "reversal_skill_motifs": dict(sorted(motifs.items(), key=lambda item: (-item[1], item[0]))),
        "reversal_route_positions": dict(sorted(positions.items())),
        "examples": examples,
    }


@dataclass(frozen=True)
class RouteNormalization:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    zero_variance: tuple[bool, ...]
    schema_id: str
    schema_version: int = 1

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        means = torch.tensor(self.means, dtype=values.dtype, device=values.device)
        scales = torch.tensor(self.scales, dtype=values.dtype, device=values.device)
        return (values - means) / scales

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_id": "enemy-route-normalization-v1",
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "zero_variance": list(self.zero_variance),
            "fit_split": "train",
        }


def fit_route_normalization(dataset: RouteDataset, *, oracle: bool = False) -> RouteNormalization:
    if dataset.split != "train":
        raise ValueError("Route normalization may only be fitted on TRAIN.")
    names = ORACLE_FEATURE_NAMES if oracle else PRIMARY_FEATURE_NAMES
    vectors = [
        (candidate.oracle_observation if oracle else candidate.primary_observation)
        for record in dataset.records
        for candidate in record.candidates
    ]
    if not vectors:
        raise ValueError("Cannot normalize an empty route dataset.")
    values = torch.tensor(vectors, dtype=torch.float64)
    means = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    zero = std == 0
    scales = torch.where(zero, torch.ones_like(std), std)
    return RouteNormalization(
        feature_names=names,
        means=tuple(float(item) for item in means.tolist()),
        scales=tuple(float(item) for item in scales.tolist()),
        zero_variance=tuple(bool(item) for item in zero.tolist()),
        schema_id=ORACLE_SCHEMA_ID if oracle else PRIMARY_SCHEMA_ID,
    )


@dataclass(frozen=True)
class RouteTrainingConfig:
    learning_rate: float = 0.005
    batch_size: int = 512
    max_epochs: int = 60
    early_stopping_patience: int = 8
    min_delta: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainedRouteRanker:
    spec: CandidateScorerSpec
    model: nn.Module
    seed: int
    selected_epoch: int
    validation_loss: float
    validation_accuracy: float
    history: list[dict[str, float | int]]


def _pair_tensors(
    dataset: RouteDataset,
    pairs: Sequence[PreferencePair],
    normalization: RouteNormalization,
    *,
    oracle: bool,
    indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    use = list(range(len(pairs))) if indices is None else list(indices)
    preferred: list[tuple[float, ...]] = []
    other: list[tuple[float, ...]] = []
    for index in use:
        pair = pairs[index]
        record = dataset.records[pair.record_index]
        first = record.candidates[pair.preferred_index]
        second = record.candidates[pair.other_index]
        preferred.append(first.oracle_observation if oracle else first.primary_observation)
        other.append(second.oracle_observation if oracle else second.primary_observation)
    return (
        normalization.normalize(torch.tensor(preferred, dtype=torch.float32)),
        normalization.normalize(torch.tensor(other, dtype=torch.float32)),
    )


def evaluate_route_ranker(
    model: nn.Module,
    dataset: RouteDataset,
    normalization: RouteNormalization,
    *,
    oracle: bool = False,
) -> dict[str, Any]:
    pairs = decisive_pairs(dataset)
    if not pairs:
        return {"pair_count": 0, "loss": 0.0, "accuracy": 0.0}
    preferred, other = _pair_tensors(dataset, pairs, normalization, oracle=oracle)
    model.eval()
    with torch.no_grad():
        diff = model(preferred) - model(other)
        loss = float(F.softplus(-diff).mean())
        correct = (diff > 0).to(torch.float64)
    output: dict[str, Any] = {
        "pair_count": len(pairs),
        "loss": loss,
        "accuracy": float(correct.mean()),
    }
    for label, selector in {
        "win_loss": lambda pair, record: pair.preference_kind == "win_loss",
        "pareto_loss": lambda pair, record: pair.preference_kind == "pareto_loss",
    }.items():
        selected_indices = [
            i for i, pair in enumerate(pairs) if selector(pair, dataset.records[pair.record_index])
        ]
        if selected_indices:
            p, o = _pair_tensors(
                dataset, pairs, normalization, oracle=oracle, indices=selected_indices
            )
            with torch.no_grad():
                d = model(p) - model(o)
            output[label] = {
                "pair_count": len(selected_indices),
                "accuracy": float((d > 0).double().mean()),
            }
        else:
            output[label] = {"pair_count": 0, "accuracy": 0.0}
    for group_name, getter in {
        "by_route": lambda record: record.route_id,
        "by_route_position": lambda record: record.route_position,
        "by_hero_policy": lambda record: record.hero_policy_id,
    }.items():
        grouped: dict[str, list[int]] = {}
        for pair_index, pair in enumerate(pairs):
            key = str(getter(dataset.records[pair.record_index]))
            grouped.setdefault(key, []).append(pair_index)
        rows: dict[str, Any] = {}
        for key, idxs in sorted(grouped.items()):
            p, o = _pair_tensors(dataset, pairs, normalization, oracle=oracle, indices=idxs)
            with torch.no_grad():
                d = model(p) - model(o)
            rows[key] = {"pair_count": len(idxs), "accuracy": float((d > 0).double().mean())}
        output[group_name] = rows
    return output


def train_route_ranker(
    spec: CandidateScorerSpec,
    train_dataset: RouteDataset,
    validation_dataset: RouteDataset,
    normalization: RouteNormalization,
    *,
    seed: int,
    config: RouteTrainingConfig,
    oracle: bool = False,
) -> TrainedRouteRanker:
    expected = ORACLE_FEATURE_NAMES if oracle else PRIMARY_FEATURE_NAMES
    if spec.feature_names != expected:
        raise ValueError("Route ranker feature order does not match the frozen route schema.")
    train_pairs = decisive_pairs(train_dataset)
    validation_pairs = decisive_pairs(validation_dataset)
    if not train_pairs or not validation_pairs:
        raise ValueError("Route ranker requires decisive TRAIN and VALIDATION pairs.")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = build_candidate_scorer(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        losses: list[tuple[float, int]] = []
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            preferred, other = _pair_tensors(
                train_dataset,
                train_pairs,
                normalization,
                oracle=oracle,
                indices=indices,
            )
            optimizer.zero_grad(set_to_none=True)
            diff = model(preferred) - model(other)
            loss = F.softplus(-diff).mean()
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), len(indices)))
        train_loss = sum(value * count for value, count in losses) / sum(
            count for _, count in losses
        )
        metrics = evaluate_route_ranker(model, validation_dataset, normalization, oracle=oracle)
        val_loss = float(metrics["loss"])
        val_acc = float(metrics["accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
            }
        )
        if val_loss < best_loss - config.min_delta:
            best_loss = val_loss
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
        raise RuntimeError("Route ranker failed to select a model state.")
    model.load_state_dict(best_state)
    final_metrics = evaluate_route_ranker(model, validation_dataset, normalization, oracle=oracle)
    return TrainedRouteRanker(
        spec=spec,
        model=model,
        seed=seed,
        selected_epoch=best_epoch,
        validation_loss=float(final_metrics["loss"]),
        validation_accuracy=float(final_metrics["accuracy"]),
        history=history,
    )


def save_normalization(normalization: RouteNormalization, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalization.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def load_normalization(path: Path) -> RouteNormalization:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RouteNormalization(
        feature_names=tuple(payload["feature_names"]),
        means=tuple(float(item) for item in payload["means"]),
        scales=tuple(float(item) for item in payload["scales"]),
        zero_variance=tuple(bool(item) for item in payload["zero_variance"]),
        schema_id=str(payload["schema_id"]),
        schema_version=int(payload["schema_version"]),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_route_ranker(
    trained: TrainedRouteRanker,
    metadata_path: Path,
    *,
    normalization_checksum: str,
    schema_id: str,
    oracle: bool,
) -> dict[str, Any]:
    model_path = metadata_path.with_suffix(".pt")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained.model.state_dict(), model_path)
    model_checksum = sha256_file(model_path)
    payload = {
        "format_id": "enemy-route-ranker-v1",
        "schema_id": schema_id,
        "schema_version": 1,
        "oracle": oracle,
        "scorer_spec": trained.spec.to_dict(),
        "training_seed": trained.seed,
        "selected_epoch": trained.selected_epoch,
        "validation_loss": trained.validation_loss,
        "validation_accuracy": trained.validation_accuracy,
        "history": trained.history,
        "normalization_checksum": normalization_checksum,
        "model_file": model_path.name,
        "model_checksum": model_checksum,
    }
    metadata_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def load_route_ranker(metadata_path: Path) -> tuple[nn.Module, dict[str, Any]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    spec_payload = payload["scorer_spec"]
    spec = CandidateScorerSpec(
        model_type=spec_payload["model_type"],
        feature_names=tuple(spec_payload["feature_names"]),
        hidden_units=tuple(spec_payload["hidden_units"]),
        activation=spec_payload["activation"],
    )
    model_path = metadata_path.with_name(str(payload["model_file"]))
    if sha256_file(model_path) != payload["model_checksum"]:
        raise ValueError("Route ranker checksum mismatch.")
    model = build_candidate_scorer(spec)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, payload


class FrozenRouteRankerPolicy:
    def __init__(
        self, model: nn.Module, normalization: RouteNormalization, *, policy_id: str
    ) -> None:
        if normalization.feature_names != PRIMARY_FEATURE_NAMES:
            raise ValueError("Shipping-eligible route policy must use the primary tactical schema.")
        self.model = model
        self.normalization = normalization
        self.policy_id = policy_id

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
        raw = torch.tensor(
            [
                encode_primary(state, definitions, candidate, runtime_context)
                for candidate in trace.candidates
            ],
            dtype=torch.float32,
        )
        values = self.normalization.normalize(raw)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(values)
        return trace.candidates[int(torch.argmax(scores).item())]


def score_record(
    model: nn.Module,
    normalization: RouteNormalization,
    record: RouteDecisionRecord,
    *,
    oracle: bool = False,
) -> int:
    vectors = [
        item.oracle_observation if oracle else item.primary_observation
        for item in record.candidates
    ]
    raw = torch.tensor(vectors, dtype=torch.float32)
    values = normalization.normalize(raw)
    model.eval()
    with torch.no_grad():
        scores = model(values)
    return int(torch.argmax(scores).item())


def paired_binary_summary(first: Sequence[bool], second: Sequence[bool]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("Paired binary inputs differ in length.")
    first_only = second_only = both = neither = 0
    diffs: list[int] = []
    for a, b in zip(first, second, strict=True):
        diffs.append(int(b) - int(a))
        if a and b:
            both += 1
        elif a:
            first_only += 1
        elif b:
            second_only += 1
        else:
            neither += 1
    count = len(first)
    first_wins = sum(first)
    second_wins = sum(second)
    return {
        "case_count": count,
        "first_wins": first_wins,
        "second_wins": second_wins,
        "first_rate": first_wins / count if count else 0.0,
        "second_rate": second_wins / count if count else 0.0,
        "first_wilson_95": wilson_interval(first_wins, count),
        "second_wilson_95": wilson_interval(second_wins, count),
        "shared_wins": both,
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "shared_losses": neither,
        "win_rate_difference": mean(diffs) if diffs else 0.0,
        "paired_difference_ci_95": paired_difference_ci(diffs),
        "mcnemar_exact_two_sided_p": mcnemar_exact(first_only, second_only),
    }
