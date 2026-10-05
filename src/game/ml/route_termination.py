"""Production-faithful research termination for stateful combat routes.

This module deliberately separates game-world combat outcomes from research
execution guards. It composes the existing deterministic episode runner one
round at a time; production combat resolution remains authoritative.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from game.combat.combat_state import CombatState
from game.combat.enemy_decision import EnemyDecisionPolicy
from game.combat.enemy_learning import HeroDecisionPolicy, run_enemy_learning_episode
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.ml.counterfactual import combat_state_digest, rng_state_digest, stable_digest


class CombatTerminationReason(StrEnum):
    HERO_VICTORY = "HERO_VICTORY"
    ENEMY_VICTORY = "ENEMY_VICTORY"
    PRODUCTION_RETREAT = "PRODUCTION_RETREAT"
    DETECTED_CYCLE = "DETECTED_CYCLE"
    SAFETY_BOUND_EXCEEDED = "SAFETY_BOUND_EXCEEDED"
    INVALID_SIMULATION = "INVALID_SIMULATION"


@dataclass(frozen=True)
class ResearchSafetyBudget:
    max_transitions: int = 2048
    max_rounds: int = 256


@dataclass(frozen=True)
class RoundAudit:
    round_number: int
    start_state_digest: str
    start_rng_digest: str
    end_state_digest: str
    end_rng_digest: str
    transition_count: int
    enemy_actions: tuple[tuple[str, str, str], ...]
    hero_actions: tuple[tuple[str, str, str], ...]
    enemy_event_digest: str
    hp_before: int
    hp_after: int
    effort_before: int
    effort_after: int


@dataclass(frozen=True)
class ResearchCombatResult:
    termination_reason: CombatTerminationReason
    rounds_elapsed: int
    transitions: int
    final_state_digest: str
    final_rng_digest: str
    enemy_action_sequence: tuple[tuple[int, str, str, str], ...]
    hero_action_sequence: tuple[tuple[int, str, str, str], ...]
    enemy_event_digest: str
    rounds: tuple[RoundAudit, ...]
    cycle_first_seen_round: int | None = None
    cycle_repeat_round: int | None = None
    exception_type: str | None = None
    exception_message: str | None = None

    @property
    def is_production_terminal(self) -> bool:
        return self.termination_reason in {
            CombatTerminationReason.HERO_VICTORY,
            CombatTerminationReason.ENEMY_VICTORY,
            CombatTerminationReason.PRODUCTION_RETREAT,
        }

    @property
    def is_censored(self) -> bool:
        return self.termination_reason in {
            CombatTerminationReason.DETECTED_CYCLE,
            CombatTerminationReason.SAFETY_BOUND_EXCEEDED,
        }


def _side_totals(state: CombatState) -> tuple[int, int]:
    hp = sum(actor.hp for actor in state.all_combatants().values())
    effort = sum(actor.effort for actor in state.all_combatants().values())
    return hp, effort


def _cycle_state_digest(state: CombatState, *, policy_name: str) -> str:
    """Digest transition-relevant round-start state conservatively.

    The handcrafted heuristic only distinguishes rounds 1/2 from the late-fight
    phase at round >=3. The learned CP3/CP4 policies consume absolute round
    number, so their cycle state must retain it and cannot claim a cycle merely
    from a repeated visible board.
    """

    normalized = copy.deepcopy(state)
    if policy_name == "pi0_heuristic" and normalized.round_number >= 3:
        normalized.round_number = 3
    return combat_state_digest(normalized)


def _cycle_key(state: CombatState, rng: GameRng, *, policy_name: str) -> str:
    return stable_digest(
        {
            "state": _cycle_state_digest(state, policy_name=policy_name),
            "rng": rng_state_digest(rng),
            "policy_name": policy_name,
            "round_semantics": (
                "heuristic-late-phase-round>=3"
                if policy_name == "pi0_heuristic"
                else "absolute-round-retained"
            ),
        }
    )


def _round_transition_count(episode: Any) -> int:
    metrics = episode.metrics
    return (
        len(episode.records)
        + len(episode.hero_actions)
        + int(metrics.enemy_waits)
        + int(metrics.enemy_recovery_moves)
        + int(metrics.enemy_proactive_moves)
        + 1
    )


def _enemy_round_actions(episode: Any) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (str(record.enemy_id), str(record.chosen_skill_id), str(record.chosen_target_id))
        for record in episode.records
    )


def _hero_round_actions(episode: Any) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (str(record.hero_id), str(record.skill_id), str(record.target_id))
        for record in episode.hero_actions
    )


def _enemy_events_payload(episode: Any) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for record in episode.records:
        payload.append(
            {
                "round": int(record.round_number),
                "enemy": str(record.enemy_id),
                "skill": str(record.chosen_skill_id),
                "target": str(record.chosen_target_id),
                "events": record.events,
            }
        )
    return payload


def run_research_combat_with_termination(
    state: CombatState,
    definitions: GameDefinitions,
    rng: GameRng,
    *,
    enemy_policy_name: str,
    enemy_policy: EnemyDecisionPolicy,
    hero_policy: HeroDecisionPolicy,
    encounter_id: str,
    encounter_name: str,
    safety_budget: ResearchSafetyBudget,
    enemy_wait_mode: str = "none",
    enemy_movement_mode: str = "recovery_only",
) -> ResearchCombatResult:
    """Run deterministic combat until a production terminal or research censor."""

    start_round = state.round_number
    transitions = 0
    round_audits: list[RoundAudit] = []
    enemy_sequence: list[tuple[int, str, str, str]] = []
    hero_sequence: list[tuple[int, str, str, str]] = []
    enemy_event_payload: list[dict[str, Any]] = []
    seen_cycle_keys: dict[str, int] = {}

    def result(
        reason: CombatTerminationReason,
        *,
        cycle_first: int | None = None,
        cycle_repeat: int | None = None,
        exception: BaseException | None = None,
    ) -> ResearchCombatResult:
        return ResearchCombatResult(
            termination_reason=reason,
            rounds_elapsed=max(0, state.round_number - start_round),
            transitions=transitions,
            final_state_digest=combat_state_digest(state),
            final_rng_digest=rng_state_digest(rng),
            enemy_action_sequence=tuple(enemy_sequence),
            hero_action_sequence=tuple(hero_sequence),
            enemy_event_digest=stable_digest(enemy_event_payload),
            rounds=tuple(round_audits),
            cycle_first_seen_round=cycle_first,
            cycle_repeat_round=cycle_repeat,
            exception_type=type(exception).__name__ if exception is not None else None,
            exception_message=str(exception) if exception is not None else None,
        )

    try:
        while True:
            if state.is_victory():
                return result(CombatTerminationReason.HERO_VICTORY)
            if state.is_defeat():
                return result(CombatTerminationReason.ENEMY_VICTORY)
            if transitions >= safety_budget.max_transitions:
                return result(CombatTerminationReason.SAFETY_BOUND_EXCEEDED)
            if state.round_number - start_round >= safety_budget.max_rounds:
                return result(CombatTerminationReason.SAFETY_BOUND_EXCEEDED)

            key = _cycle_key(state, rng, policy_name=enemy_policy_name)
            prior_round = seen_cycle_keys.get(key)
            if prior_round is not None:
                return result(
                    CombatTerminationReason.DETECTED_CYCLE,
                    cycle_first=prior_round,
                    cycle_repeat=state.round_number,
                )
            seen_cycle_keys[key] = state.round_number

            round_number = state.round_number
            start_state = combat_state_digest(state)
            start_rng = rng_state_digest(rng)
            hp_before, effort_before = _side_totals(state)
            episode = run_enemy_learning_episode(
                state,
                definitions,
                rng,
                policy=enemy_policy,
                max_rounds=round_number,
                encounter_id=encounter_id,
                encounter_name=encounter_name,
                hero_policy=hero_policy,
                enemy_wait_mode=enemy_wait_mode,
                enemy_movement_mode=enemy_movement_mode,
            )
            round_transitions = _round_transition_count(episode)
            transitions += round_transitions
            enemy_actions = _enemy_round_actions(episode)
            hero_actions = _hero_round_actions(episode)
            for enemy_id, skill_id, target_id in enemy_actions:
                enemy_sequence.append((round_number, enemy_id, skill_id, target_id))
            for hero_id, skill_id, target_id in hero_actions:
                hero_sequence.append((round_number, hero_id, skill_id, target_id))
            enemy_event_payload.extend(_enemy_events_payload(episode))
            hp_after, effort_after = _side_totals(state)
            round_audits.append(
                RoundAudit(
                    round_number=round_number,
                    start_state_digest=start_state,
                    start_rng_digest=start_rng,
                    end_state_digest=combat_state_digest(state),
                    end_rng_digest=rng_state_digest(rng),
                    transition_count=round_transitions,
                    enemy_actions=enemy_actions,
                    hero_actions=hero_actions,
                    enemy_event_digest=stable_digest(_enemy_events_payload(episode)),
                    hp_before=hp_before,
                    hp_after=hp_after,
                    effort_before=effort_before,
                    effort_after=effort_after,
                )
            )
    except Exception as exc:  # pragma: no cover - exercised by explicit invalid test
        return result(CombatTerminationReason.INVALID_SIMULATION, exception=exc)


def result_payload(result: ResearchCombatResult) -> dict[str, Any]:
    return {
        "termination_reason": result.termination_reason.value,
        "rounds_elapsed": result.rounds_elapsed,
        "transitions": result.transitions,
        "final_state_digest": result.final_state_digest,
        "final_rng_digest": result.final_rng_digest,
        "enemy_action_sequence": [list(item) for item in result.enemy_action_sequence],
        "hero_action_sequence": [list(item) for item in result.hero_action_sequence],
        "enemy_event_digest": result.enemy_event_digest,
        "cycle_first_seen_round": result.cycle_first_seen_round,
        "cycle_repeat_round": result.cycle_repeat_round,
        "exception_type": result.exception_type,
        "exception_message": result.exception_message,
        "rounds": [
            {
                "round_number": row.round_number,
                "start_state_digest": row.start_state_digest,
                "start_rng_digest": row.start_rng_digest,
                "end_state_digest": row.end_state_digest,
                "end_rng_digest": row.end_rng_digest,
                "transition_count": row.transition_count,
                "enemy_actions": [list(item) for item in row.enemy_actions],
                "hero_actions": [list(item) for item in row.hero_actions],
                "enemy_event_digest": row.enemy_event_digest,
                "hp_before": row.hp_before,
                "hp_after": row.hp_after,
                "effort_before": row.effort_before,
                "effort_after": row.effort_after,
            }
            for row in result.rounds
        ],
    }


class RouteTerminalStatus(StrEnum):
    ROUTE_COMPLETED = "ROUTE_COMPLETED"
    ROUTE_FAILED_ENEMY_VICTORY = "ROUTE_FAILED_ENEMY_VICTORY"
    ROUTE_ENDED_PRODUCTION_RETREAT = "ROUTE_ENDED_PRODUCTION_RETREAT"
    ROUTE_CENSORED_CYCLE = "ROUTE_CENSORED_CYCLE"
    ROUTE_CENSORED_SAFETY_BOUND = "ROUTE_CENSORED_SAFETY_BOUND"
    ROUTE_INVALID = "ROUTE_INVALID"


def _combat_snapshot(state: CombatState) -> dict[str, Any]:
    heroes = [state.heroes[key] for key in sorted(state.heroes)]
    max_hp = sum(hero.max_hp for hero in heroes)
    max_effort = sum(hero.max_effort for hero in heroes)
    return {
        "active_hero_count": sum(hero.life_state.value != "dead" for hero in heroes),
        "downed_count": sum(hero.life_state.value == "downed" for hero in heroes),
        "hp_total": sum(hero.hp for hero in heroes),
        "max_hp_total": max_hp,
        "hp_fraction": sum(hero.hp for hero in heroes) / max_hp if max_hp else 0.0,
        "effort_total": sum(hero.effort for hero in heroes),
        "max_effort_total": max_effort,
        "mortal_wounds_total": sum(hero.mortal_wounds for hero in heroes),
        "strain_marks_total": sum(len(hero.strain_marks) for hero in heroes),
    }


def _deterioration(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        "hp_loss": int(before["hp_total"]) - int(after["hp_total"]),
        "effort_loss": int(before["effort_total"]) - int(after["effort_total"]),
        "active_hero_delta": int(after["active_hero_count"]) - int(before["active_hero_count"]),
        "downed_delta": int(after["downed_count"]) - int(before["downed_count"]),
        "mortal_wounds_delta": int(after["mortal_wounds_total"])
        - int(before["mortal_wounds_total"]),
        "strain_marks_delta": int(after["strain_marks_total"]) - int(before["strain_marks_total"]),
    }


def behavior_counts(
    result: ResearchCombatResult,
    definitions: GameDefinitions,
) -> dict[str, Any]:
    from collections import Counter

    skill_counts: Counter[str] = Counter()
    marks = guards = grabs = bites = effort_drains = direct_pressure = 0
    for _round, _enemy, skill_id, _target in result.enemy_action_sequence:
        skill_counts[skill_id] += 1
        skill = definitions.skills[skill_id]
        tags = set(skill.tags)
        marks += int("mark" in tags)
        guards += int("guard" in tags)
        grabs += int("drag_forward" in tags)
        effort_drains += int("effort_drain" in tags)
        direct_pressure += int((skill.damage_max or 0) > 0)
        bites += int("bite" in skill_id or "bite" in tags)
    no_progress_rounds = sum(
        row.hp_before == row.hp_after and row.effort_before == row.effort_after
        for row in result.rounds
    )
    return {
        "enemy_actions": len(result.enemy_action_sequence),
        "hero_actions": len(result.hero_action_sequence),
        "mark_uses": marks,
        "guard_uses": guards,
        "grab_uses": grabs,
        "bite_uses": bites,
        "effort_drain_uses": effort_drains,
        "direct_pressure_uses": direct_pressure,
        "skill_counts": dict(sorted(skill_counts.items())),
        "no_progress_rounds": no_progress_rounds,
        "no_progress_round_rate": (
            no_progress_rounds / len(result.rounds) if result.rounds else 0.0
        ),
    }


def run_corrected_stateful_route(
    definitions: GameDefinitions,
    route: Any,
    *,
    root_seed: int,
    enemy_policy_name: str,
    enemy_policy: EnemyDecisionPolicy,
    hero_policy_id: str,
    safety_budget: ResearchSafetyBudget,
) -> dict[str, Any]:
    """Run one CP6 route with production-valid terminal semantics only."""

    from game.campaign.company import create_new_company
    from game.campaign.roster import sync_company_from_combat
    from game.combat.enemy_learning import create_hero_policy
    from game.expedition.cave import create_encounter_combat
    from game.ml.stateful_route import (
        derive_encounter_seed,
        snapshot_combat_entry,
        snapshot_company,
    )

    company = create_new_company(definitions)
    initial = snapshot_company(company, definitions)
    encounters: list[dict[str, Any]] = []
    completed_count = 0
    terminal_status: RouteTerminalStatus | None = None
    stopped_at: str | None = None

    for encounter_index, encounter_id in enumerate(route.encounter_ids):
        company_before = snapshot_company(company, definitions)
        combat = create_encounter_combat(company, definitions, encounter_id)
        combat_entry = snapshot_combat_entry(combat)
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
        combat_result = run_research_combat_with_termination(
            combat,
            definitions,
            rng,
            enemy_policy_name=enemy_policy_name,
            enemy_policy=enemy_policy,
            hero_policy=create_hero_policy(
                hero_policy_id,
                encounter_id=encounter_id,
                seed=encounter_seed,
            ),
            encounter_id=encounter_id,
            encounter_name=encounter_id.replace("_", " ").title(),
            safety_budget=safety_budget,
            enemy_wait_mode="none",
            enemy_movement_mode="recovery_only",
        )
        partial = _combat_snapshot(combat)
        terminal = combat_result.termination_reason
        company_after: dict[str, Any] | None = None
        deterioration: dict[str, int] | None = None
        if terminal in {
            CombatTerminationReason.HERO_VICTORY,
            CombatTerminationReason.ENEMY_VICTORY,
            CombatTerminationReason.PRODUCTION_RETREAT,
        }:
            sync_company_from_combat(company, combat.heroes, combat.party_formation)
            company_after = snapshot_company(company, definitions)
            deterioration = _deterioration(combat_entry, company_after)

        encounters.append(
            {
                "encounter_index": encounter_index,
                "encounter_id": encounter_id,
                "encounter_seed": encounter_seed,
                "company_before_create": company_before,
                "combat_entry": combat_entry,
                "termination_reason": terminal.value,
                "production_terminal": combat_result.is_production_terminal,
                "censored": combat_result.is_censored,
                "company_after_sync": company_after,
                "partial_combat_state": partial,
                "deterioration": deterioration,
                "rounds": combat_result.rounds_elapsed,
                "transitions": combat_result.transitions,
                "cycle_first_seen_round": combat_result.cycle_first_seen_round,
                "cycle_repeat_round": combat_result.cycle_repeat_round,
                "final_state_digest": combat_result.final_state_digest,
                "final_rng_digest": combat_result.final_rng_digest,
                "enemy_event_digest": combat_result.enemy_event_digest,
                "enemy_action_sequence": [
                    list(item) for item in combat_result.enemy_action_sequence
                ],
                "hero_action_sequence": [list(item) for item in combat_result.hero_action_sequence],
                "behavior": behavior_counts(combat_result, definitions),
                "round_audit": result_payload(combat_result)["rounds"],
            }
        )

        if terminal == CombatTerminationReason.HERO_VICTORY:
            completed_count += 1
            continue
        stopped_at = encounter_id
        if terminal == CombatTerminationReason.ENEMY_VICTORY:
            terminal_status = RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY
        elif terminal == CombatTerminationReason.PRODUCTION_RETREAT:
            terminal_status = RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT
        elif terminal == CombatTerminationReason.DETECTED_CYCLE:
            terminal_status = RouteTerminalStatus.ROUTE_CENSORED_CYCLE
        elif terminal == CombatTerminationReason.SAFETY_BOUND_EXCEEDED:
            terminal_status = RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND
        else:
            terminal_status = RouteTerminalStatus.ROUTE_INVALID
        break

    if terminal_status is None:
        terminal_status = RouteTerminalStatus.ROUTE_COMPLETED
    final_company = snapshot_company(company, definitions)
    return {
        "case_id": f"{route.route_id}|{hero_policy_id}|{root_seed}",
        "route_id": route.route_id,
        "route_structure": route.structure,
        "root_seed": root_seed,
        "enemy_policy": enemy_policy_name,
        "hero_policy_id": hero_policy_id,
        "encounter_sequence": list(route.encounter_ids),
        "route_terminal_status": terminal_status.value,
        "route_completed": terminal_status == RouteTerminalStatus.ROUTE_COMPLETED,
        "route_failed_enemy_victory": (
            terminal_status == RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY
        ),
        "route_censored": terminal_status
        in {
            RouteTerminalStatus.ROUTE_CENSORED_CYCLE,
            RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND,
        },
        "route_invalid": terminal_status == RouteTerminalStatus.ROUTE_INVALID,
        "encounters_completed": completed_count,
        "stopped_at_encounter_id": stopped_at,
        "initial_state": initial,
        "final_company_state": final_company,
        "encounters": encounters,
        "total_rounds": sum(int(row["rounds"]) for row in encounters),
        "total_transitions": sum(int(row["transitions"]) for row in encounters),
    }
