from __future__ import annotations

from game.campaign.company import create_new_company
from game.combat.enemy_decision import HeuristicEnemyDecisionPolicy
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.expedition.cave import create_encounter_combat
from game.ml.counterfactual import combat_state_digest, rng_state_digest
from game.ml.route_termination import (
    CombatTerminationReason,
    ResearchSafetyBudget,
    RouteTerminalStatus,
    _cycle_state_digest,
    run_corrected_stateful_route,
    run_research_combat_with_termination,
)
from game.ml.stateful_route import ResearchRoute, run_stateful_route


def _fresh(encounter_id: str):
    definitions = load_game_definitions()
    state = create_encounter_combat(create_new_company(definitions), definitions, encounter_id)
    return definitions, state


def test_short_combat_matches_old_execution_exactly() -> None:
    definitions, old_state = _fresh("shallow_cave")
    old_rng = GameRng(20001)
    policy = HeuristicEnemyDecisionPolicy()
    old_episode = run_enemy_learning_episode(
        old_state,
        definitions,
        old_rng,
        policy=policy,
        max_rounds=20,
        encounter_id="shallow_cave",
        encounter_name="Shallow Cave",
        hero_policy=create_hero_policy("tactical", encounter_id="shallow_cave", seed=20001),
        enemy_wait_mode="none",
        enemy_movement_mode="recovery_only",
    )
    definitions, new_state = _fresh("shallow_cave")
    new_rng = GameRng(20001)
    result = run_research_combat_with_termination(
        new_state,
        definitions,
        new_rng,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=policy,
        hero_policy=create_hero_policy("tactical", encounter_id="shallow_cave", seed=20001),
        encounter_id="shallow_cave",
        encounter_name="Shallow Cave",
        safety_budget=ResearchSafetyBudget(),
    )
    old_enemy = tuple(
        (record.round_number, record.enemy_id, record.chosen_skill_id, record.chosen_target_id)
        for record in old_episode.records
    )
    old_hero = tuple(
        (record.round_number, record.hero_id, record.skill_id, record.target_id)
        for record in old_episode.hero_actions
    )
    assert old_episode.final_victor == "heroes"
    assert result.termination_reason == CombatTerminationReason.HERO_VICTORY
    assert combat_state_digest(old_state) == result.final_state_digest
    assert rng_state_digest(old_rng) == result.final_rng_digest
    assert old_enemy == result.enemy_action_sequence
    assert old_hero == result.hero_action_sequence


def test_safety_budget_is_censor_not_gameplay_outcome() -> None:
    definitions, state = _fresh("generated_maze_hunt")
    result = run_research_combat_with_termination(
        state,
        definitions,
        GameRng(7),
        enemy_policy_name="pi0_heuristic",
        enemy_policy=HeuristicEnemyDecisionPolicy(),
        hero_policy=create_hero_policy(
            "company_survival", encounter_id="generated_maze_hunt", seed=7
        ),
        encounter_id="generated_maze_hunt",
        encounter_name="Generated Maze Hunt",
        safety_budget=ResearchSafetyBudget(max_transitions=0, max_rounds=0),
    )
    assert result.termination_reason == CombatTerminationReason.SAFETY_BOUND_EXCEEDED
    assert result.is_censored
    assert not result.is_production_terminal
    assert not state.is_victory()
    assert not state.is_defeat()


def test_pi0_cycle_round_phase_normalizes_but_learned_round_does_not() -> None:
    _definitions, state = _fresh("generated_maze_probe")
    state.round_number = 3
    pi0_round3 = _cycle_state_digest(state, policy_name="pi0_heuristic")
    learned_round3 = _cycle_state_digest(state, policy_name="pi1_cp3")
    state.round_number = 9
    assert _cycle_state_digest(state, policy_name="pi0_heuristic") == pi0_round3
    assert _cycle_state_digest(state, policy_name="pi1_cp3") != learned_round3


def test_censored_route_does_not_sync_partial_combat_state() -> None:
    definitions = load_game_definitions()
    route = ResearchRoute("censor", ("generated_maze_hunt",), "TEST", "test")
    row = run_corrected_stateful_route(
        definitions,
        route,
        root_seed=40001,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=HeuristicEnemyDecisionPolicy(),
        hero_policy_id="company_survival",
        safety_budget=ResearchSafetyBudget(max_transitions=0, max_rounds=0),
    )
    assert row["route_terminal_status"] == RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND.value
    assert row["route_censored"] is True
    assert row["route_completed"] is False
    assert row["final_company_state"] == row["initial_state"]
    assert row["encounters"][0]["company_after_sync"] is None


def test_corrected_route_matches_cp6_persistence_for_short_route() -> None:
    definitions = load_game_definitions()
    route = ResearchRoute("probe", ("generated_maze_probe",), "TEST", "test")
    policy = HeuristicEnemyDecisionPolicy()
    old = run_stateful_route(
        definitions,
        route,
        root_seed=20001,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=policy,
        hero_policy_id="tactical",
    )
    new = run_corrected_stateful_route(
        definitions,
        route,
        root_seed=20001,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=policy,
        hero_policy_id="tactical",
        safety_budget=ResearchSafetyBudget(),
    )
    assert old["route_completed"] is True
    assert new["route_terminal_status"] == RouteTerminalStatus.ROUTE_COMPLETED.value
    assert old["final_state"] == new["final_company_state"]
