from __future__ import annotations

import game.combat.actions as combat_actions
import game.combat.damage as combat_damage
import game.combat.damage_range as damage_range
from game.campaign.company import create_new_company
from game.combat.combat_state import Tag
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import _load_frozen_policies
from game.dev.ml_cp6 import PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.expedition.cave import create_encounter_combat
from game.ml.balance_perturbation import (
    PERTURBATIONS,
    MechanicOverrideContext,
    ResolutionRecorder,
    cloned_definitions,
    run_perturbed_route,
)
from game.ml.route_termination import run_corrected_stateful_route


def _route(route_id: str):
    return next(route for route in PRIMARY_ROUTES if route.route_id == route_id)


def test_data_driven_variants_do_not_mutate_production_definitions() -> None:
    definitions = load_game_definitions()
    baseline = {
        skill_id: (skill.effort_cost, skill.damage_min, skill.damage_max)
        for skill_id, skill in definitions.skills.items()
    }

    m2 = cloned_definitions(definitions, PERTURBATIONS["M2"])
    d2 = cloned_definitions(definitions, PERTURBATIONS["D2"])
    e1 = cloned_definitions(definitions, PERTURBATIONS["E1"])
    g1 = cloned_definitions(definitions, PERTURBATIONS["G1"])

    assert m2.skills["mark_the_pattern"].effort_cost == 0
    assert m2.skills["splinter_mark"].effort_cost == 0
    assert d2.skills["ward_pattern"].effort_cost == 0
    assert d2.skills["shielding_dead"].effort_cost == 0
    assert e1.skills["effort_drain"].effort_cost == 0
    assert (g1.skills["drag_forward"].damage_min, g1.skills["drag_forward"].damage_max) == (2, 3)
    assert (g1.skills["stalker_hook"].damage_min, g1.skills["stalker_hook"].damage_max) == (2, 3)
    assert baseline == {
        skill_id: (skill.effort_cost, skill.damage_min, skill.damage_max)
        for skill_id, skill in definitions.skills.items()
    }


def test_m1_changes_only_enemy_marked_vulnerable_payoff() -> None:
    definitions = load_game_definitions()
    company = create_new_company(definitions)
    state = create_encounter_combat(company, definitions, "generated_maze_hunt")
    enemy = next(item for item in state.enemies.values() if "maw_slam" in item.skills)
    hero = next(item for item in state.heroes.values() if item.can_act())
    hero.tags.add(Tag.MARKED)
    skill = definitions.skills["maw_slam"]
    baseline = damage_range.skill_damage_bonus(state, enemy, hero, set(skill.tags))
    recorder = ResolutionRecorder()
    with MechanicOverrideContext(PERTURBATIONS["M1"], recorder):
        changed = damage_range.skill_damage_bonus(state, enemy, hero, set(skill.tags))
    assert changed == baseline + 1
    assert damage_range.skill_damage_bonus(state, enemy, hero, set(skill.tags)) == baseline


def test_g2_extends_only_drag_applied_mark_inside_context() -> None:
    definitions = load_game_definitions()
    changed = cloned_definitions(definitions, PERTURBATIONS["G2"])
    found_success = False
    for seed in range(1, 100):
        company = create_new_company(changed)
        state = create_encounter_combat(company, changed, "cave_mini_boss")
        enemy = next(item for item in state.enemies.values() if "drag_forward" in item.skills)
        target = next(
            hero for hero in state.heroes.values() if hero.formation_slot.value.startswith("BACK")
        )
        recorder = ResolutionRecorder()
        with MechanicOverrideContext(PERTURBATIONS["G2"], recorder):
            combat_actions.use_skill(
                state,
                enemy.actor_id,
                changed.skills["drag_forward"],
                target.actor_id,
                GameRng(seed),
                ignore_target_legality=True,
            )
        if Tag.MARKED in target.tags:
            assert target.tag_turns[Tag.MARKED] == 3
            found_success = True
            break
    assert found_success


def test_d1_enemy_guard_mitigation_is_process_local_and_restored() -> None:
    definitions = load_game_definitions()
    company = create_new_company(definitions)
    state = create_encounter_combat(company, definitions, "generated_maze_pattern_cell")
    hero = next(item for item in state.heroes.values() if item.can_act())
    enemy = next(item for item in state.enemies.values() if item.is_alive())
    enemy.tags.add(Tag.GUARDED)
    hp_before = enemy.hp
    original_constant = combat_damage.GUARDED_DAMAGE_REDUCTION
    recorder = ResolutionRecorder()
    with MechanicOverrideContext(PERTURBATIONS["D1"], recorder):
        combat_actions.apply_damage(state, hero.actor_id, enemy.actor_id, 5)
    assert enemy.hp == hp_before - 1
    assert combat_damage.GUARDED_DAMAGE_REDUCTION == original_constant == 3


def test_baseline_perturbation_route_reproduces_cp7_exactly() -> None:
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    route = _route("fresh_hunt")
    expected = run_corrected_stateful_route(
        definitions,
        route,
        root_seed=40001,
        enemy_policy_name="pi2_cp4",
        enemy_policy=policies["pi2_cp4"],
        hero_policy_id="company_survival",
        safety_budget=SAFETY_BUDGET,
    )
    actual = run_perturbed_route(
        definitions,
        route,
        root_seed=40001,
        enemy_policy_name="pi2_cp4",
        enemy_policy=policies["pi2_cp4"],
        hero_policy_id="company_survival",
        spec=PERTURBATIONS["BASELINE"],
        safety_budget=SAFETY_BUDGET,
    )
    assert actual["route_terminal_status"] == expected["route_terminal_status"]
    assert actual["final_company_state"] == expected["final_company_state"]
    assert len(actual["encounters"]) == len(expected["encounters"])
    for actual_encounter, expected_encounter in zip(
        actual["encounters"], expected["encounters"], strict=True
    ):
        assert actual_encounter["final_state_digest"] == expected_encounter["final_state_digest"]
        assert actual_encounter["final_rng_digest"] == expected_encounter["final_rng_digest"]
        assert (
            actual_encounter["enemy_action_sequence"] == expected_encounter["enemy_action_sequence"]
        )
        assert (
            actual_encounter["hero_action_sequence"] == expected_encounter["hero_action_sequence"]
        )
        assert actual_encounter["enemy_event_digest"] == expected_encounter["enemy_event_digest"]


def test_baseline_context_does_not_leak_function_overrides() -> None:
    before_use_skill = combat_actions.use_skill
    before_bonus = damage_range.skill_damage_bonus
    recorder = ResolutionRecorder()
    with MechanicOverrideContext(PERTURBATIONS["BASELINE"], recorder):
        assert combat_actions.use_skill is not before_use_skill
    assert combat_actions.use_skill is before_use_skill
    assert damage_range.skill_damage_bonus is before_bonus
