from __future__ import annotations

from game.campaign.company import create_new_company
from game.campaign.gear import effective_hero_stats
from game.campaign.roster import sync_company_from_combat
from game.combat.combat_state import LifeState, MoraleState, StrainMark, StrainTier, Tag
from game.combat.enemy_decision import HeuristicEnemyDecisionPolicy
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.combat.formation import FormationSlot
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.expedition.cave import create_encounter_combat
from game.ml.stateful_route import (
    ResearchRoute,
    aggregate_route_cases,
    derive_encounter_seed,
    paired_route_comparison,
    route_digest,
    run_stateful_route,
    snapshot_company,
)


def _definitions():
    return load_game_definitions()


def _company_hero(company, hero_id: str):
    return next(hero for hero in company.roster if hero.hero_id == hero_id)


def test_sync_persists_attritional_hero_state_and_clears_mark() -> None:
    definitions = _definitions()
    company = create_new_company(definitions)
    combat = create_encounter_combat(company, definitions, "shallow_cave")
    hero = combat.heroes["hero_field_surgeon"]
    hero.hp = 6
    hero.effort = 2
    hero.life_state = LifeState.DOWNED
    hero.morale = MoraleState.SHAKEN
    hero.strain = StrainTier.STRAINED
    hero.strain_marks = {StrainMark.BATTERED}
    hero.mortal_wounds = 2
    hero.tags.update({Tag.MARKED, Tag.GUARDED, Tag.WET})

    sync_company_from_combat(company, combat.heroes, combat.party_formation)
    persisted = _company_hero(company, hero.actor_id)

    assert persisted.hp == 6
    assert persisted.effort == 2
    assert persisted.life_state is LifeState.DOWNED
    assert persisted.morale is MoraleState.SHAKEN
    assert persisted.strain is StrainTier.STRAINED
    assert persisted.strain_marks == {StrainMark.BATTERED}
    assert persisted.mortal_wounds == 2
    assert Tag.MARKED not in persisted.tags
    assert {Tag.GUARDED, Tag.WET} <= persisted.tags

    next_combat = create_encounter_combat(company, definitions, "cave_mini_boss")
    next_hero = next_combat.heroes[hero.actor_id]
    assert next_hero.hp == 6
    assert next_hero.effort == 2
    assert next_hero.life_state is LifeState.DOWNED
    assert next_hero.mortal_wounds == 2
    assert Tag.MARKED not in next_hero.tags
    assert {Tag.GUARDED, Tag.WET} <= next_hero.tags


def test_persistent_drained_mark_reapplies_combat_start_penalty() -> None:
    definitions = _definitions()
    company = create_new_company(definitions)
    combat = create_encounter_combat(company, definitions, "shallow_cave")
    hero = combat.heroes["hero_scribe"]
    hero.effort = 3
    hero.strain_marks.add(StrainMark.DRAINED)
    sync_company_from_combat(company, combat.heroes, combat.party_formation)

    persisted = _company_hero(company, hero.actor_id)
    assert persisted.effort == 3
    next_combat = create_encounter_combat(company, definitions, "cave_mini_boss")
    assert next_combat.heroes[hero.actor_id].effort == 2


def test_death_removes_hero_from_next_encounter_and_active_party() -> None:
    definitions = _definitions()
    company = create_new_company(definitions)
    combat = create_encounter_combat(company, definitions, "shallow_cave")
    hero = combat.heroes["hero_cutpurse"]
    hero.hp = 0
    hero.life_state = LifeState.DEAD
    sync_company_from_combat(company, combat.heroes, combat.party_formation)

    assert all(item.hero_id != hero.actor_id for item in company.roster)
    assert hero.actor_id in {item.hero_id for item in company.deceased_heroes}
    assert hero.actor_id not in set(company.active_party_slots.values())
    next_combat = create_encounter_combat(company, definitions, "cave_mini_boss")
    assert hero.actor_id not in next_combat.heroes


def test_interactive_party_formation_persists_between_combats() -> None:
    definitions = _definitions()
    company = create_new_company(definitions)
    combat = create_encounter_combat(company, definitions, "shallow_cave")
    assert combat.party_formation.swap_slots(
        FormationSlot.FRONT_LEFT,
        FormationSlot.BACK_LEFT,
        require_adjacent=False,
    )
    sync_company_from_combat(company, combat.heroes, combat.party_formation)

    next_combat = create_encounter_combat(company, definitions, "cave_mini_boss")
    assert next_combat.party_formation.actor_at(FormationSlot.FRONT_LEFT) == "hero_field_surgeon"
    assert next_combat.party_formation.actor_at(FormationSlot.BACK_LEFT) == "hero_watchman"


def test_gear_derived_stats_are_recomputed_from_persistent_company_state() -> None:
    definitions = _definitions()
    company = create_new_company(definitions)
    hero = _company_hero(company, "hero_watchman")
    gear_id = next(iter(definitions.gear))
    hero.equipped_gear_id = gear_id
    expected = effective_hero_stats(hero, definitions)

    combat = create_encounter_combat(company, definitions, "shallow_cave")
    assert combat.heroes[hero.hero_id].max_hp == expected.max_hp
    assert combat.heroes[hero.hero_id].max_effort == expected.max_effort


def test_encounter_seed_is_stable_and_enemy_policy_independent() -> None:
    first = derive_encounter_seed(
        40001,
        route_id="light_to_hunt",
        encounter_index=1,
        encounter_id="generated_maze_hunt",
        hero_policy_id="company_survival",
    )
    second = derive_encounter_seed(
        40001,
        route_id="light_to_hunt",
        encounter_index=1,
        encounter_id="generated_maze_hunt",
        hero_policy_id="company_survival",
    )
    assert first == second


def test_stateful_route_is_exactly_deterministic() -> None:
    definitions = _definitions()
    route = ResearchRoute(
        route_id="test_authored_cave",
        encounter_ids=("shallow_cave", "cave_mini_boss"),
        structure="test",
        purpose="determinism",
    )
    first = run_stateful_route(
        definitions,
        route,
        root_seed=40001,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=HeuristicEnemyDecisionPolicy(),
        hero_policy_id="company_survival",
    )
    second = run_stateful_route(
        definitions,
        route,
        root_seed=40001,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=HeuristicEnemyDecisionPolicy(),
        hero_policy_id="company_survival",
    )
    assert route_digest(first) == route_digest(second)


def test_single_encounter_route_matches_direct_production_transition() -> None:
    definitions = _definitions()
    route = ResearchRoute(
        route_id="single_shallow",
        encounter_ids=("shallow_cave",),
        structure="isolated",
        purpose="baseline",
    )
    root_seed = 40002
    policy = HeuristicEnemyDecisionPolicy()
    row = run_stateful_route(
        definitions,
        route,
        root_seed=root_seed,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=policy,
        hero_policy_id="company_survival",
    )

    company = create_new_company(definitions)
    combat = create_encounter_combat(company, definitions, "shallow_cave")
    seed = derive_encounter_seed(
        root_seed,
        route_id=route.route_id,
        encounter_index=0,
        encounter_id="shallow_cave",
        hero_policy_id="company_survival",
    )
    run_enemy_learning_episode(
        combat,
        definitions,
        GameRng(seed),
        policy=HeuristicEnemyDecisionPolicy(),
        max_rounds=20,
        encounter_id="shallow_cave",
        encounter_name="Shallow Cave",
        hero_policy=create_hero_policy(
            "company_survival",
            encounter_id="shallow_cave",
            seed=seed,
        ),
        enemy_wait_mode="none",
        enemy_movement_mode="recovery_only",
    )
    sync_company_from_combat(company, combat.heroes, combat.party_formation)
    assert row["final_state"] == snapshot_company(company, definitions)


def test_route_aggregation_and_pairing_preserve_case_identity() -> None:
    base = [
        {
            "case_id": "a",
            "route_failed": False,
            "encounters_completed": 2,
            "failed_at_encounter_id": None,
            "final_state": {
                "active_hero_count": 4,
                "hp_total": 30,
                "hp_fraction": 0.5,
                "effort_total": 5,
                "mortal_wounds_total": 0,
            },
            "total_downs": 0,
            "total_deaths": 0,
            "total_mortal_wounds": 0,
            "total_rounds": 4,
            "cumulative_damage_taken": 10,
            "hero_effort_spent": 3,
            "cumulative_enemy_effort_drain": 1,
        }
    ]
    comparison = [
        {
            "case_id": "a",
            "route_failed": True,
            "encounters_completed": 2,
            "failed_at_encounter_id": "hard",
            "final_state": {
                "active_hero_count": 4,
                "hp_total": 10,
                "hp_fraction": 0.2,
                "effort_total": 5,
                "mortal_wounds_total": 0,
            },
            "total_downs": 0,
            "total_deaths": 0,
            "total_mortal_wounds": 0,
            "total_rounds": 4,
            "cumulative_damage_taken": 10,
            "hero_effort_spent": 3,
            "cumulative_enemy_effort_drain": 1,
        }
    ]
    assert aggregate_route_cases(base)["route_failure_rate"] == 0.0
    paired = paired_route_comparison(base, comparison)
    assert paired["route_failure_rate_difference"] == 1.0
    assert paired["transition_matrix"]["comparison_only_failure"] == 1


def test_fresh_and_conditioned_target_use_same_target_rng_stream() -> None:
    definitions = _definitions()
    policy = HeuristicEnemyDecisionPolicy()
    fresh = ResearchRoute(
        route_id="freshcmp_hunt_fresh_test",
        encounter_ids=("generated_maze_hunt",),
        structure="fresh",
        purpose="seed pairing",
        seed_namespace="freshcmp_generated_maze_hunt_test",
        seed_by_encounter_identity=True,
    )
    conditioned = ResearchRoute(
        route_id="freshcmp_hunt_conditioned_test",
        encounter_ids=("generated_maze_probe", "generated_maze_hunt"),
        structure="conditioned",
        purpose="seed pairing",
        seed_namespace="freshcmp_generated_maze_hunt_test",
        seed_by_encounter_identity=True,
    )
    fresh_row = run_stateful_route(
        definitions,
        fresh,
        root_seed=40003,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=policy,
        hero_policy_id="company_survival",
    )
    conditioned_row = run_stateful_route(
        definitions,
        conditioned,
        root_seed=40003,
        enemy_policy_name="pi0_heuristic",
        enemy_policy=policy,
        hero_policy_id="company_survival",
    )
    fresh_seed = fresh_row["encounters"][0]["encounter_seed"]
    conditioned_target = next(
        row
        for row in conditioned_row["encounters"]
        if row["encounter_id"] == "generated_maze_hunt"
    )
    assert conditioned_target["encounter_seed"] == fresh_seed
