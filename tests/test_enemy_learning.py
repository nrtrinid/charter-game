from __future__ import annotations

from game.campaign.company import create_new_company
from game.combat.combat_state import ActorStatus, Combatant, CombatState, Tag, Team
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionRuntimeContext,
    EnemyDecisionTrace,
    choose_enemy_skill_and_target,
    explain_enemy_decision,
)
from game.combat.enemy_learning import (
    SUPPORTED_HERO_POLICY_IDS,
    AntiMarkHeroPolicy,
    CompanySurvivalHeroPolicy,
    ConservativeHeroPolicy,
    DamageRaceHeroPolicy,
    EnemyDecisionEpisode,
    EnemyDecisionRecord,
    LinearEnemyDecisionPolicy,
    MixedHeroPolicy,
    NaiveHeroPolicy,
    PartyCollapseRewardWeights,
    SurvivalHeroPolicy,
    TacticalHeroPolicy,
    _company_survival_score_entries,
    _explain_company_survival_choice,
    _PendingEnemyTiming,
    _TimingOutcome,
    create_hero_policy,
    learn_linear_enemy_weights,
    run_enemy_learning_episode,
    score_enemy_action_events,
    score_enemy_episode,
    score_enemy_timing_outcome,
)
from game.combat.formation import Formation, FormationSlot
from game.core.events import (
    DamageEvent,
    DeathEvent,
    DownedEvent,
    MissEvent,
    SkillUsedEvent,
    StatusChangedEvent,
)
from game.core.rng import GameRng
from game.dev.hero_policy_audit import aggregate_hero_policy_audit
from game.dev.route_lab import GeneratedRouteLabConfig, run_generated_route_lab
from tests.conftest import get_definitions


def test_learning_episode_records_enemy_decision_trace_and_features() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone", "cheap_shot"])
    state.heroes["front_right"].tags.add(Tag.MARKED)

    episode = run_enemy_learning_episode(state, definitions, GameRng(1), max_rounds=1)

    assert episode.records
    record = episode.records[0]
    assert record.trace.candidates
    assert record.trace.chosen is not None
    assert record.chosen_skill_id == record.trace.chosen.skill_id
    assert record.chosen_target_id == record.trace.chosen.target_id
    assert record.chosen_features
    assert record.events


def test_naive_hero_policy_uses_first_usable_skill_and_first_legal_target() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    policy = NaiveHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "enemy")


def test_learning_episode_default_hero_policy_matches_explicit_naive_policy() -> None:
    definitions = get_definitions()
    default_episode = run_enemy_learning_episode(
        _combat_state(enemy_skills=["skulker_stone", "cheap_shot"]),
        definitions,
        GameRng(1),
        max_rounds=1,
    )
    explicit_episode = run_enemy_learning_episode(
        _combat_state(enemy_skills=["skulker_stone", "cheap_shot"]),
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=NaiveHeroPolicy(),
    )

    assert default_episode.total_reward == explicit_episode.total_reward
    assert default_episode.final_victor == explicit_episode.final_victor
    assert [
        (record.chosen_skill_id, record.chosen_target_id)
        for record in default_episode.records
    ] == [
        (record.chosen_skill_id, record.chosen_target_id)
        for record in explicit_episode.records
    ]


def test_learning_episode_uses_explicit_hero_policy() -> None:
    definitions = get_definitions()
    policy = _RecordingHeroPolicy()

    run_enemy_learning_episode(
        _combat_state(enemy_skills=["skulker_stone"]),
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=policy,
    )

    assert policy.hero_ids


def test_damage_race_policy_prefers_free_kill_over_effort_kill() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["guard_strike", "shield_drive"]
    state.heroes["front_left"].effort = 1
    state.enemies["enemy"].hp = 3
    policy = DamageRaceHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "enemy")


def test_damage_race_policy_prefers_higher_expected_damage() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["knife_work", "exposed_cut"]
    state.heroes["front_left"].effort = 1
    state.enemies["enemy"].tags.add(Tag.MARKED)
    state.enemies["enemy"].hp = 10
    policy = DamageRaceHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("exposed_cut", "enemy")


def test_survival_policy_prioritizes_downed_or_low_hp_ally_healing() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["bone_saw", "emergency_stitch"]
    state.heroes["front_left"].effort = 1
    state.heroes["front_right"].hp = 0
    state.heroes["front_right"].statuses.add(ActorStatus.DOWNED)
    policy = SurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == (
        "emergency_stitch",
        "front_right",
    )


def test_survival_policy_falls_back_to_offense_without_urgent_heal() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["bone_saw", "emergency_stitch"]
    state.heroes["front_left"].effort = 1
    policy = SurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("bone_saw", "enemy")


def test_anti_mark_policy_targets_killable_marker_when_marked_hero_exists() -> None:
    definitions = get_definitions()
    state = _anti_mark_state()
    policy = AntiMarkHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "z_marker")


def test_anti_mark_policy_falls_back_safely_without_clear_response() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_right"].tags.add(Tag.MARKED)
    policy = AntiMarkHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "enemy")


def test_mixed_hero_policy_is_stable_and_varies_across_seeds() -> None:
    stable = MixedHeroPolicy(encounter_id="road_bandits", seed=1)
    assert stable.selected_policy_id == MixedHeroPolicy(
        encounter_id="road_bandits",
        seed=1,
    ).selected_policy_id

    selected = {
        MixedHeroPolicy(encounter_id="road_bandits", seed=seed).selected_policy_id
        for seed in range(1, 9)
    }
    assert len(selected) > 1


def test_learning_episode_can_execute_treatment_policy_action() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["mark_the_pattern"])
    state.heroes["front_left"].skills = ["emergency_stitch"]
    state.heroes["front_left"].effort = 1
    state.heroes["front_right"].hp = 1

    run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_FixedHeroPolicy("front_left", "emergency_stitch", "front_right"),
    )

    assert state.heroes["front_right"].hp > 1


def test_learning_episode_uses_enemy_recovery_move_when_no_skill_is_legal() -> None:
    definitions = get_definitions()
    state = _two_enemy_state(
        acting_skills=["rusted_chop"],
        acting_slot=FormationSlot.BACK_LEFT,
        blocker_slot=FormationSlot.FRONT_LEFT,
    )

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_RecordingHeroPolicy(),
    )

    assert episode.metrics.enemy_recovery_moves == 1
    assert state.enemy_formation.actor_at(FormationSlot.FRONT_LEFT) == "acting_enemy"


def test_enemy_package_wait_mode_delays_payoff_until_later_marker() -> None:
    definitions = get_definitions()
    definitions.skills["sure_dirty_finish"] = definitions.skills["dirty_finish"].model_copy(
        update={
            "id": "sure_dirty_finish",
            "accuracy": 100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    state = _mark_setup_state()
    state.enemies["acting_enemy"].skills = ["sure_dirty_finish"]

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_RecordingHeroPolicy(),
        enemy_wait_mode="package_only",
    )
    events = _events(episode)

    assert episode.metrics.enemy_waits == 1
    assert episode.metrics.waited_then_attacked_next_activation == 1
    assert episode.metrics.waited_then_marked_hit == 1
    assert episode.metrics.waited_then_payoff == 1
    assert episode.metrics.waits_without_payoff == 0
    assert any(
        isinstance(event, SkillUsedEvent) and event.actor_id == "marker_enemy"
        for event in events
    )


def test_enemy_package_move_mode_moves_into_marked_payoff_lane() -> None:
    definitions = get_definitions()
    definitions.skills["sure_dirty_finish"] = definitions.skills["dirty_finish"].model_copy(
        update={
            "id": "sure_dirty_finish",
            "accuracy": 100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    state = _two_enemy_state(
        acting_skills=["sure_dirty_finish"],
        acting_slot=FormationSlot.BACK_LEFT,
        blocker_slot=FormationSlot.FRONT_LEFT,
    )
    state.heroes["front_left"].tags.add(Tag.MARKED)

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=2,
        hero_policy=_RecordingHeroPolicy(),
        enemy_movement_mode="package_only",
    )

    assert episode.metrics.enemy_proactive_moves == 1
    assert episode.metrics.move_then_attack_next_activation == 1
    assert episode.metrics.move_then_marked_hit == 1
    assert episode.metrics.move_then_payoff == 1
    assert episode.metrics.move_into_marked_lane == 1
    assert episode.metrics.move_wasted_no_followup == 0
    assert state.enemy_formation.actor_at(FormationSlot.FRONT_LEFT) == "acting_enemy"


def test_wait_quality_counts_down_and_no_payoff() -> None:
    definitions = get_definitions()
    definitions.skills["sure_dirty_finish"] = definitions.skills["dirty_finish"].model_copy(
        update={
            "id": "sure_dirty_finish",
            "accuracy": 100,
            "damage": 10,
            "damage_min": 10,
            "damage_max": 10,
        }
    )
    state = _mark_setup_state()
    state.enemies["acting_enemy"].skills = ["sure_dirty_finish"]
    state.heroes["front_left"].hp = 2

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_RecordingHeroPolicy(),
        enemy_wait_mode="package_only",
    )

    assert episode.metrics.waited_then_down == 1

    definitions.skills["miss_payoff"] = definitions.skills["dirty_finish"].model_copy(
        update={
            "id": "miss_payoff",
            "accuracy": -100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    no_payoff_state = _mark_setup_state()
    no_payoff_state.enemies["acting_enemy"].skills = ["miss_payoff"]
    no_payoff_episode = run_enemy_learning_episode(
        no_payoff_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_RecordingHeroPolicy(),
        enemy_wait_mode="unrestricted",
    )

    assert no_payoff_episode.metrics.waited_then_no_payoff == 1
    assert no_payoff_episode.metrics.waits_without_payoff == 1


def test_wait_quality_counts_no_attack_when_wait_has_no_next_attack() -> None:
    definitions = get_definitions()
    definitions.skills["costly_dirty_finish"] = definitions.skills["dirty_finish"].model_copy(
        update={
            "id": "costly_dirty_finish",
            "effort_cost": 3,
        }
    )
    state = _mark_setup_state()
    state.enemies["acting_enemy"].skills = ["costly_dirty_finish"]
    state.enemies["acting_enemy"].effort = 2

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_RecordingHeroPolicy(),
        enemy_wait_mode="unrestricted",
    )

    assert episode.metrics.enemy_waits == 1
    assert episode.metrics.waited_then_no_attack == 1
    assert episode.metrics.waits_without_payoff == 1


def test_recovery_quality_counts_followup_and_waste() -> None:
    definitions = get_definitions()
    definitions.skills["sure_rusted_chop"] = definitions.skills["rusted_chop"].model_copy(
        update={
            "id": "sure_rusted_chop",
            "accuracy": 100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    state = _two_enemy_state(
        acting_skills=["sure_rusted_chop"],
        acting_slot=FormationSlot.BACK_LEFT,
        blocker_slot=FormationSlot.FRONT_LEFT,
    )
    state.heroes["front_left"].tags.add(Tag.MARKED)

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=2,
        hero_policy=_RecordingHeroPolicy(),
    )

    assert episode.metrics.enemy_recovery_moves == 1
    assert episode.metrics.recovery_move_then_attack == 1
    assert episode.metrics.recovery_move_then_payoff == 1
    assert episode.metrics.recovery_move_wasted == 0

    wasted_state = _two_enemy_state(
        acting_skills=["rusted_chop"],
        acting_slot=FormationSlot.BACK_LEFT,
        blocker_slot=FormationSlot.FRONT_LEFT,
    )
    wasted_episode = run_enemy_learning_episode(
        wasted_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_RecordingHeroPolicy(),
    )

    assert wasted_episode.metrics.recovery_move_wasted == 1
    assert wasted_episode.metrics.recovery_moves_without_followup == 1


def test_learning_episode_metrics_count_damage_mark_exploit_rounds_and_final_state() -> None:
    definitions = get_definitions()
    definitions.skills["sure_cheap_shot"] = definitions.skills["cheap_shot"].model_copy(
        update={
            "id": "sure_cheap_shot",
            "accuracy": 100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    state = _combat_state(enemy_skills=["sure_cheap_shot"])
    state.heroes["front_left"].tags.add(Tag.MARKED)

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_FixedEnemyPolicy("sure_cheap_shot", "front_left"),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert episode.metrics.rounds_elapsed == 1
    assert episode.metrics.enemy_decisions == len(episode.records)
    assert episode.metrics.total_hero_damage > 0
    assert episode.metrics.lowest_hero_hp_reached < 10
    assert episode.metrics.final_hero_hp_total == sum(hero.hp for hero in state.heroes.values())
    assert episode.metrics.final_hero_effort_total == sum(
        hero.effort for hero in state.heroes.values()
    )
    assert episode.metrics.marks_exploited >= 1
    assert episode.metrics.skill_uses["sure_cheap_shot"] >= 1


def test_learning_episode_metrics_count_healing_and_guard_actions() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["mark_the_pattern"])
    state.heroes["front_left"].skills = ["stand_watch", "emergency_stitch"]
    state.heroes["front_left"].effort = 2
    state.heroes["front_right"].hp = 1

    heal_episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_FixedHeroPolicy("front_left", "emergency_stitch", "front_right"),
    )

    assert heal_episode.metrics.healing_actions >= 1
    assert heal_episode.metrics.healing_amount > 0

    guard_state = _combat_state(enemy_skills=["mark_the_pattern"])
    guard_state.heroes["front_left"].skills = ["stand_watch"]
    guard_state.heroes["front_left"].effort = 1
    guard_episode = run_enemy_learning_episode(
        guard_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        hero_policy=_FixedHeroPolicy("front_left", "stand_watch", "front_right"),
    )

    assert guard_episode.metrics.guard_actions >= 1


def test_learning_episode_guard_flow_tracks_dead_guard_use_and_target() -> None:
    definitions = get_definitions()
    state = _guard_flow_state()

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_PerEnemyPolicy(
            {
                "guarder_enemy": ("shielding_dead", "guarded_enemy"),
                "guarded_enemy": ("cheap_shot", "front_left"),
            }
        ),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert episode.metrics.guard_flow.guard_uses == 1
    assert episode.metrics.guard_flow.dead_guard_uses == 1
    assert episode.metrics.guard_flow.guard_targets["skulker"] == 1
    assert episode.metrics.guard_flow.guard_targets_by_enemy_id["guarded_enemy"] == 1
    assert episode.metrics.guard_flow.guarded_ally_survived_to_next_activation == 1
    assert episode.metrics.guard_flow.guarded_ally_acted_after_guard == 1


def test_learning_episode_guard_flow_tracks_blocked_damage() -> None:
    definitions = get_definitions()
    state = _guard_flow_state()
    state.heroes["front_left"].speed = 90

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_PerEnemyPolicy(
            {
                "guarder_enemy": ("shielding_dead", "guarded_enemy"),
                "guarded_enemy": ("cheap_shot", "front_left"),
            }
        ),
        hero_policy=_FixedHeroPolicy("front_left", "guard_strike", "guarded_enemy"),
    )

    assert episode.metrics.guard_flow.guard_damage_blocked == 3
    assert episode.metrics.guard_flow.guard_expired_or_consumed_without_payoff == 0


def test_learning_episode_guard_flow_tracks_guarded_payoff_and_down() -> None:
    definitions = get_definitions()
    definitions.skills["guarded_payoff"] = definitions.skills["cheap_shot"].model_copy(
        update={
            "id": "guarded_payoff",
            "accuracy": 100,
            "damage": 5,
            "damage_min": 5,
            "damage_max": 5,
            "tags": ["enemy", "vulnerable_bonus"],
        }
    )
    state = _guard_flow_state(guarded_skills=["guarded_payoff"])
    state.heroes["front_left"].hp = 1

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_PerEnemyPolicy(
            {
                "guarder_enemy": ("shielding_dead", "guarded_enemy"),
                "guarded_enemy": ("guarded_payoff", "front_left"),
            }
        ),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert episode.metrics.guard_flow.guarded_ally_used_payoff_after_guard == 1
    assert episode.metrics.guard_flow.guarded_ally_downs_after_guard == 1
    assert episode.metrics.guard_flow.guard_wasted_no_followup == 0


def test_learning_episode_guard_flow_tracks_wasted_guard() -> None:
    definitions = get_definitions()
    state = _guard_flow_state()

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_PerEnemyPolicy({"guarder_enemy": ("shielding_dead", "guarded_enemy")}),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert episode.metrics.guard_flow.guard_uses == 1
    assert episode.metrics.guard_flow.guard_wasted_no_followup == 1


def test_learning_episode_metrics_count_downed_death_mortal_wound_and_forced_move() -> None:
    definitions = get_definitions()
    definitions.skills["sure_hit"] = definitions.skills["black_pulse"].model_copy(
        update={"id": "sure_hit", "accuracy": 100, "damage": 5, "damage_min": 5, "damage_max": 5}
    )
    down_state = _combat_state(enemy_skills=["sure_hit"])
    down_state.heroes["front_left"].hp = 1
    down_episode = run_enemy_learning_episode(
        down_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_FixedEnemyPolicy("sure_hit", "front_left"),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert down_episode.metrics.hero_downs == 1

    death_state = _combat_state(enemy_skills=["sure_hit"])
    death_state.heroes["front_left"].hp = 0
    death_state.heroes["front_left"].statuses.add(ActorStatus.DOWNED)
    death_state.heroes["front_left"].mortal_wounds = 2
    death_episode = run_enemy_learning_episode(
        death_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_FixedEnemyPolicy("sure_hit", "front_left"),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert death_episode.metrics.mortal_wounds == 1
    assert death_episode.metrics.hero_deaths == 1

    definitions.skills["sure_drag_forward"] = definitions.skills["drag_forward"].model_copy(
        update={
            "id": "sure_drag_forward",
            "accuracy": 100,
            "damage": 0,
            "damage_min": 0,
            "damage_max": 0,
        }
    )
    move_state = _combat_state(enemy_skills=["sure_drag_forward"])
    move_episode = run_enemy_learning_episode(
        move_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_FixedEnemyPolicy("sure_drag_forward", "back_left"),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert move_episode.metrics.forced_movement == 1


def test_learning_episode_tracks_boss_grab_to_bite_sequence() -> None:
    definitions = get_definitions()
    definitions.skills["sure_drag_forward"] = definitions.skills["drag_forward"].model_copy(
        update={
            "id": "sure_drag_forward",
            "accuracy": 100,
            "damage": 1,
            "damage_min": 1,
            "damage_max": 1,
        }
    )
    definitions.skills["sure_maw_slam"] = definitions.skills["maw_slam"].model_copy(
        update={
            "id": "sure_maw_slam",
            "accuracy": 100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    state = _combat_state(
        enemy_skills=["sure_drag_forward", "sure_maw_slam"],
        enemy_class_id="cave_maw_brute",
    )

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=2,
        policy=_QueuedEnemyPolicy(
            [
                ("sure_drag_forward", "back_left"),
                ("sure_maw_slam", "back_left"),
            ]
        ),
        hero_policy=_RecordingHeroPolicy(),
    )

    boss = episode.metrics.boss_sequence
    assert boss.grab_uses == 1
    assert boss.bite_uses == 1
    assert boss.grab_damage > 0
    assert boss.bite_damage > 0
    assert boss.grab_to_bite_any_target == 1
    assert boss.grab_to_bite_same_target == 1
    assert boss.bite_hit_dragged_target == 1
    assert episode.records[0].chosen_features["maw_grab_setup"] > 0
    assert episode.records[0].action_reward >= 120
    assert episode.records[1].chosen_features["maw_bite_payoff"] > 0
    assert episode.records[1].action_reward >= 220


def test_learning_episode_tracks_boss_support_targeting_metrics() -> None:
    definitions = get_definitions()
    definitions.skills["sure_drag_forward"] = definitions.skills["drag_forward"].model_copy(
        update={
            "id": "sure_drag_forward",
            "accuracy": 100,
            "damage": 1,
            "damage_min": 1,
            "damage_max": 1,
        }
    )
    definitions.skills["sure_maw_slam"] = definitions.skills["maw_slam"].model_copy(
        update={
            "id": "sure_maw_slam",
            "accuracy": 100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    state = _combat_state(
        enemy_skills=["sure_drag_forward", "sure_maw_slam"],
        enemy_class_id="cave_maw_brute",
    )
    state.heroes["back_left"].class_id = "field_surgeon"
    state.heroes["back_left"].skills = ["emergency_stitch"]
    state.heroes["back_left"].effort = 1

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=2,
        policy=_QueuedEnemyPolicy(
            [
                ("sure_drag_forward", "back_left"),
                ("sure_maw_slam", "back_left"),
            ]
        ),
        hero_policy=_RecordingHeroPolicy(),
    )

    targeting = episode.metrics.boss_targeting
    assert targeting.grab_target_classes["field_surgeon"] == 1
    assert targeting.bite_target_classes["field_surgeon"] == 1
    assert targeting.support_grabs == 1
    assert targeting.support_grabs_with_effort == 1
    assert targeting.support_grab_to_bite_same_target == 1


def test_learning_episode_boss_targeting_metrics_remain_zero_without_boss_actions() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["sling_stone"], enemy_class_id="bandit_slinger")

    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_FixedEnemyPolicy("sling_stone", "front_left"),
        hero_policy=_RecordingHeroPolicy(),
    )

    targeting = episode.metrics.boss_targeting
    assert targeting.grab_target_classes == {}
    assert targeting.bite_target_classes == {}
    assert targeting.support_grabs == 0


def test_learning_episode_tracks_mark_refresh_and_payoff_flow() -> None:
    definitions = get_definitions()
    marked_state = _combat_state(enemy_skills=["mark_the_pattern"])
    marked_state.heroes["front_left"].tags.add(Tag.MARKED)
    marked_state.heroes["front_left"].tag_turns[Tag.MARKED] = 1

    refresh_episode = run_enemy_learning_episode(
        marked_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_FixedEnemyPolicy("mark_the_pattern", "front_left"),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert refresh_episode.metrics.mark_flow.marks_refreshed == 1
    assert refresh_episode.metrics.mark_flow.marks_applied_to_already_marked == 1

    definitions.skills["sure_cheap_shot"] = definitions.skills["cheap_shot"].model_copy(
        update={
            "id": "sure_cheap_shot",
            "accuracy": 100,
            "damage": 2,
            "damage_min": 2,
            "damage_max": 2,
        }
    )
    payoff_state = _combat_state(enemy_skills=["sure_cheap_shot"])
    payoff_state.heroes["front_left"].tags.add(Tag.MARKED)
    payoff_episode = run_enemy_learning_episode(
        payoff_state,
        definitions,
        GameRng(1),
        max_rounds=1,
        policy=_FixedEnemyPolicy("sure_cheap_shot", "front_left"),
        hero_policy=_RecordingHeroPolicy(),
    )

    assert payoff_episode.metrics.mark_flow.exploited_by_enemy_hit == 1
    assert payoff_episode.metrics.mark_flow.vulnerable_payoffs == 1
    assert payoff_episode.metrics.mark_flow.total_damage_to_marked > 0


def test_mark_package_reward_shaping_rewards_focus_and_penalizes_ignored_mark() -> None:
    collapse_mark = score_enemy_action_events(
        [
            StatusChangedEvent(
                message="",
                actor_id="hero",
                status="marked",
                added=True,
            )
        ],
        _trace_with_features(
            {
                "mark": 1,
                "bandit_mark_collapse": 1,
                "mark_followup_can_down": 1,
            }
        ),
    )
    marked_attack = score_enemy_action_events(
        [DamageEvent(message="", source_id="bandit", target_id="hero", amount=2, hp_before=10)],
        _trace_with_features(
            {
                "bandit_marked_attack": 1,
                "bandit_marked_payoff": 1,
                "bandit_ignored_marked_legal": 0,
            }
        ),
    )
    ignored = score_enemy_action_events(
        [DamageEvent(message="", source_id="bandit", target_id="other", amount=2, hp_before=10)],
        _trace_with_features({"bandit_ignored_marked_legal": 1}),
    )

    assert collapse_mark >= 1600
    assert marked_attack >= 210
    assert ignored < 0


def test_wait_payoff_increases_timing_reward() -> None:
    reward = score_enemy_timing_outcome(
        _PendingEnemyTiming(kind="wait"),
        _TimingOutcome(attacked=True, marked_hit=True, payoff=True, down=False),
    )

    assert reward >= PartyCollapseRewardWeights().waited_then_payoff


def test_wasted_wait_penalizes_timing_reward() -> None:
    reward = score_enemy_timing_outcome(
        _PendingEnemyTiming(kind="wait"),
        _TimingOutcome(attacked=False, marked_hit=False, payoff=False, down=False),
    )

    assert reward <= PartyCollapseRewardWeights().waited_then_no_attack


def test_move_payoff_and_waste_rewards() -> None:
    payoff_reward = score_enemy_timing_outcome(
        _PendingEnemyTiming(kind="move"),
        _TimingOutcome(attacked=True, marked_hit=True, payoff=True, down=False),
    )
    wasted_reward = score_enemy_timing_outcome(
        _PendingEnemyTiming(kind="move"),
        _TimingOutcome(attacked=True, marked_hit=False, payoff=False, down=False),
    )

    assert payoff_reward > 0
    assert wasted_reward < 0


def test_recovery_move_reward_and_penalty() -> None:
    rewarded = score_enemy_timing_outcome(
        _PendingEnemyTiming(kind="recovery"),
        _TimingOutcome(attacked=True, marked_hit=False, payoff=True, down=False),
    )
    wasted = score_enemy_timing_outcome(
        _PendingEnemyTiming(kind="recovery"),
        _TimingOutcome(attacked=False, marked_hit=False, payoff=False, down=False),
    )

    assert rewarded > 0
    assert wasted < 0


def test_wait_when_current_action_good_penalized() -> None:
    reward = score_enemy_timing_outcome(
        _PendingEnemyTiming(
            kind="wait",
            features={"wait_when_current_action_good": 1},
        ),
        _TimingOutcome(attacked=True, marked_hit=True, payoff=True, down=False),
    )
    baseline = score_enemy_timing_outcome(
        _PendingEnemyTiming(kind="wait"),
        _TimingOutcome(attacked=True, marked_hit=True, payoff=True, down=False),
    )

    assert reward < baseline


def test_tactical_policy_kills_payoff_enemy_when_marked_and_killable() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["payoff_enemy"].hp = 2
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "payoff_enemy")


def test_tactical_policy_prefers_filler_kill_over_nonlethal_payoff_when_marked() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_tactical_policy_prefers_killable_enemy_over_non_lethal_damage() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["guard_strike", "shield_drive"]
    state.heroes["front_left"].effort = 1
    state.enemies["enemy"].hp = 3
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "enemy")


def test_tactical_policy_kills_setup_enemy_when_killable_without_payoff_threat() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["marker_enemy"].hp = 2
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "marker_enemy")


def test_tactical_policy_setup_kill_beats_generic_kill_when_both_killable_and_marked() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["marker_enemy"].hp = 2
    state.enemies["filler_enemy"].hp = 2
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "marker_enemy")


def test_tactical_policy_prefers_filler_kill_over_nonlethal_setup_damage() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.heroes["front_left"].skills = ["guard_strike", "exposed_cut"]
    state.heroes["front_left"].effort = 1
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_tactical_policy_allows_urgent_nonlethal_payoff_only_without_kills() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["filler_enemy"].hp = 10
    state.heroes["front_right"].hp = 2
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "payoff_enemy")


def test_tactical_policy_blocks_nonlethal_payoff_when_killable_filler_exists() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.heroes["front_right"].hp = 2
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_tactical_policy_heals_downed_or_low_hp_allies_before_offense() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["bone_saw", "emergency_stitch"]
    state.heroes["front_left"].effort = 1
    state.heroes["front_right"].hp = 0
    state.heroes["front_right"].statuses.add(ActorStatus.DOWNED)
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == (
        "emergency_stitch",
        "front_right",
    )


def test_tactical_policy_avoids_effort_overkill_when_free_useful_attack_exists() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["guard_strike", "exposed_cut"]
    state.heroes["front_left"].effort = 1
    state.enemies["enemy"].hp = 10
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "enemy")


def test_tactical_policy_avoids_nonlethal_package_without_low_hp_when_generic_exists() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["filler_enemy"].hp = 10
    state.enemies["payoff_enemy"].hp = 10
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_tactical_policy_allows_zero_effort_nonlethal_package_when_creates_low_hp() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["filler_enemy"].hp = 10
    state.enemies["payoff_enemy"].hp = 3
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "payoff_enemy")


def test_tactical_policy_blocks_effort_nonlethal_package_chip_when_not_killable() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["filler_enemy"].hp = 10
    state.enemies["marker_enemy"].hp = 10
    state.heroes["front_left"].skills = ["guard_strike", "exposed_cut"]
    state.heroes["front_left"].effort = 1
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_tactical_policy_avoids_effort_on_nonlethal_payoff_when_free_kill_exists() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.heroes["front_left"].skills = ["guard_strike", "exposed_cut"]
    state.heroes["front_left"].effort = 1
    state.enemies["filler_enemy"].hp = 2
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_tactical_policy_spends_effort_for_worthwhile_kills() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"], place_enemy=False)
    state.heroes["front_left"].skills = ["exposed_cut"]
    state.heroes["front_left"].effort = 1
    back_enemy = _combatant(
        "enemy",
        Team.ENEMY,
        FormationSlot.BACK_RIGHT,
        skills=["skulker_stone"],
        effort=2,
    )
    back_enemy.hp = 3
    state.enemies["enemy"] = back_enemy
    state.enemy_formation.place("enemy", FormationSlot.BACK_RIGHT)
    policy = TacticalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("exposed_cut", "enemy")


def test_create_hero_policy_accepts_tactical() -> None:
    policy = create_hero_policy("tactical")
    assert policy.policy_id == "tactical"


def test_create_hero_policy_accepts_company_survival() -> None:
    policy = create_hero_policy("company_survival")
    assert policy.policy_id == "company_survival"
    assert isinstance(policy, CompanySurvivalHeroPolicy)


def test_supported_hero_policy_ids_includes_company_survival() -> None:
    assert "company_survival" in SUPPORTED_HERO_POLICY_IDS


def test_mixed_hero_policy_rotation_excludes_company_survival() -> None:
    selected = {
        MixedHeroPolicy(encounter_id="road_bandits", seed=seed).selected_policy_id
        for seed in range(1, 32)
    }
    assert "company_survival" not in selected


def test_company_survival_heals_urgent_ally_before_offense() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["bone_saw", "emergency_stitch"]
    state.heroes["front_left"].effort = 1
    state.heroes["front_right"].hp = 0
    state.heroes["front_right"].statuses.add(ActorStatus.DOWNED)
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == (
        "emergency_stitch",
        "front_right",
    )


def test_company_survival_prefers_generic_kill_over_nonlethal_package_chip() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["filler_enemy"].hp = 10
    state.enemies["payoff_enemy"].hp = 10
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_company_survival_kills_urgent_payoff_when_marked() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["payoff_enemy"].hp = 2
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "payoff_enemy")


def test_company_survival_prefers_high_threat_generic_kill_over_nonlethal_setup() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["filler_enemy"].hp = 2
    state.enemies["marker_enemy"].hp = 10
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_company_survival_spends_effort_for_decisive_kill() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"], place_enemy=False)
    state.heroes["front_left"].skills = ["exposed_cut"]
    state.heroes["front_left"].effort = 1
    back_enemy = _combatant(
        "enemy",
        Team.ENEMY,
        FormationSlot.BACK_RIGHT,
        skills=["skulker_stone"],
        effort=2,
    )
    back_enemy.hp = 3
    state.enemies["enemy"] = back_enemy
    state.enemy_formation.place("enemy", FormationSlot.BACK_RIGHT)
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("exposed_cut", "enemy")


def test_company_survival_blocks_effort_nonlethal_chip() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["filler_enemy"].hp = 10
    state.enemies["marker_enemy"].hp = 10
    state.heroes["front_left"].skills = ["guard_strike", "exposed_cut"]
    state.heroes["front_left"].effort = 1
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_company_survival_prefers_generic_kill_over_marker_when_marker_not_dominant() -> None:
    definitions = get_definitions()
    state = _anti_mark_state()
    state.enemies["a_enemy"].speed = 25
    state.enemies["z_marker"].speed = 5
    company = CompanySurvivalHeroPolicy()
    anti_mark = AntiMarkHeroPolicy()

    assert company.choose(state, definitions, "front_left") == ("guard_strike", "a_enemy")
    assert anti_mark.choose(state, definitions, "front_left") == ("guard_strike", "z_marker")


def test_company_survival_values_setup_kill_when_marker_enables_pressure() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["filler_enemy"].hp = 2
    state.enemies["filler_enemy"].speed = 5
    state.enemies["marker_enemy"].hp = 2
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "marker_enemy")


def test_company_survival_killable_generic_beats_nonlethal_package_chip() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["filler_enemy"].hp = 10
    state.enemies["payoff_enemy"].hp = 10
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_company_survival_killable_generic_beats_zero_effort_high_threat_chip() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["filler_enemy"].hp = 2
    state.enemies["marker_enemy"].hp = 10
    state.enemies["marker_enemy"].speed = 30
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "filler_enemy")


def test_company_survival_setup_beats_low_value_filler_when_marked() -> None:
    definitions = get_definitions()
    state = _tactical_setup_state()
    state.enemies["filler_enemy"].hp = 2
    state.enemies["filler_enemy"].speed = 5
    state.enemies["marker_enemy"].hp = 2
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "marker_enemy")


def test_company_survival_nonlethal_tier_never_beats_kill_tier_70() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["filler_enemy"].hp = 2
    state.enemies["payoff_enemy"].hp = 10
    entries = _company_survival_score_entries(state, definitions, "front_left")
    assert any(entry.tier >= 70 for entry in entries)
    for low in entries:
        for high in entries:
            if low.tier <= 55 and high.tier >= 70:
                assert low.score < high.score
    assert (70, 40, 0, 0, 0, 0, "a") > (55, 40, 0, 0, 0, 0, "b")
    assert (70, 40, 0, 0, 0, 0, "a") > (45, 40, 0, 0, 0, 0, "b")


def test_company_survival_skips_heal_for_noncritical_ally_when_kill_available() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["lookout_poke"])
    state.heroes["front_left"].skills = ["guard_strike", "emergency_stitch"]
    state.heroes["front_right"].hp = 8
    state.heroes["front_right"].max_hp = 10
    state.enemies["enemy"].hp = 2
    policy = CompanySurvivalHeroPolicy()

    assert policy.choose(state, definitions, "front_left") == ("guard_strike", "enemy")


def test_company_survival_score_trace_reports_tier() -> None:
    definitions = get_definitions()
    state = _tactical_payoff_state()
    state.enemies["payoff_enemy"].hp = 2
    choice, top_entries = _explain_company_survival_choice(state, definitions, "front_left")

    assert choice is not None
    assert top_entries
    assert top_entries[0].tier >= 70
    assert choice == (top_entries[0].skill_id, top_entries[0].target_id)


def test_company_survival_attrition_hunt_not_catastrophic_vs_mixed() -> None:
    definitions = get_definitions()
    config_base = {
        "definitions": definitions,
        "seeds": 5,
        "max_rounds": 10,
        "preset_id": "attrition",
        "pressure_profile_id": "marked_hunt",
    }
    mixed = run_generated_route_lab(
        GeneratedRouteLabConfig(hero_policy_id="mixed", **config_base)
    )
    company = run_generated_route_lab(
        GeneratedRouteLabConfig(hero_policy_id="company_survival", **config_base)
    )
    mixed_rate = sum(run.completed for run in mixed.runs) / len(mixed.runs)
    company_rate = sum(run.completed for run in company.runs) / len(company.runs)

    mixed_episodes = tuple(episode for run in mixed.runs for episode in run.episodes)
    company_episodes = tuple(episode for run in company.runs for episode in run.episodes)
    mixed_audit = aggregate_hero_policy_audit(mixed_episodes, policy_id="mixed")
    company_audit = aggregate_hero_policy_audit(
        company_episodes,
        policy_id="company_survival",
    )

    assert company_rate >= mixed_rate - 0.35
    assert company_audit.metrics.ignored_killable_rate <= (
        mixed_audit.metrics.ignored_killable_rate + 0.25
    )


def test_tactical_attrition_hunt_not_dramatically_worse_than_mixed() -> None:
    definitions = get_definitions()
    config_base = {
        "definitions": definitions,
        "seeds": 5,
        "max_rounds": 10,
        "preset_id": "attrition",
        "pressure_profile_id": "marked_hunt",
    }
    mixed = run_generated_route_lab(
        GeneratedRouteLabConfig(hero_policy_id="mixed", **config_base)
    )
    tactical = run_generated_route_lab(
        GeneratedRouteLabConfig(hero_policy_id="tactical", **config_base)
    )
    mixed_rate = sum(run.completed for run in mixed.runs) / len(mixed.runs)
    tactical_rate = sum(run.completed for run in tactical.runs) / len(tactical.runs)
    assert tactical_rate >= mixed_rate - 0.20


def test_mixed_hero_policy_rotation_excludes_tactical() -> None:
    selected = {
        MixedHeroPolicy(encounter_id="road_bandits", seed=seed).selected_policy_id
        for seed in range(1, 32)
    }
    assert "tactical" not in selected
    assert selected <= {"naive", "damage_race", "survival", "anti_mark", "conservative"}


def test_conservative_policy_preserves_effort_for_ordinary_damage() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_left"].skills = ["exposed_cut"]
    state.heroes["front_left"].effort = 1
    state.enemies["enemy"].hp = 10

    assert ConservativeHeroPolicy().choose(state, definitions, "front_left") is None


def test_conservative_policy_spends_effort_for_kill_and_urgent_heal_only() -> None:
    definitions = get_definitions()
    kill_state = _combat_state(enemy_skills=["skulker_stone"])
    kill_state.heroes["front_left"].skills = ["exposed_cut"]
    kill_state.heroes["front_left"].effort = 1
    kill_state.enemies["enemy"].hp = 3

    assert ConservativeHeroPolicy().choose(kill_state, definitions, "front_left") == (
        "exposed_cut",
        "enemy",
    )

    non_urgent_state = _combat_state(enemy_skills=["skulker_stone"])
    non_urgent_state.heroes["front_left"].skills = ["emergency_stitch"]
    non_urgent_state.heroes["front_left"].effort = 1
    non_urgent_state.heroes["front_right"].hp = 4
    assert ConservativeHeroPolicy().choose(non_urgent_state, definitions, "front_left") is None

    urgent_state = _combat_state(enemy_skills=["skulker_stone"])
    urgent_state.heroes["front_left"].skills = ["emergency_stitch"]
    urgent_state.heroes["front_left"].effort = 1
    urgent_state.heroes["front_right"].hp = 2
    assert ConservativeHeroPolicy().choose(urgent_state, definitions, "front_left") == (
        "emergency_stitch",
        "front_right",
    )


def test_action_rewards_order_party_collapse_events() -> None:
    damage = score_enemy_action_events(
        [DamageEvent(message="", source_id="enemy", target_id="hero", amount=2, hp_before=10)]
    )
    kill_range = score_enemy_action_events(
        [DamageEvent(message="", source_id="enemy", target_id="hero", amount=6, hp_before=10)]
    )
    downed = score_enemy_action_events([DownedEvent(message="", actor_id="hero")])
    death = score_enemy_action_events([DeathEvent(message="", actor_id="hero")])
    mortal_wound = score_enemy_action_events(
        [
            StatusChangedEvent(
                message="",
                actor_id="hero",
                status="mortal_wound",
                added=True,
            )
        ]
    )
    effort = score_enemy_action_events(
        [StatusChangedEvent(message="", actor_id="hero", status="effort", added=False)]
    )

    assert damage > 0
    assert kill_range > damage
    assert effort > damage
    assert downed > effort
    assert mortal_wound > downed
    assert death > mortal_wound


def test_miss_is_penalized() -> None:
    assert (
        score_enemy_action_events([MissEvent(message="", actor_id="enemy", target_id="hero")])
        < 0
    )


def test_mark_setup_and_payoff_rewards_use_chosen_features() -> None:
    definitions = get_definitions()
    mark_state = _mark_state()
    mark_trace = explain_enemy_decision(mark_state, definitions, "lookout")
    payoff_state = _combat_state(enemy_skills=["skulker_stone", "cheap_shot"])
    payoff_state.heroes["front_right"].tags.add(Tag.MARKED)
    payoff_trace = explain_enemy_decision(payoff_state, definitions, "enemy")

    assert mark_trace is not None
    assert payoff_trace is not None
    setup_reward = score_enemy_action_events(
        [
            StatusChangedEvent(
                message="",
                actor_id=mark_trace.chosen.target_id if mark_trace.chosen else "hero",
                status="marked",
                added=True,
            )
        ],
        mark_trace,
    )
    payoff_reward = score_enemy_action_events(
        [
            DamageEvent(
                message="",
                source_id="enemy",
                target_id=payoff_trace.chosen.target_id if payoff_trace.chosen else "hero",
                amount=3,
                hp_before=10,
            )
        ],
        payoff_trace,
    )

    assert setup_reward >= 90
    assert payoff_reward >= 160


def test_enemy_victory_and_withdrawal_add_large_episode_rewards() -> None:
    base_record = _record(action_reward=10)
    enemy_episode = score_enemy_episode(
        EnemyDecisionEpisode("combat", "Combat", 1, (base_record,), "enemies", 10)
    )
    withdrawal_episode = score_enemy_episode(
        EnemyDecisionEpisode("combat", "Combat", 1, (base_record,), "withdrawal", 10)
    )

    assert enemy_episode.total_reward > withdrawal_episode.total_reward
    assert withdrawal_episode.total_reward >= 900


def test_linear_policy_uses_weights_and_stable_tie_breaking() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone", "cheap_shot"])
    state.heroes["front_right"].tags.add(Tag.MARKED)
    policy = LinearEnemyDecisionPolicy({"vulnerable_payoff": 100})

    choice = policy.choose(state, definitions, "enemy")

    assert choice is not None
    assert (choice.skill_id, choice.target_id) == ("cheap_shot", "front_right")

    tie_state = _combat_state(enemy_skills=["glass_bite", "black_pulse"], place_enemy=False)
    tie_policy = LinearEnemyDecisionPolicy({})
    tie_choices = [tie_policy.choose(tie_state, definitions, "enemy") for _ in range(3)]
    assert [(choice.skill_id, choice.target_id) for choice in tie_choices if choice] == [
        ("glass_bite", "front_left")
    ] * 3


def test_linear_policy_prior_penalizes_ignored_marked_targets() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["sling_stone", "pinning_shot"],
        enemy_class_id="bandit_slinger",
    )
    state.heroes["front_right"].tags.add(Tag.MARKED)

    choice = LinearEnemyDecisionPolicy({}).choose(state, definitions, "enemy")

    assert choice is not None
    assert choice.target_id == "front_right"


def test_training_returns_non_empty_weights_for_rewarded_features() -> None:
    record = _record(
        action_reward=5,
        chosen_features={"marked_focus": 2, "vulnerable_payoff": 4, "guard": 0},
    )

    weights = learn_linear_enemy_weights(
        [EnemyDecisionEpisode("combat", "Combat", 1, (record,), "enemies", 5)]
    )

    assert weights == {"marked_focus": 10, "vulnerable_payoff": 20}


def test_training_learns_negative_weight_for_ignored_marked_targets() -> None:
    record = _record(
        action_reward=5,
        chosen_features={"damage_pressure": 2, "bandit_ignored_marked_legal": 16},
    )

    weights = learn_linear_enemy_weights(
        [
            EnemyDecisionEpisode(
                encounter_id="test",
                encounter_name="Test",
                seed=1,
                records=(record,),
                final_victor="heroes",
                total_reward=5,
            )
        ]
    )

    assert weights["damage_pressure"] == 10
    assert weights["bandit_ignored_marked_legal"] == -80


def test_normal_selector_unchanged_without_learned_policy() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone", "cheap_shot"])
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "cheap_shot",
        "front_right",
    )


def test_learning_episode_does_not_mutate_campaign_memory() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    dungeon_memory = dict(company.dungeon_memory)
    world_memory = dict(company.world_memory)
    breach_memory = dict(company.breach_memory)

    run_enemy_learning_episode(
        _combat_state(enemy_skills=["skulker_stone"]),
        definitions,
        GameRng(1),
        max_rounds=1,
    )

    assert company.dungeon_memory == dungeon_memory
    assert company.world_memory == world_memory
    assert company.breach_memory == breach_memory


def _record(
    *,
    action_reward: int = 0,
    chosen_features: dict[str, int] | None = None,
) -> EnemyDecisionRecord:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone"])
    trace = explain_enemy_decision(state, definitions, "enemy")
    assert trace is not None
    assert trace.chosen is not None
    return EnemyDecisionRecord(
        enemy_id="enemy",
        round_number=1,
        action_index=0,
        trace=trace,
        chosen_skill_id=trace.chosen.skill_id,
        chosen_target_id=trace.chosen.target_id,
        chosen_features=chosen_features or dict(trace.chosen.features),
        events=(),
        action_reward=action_reward,
    )


def _trace_with_features(features: dict[str, int]) -> EnemyDecisionTrace:
    candidate = EnemyDecisionCandidate(
        skill_id="test_skill",
        target_id="hero",
        score=0,
        skill_order=0,
        target_order=0,
        skill_tags=frozenset(),
        features=features,
    )
    return EnemyDecisionTrace(
        enemy_id="enemy",
        runtime_context=EnemyDecisionRuntimeContext(),
        candidates=(candidate,),
        chosen=candidate,
    )


def _mark_state() -> CombatState:
    state = _combat_state(
        enemy_skills=["lookout_poke", "spot_target"],
        enemy_class_id="bandit_lookout",
    )
    lookout = state.enemies.pop("enemy")
    lookout.actor_id = "lookout"
    lookout.formation_slot = FormationSlot.BACK_RIGHT
    state.enemies["lookout"] = lookout
    state.enemy_formation = Formation.empty()
    state.enemy_formation.place("lookout", FormationSlot.BACK_RIGHT)
    payoff = _combatant(
        "cutthroat",
        Team.ENEMY,
        FormationSlot.FRONT_LEFT,
        skills=["bandit_blade", "dirty_finish"],
        effort=2,
        class_id="bandit_cutthroat",
    )
    state.enemies["cutthroat"] = payoff
    state.enemy_formation.place(payoff.actor_id, payoff.formation_slot)
    return state


def _tactical_payoff_state() -> CombatState:
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_right"].tags.add(Tag.MARKED)
    filler = state.enemies.pop("enemy")
    filler.actor_id = "filler_enemy"
    filler.hp = 2
    payoff = _combatant(
        "payoff_enemy",
        Team.ENEMY,
        FormationSlot.FRONT_RIGHT,
        skills=["bandit_blade", "dirty_finish"],
        effort=2,
        class_id="bandit_cutthroat",
    )
    payoff.hp = 10
    state.enemies = {"filler_enemy": filler, "payoff_enemy": payoff}
    state.enemy_formation = Formation.empty()
    state.enemy_formation.place("filler_enemy", FormationSlot.FRONT_LEFT)
    state.enemy_formation.place("payoff_enemy", FormationSlot.FRONT_RIGHT)
    return state


def _tactical_setup_state() -> CombatState:
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_right"].tags.add(Tag.MARKED)
    filler = state.enemies.pop("enemy")
    filler.actor_id = "filler_enemy"
    filler.hp = 2
    marker = _combatant(
        "marker_enemy",
        Team.ENEMY,
        FormationSlot.BACK_RIGHT,
        skills=["lookout_poke", "spot_target"],
        effort=1,
        class_id="bandit_spotter",
    )
    marker.hp = 10
    state.enemies = {"filler_enemy": filler, "marker_enemy": marker}
    state.enemy_formation = Formation.empty()
    state.enemy_formation.place("filler_enemy", FormationSlot.FRONT_LEFT)
    state.enemy_formation.place("marker_enemy", FormationSlot.BACK_RIGHT)
    return state


def _anti_mark_state() -> CombatState:
    state = _combat_state(enemy_skills=["skulker_stone"])
    state.heroes["front_right"].tags.add(Tag.MARKED)
    normal = state.enemies.pop("enemy")
    normal.actor_id = "a_enemy"
    normal.hp = 2
    marker = _combatant(
        "z_marker",
        Team.ENEMY,
        FormationSlot.FRONT_RIGHT,
        skills=["lookout_poke", "spot_target"],
        effort=1,
    )
    marker.hp = 2
    state.enemies = {"a_enemy": normal, "z_marker": marker}
    state.enemy_formation = Formation.empty()
    state.enemy_formation.place("a_enemy", FormationSlot.FRONT_LEFT)
    state.enemy_formation.place("z_marker", FormationSlot.FRONT_RIGHT)
    return state


def _two_enemy_state(
    *,
    acting_skills: list[str],
    acting_slot: FormationSlot,
    blocker_slot: FormationSlot,
) -> CombatState:
    state = _combat_state(enemy_skills=acting_skills, place_enemy=False)
    acting = state.enemies.pop("enemy")
    acting.actor_id = "acting_enemy"
    acting.name = "Acting Enemy"
    acting.skills = acting_skills
    acting.speed = 20
    acting.formation_slot = acting_slot
    blocker = _combatant(
        "blocker_enemy",
        Team.ENEMY,
        blocker_slot,
        skills=["lookout_poke"],
        class_id="bone_soldier",
    )
    blocker.speed = 1
    state.enemies = {"acting_enemy": acting, "blocker_enemy": blocker}
    state.enemy_formation = Formation.empty()
    state.enemy_formation.place("acting_enemy", acting_slot)
    state.enemy_formation.place("blocker_enemy", blocker_slot)
    return state


def _mark_setup_state() -> CombatState:
    state = _combat_state(enemy_skills=["dirty_finish"], place_enemy=False)
    acting = state.enemies.pop("enemy")
    acting.actor_id = "acting_enemy"
    acting.name = "Acting Enemy"
    acting.skills = ["dirty_finish"]
    acting.speed = 20
    acting.formation_slot = FormationSlot.FRONT_LEFT
    marker = _combatant(
        "marker_enemy",
        Team.ENEMY,
        FormationSlot.FRONT_RIGHT,
        skills=["spot_target"],
        effort=1,
        class_id="bandit_spotter",
    )
    marker.speed = 1
    state.enemies = {"acting_enemy": acting, "marker_enemy": marker}
    state.enemy_formation = Formation.empty()
    state.enemy_formation.place("acting_enemy", FormationSlot.FRONT_LEFT)
    state.enemy_formation.place("marker_enemy", FormationSlot.FRONT_RIGHT)
    return state


def _events(episode: EnemyDecisionEpisode) -> tuple[object, ...]:
    return tuple(event for record in episode.records for event in record.events)


def _combat_state(
    *,
    enemy_skills: list[str],
    place_enemy: bool = True,
    enemy_class_id: str = "",
) -> CombatState:
    heroes = {
        "front_left": _combatant("front_left", Team.HERO, FormationSlot.FRONT_LEFT),
        "front_right": _combatant("front_right", Team.HERO, FormationSlot.FRONT_RIGHT),
        "back_left": _combatant("back_left", Team.HERO, FormationSlot.BACK_LEFT),
        "back_right": _combatant("back_right", Team.HERO, FormationSlot.BACK_RIGHT),
    }
    enemy = _combatant(
        "enemy",
        Team.ENEMY,
        FormationSlot.FRONT_LEFT,
        skills=enemy_skills,
        effort=2,
        class_id=enemy_class_id,
    )
    party_formation = Formation.empty()
    for hero in heroes.values():
        party_formation.place(hero.actor_id, hero.formation_slot)
    enemy_formation = Formation.empty()
    if place_enemy:
        enemy_formation.place(enemy.actor_id, enemy.formation_slot)
    return CombatState(
        heroes=heroes,
        enemies={enemy.actor_id: enemy},
        party_formation=party_formation,
        enemy_formation=enemy_formation,
    )


def _guard_flow_state(guarded_skills: list[str] | None = None) -> CombatState:
    state = _combat_state(enemy_skills=["shielding_dead"], enemy_class_id="bone_soldier")
    guarder = state.enemies.pop("enemy")
    state.enemy_formation.remove("enemy")
    guarder.actor_id = "guarder_enemy"
    guarder.name = "Guarder Enemy"
    guarder.speed = 100
    guarder.formation_slot = FormationSlot.FRONT_LEFT
    guarded = _combatant(
        "guarded_enemy",
        Team.ENEMY,
        FormationSlot.FRONT_RIGHT,
        skills=guarded_skills or ["cheap_shot"],
        effort=2,
        class_id="skulker",
    )
    guarded.speed = 80
    state.enemies = {
        guarder.actor_id: guarder,
        guarded.actor_id: guarded,
    }
    state.enemy_formation.place(guarder.actor_id, FormationSlot.FRONT_LEFT)
    state.enemy_formation.place(guarded.actor_id, FormationSlot.FRONT_RIGHT)
    for hero in state.heroes.values():
        hero.speed = 0
    return state


def _combatant(
    actor_id: str,
    team: Team,
    slot: FormationSlot,
    *,
    skills: list[str] | None = None,
    effort: int = 0,
    class_id: str = "",
) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id.replace("_", " ").title(),
        team=team,
        max_hp=10,
        hp=10,
        speed=5,
        accuracy=0,
        defense=0,
        damage=0,
        max_effort=2,
        effort=effort,
        skills=skills or ["guard_strike"],
        formation_slot=slot,
        class_id=class_id,
    )


class _RecordingHeroPolicy:
    policy_id = "recording"

    def __init__(self) -> None:
        self.hero_ids: list[str] = []

    def choose(
        self,
        state: CombatState,
        definitions: object,
        hero_id: str,
    ) -> tuple[str, str] | None:
        self.hero_ids.append(hero_id)
        return None


class _FixedHeroPolicy:
    policy_id = "fixed"

    def __init__(self, hero_id: str, skill_id: str, target_id: str) -> None:
        self.hero_id = hero_id
        self.skill_id = skill_id
        self.target_id = target_id

    def choose(
        self,
        state: CombatState,
        definitions: object,
        hero_id: str,
    ) -> tuple[str, str] | None:
        if hero_id != self.hero_id:
            return None
        return self.skill_id, self.target_id


class _FixedEnemyPolicy:
    def __init__(self, skill_id: str, target_id: str) -> None:
        self.skill_id = skill_id
        self.target_id = target_id

    def choose(
        self,
        state: CombatState,
        definitions: object,
        enemy_id: str,
        runtime_context: object,
    ) -> EnemyDecisionCandidate | None:
        enemy = state.actor(enemy_id)
        if self.skill_id not in enemy.skills:
            return None
        skill = definitions.skills[self.skill_id]
        return EnemyDecisionCandidate(
            skill_id=self.skill_id,
            target_id=self.target_id,
            score=0,
            skill_order=enemy.skills.index(self.skill_id),
            target_order=0,
            skill_tags=frozenset(skill.tags),
            features={},
        )


class _QueuedEnemyPolicy:
    def __init__(self, actions: list[tuple[str, str]]) -> None:
        self.actions = actions
        self.index = 0

    def choose(
        self,
        state: CombatState,
        definitions: object,
        enemy_id: str,
        runtime_context: object,
    ) -> EnemyDecisionCandidate | None:
        if self.index >= len(self.actions):
            return None
        skill_id, target_id = self.actions[self.index]
        self.index += 1
        enemy = state.actor(enemy_id)
        skill = definitions.skills[skill_id]
        trace = explain_enemy_decision(state, definitions, enemy_id)
        if trace is not None:
            for candidate in trace.candidates:
                if candidate.skill_id == skill_id and candidate.target_id == target_id:
                    return candidate
        return EnemyDecisionCandidate(
            skill_id=skill_id,
            target_id=target_id,
            score=0,
            skill_order=enemy.skills.index(skill_id),
            target_order=0,
            skill_tags=frozenset(skill.tags),
            features={},
        )


class _PerEnemyPolicy:
    def __init__(self, actions: dict[str, tuple[str, str]]) -> None:
        self.actions = actions

    def choose(
        self,
        state: CombatState,
        definitions: object,
        enemy_id: str,
        runtime_context: object,
    ) -> EnemyDecisionCandidate | None:
        action = self.actions.get(enemy_id)
        if action is None:
            return None
        skill_id, target_id = action
        enemy = state.actor(enemy_id)
        skill = definitions.skills[skill_id]
        return EnemyDecisionCandidate(
            skill_id=skill_id,
            target_id=target_id,
            score=0,
            skill_order=enemy.skills.index(skill_id),
            target_order=0,
            skill_tags=frozenset(skill.tags),
            features={},
        )
