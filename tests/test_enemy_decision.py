from __future__ import annotations

import pytest

from game.combat.actions import use_skill
from game.combat.combat_state import Combatant, CombatState, Tag, Team
from game.combat.enemy_decision import (
    EnemyDecisionRuntimeContext,
    HeuristicEnemyDecisionPolicy,
    StaticLearnedEnemyDecisionPolicy,
    choose_enemy_skill_and_target,
    explain_enemy_decision,
    production_enemy_decision_policy,
    production_enemy_movement_mode,
    production_enemy_wait_mode,
)
from game.combat.formation import Formation, FormationSlot
from game.combat.targeting import legal_targets
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from tests.conftest import get_definitions

_ORIGINAL_GLASS_BITE = load_game_definitions().skills["glass_bite"]


@pytest.fixture(autouse=True)
def reset_glass_bite_skill() -> None:
    get_definitions().skills["glass_bite"] = _ORIGINAL_GLASS_BITE
    yield
    get_definitions().skills["glass_bite"] = _ORIGINAL_GLASS_BITE


def test_exploit_vulnerable_prefers_marked_targets() -> None:
    definitions = get_definitions()
    definitions.skills["glass_bite"] = definitions.skills["glass_bite"].model_copy(
        update={"tags": ["enemy", "exploit_vulnerable"]}
    )
    state = _combat_state(enemy_skills=["glass_bite"])
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "glass_bite",
        "front_right",
    )


def test_exploit_vulnerable_prefers_wounded_targets() -> None:
    definitions = get_definitions()
    definitions.skills["glass_bite"] = definitions.skills["glass_bite"].model_copy(
        update={"tags": ["enemy", "exploit_vulnerable"]}
    )
    state = _combat_state(enemy_skills=["glass_bite"])
    state.heroes["back_right"].hp = 7

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "glass_bite",
        "back_right",
    )


def test_drain_effort_prefers_targets_with_effort() -> None:
    definitions = get_definitions()
    definitions.skills["glass_bite"] = definitions.skills["glass_bite"].model_copy(
        update={"tags": ["enemy", "drain_effort"]}
    )
    state = _combat_state(enemy_skills=["glass_bite"])
    state.heroes["front_left"].effort = 0
    state.heroes["front_right"].effort = 2

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "glass_bite",
        "front_right",
    )


def test_mark_target_avoids_already_marked_targets() -> None:
    definitions = get_definitions()
    definitions.skills["glass_bite"] = definitions.skills["glass_bite"].model_copy(
        update={"tags": ["enemy", "mark_target"]}
    )
    state = _combat_state(enemy_skills=["glass_bite"])
    state.heroes["front_left"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "glass_bite",
        "back_left",
    )


def test_pull_forward_prefers_backline_targets() -> None:
    definitions = get_definitions()
    definitions.skills["glass_bite"] = definitions.skills["glass_bite"].model_copy(
        update={"tags": ["enemy", "pull_forward"]}
    )
    state = _combat_state(enemy_skills=["glass_bite"])

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "glass_bite",
        "back_left",
    )


def test_fallback_returns_a_legal_action() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["rusted_chop"])

    choice = choose_enemy_skill_and_target(state, definitions, "enemy")

    assert choice is not None
    skill_id, target_id = choice
    assert skill_id == "rusted_chop"
    assert target_id in legal_targets(state, "enemy", definitions.skills[skill_id].attack_type)


def test_back_row_enemy_cannot_choose_front_only_skill() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["rusted_chop"])
    enemy = state.enemies["enemy"]
    state.enemy_formation.remove(enemy.actor_id)
    state.enemy_formation.place(enemy.actor_id, FormationSlot.BACK_LEFT)
    enemy.formation_slot = FormationSlot.BACK_LEFT

    assert choose_enemy_skill_and_target(state, definitions, "enemy") is None


def test_real_vulnerable_bonus_skill_prefers_marked_targets() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone", "cheap_shot"])
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "cheap_shot",
        "front_right",
    )


def test_real_effort_drain_skill_prefers_targets_with_effort() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["glass_bite", "effort_drain"])
    state.heroes["front_left"].effort = 0
    state.heroes["front_right"].effort = 2

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "effort_drain",
        "front_right",
    )


def test_real_mark_skill_prefers_target_reachable_by_enemy_allies() -> None:
    definitions = get_definitions()
    state = _mark_setup_state()
    state.heroes["front_left"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "lookout") == (
        "spot_target",
        "front_right",
    )


def test_feature_scoring_matches_public_selector() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    state.heroes["front_right"].tags.add(Tag.MARKED)
    runtime_context = EnemyDecisionRuntimeContext(
        initiative_actor_ids=("lookout", "slinger", "front_right"),
        current_turn_index=0,
    )

    trace = explain_enemy_decision(state, definitions, "slinger", runtime_context)

    assert trace is not None
    assert trace.chosen is not None
    assert (trace.chosen.skill_id, trace.chosen.target_id) == choose_enemy_skill_and_target(
        state,
        definitions,
        "slinger",
        runtime_context,
    )


def test_mark_skill_prefers_target_with_more_ally_followup() -> None:
    definitions = get_definitions()
    state = _mark_setup_state()

    assert choose_enemy_skill_and_target(state, definitions, "lookout") == (
        "spot_target",
        "front_right",
    )


def test_mark_feature_values_payoff_followup_above_generic_followup() -> None:
    definitions = get_definitions()
    generic_state = _mark_setup_state()
    payoff_state = _mark_setup_state()
    generic_state.enemies["cutthroat"].skills = ["bandit_blade"]
    payoff_state.enemies["cutthroat"].skills = ["dirty_finish"]
    target_id = "front_right"

    generic_trace = explain_enemy_decision(generic_state, definitions, "lookout")
    payoff_trace = explain_enemy_decision(payoff_state, definitions, "lookout")

    assert generic_trace is not None
    assert payoff_trace is not None
    generic_mark = _candidate_feature(
        generic_trace,
        "spot_target",
        target_id,
        "bandit_mark_collapse",
    )
    payoff_mark = _candidate_feature(
        payoff_trace,
        "spot_target",
        target_id,
        "bandit_mark_collapse",
    )
    assert payoff_mark > generic_mark


def test_bandit_mark_learning_feature_values_real_kill_lanes() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    trace = explain_enemy_decision(state, definitions, "lookout")

    assert trace is not None
    assert _candidate_feature(trace, "spot_target", "front_right", "bandit_mark_kill_lane") > 0
    assert _candidate_feature(trace, "spot_target", "front_right", "bandit_mark_collapse") > 0


def test_bandit_spotter_prefers_downable_kill_lane_over_raw_reach() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    state.heroes["back_left"].hp = 2
    state.heroes["back_left"].class_id = "scribe"

    choice = choose_enemy_skill_and_target(
        state,
        definitions,
        "lookout",
        EnemyDecisionRuntimeContext(
            initiative_actor_ids=("lookout", "slinger", "back_left", "cutthroat"),
            current_turn_index=0,
        ),
    )

    assert choice == ("spot_target", "back_left")


def test_bandit_spotter_keeps_reach_central_when_low_hp_target_has_no_followup() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    state.heroes["back_left"].hp = 2
    state.enemies["slinger"].skills = []

    assert choose_enemy_skill_and_target(state, definitions, "lookout") == (
        "spot_target",
        "front_right",
    )


def test_bandit_spotter_avoids_guarded_protected_target_without_down_threat() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    state.heroes["back_left"].hp = 4
    state.heroes["back_left"].class_id = "scribe"
    state.heroes["back_left"].tags.add(Tag.GUARDED)

    trace = explain_enemy_decision(state, definitions, "lookout")

    assert trace is not None
    assert choose_enemy_skill_and_target(state, definitions, "lookout") == (
        "spot_target",
        "front_right",
    )
    assert _candidate_feature(trace, "spot_target", "back_left", "mark_target_guarded") == 1
    assert _candidate_feature(trace, "spot_target", "back_left", "mark_target_protected") == 1


def test_bandit_mark_collapse_features_track_down_threat_and_followup_damage() -> None:
    definitions = get_definitions()
    steady_state = _coordinated_mark_state()
    downable_state = _coordinated_mark_state()
    downable_state.heroes["back_left"].hp = 2

    steady_trace = explain_enemy_decision(steady_state, definitions, "lookout")
    downable_trace = explain_enemy_decision(downable_state, definitions, "lookout")

    assert steady_trace is not None
    assert downable_trace is not None
    steady_damage = _candidate_feature(
        steady_trace,
        "spot_target",
        "back_left",
        "mark_expected_followup_damage",
    )
    downable_damage = _candidate_feature(
        downable_trace,
        "spot_target",
        "back_left",
        "mark_expected_followup_damage",
    )
    assert downable_damage == steady_damage
    assert (
        _candidate_feature(steady_trace, "spot_target", "back_left", "mark_followup_can_down")
        == 0
    )
    assert (
        _candidate_feature(downable_trace, "spot_target", "back_left", "mark_followup_can_down")
        == 1
    )


def test_wolf_mark_does_not_use_bandit_mark_learning_feature() -> None:
    definitions = get_definitions()
    state = _wolf_pack_state()
    trace = explain_enemy_decision(state, definitions, "alpha")

    assert trace is not None
    assert _candidate_feature(trace, "howl", "front_left", "bandit_mark_kill_lane") == 0


def test_initiative_context_prefers_mark_target_allies_can_exploit_before_target_acts() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()

    assert choose_enemy_skill_and_target(
        state,
        definitions,
        "lookout",
        EnemyDecisionRuntimeContext(
            initiative_actor_ids=("lookout", "front_left", "slinger", "front_right"),
            current_turn_index=0,
        ),
    ) == ("spot_target", "front_right")


def test_mark_refresh_requires_low_duration_and_high_followup() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    state.heroes["front_left"].tags.add(Tag.MARKED)
    state.heroes["front_left"].tag_turns[Tag.MARKED] = 2
    state.heroes["front_right"].tags.add(Tag.MARKED)
    state.heroes["front_right"].tag_turns[Tag.MARKED] = 1
    state.heroes["front_right"].hp = 4

    assert choose_enemy_skill_and_target(
        state,
        definitions,
        "lookout",
        EnemyDecisionRuntimeContext(
            initiative_actor_ids=("lookout", "slinger", "front_right"),
            current_turn_index=0,
        ),
    ) == ("spot_target", "front_right")


def test_explain_enemy_decision_includes_candidates_features_scores_and_choice() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone", "cheap_shot"])
    state.heroes["front_right"].tags.add(Tag.MARKED)

    trace = explain_enemy_decision(state, definitions, "enemy")

    assert trace is not None
    assert trace.enemy_id == "enemy"
    assert trace.chosen is not None
    assert trace.candidates
    assert trace.chosen in trace.candidates
    assert all(candidate.features for candidate in trace.candidates)
    assert all(
        candidate.score == sum(candidate.features.values())
        for candidate in trace.candidates
    )
    assert {candidate.skill_id for candidate in trace.candidates} == {
        "skulker_stone",
        "cheap_shot",
    }


def test_real_drag_forward_skill_prefers_backline_targets() -> None:
    definitions = get_definitions()
    state = _maw_drag_state()

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "drag_forward",
        "back_left",
    )


def test_maw_package_learning_feature_does_not_change_heuristic_score() -> None:
    definitions = get_definitions()
    state = _maw_drag_state()
    trace = explain_enemy_decision(state, definitions, "enemy")

    assert trace is not None
    candidate = _candidate(trace, "drag_forward", "back_left")
    assert candidate.features["maw_grab_setup"] > 0
    assert candidate.score < sum(candidate.features.values())


def test_production_policy_uses_static_learned_features() -> None:
    definitions = get_definitions()
    state = _maw_drag_state()
    policy = production_enemy_decision_policy()

    choice = choose_enemy_skill_and_target(state, definitions, "enemy", policy=policy)
    trace = explain_enemy_decision(state, definitions, "enemy")

    assert choice == ("drag_forward", "back_left")
    assert trace is not None
    candidate = _candidate(trace, "drag_forward", "back_left")
    assert candidate.features["maw_grab_setup"] > 0
    assert policy.choose(state, definitions, "enemy") == candidate


def test_production_policy_mode_selects_static_learned_or_heuristic() -> None:
    assert isinstance(
        production_enemy_decision_policy("learned_static"),
        StaticLearnedEnemyDecisionPolicy,
    )
    assert isinstance(
        production_enemy_decision_policy("heuristic"),
        HeuristicEnemyDecisionPolicy,
    )


def test_production_static_weights_include_promoted_package_emphasis() -> None:
    policy = production_enemy_decision_policy("learned_static")

    assert isinstance(policy, StaticLearnedEnemyDecisionPolicy)
    assert policy.weights["maw_bite_payoff"] == 36
    assert policy.weights["maw_grab_setup"] == 15
    assert policy.weights["maw_grab_high_value_support"] == 15
    assert policy.weights["boss_guard_package"] == 16
    assert policy.weights["bandit_mark_collapse"] == 36
    assert policy.weights["bandit_mark_kill_lane"] == 21
    assert policy.weights["bandit_marked_payoff"] == 15
    assert policy.weights["mark_expected_followup_damage"] == 15


def test_production_policy_mode_rejects_unknown_mode() -> None:
    try:
        production_enemy_decision_policy("missing")
    except ValueError as exc:
        assert str(exc) == "Unknown enemy AI mode: missing"
    else:
        raise AssertionError("expected unknown enemy AI mode to fail")


def test_production_ai_mode_bundles_wait_and_move_timing() -> None:
    assert production_enemy_wait_mode("learned_static") == "package_only"
    assert production_enemy_movement_mode("learned_static") == "package_only"
    assert production_enemy_wait_mode("heuristic") == "none"
    assert production_enemy_movement_mode("heuristic") == "recovery_only"


def test_maw_drag_prefers_backline_support_with_effort_over_damage_target() -> None:
    definitions = get_definitions()
    state = _maw_drag_state()
    state.heroes["back_left"].class_id = "cutpurse"
    state.heroes["back_left"].skills = ["knife_work"]
    state.heroes["back_left"].effort = 4
    state.heroes["back_right"].class_id = "field_surgeon"
    state.heroes["back_right"].skills = ["emergency_stitch"]
    state.heroes["back_right"].effort = 1

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "drag_forward",
        "back_right",
    )

    trace = explain_enemy_decision(state, definitions, "enemy")
    assert trace is not None
    assert _candidate_feature(trace, "drag_forward", "back_right", "target_has_heal_skill") == 1
    assert (
        _candidate_feature(trace, "drag_forward", "back_right", "target_has_effort_for_support")
        == 1
    )
    assert (
        _candidate_feature(trace, "drag_forward", "back_right", "maw_grab_high_value_support")
        > 0
    )
    assert _candidate_feature(trace, "drag_forward", "back_left", "target_has_heal_skill") == 0


def test_maw_drag_prefers_downable_backliner_over_support_when_collapse_is_clear() -> None:
    definitions = get_definitions()
    state = _maw_drag_state()
    state.heroes["back_left"].class_id = "cutpurse"
    state.heroes["back_left"].skills = ["knife_work"]
    state.heroes["back_left"].hp = 3
    state.heroes["back_right"].class_id = "field_surgeon"
    state.heroes["back_right"].skills = ["emergency_stitch"]
    state.heroes["back_right"].effort = 1
    state.heroes["back_right"].hp = 10

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "drag_forward",
        "back_left",
    )

    trace = explain_enemy_decision(state, definitions, "enemy")
    assert trace is not None
    assert _candidate_feature(trace, "drag_forward", "back_left", "maw_grab_down_threat") > 0


def test_maw_support_value_decreases_without_effort_or_pending_action() -> None:
    definitions = get_definitions()
    state = _maw_drag_state()
    state.heroes["back_left"].class_id = "field_surgeon"
    state.heroes["back_left"].skills = ["emergency_stitch"]
    state.heroes["back_left"].effort = 1
    state.heroes["back_right"].class_id = "field_surgeon"
    state.heroes["back_right"].skills = ["emergency_stitch"]
    state.heroes["back_right"].effort = 0

    trace = explain_enemy_decision(
        state,
        definitions,
        "enemy",
        EnemyDecisionRuntimeContext(
            initiative_actor_ids=("enemy", "back_left", "back_right"),
            current_turn_index=0,
        ),
    )

    assert trace is not None
    assert (
        _candidate_feature(trace, "drag_forward", "back_left", "target_has_effort_for_support")
        == 1
    )
    assert (
        _candidate_feature(trace, "drag_forward", "back_right", "target_has_effort_for_support")
        == 0
    )
    assert _candidate_feature(trace, "drag_forward", "back_left", "grab_action_tax_value") > 0
    assert _candidate_feature(trace, "drag_forward", "back_right", "grab_action_tax_value") > 0


def test_maw_support_features_do_not_apply_to_unrelated_enemies() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["sling_stone"], enemy_class_id="bandit_slinger")
    state.heroes["back_left"].class_id = "field_surgeon"
    state.heroes["back_left"].skills = ["emergency_stitch"]
    state.heroes["back_left"].effort = 1

    trace = explain_enemy_decision(state, definitions, "enemy")

    assert trace is not None
    assert _candidate_feature(trace, "sling_stone", "back_left", "target_has_heal_skill") == 1
    assert _candidate_feature(trace, "sling_stone", "back_left", "maw_grab_high_value_support") == 0


def test_maw_prefers_slam_after_drag_exposes_backliner() -> None:
    definitions = get_definitions()
    state = _maw_after_drag_state()

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "maw_slam",
        "front_left",
    )


def test_maw_drag_does_not_undo_previous_drag() -> None:
    definitions = get_definitions()
    state = _maw_after_drag_state()

    choice = choose_enemy_skill_and_target(state, definitions, "enemy")

    assert choice != ("drag_forward", "back_left")
    assert choice == ("maw_slam", "front_left")


def test_maw_can_still_slam_without_effort() -> None:
    definitions = get_definitions()
    state = _maw_drag_state()
    state.enemies["enemy"].effort = 0

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "maw_slam",
        "front_left",
    )


def test_acolyte_can_still_black_pulse_without_effort() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["black_pulse", "mark_the_pattern"],
        enemy_class_id="maze_acolyte",
        enemy_effort=0,
    )

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "black_pulse",
        "front_left",
    )


def test_slinger_pin_skill_pays_off_marked_targets() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["sling_stone", "pinning_shot"],
        enemy_class_id="bandit_slinger",
    )
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "pinning_shot",
        "front_right",
    )


def test_bandit_marked_attack_learning_features_are_present() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    state.heroes["front_right"].tags.add(Tag.MARKED)
    trace = explain_enemy_decision(state, definitions, "slinger")

    assert trace is not None
    assert _candidate_feature(trace, "pinning_shot", "front_right", "bandit_marked_attack") > 0
    assert _candidate_feature(trace, "pinning_shot", "front_right", "bandit_marked_payoff") > 0


def test_bandit_scored_decision_flags_ignored_legal_marked_targets() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()
    state.heroes["front_right"].tags.add(Tag.MARKED)

    trace = explain_enemy_decision(state, definitions, "slinger")

    assert trace is not None
    assert trace.chosen is not None
    assert trace.chosen.target_id == "front_right"
    assert _candidate_feature(trace, "sling_stone", "front_left", "bandit_ignored_marked_legal") > 0


def test_slinger_uses_stone_without_payoff_target() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["sling_stone", "pinning_shot"],
        enemy_class_id="bandit_slinger",
    )

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "sling_stone",
        "front_left",
    )


def test_regular_damage_skill_focuses_marked_targets() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["skulker_stone"],
        enemy_class_id="skulker",
    )
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "skulker_stone",
        "front_right",
    )


def test_alpha_wolf_opens_with_howl_when_pack_can_pay_off_mark() -> None:
    definitions = get_definitions()
    state = _wolf_pack_state()

    choice = choose_enemy_skill_and_target(state, definitions, "alpha")

    assert choice is not None
    skill_id, target_id = choice
    assert skill_id == "howl"
    assert target_id == "front_right"


def test_wolf_pack_bite_pays_off_marked_targets() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["wolf_bite", "pack_bite"],
        enemy_class_id="wolf",
    )
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "pack_bite",
        "front_right",
    )


def test_real_guard_skill_can_protect_enemy_allies() -> None:
    definitions = get_definitions()
    state = _guard_state()

    assert choose_enemy_skill_and_target(state, definitions, "bone") == (
        "shielding_dead",
        "acolyte",
    )


def test_enemy_specials_are_skipped_without_effort() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["skulker_stone", "cheap_shot"],
        enemy_effort=0,
    )
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "skulker_stone",
        "front_right",
    )


def test_regular_enemy_uses_limited_special_effort_then_falls_back() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["skulker_stone", "cheap_shot"])
    state.heroes["front_right"].tags.add(Tag.MARKED)

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "cheap_shot",
        "front_right",
    )

    state.enemies["enemy"].effort = 1
    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "cheap_shot",
        "front_right",
    )

    state.enemies["enemy"].effort = 0
    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "skulker_stone",
        "front_right",
    )


def test_marked_target_stays_focused_after_basic_hit() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["skulker_stone"],
        enemy_class_id="skulker",
    )
    state.heroes["front_right"].tags.add(Tag.MARKED)
    basic_stone = definitions.skills["skulker_stone"].model_copy(
        update={"tags": ["enemy", "basic"]}
    )

    result = use_skill(
        state,
        "enemy",
        basic_stone,
        "front_right",
        GameRng(1),
    )

    assert result.success
    assert Tag.MARKED in state.heroes["front_right"].tags
    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "skulker_stone",
        "front_right",
    )


def test_marked_survives_ally_hit_for_later_payoff_skill() -> None:
    definitions = get_definitions()
    state = _coordinated_mark_state()

    mark_result = use_skill(
        state,
        "lookout",
        definitions.skills["spot_target"],
        "front_right",
        GameRng(1),
    )
    basic_blade = definitions.skills["bandit_blade"].model_copy(
        update={"tags": ["enemy", "bandit", "basic"]}
    )
    hit_result = use_skill(
        state,
        "cutthroat",
        basic_blade,
        "front_right",
        GameRng(1),
    )

    assert mark_result.success
    assert hit_result.success
    assert Tag.MARKED in state.heroes["front_right"].tags
    assert choose_enemy_skill_and_target(state, definitions, "slinger") == (
        "pinning_shot",
        "front_right",
    )


def test_bone_soldier_attacks_instead_of_guarding_when_alone() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["rusted_chop", "shielding_dead"],
        enemy_class_id="bone_soldier",
    )

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "rusted_chop",
        "front_left",
    )


def test_bone_soldier_avoids_guarding_already_guarded_allies() -> None:
    definitions = get_definitions()
    state = _guard_state()
    state.enemies["acolyte"].tags.add(Tag.GUARDED)

    assert choose_enemy_skill_and_target(state, definitions, "bone") == (
        "rusted_chop",
        "front_left",
    )


def test_acolyte_prefers_damage_when_alone_late() -> None:
    definitions = get_definitions()
    state = _combat_state(
        enemy_skills=["black_pulse", "mark_the_pattern"],
        enemy_class_id="maze_acolyte",
        round_number=3,
    )

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "black_pulse",
        "front_left",
    )


def test_leech_avoids_effort_drain_against_empty_targets() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["glass_bite", "effort_drain"])
    for hero in state.heroes.values():
        hero.effort = 0

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "glass_bite",
        "front_left",
    )


def test_damage_that_can_down_a_hero_gets_priority() -> None:
    definitions = get_definitions()
    state = _combat_state(enemy_skills=["black_pulse", "mark_the_pattern"])
    state.heroes["front_left"].hp = 3

    assert choose_enemy_skill_and_target(state, definitions, "enemy") == (
        "black_pulse",
        "front_left",
    )


def test_deterministic_tie_breaking_is_stable() -> None:
    definitions = get_definitions()
    definitions.skills["glass_bite"] = definitions.skills["glass_bite"].model_copy(
        update={"tags": ["enemy"]}
    )
    definitions.skills["black_pulse"] = definitions.skills["black_pulse"].model_copy(
        update={"tags": ["enemy"]}
    )
    state = _combat_state(
        enemy_skills=["black_pulse", "glass_bite"],
        place_enemy=False,
    )

    choices = [
        choose_enemy_skill_and_target(state, definitions, "enemy")
        for _ in range(3)
    ]

    assert choices == [("black_pulse", "front_left")] * 3


def _combat_state(
    *,
    enemy_skills: list[str],
    place_enemy: bool = True,
    enemy_effort: int = 2,
    enemy_class_id: str = "",
    round_number: int = 1,
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
        effort=enemy_effort,
        max_effort=max(2, enemy_effort),
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
        round_number=round_number,
    )


def _mark_setup_state() -> CombatState:
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
        max_effort=2,
        class_id="bandit_cutthroat",
    )
    state.enemies["cutthroat"] = payoff
    state.enemy_formation.place(payoff.actor_id, payoff.formation_slot)
    return state


def _maw_drag_state() -> CombatState:
    state = _combat_state(
        enemy_skills=["maw_slam", "drag_forward"],
        enemy_class_id="cave_maw_brute",
    )
    state.heroes["front_left"].class_id = "watchman"
    state.heroes["back_left"].class_id = "scribe"
    return state


def _maw_after_drag_state() -> CombatState:
    state = _combat_state(
        enemy_skills=["maw_slam", "drag_forward"],
        enemy_class_id="cave_maw_brute",
    )
    state.heroes["front_left"].class_id = "scribe"
    state.heroes["back_left"].class_id = "watchman"
    return state


def _wolf_pack_state() -> CombatState:
    state = _combat_state(
        enemy_skills=["wolf_bite", "howl"],
        enemy_class_id="alpha_wolf",
    )
    alpha = state.enemies.pop("enemy")
    alpha.actor_id = "alpha"
    alpha.formation_slot = FormationSlot.FRONT_RIGHT
    state.enemies["alpha"] = alpha
    state.enemy_formation = Formation.empty()
    state.enemy_formation.place("alpha", FormationSlot.FRONT_RIGHT)
    wolf = _combatant(
        "wolf",
        Team.ENEMY,
        FormationSlot.FRONT_LEFT,
        skills=["wolf_bite", "pack_bite"],
        effort=2,
        max_effort=2,
        class_id="wolf",
    )
    state.enemies["wolf"] = wolf
    state.enemy_formation.place(wolf.actor_id, wolf.formation_slot)
    return state


def _guard_state() -> CombatState:
    heroes = {
        "front_left": _combatant("front_left", Team.HERO, FormationSlot.FRONT_LEFT),
        "front_right": _combatant("front_right", Team.HERO, FormationSlot.FRONT_RIGHT),
    }
    bone = _combatant(
        "bone",
        Team.ENEMY,
        FormationSlot.FRONT_LEFT,
        skills=["rusted_chop", "shielding_dead"],
        effort=2,
        max_effort=2,
        class_id="bone_soldier",
    )
    acolyte = _combatant(
        "acolyte",
        Team.ENEMY,
        FormationSlot.BACK_LEFT,
        skills=["black_pulse", "mark_the_pattern"],
        effort=3,
        max_effort=3,
        class_id="maze_acolyte",
    )

    party_formation = Formation.empty()
    for hero in heroes.values():
        party_formation.place(hero.actor_id, hero.formation_slot)

    enemy_formation = Formation.empty()
    enemy_formation.place(bone.actor_id, bone.formation_slot)
    enemy_formation.place(acolyte.actor_id, acolyte.formation_slot)

    return CombatState(
        heroes=heroes,
        enemies={bone.actor_id: bone, acolyte.actor_id: acolyte},
        party_formation=party_formation,
        enemy_formation=enemy_formation,
    )


def _coordinated_mark_state() -> CombatState:
    state = _mark_setup_state()
    slinger = _combatant(
        "slinger",
        Team.ENEMY,
        FormationSlot.BACK_LEFT,
        skills=["sling_stone", "pinning_shot"],
        effort=2,
        max_effort=2,
        class_id="bandit_slinger",
    )
    state.enemies["slinger"] = slinger
    state.enemy_formation.place(slinger.actor_id, slinger.formation_slot)
    return state


def _candidate_feature(
    trace,
    skill_id: str,
    target_id: str,
    feature_name: str,
) -> int:
    return _candidate(trace, skill_id, target_id).features[feature_name]


def _candidate(trace, skill_id: str, target_id: str):
    for candidate in trace.candidates:
        if candidate.skill_id == skill_id and candidate.target_id == target_id:
            return candidate
    raise AssertionError(f"Missing candidate {skill_id} -> {target_id}")


def _combatant(
    actor_id: str,
    team: Team,
    slot: FormationSlot,
    *,
    skills: list[str] | None = None,
    effort: int = 0,
    max_effort: int = 2,
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
        max_effort=max_effort,
        effort=effort,
        skills=skills or ["guard_strike"],
        formation_slot=slot,
        class_id=class_id,
    )
