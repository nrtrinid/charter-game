from __future__ import annotations

from game.combat.combat_state import Combatant, CombatState, Tag, Team
from game.combat.enemy_actions import (
    enemy_proactive_move,
    enemy_wait_reason,
    extract_move_timing_features,
    extract_wait_timing_features,
)
from game.combat.enemy_decision import EnemyDecisionRuntimeContext
from game.combat.formation import Formation, FormationSlot
from tests.conftest import get_definitions


def test_wait_features_when_current_action_good_with_marked_payoff() -> None:
    definitions = get_definitions()
    state = _mark_setup_state()
    state.heroes["front_left"].tags.add(Tag.MARKED)
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "marker_enemy"), 0)

    features = extract_wait_timing_features(
        state,
        definitions,
        actor,
        runtime_context,
    )

    assert features["current_best_action_is_good"] == 1
    assert features["wait_when_current_action_good"] == 1
    assert features["wait_no_current_good_action"] == 0


def test_wait_features_when_no_good_action_before_mark() -> None:
    definitions = get_definitions()
    state = _mark_setup_state()
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "marker_enemy"), 0)

    features = extract_wait_timing_features(
        state,
        definitions,
        actor,
        runtime_context,
    )

    assert features["wait_no_current_good_action"] == 1
    assert features["wait_when_current_action_good"] == 0
    assert features["wait_expected_value_delta"] >= 0


def test_move_features_unlock_and_marked_lane() -> None:
    definitions = get_definitions()
    state = _two_enemy_state(
        acting_skills=["dirty_finish"],
        acting_slot=FormationSlot.BACK_LEFT,
        blocker_slot=FormationSlot.FRONT_LEFT,
    )
    state.heroes["front_left"].tags.add(Tag.MARKED)
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "blocker_enemy"), 0)

    features = extract_move_timing_features(
        state,
        definitions,
        actor,
        runtime_context,
    )

    assert features["move_unlocks_future_skill"] == 1
    assert features["move_into_marked_lane"] == 1
    assert features["move_toward_payoff_target"] == 1


def test_timing_features_are_deterministic() -> None:
    definitions = get_definitions()
    state = _mark_setup_state()
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "marker_enemy"), 0)

    first = extract_wait_timing_features(state, definitions, actor, runtime_context)
    second = extract_wait_timing_features(state, definitions, actor, runtime_context)

    assert first == second


def test_bandit_waits_for_spotter_when_mark_setup_is_plausible() -> None:
    definitions = get_definitions()
    state = _mark_setup_state()
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "marker_enemy"), 0)

    assert (
        enemy_wait_reason(
            state,
            definitions,
            actor,
            runtime_context,
            "package_only",
            set(),
        )
        == "mark setup"
    )


def test_bandit_does_not_wait_when_marked_payoff_is_already_available() -> None:
    definitions = get_definitions()
    state = _mark_setup_state()
    state.heroes["front_left"].tags.add(Tag.MARKED)
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "marker_enemy"), 0)

    assert (
        enemy_wait_reason(
            state,
            definitions,
            actor,
            runtime_context,
            "package_only",
            set(),
        )
        is None
    )


def test_wolf_waits_for_alpha_when_mark_payoff_is_plausible() -> None:
    definitions = get_definitions()
    state = _wolf_pack_state()
    wolf = state.enemies["wolf"]
    wolf.speed = 20
    alpha = state.enemies["alpha"]
    alpha.speed = 1
    runtime_context = _runtime_context(("wolf", "alpha"), 0)

    assert (
        enemy_wait_reason(
            state,
            definitions,
            wolf,
            runtime_context,
            "package_only",
            set(),
        )
        == "mark setup"
    )


def test_wolf_does_not_wait_when_payoff_target_is_unreachable() -> None:
    definitions = get_definitions()
    state = _wolf_pack_state()
    wolf = state.enemies["wolf"]
    wolf.speed = 20
    wolf.skills = ["pack_bite"]
    wolf.formation_slot = FormationSlot.BACK_LEFT
    state.enemy_formation.remove("wolf")
    state.enemy_formation.place("wolf", FormationSlot.BACK_LEFT)
    alpha = state.enemies["alpha"]
    alpha.speed = 1
    runtime_context = _runtime_context(("wolf", "alpha"), 0)

    assert (
        enemy_wait_reason(
            state,
            definitions,
            wolf,
            runtime_context,
            "package_only",
            set(),
        )
        is None
    )


def test_bone_soldier_does_not_get_package_only_wait() -> None:
    definitions = get_definitions()
    state = _two_enemy_state(
        acting_skills=["rusted_chop"],
        acting_slot=FormationSlot.FRONT_LEFT,
        blocker_slot=FormationSlot.FRONT_RIGHT,
    )
    state.enemies["blocker_enemy"].skills = ["spot_target"]
    state.enemies["blocker_enemy"].class_id = "bandit_spotter"
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "blocker_enemy"), 0)

    assert (
        enemy_wait_reason(
            state,
            definitions,
            actor,
            runtime_context,
            "package_only",
            set(),
        )
        is None
    )


def test_cutthroat_proactive_move_into_marked_lane_when_useful() -> None:
    definitions = get_definitions()
    state = _two_enemy_state(
        acting_skills=["dirty_finish"],
        acting_slot=FormationSlot.BACK_LEFT,
        blocker_slot=FormationSlot.FRONT_LEFT,
    )
    state.heroes["front_left"].tags.add(Tag.MARKED)
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "blocker_enemy"), 0)

    events = enemy_proactive_move(
        state,
        definitions,
        actor,
        "package_only",
        runtime_context,
    )

    assert events
    assert state.enemy_formation.slot_of("acting_enemy") == FormationSlot.FRONT_LEFT


def test_proactive_move_does_not_appear_when_current_marked_payoff_exists() -> None:
    definitions = get_definitions()
    state = _two_enemy_state(
        acting_skills=["dirty_finish"],
        acting_slot=FormationSlot.FRONT_LEFT,
        blocker_slot=FormationSlot.FRONT_RIGHT,
    )
    state.heroes["front_left"].tags.add(Tag.MARKED)
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "blocker_enemy"), 0)

    assert (
        enemy_proactive_move(
            state,
            definitions,
            actor,
            "package_only",
            runtime_context,
        )
        == ()
    )


def test_proactive_move_does_not_appear_when_swap_would_not_unlock_payoff() -> None:
    definitions = get_definitions()
    state = _two_enemy_state(
        acting_skills=["dirty_finish"],
        acting_slot=FormationSlot.BACK_RIGHT,
        blocker_slot=FormationSlot.FRONT_LEFT,
    )
    state.heroes["front_left"].tags.add(Tag.MARKED)
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "blocker_enemy"), 0)

    assert (
        enemy_proactive_move(
            state,
            definitions,
            actor,
            "package_only",
            runtime_context,
        )
        == ()
    )


def test_proactive_move_respects_adjacency() -> None:
    definitions = get_definitions()
    state = _two_enemy_state(
        acting_skills=["dirty_finish"],
        acting_slot=FormationSlot.BACK_LEFT,
        blocker_slot=FormationSlot.BACK_RIGHT,
    )
    state.heroes["front_left"].tags.add(Tag.MARKED)
    actor = state.enemies["acting_enemy"]
    runtime_context = _runtime_context(("acting_enemy", "blocker_enemy"), 0)

    assert (
        enemy_proactive_move(
            state,
            definitions,
            actor,
            "package_only",
            runtime_context,
        )
        == ()
    )


def _runtime_context(actor_ids: tuple[str, ...], current_index: int) -> EnemyDecisionRuntimeContext:
    return EnemyDecisionRuntimeContext(
        initiative_actor_ids=actor_ids,
        current_turn_index=current_index,
    )


def _mark_setup_state() -> CombatState:
    state = _combat_state(enemy_skills=["dirty_finish"], place_enemy=False)
    acting = state.enemies.pop("enemy")
    acting.actor_id = "acting_enemy"
    acting.name = "Acting Enemy"
    acting.skills = ["dirty_finish"]
    acting.effort = 2
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
    acting.effort = 2
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
    enemies = {"enemy": enemy}
    if place_enemy:
        enemy_formation.place(enemy.actor_id, enemy.formation_slot)
    return CombatState(
        heroes=heroes,
        enemies=enemies,
        party_formation=party_formation,
        enemy_formation=enemy_formation,
    )


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
