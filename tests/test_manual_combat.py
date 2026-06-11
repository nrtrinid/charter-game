from __future__ import annotations

from game.app.commands import (
    ChooseCombatSkill,
    DelayCombatTurn,
    MoveCombatActor,
    PassCombatTurn,
    ResolveCombatAction,
    ResolveCombatReaction,
    StartManualCombat,
    StartNewCompany,
    ViewCombat,
)
from game.app.controller import AppController
from game.app.manual_combat import (
    EnemyIntent,
    ManualCombatSession,
    auto_advance_to_hero,
    heal_amount_range_for_skill,
    legal_move_slots,
    legal_reaction_options,
    legal_skill_ids,
    legal_target_ids,
    resolve_enemy_reaction,
    resolve_hero_pass,
    resolve_hero_retreat,
)
from game.app.views import CombatView, build_combat_view
from game.combat.combat_state import ActorStatus, Combatant, CombatState, Tag, Team, apply_marked
from game.combat.damage_range import combatant_damage_range
from game.combat.formation import Formation, FormationSlot
from game.combat.preview import preview_attack
from game.combat.traits import QUIRK_FIELD_MEDIC
from game.combat.turn_order import InitiativeEntry
from game.content.definitions import GameDefinitions
from game.core.events import (
    CombatRetreatDeclaredEvent,
    DamageEvent,
    EncounterEndedEvent,
    EnemyIntentEvent,
    HealingEvent,
    MissEvent,
    MoveEvent,
    ReactionSkippedEvent,
    ReactionUsedEvent,
    RoundEndedEvent,
    RoundStartedEvent,
    SkillUsedEvent,
    StatusChangedEvent,
    TurnDelayedEvent,
    TurnPassedEvent,
)
from game.core.rng import GameRng
from game.expedition.cave import create_shallow_cave_combat
from tests.conftest import get_definitions


def test_manual_combat_lists_legal_affordable_skills_and_targets() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    result = controller.handle(StartManualCombat("shallow_cave"))

    assert result.success
    assert controller.manual_combat is not None
    session = controller.manual_combat
    skill_ids = legal_skill_ids(session, controller.definitions)

    assert skill_ids
    for skill_id in skill_ids:
        skill = controller.definitions.skills[skill_id]
        pending_hero = session.pending_hero()
        assert pending_hero is not None
        assert pending_hero.effort >= skill.effort_cost
        assert legal_target_ids(session, controller.definitions, skill_id)


def test_manual_combat_session_uses_controller_enemy_ai_mode_and_timing() -> None:
    controller = AppController(
        definitions=get_definitions(),
        enemy_ai_mode="heuristic",
    )
    controller.handle(StartNewCompany())
    result = controller.handle(StartManualCombat("shallow_cave"))

    assert result.success
    assert controller.manual_combat is not None
    assert controller.manual_combat.enemy_ai_mode == "heuristic"
    assert controller.manual_combat.enemy_wait_mode == "none"
    assert controller.manual_combat.enemy_movement_mode == "recovery_only"


def test_combat_view_model_includes_defaults_and_target_previews() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert view_result.success
    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    assert view.current_actor is not None
    assert view.party[0].art_lines
    assert view.enemies[0].art_lines
    assert view.commands
    assert {action.value for action in view.commands} == {
        "skill",
        "move",
        "delay",
        "pass",
        "retreat",
    }
    assert view.skills
    default_skill = next(skill for skill in view.skills if skill.action.default)
    assert default_skill.action.enabled

    skill_result = controller.handle(ChooseCombatSkill(default_skill.skill_id))

    assert skill_result.success
    assert isinstance(skill_result.value, CombatView)
    selected = skill_result.value
    assert selected.targets
    assert selected.targets[0].action.default
    assert selected.targets[0].hit_chance >= 0
    assert selected.targets[0].damage_estimate >= 0
    assert selected.targets[0].legality_reason in {
        "frontline",
        "exposed",
        "ranged",
        "magic",
        "same lane",
        "living ally",
    }


def test_combat_target_defaults_to_front_left_before_front_right() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))
    view_result = controller.handle(ViewCombat())
    assert isinstance(view_result.value, CombatView)
    skill = next(skill for skill in view_result.value.skills if skill.action.enabled)

    skill_result = controller.handle(ChooseCombatSkill(skill.skill_id))

    assert isinstance(skill_result.value, CombatView)
    view = skill_result.value
    assert view.targets
    assert view.targets[0].slot == FormationSlot.FRONT_LEFT.value
    assert view.targets[0].action.default


def test_attack_preview_reports_tactical_targeting_reasons() -> None:
    definitions = get_definitions()
    assert definitions is not None
    controller = AppController(definitions=definitions)
    controller.handle(StartNewCompany())
    assert controller.company is not None
    combat = create_shallow_cave_combat(controller.company, definitions)

    frontline = preview_attack(
        combat,
        "hero_watchman",
        definitions.skills["guard_strike"],
        "bone_soldier_1",
    )
    ranged = preview_attack(
        combat,
        "hero_cutpurse",
        definitions.skills["exposed_cut"],
        "skulker_1",
    )
    magic = preview_attack(
        combat,
        "hero_scribe",
        definitions.skills["abyssal_spark"],
        "skulker_1",
    )
    combat.enemies["bone_soldier_1"].statuses.add(ActorStatus.DEAD)
    exposed = preview_attack(
        combat,
        "hero_watchman",
        definitions.skills["guard_strike"],
        "skulker_1",
    )

    assert frontline.legality_reason == "frontline"
    assert ranged.legality_reason == "ranged"
    assert magic.legality_reason == "magic"
    assert exposed.legality_reason == "exposed"


def test_attack_preview_includes_conditional_damage_bonuses() -> None:
    definitions = get_definitions()
    assert definitions is not None
    controller = AppController(definitions=definitions)
    controller.handle(StartNewCompany())
    assert controller.company is not None
    combat = create_shallow_cave_combat(controller.company, definitions)
    cutpurse = combat.heroes["hero_cutpurse"]
    skill = definitions.skills["exposed_cut"]
    target_id = next(iter(combat.enemies))
    target = combat.enemies[target_id]
    base_low, base_high = combatant_damage_range(skill, cutpurse)

    unmarked = preview_attack(combat, cutpurse.actor_id, skill, target_id)
    assert unmarked.damage_min == base_low
    assert unmarked.damage_max == base_high

    apply_marked(target)
    marked = preview_attack(combat, cutpurse.actor_id, skill, target_id)
    assert marked.damage_min == base_low + 3
    assert marked.damage_max == base_high + 3


def test_manual_combat_resolves_hero_actions_and_ends_on_victory() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    final_events = []
    while controller.manual_combat is not None:
        session = controller.manual_combat
        skill_id = legal_skill_ids(session, controller.definitions)[0]
        target_id = legal_target_ids(session, controller.definitions, skill_id)[0]
        result = controller.handle(ResolveCombatAction(skill_id, target_id))
        assert result.success
        final_events = result.events

    assert any(isinstance(event, SkillUsedEvent) for event in final_events)
    assert any(isinstance(event, DamageEvent) for event in final_events)
    assert any(isinstance(event, EncounterEndedEvent) for event in final_events)
    assert controller.company is not None
    assert controller.company.roster


def test_combat_movement_lists_adjacent_slots_and_persists_formation() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    session.turn_index = 0

    assert set(legal_move_slots(session)) == {
        FormationSlot.BACK_LEFT,
        FormationSlot.FRONT_RIGHT,
    }

    blocked = controller.handle(MoveCombatActor(FormationSlot.BACK_RIGHT))

    assert not blocked.success
    assert session.state.party_formation.actor_at(FormationSlot.FRONT_LEFT) == "hero_watchman"

    result = controller.handle(MoveCombatActor(FormationSlot.BACK_LEFT))

    assert result.success
    assert any(isinstance(event, MoveEvent) for event in result.events)
    assert session.state.party_formation.actor_at(FormationSlot.BACK_LEFT) == "hero_watchman"
    assert session.state.party_formation.actor_at(FormationSlot.FRONT_LEFT) == "hero_field_surgeon"
    assert session.state.heroes["hero_watchman"].formation_slot == FormationSlot.BACK_LEFT
    assert session.state.heroes["hero_field_surgeon"].formation_slot == FormationSlot.FRONT_LEFT


def test_drag_forward_rebuilds_movement_options_from_updated_formation() -> None:
    controller = AppController(definitions=get_definitions())
    controller.definitions.skills["drag_forward"] = controller.definitions.skills[
        "drag_forward"
    ].model_copy(update={"accuracy": 100})
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("cave_maw_brute_1", 99),
        InitiativeEntry("hero_watchman", 98),
        InitiativeEntry("hero_cutpurse", 97),
    ]
    session.turn_index = 0

    events = auto_advance_to_hero(session, controller.definitions, GameRng(1))

    assert any(isinstance(event, MoveEvent) for event in events)
    assert session.state.party_formation.actor_at(FormationSlot.FRONT_LEFT) == "hero_field_surgeon"
    assert session.state.party_formation.actor_at(FormationSlot.BACK_LEFT) == "hero_watchman"

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    assert view.current_actor is not None
    assert view.current_actor.actor_id == "hero_watchman"
    assert view.current_actor.slot == "BACK_LEFT"
    assert {(move.from_slot, move.to_slot) for move in view.moves} == {
        ("BACK_LEFT", "BACK_RIGHT"),
        ("BACK_LEFT", "FRONT_LEFT"),
    }
    assert {actor.actor_id: actor.slot for actor in view.party} == {
        "hero_watchman": "BACK_LEFT",
        "hero_scribe": "BACK_RIGHT",
        "hero_field_surgeon": "FRONT_LEFT",
        "hero_cutpurse": "FRONT_RIGHT",
    }


def test_pass_emits_structured_event_and_advances_turn() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    session.turn_index = 0

    result = controller.handle(PassCombatTurn())

    assert result.success
    assert any(isinstance(event, TurnPassedEvent) for event in result.events)
    assert session.turn_index == 1
    pending_hero = session.pending_hero()
    assert pending_hero is not None
    assert pending_hero.actor_id == "hero_cutpurse"


def test_delay_moves_actor_to_end_of_current_round_without_spending_action() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    session.turn_index = 0

    result = controller.handle(DelayCombatTurn())

    assert result.success
    assert any(isinstance(event, TurnDelayedEvent) for event in result.events)
    assert [entry.actor_id for entry in session.initiative] == ["hero_cutpurse", "hero_watchman"]
    assert session.turn_index == 0
    pending_hero = session.pending_hero()
    assert pending_hero is not None
    assert pending_hero.actor_id == "hero_cutpurse"


def test_delay_does_not_tick_marked_duration() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    session.turn_index = 0
    watchman = session.state.heroes["hero_watchman"]
    apply_marked(watchman)

    result = controller.handle(DelayCombatTurn())

    assert result.success
    assert Tag.MARKED in watchman.tags
    assert watchman.tag_turns[Tag.MARKED] == 2


def test_delay_is_once_per_round_and_disabled_without_later_slot() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    session.turn_index = 0

    first_delay = controller.handle(DelayCombatTurn())
    pass_thief = controller.handle(PassCombatTurn())
    second_delay = controller.handle(DelayCombatTurn())

    assert first_delay.success
    assert pass_thief.success
    assert not second_delay.success
    pending_hero = session.pending_hero()
    assert pending_hero is not None
    assert pending_hero.actor_id == "hero_watchman"

    session.initiative = [InitiativeEntry("hero_watchman", 99)]
    session.turn_index = 0
    session.delayed_actor_ids.clear()
    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    delay = next(action for action in view_result.value.commands if action.value == "delay")
    assert not delay.enabled
    assert not controller.handle(DelayCombatTurn()).success


def test_retreat_declares_pending_escape_and_resolves_at_round_end() -> None:
    definitions = get_definitions()
    session = _reaction_test_session(enemy_skill_id="rusted_chop", enemy_accuracy=100)
    session.initiative = [
        InitiativeEntry("a_cutpurse", 99),
        InitiativeEntry("enemy_brute", 98),
    ]
    session.turn_index = 0

    events = resolve_hero_retreat(session, definitions, GameRng(1))

    assert session.ended
    assert session.outcome == "retreat"
    assert session.retreat_pending
    assert session.retreat_actor_id == "a_cutpurse"
    assert any(isinstance(event, CombatRetreatDeclaredEvent) for event in events)
    assert any(
        isinstance(event, SkillUsedEvent) and event.actor_id == "enemy_brute"
        for event in events
    )
    assert any(isinstance(event, RoundEndedEvent) for event in events)
    assert not any(
        isinstance(event, RoundStartedEvent) and event.round_number == 2
        for event in events
    )


def test_later_heroes_cover_withdrawal_instead_of_attacking() -> None:
    definitions = get_definitions()
    session = _reaction_test_session(enemy_skill_id="rusted_chop", enemy_accuracy=100)
    session.initiative = [
        InitiativeEntry("a_cutpurse", 99),
        InitiativeEntry("b_field_surgeon", 98),
        InitiativeEntry("enemy_brute", 97),
    ]
    session.turn_index = 0

    events = resolve_hero_retreat(session, definitions, GameRng(1))

    assert session.ended
    assert session.outcome == "retreat"
    assert any(
        isinstance(event, TurnPassedEvent)
        and event.actor_id == "b_field_surgeon"
        and "withdrawal" in event.message
        for event in events
    )
    assert not any(
        isinstance(event, SkillUsedEvent) and event.actor_id == "b_field_surgeon"
        for event in events
    )


def test_marked_ticks_only_when_marked_combatant_finishes_activation() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    marked_hero = session.state.heroes["a_cutpurse"]
    apply_marked(marked_hero)

    events = auto_advance_to_hero(session, definitions, GameRng(1))

    assert events
    assert Tag.MARKED in marked_hero.tags
    assert marked_hero.tag_turns[Tag.MARKED] == 2


def test_marked_expires_after_two_own_finished_activations() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    session.initiative = [
        InitiativeEntry("a_cutpurse", 99),
        InitiativeEntry("b_field_surgeon", 98),
    ]
    session.turn_index = 0
    marked_hero = session.state.heroes["a_cutpurse"]
    apply_marked(marked_hero)

    first_events = resolve_hero_pass(session, definitions, GameRng(1))

    assert first_events
    assert Tag.MARKED in marked_hero.tags
    assert marked_hero.tag_turns[Tag.MARKED] == 1

    session.initiative = [
        InitiativeEntry("a_cutpurse", 99),
        InitiativeEntry("b_field_surgeon", 98),
    ]
    session.turn_index = 0

    second_events = resolve_hero_pass(session, definitions, GameRng(1))

    assert Tag.MARKED not in marked_hero.tags
    assert Tag.MARKED not in marked_hero.tag_turns
    assert any(
        isinstance(event, StatusChangedEvent)
        and event.actor_id == marked_hero.actor_id
        and event.status == Tag.MARKED.name.lower()
        and not event.added
        for event in second_events
    )


def test_treatment_spends_effort_heals_and_lifts_downed() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_field_surgeon", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0
    guard = session.state.heroes["hero_watchman"]
    guard.hp = 0
    guard.statuses.add(ActorStatus.DOWNED)

    assert "emergency_stitch" in legal_skill_ids(session, controller.definitions)
    assert "hero_watchman" in legal_target_ids(
        session,
        controller.definitions,
        "emergency_stitch",
    )

    result = controller.handle(ResolveCombatAction("emergency_stitch", "hero_watchman"))

    assert result.success
    assert session.state.heroes["hero_field_surgeon"].effort == 3
    assert 3 <= guard.hp <= 4
    assert ActorStatus.DOWNED not in guard.statuses
    assert any(isinstance(event, HealingEvent) for event in result.events)
    assert any(
        isinstance(event, StatusChangedEvent)
        and event.status == ActorStatus.DOWNED.value
        and not event.added
        for event in result.events
    )


def test_field_medic_first_treatment_heals_more_and_updates_preview() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    surgeon = session.state.heroes["hero_field_surgeon"]
    surgeon.effort = 3
    guard = session.state.heroes["hero_watchman"]
    guard.hp = 7
    session.initiative = [
        InitiativeEntry("hero_field_surgeon", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0

    normal_range = heal_amount_range_for_skill(
        session,
        controller.definitions,
        "emergency_stitch",
        "hero_watchman",
    )
    surgeon.quirks = [QUIRK_FIELD_MEDIC]
    boosted_range = heal_amount_range_for_skill(
        session,
        controller.definitions,
        "emergency_stitch",
        "hero_watchman",
    )

    assert boosted_range == (normal_range[0] + 1, normal_range[1] + 1)

    result = controller.handle(ResolveCombatAction("emergency_stitch", "hero_watchman"))

    assert result.success
    heal_event = next(
        event for event in result.events if isinstance(event, HealingEvent)
    )
    assert boosted_range[0] <= heal_event.amount <= boosted_range[1]
    assert guard.hp == 7 + heal_event.amount

    surgeon.effort = 3
    guard.hp = 7
    session.initiative = [
        InitiativeEntry("hero_field_surgeon", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0

    assert heal_amount_range_for_skill(
        session,
        controller.definitions,
        "emergency_stitch",
        "hero_watchman",
    ) == normal_range


def test_shield_drive_targets_enemies_not_allies() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    session.turn_index = 0

    assert "shield_drive" in legal_skill_ids(session, controller.definitions)
    assert legal_target_ids(session, controller.definitions, "shield_drive") == [
        "bone_soldier_1",
        "bone_soldier_2",
        "skulker_1",
    ]


def test_front_only_skills_are_visible_disabled_from_back_row() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    session.turn_index = 0
    old_back_actor_id = session.state.party_formation.actor_at(FormationSlot.BACK_LEFT)
    assert session.state.party_formation.swap_slots(
        FormationSlot.FRONT_LEFT,
        FormationSlot.BACK_LEFT,
    )
    session.state.heroes["hero_watchman"].formation_slot = FormationSlot.BACK_LEFT
    if old_back_actor_id is not None:
        session.state.actor(old_back_actor_id).formation_slot = FormationSlot.FRONT_LEFT

    view = build_combat_view(session, controller.definitions)
    guard_strike = next(skill for skill in view.skills if skill.skill_id == "guard_strike")
    shield_drive = next(skill for skill in view.skills if skill.skill_id == "shield_drive")

    assert "guard_strike" not in legal_skill_ids(session, controller.definitions)
    assert not guard_strike.action.enabled
    assert guard_strike.usable_from_label == "Front"
    assert guard_strike.unavailable_reason == "Requires front row."
    assert not shield_drive.action.enabled
    assert shield_drive.unavailable_reason == "Requires front row."
    assert controller.handle(ChooseCombatSkill("guard_strike")).error == "Choose a listed skill."


def test_any_position_skills_remain_usable_from_back_row() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_scribe", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0

    assert "staff_jab" in legal_skill_ids(session, controller.definitions)
    assert "abyssal_spark" in legal_skill_ids(session, controller.definitions)
    view = build_combat_view(session, controller.definitions)
    staff_jab = next(skill for skill in view.skills if skill.skill_id == "staff_jab")

    assert staff_jab.action.enabled
    assert staff_jab.usable_from_label == "Any"


def test_back_row_field_surgeon_can_use_bone_saw() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_field_surgeon", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0

    assert "bone_saw" in legal_skill_ids(session, controller.definitions)
    view = build_combat_view(session, controller.definitions)
    bone_saw = next(skill for skill in view.skills if skill.skill_id == "bone_saw")

    assert bone_saw.action.enabled
    assert bone_saw.usable_from_label == "Any"


def test_enemy_skill_special_move_metadata_defaults_and_authored_windows() -> None:
    definitions = get_definitions()

    assert not definitions.skills["rusted_chop"].reaction_window
    assert not definitions.skills["skulker_stone"].reaction_window
    assert not definitions.skills["glass_bite"].reaction_window
    assert definitions.skills["maw_slam"].reaction_window
    assert definitions.skills["maw_slam"].threat_level == "high"
    assert definitions.skills["maw_slam"].obvious_effect
    assert definitions.skills["black_pulse"].reaction_window
    assert definitions.skills["black_pulse"].intent_label == "Black Pulse"


def test_basic_enemy_action_does_not_pause_for_reaction_window() -> None:
    definitions = get_definitions()
    session = _reaction_test_session(enemy_skill_id="rusted_chop")

    events = auto_advance_to_hero(session, definitions, GameRng(1))

    assert session.pending_enemy_intent is None
    assert session.turn_index == 1
    assert any(isinstance(event, SkillUsedEvent) for event in events)
    assert not any(isinstance(event, EnemyIntentEvent) for event in events)


def test_back_row_enemy_moves_to_unlock_front_only_skill() -> None:
    definitions = get_definitions()
    session = _reaction_test_session(enemy_skill_id="rusted_chop")
    enemy = session.state.enemies["enemy_brute"]
    session.state.enemy_formation.remove(enemy.actor_id)
    session.state.enemy_formation.place(enemy.actor_id, FormationSlot.BACK_LEFT)
    enemy.formation_slot = FormationSlot.BACK_LEFT

    events = auto_advance_to_hero(session, definitions, GameRng(1))

    assert session.state.enemy_formation.actor_at(FormationSlot.FRONT_LEFT) == enemy.actor_id
    assert enemy.formation_slot == FormationSlot.FRONT_LEFT
    assert any(
        isinstance(event, MoveEvent) and event.actor_id == enemy.actor_id
        for event in events
    )
    assert not any(
        isinstance(event, SkillUsedEvent) and event.actor_id == enemy.actor_id
        for event in events
    )


def test_back_row_enemy_with_any_position_skill_attacks_without_moving() -> None:
    definitions = get_definitions()
    session = _reaction_test_session(enemy_skill_id="skulker_stone")
    enemy = session.state.enemies["enemy_brute"]
    session.state.enemy_formation.remove(enemy.actor_id)
    session.state.enemy_formation.place(enemy.actor_id, FormationSlot.BACK_LEFT)
    enemy.formation_slot = FormationSlot.BACK_LEFT

    events = auto_advance_to_hero(session, definitions, GameRng(1))

    assert session.state.enemy_formation.actor_at(FormationSlot.BACK_LEFT) == enemy.actor_id
    assert any(
        isinstance(event, SkillUsedEvent) and event.actor_id == enemy.actor_id
        for event in events
    )
    assert not any(
        isinstance(event, MoveEvent) and event.actor_id == enemy.actor_id
        for event in events
    )


def test_special_enemy_action_resolves_after_danger_tell_without_reaction_pause() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()

    events = auto_advance_to_hero(session, definitions, GameRng(1))

    assert session.pending_enemy_intent is None
    assert session.turn_index == 1
    assert any(isinstance(event, EnemyIntentEvent) for event in events)
    assert any(isinstance(event, SkillUsedEvent) for event in events)
    assert any("rears back" in event.message for event in events)

    view = build_combat_view(session, definitions)
    assert view.pending_enemy_intent is None
    assert view.reaction_options == ()
    assert view.current_actor is not None
    assert view.current_actor.actor_id == "a_cutpurse"


def test_skip_reaction_resolves_pending_enemy_action() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    _prime_reaction_intent(session, definitions)

    events = resolve_enemy_reaction(session, definitions, GameRng(1), None)

    assert session.pending_enemy_intent is None
    assert session.turn_index == 1
    assert any(isinstance(event, ReactionSkippedEvent) for event in events)
    assert any(isinstance(event, SkillUsedEvent) for event in events)
    assert session.state.heroes["a_cutpurse"].hp == 7


def test_watchman_intercede_redirects_incoming_action_and_spends_effort() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    _prime_reaction_intent(session, definitions)

    reaction_id = _reaction_id(session, "watchman_intercede")
    events = resolve_enemy_reaction(session, definitions, GameRng(1), reaction_id)

    assert any(isinstance(event, ReactionUsedEvent) for event in events)
    assert any(
        isinstance(event, SkillUsedEvent) and event.target_id == "z_watchman"
        for event in events
    )
    assert session.state.heroes["z_watchman"].effort == 1
    assert session.state.heroes["z_watchman"].hp == 7
    assert session.state.heroes["a_cutpurse"].hp == 10


def test_cutpurse_evade_halves_incoming_damage_and_spends_effort() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    _prime_reaction_intent(session, definitions)

    reaction_id = _reaction_id(session, "cutpurse_evade")
    events = resolve_enemy_reaction(session, definitions, GameRng(1), reaction_id)

    assert any(isinstance(event, ReactionUsedEvent) for event in events)
    assert session.state.heroes["a_cutpurse"].effort == 1
    assert session.state.heroes["a_cutpurse"].hp == 9


def test_field_surgeon_stabilize_reduces_damage_and_prevents_collapse() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    session.state.heroes["a_cutpurse"].hp = 2
    _prime_reaction_intent(session, definitions)

    reaction_id = _reaction_id(session, "field_surgeon_stabilize")
    events = resolve_enemy_reaction(session, definitions, GameRng(1), reaction_id)

    assert any(isinstance(event, ReactionUsedEvent) for event in events)
    assert session.state.heroes["b_field_surgeon"].effort == 1
    assert session.state.heroes["a_cutpurse"].hp == 1
    assert ActorStatus.DOWNED not in session.state.heroes["a_cutpurse"].statuses


def test_scribe_disrupt_reduces_hit_chance_and_can_force_a_miss() -> None:
    definitions = get_definitions()
    definitions.skills["maw_slam"] = definitions.skills["maw_slam"].model_copy(
        update={"accuracy": 50}
    )
    session = _reaction_test_session(enemy_accuracy=0)
    _prime_reaction_intent(session, definitions)

    reaction_id = _reaction_id(session, "scribe_disrupt")
    events = resolve_enemy_reaction(session, definitions, GameRng(16), reaction_id)

    assert any(isinstance(event, ReactionUsedEvent) for event in events)
    assert any(isinstance(event, MissEvent) for event in events)
    assert session.state.heroes["c_scribe"].effort == 1
    assert session.state.heroes["a_cutpurse"].hp == 10


def test_invalid_reaction_does_not_resolve_pending_intent() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    _prime_reaction_intent(session, definitions)

    events = resolve_enemy_reaction(
        session,
        definitions,
        GameRng(1),
        "scribe_disrupt:a_cutpurse",
    )

    assert events == []
    assert session.pending_enemy_intent is not None
    assert session.turn_index == 0


def test_controller_resolves_reaction_command() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.manual_combat = _reaction_test_session()
    _prime_reaction_intent(controller.manual_combat, controller.definitions)

    result = controller.handle(ResolveCombatReaction(None))

    assert result.success
    assert isinstance(result.value, CombatView)
    assert any(isinstance(event, ReactionSkippedEvent) for event in result.events)


def test_pending_maw_intent_displays_damage_range() -> None:
    definitions = get_definitions()
    session = _reaction_test_session()
    session.state.enemies["enemy_brute"].damage = definitions.enemies["cave_maw_brute"].damage
    _prime_reaction_intent(session, definitions)

    view = build_combat_view(session, definitions, debug_combat_preview=True)

    assert view.pending_enemy_intent is not None
    assert view.pending_enemy_intent.debug_damage_estimate == 6
    assert view.pending_enemy_intent.debug_damage_label == "5-6"


def _reaction_test_session(
    *,
    enemy_skill_id: str = "maw_slam",
    enemy_accuracy: int = 100,
) -> ManualCombatSession:
    heroes = {
        "a_cutpurse": _test_combatant(
            "a_cutpurse",
            "Test Cutpurse",
            Team.HERO,
            FormationSlot.FRONT_LEFT,
            class_id="cutpurse",
        ),
        "b_field_surgeon": _test_combatant(
            "b_field_surgeon",
            "Test Field Surgeon",
            Team.HERO,
            FormationSlot.FRONT_RIGHT,
            class_id="field_surgeon",
        ),
        "c_scribe": _test_combatant(
            "c_scribe",
            "Test Scribe",
            Team.HERO,
            FormationSlot.BACK_RIGHT,
            class_id="scribe",
        ),
        "z_watchman": _test_combatant(
            "z_watchman",
            "Test Watchman",
            Team.HERO,
            FormationSlot.BACK_LEFT,
            class_id="watchman",
        ),
    }
    enemy = _test_combatant(
        "enemy_brute",
        "Test Brute",
        Team.ENEMY,
        FormationSlot.FRONT_LEFT,
        skills=[enemy_skill_id],
        accuracy=enemy_accuracy,
        effort=2,
        max_effort=2,
    )
    party_formation = Formation.empty()
    for hero in heroes.values():
        party_formation.place(hero.actor_id, hero.formation_slot)
    enemy_formation = Formation.empty()
    enemy_formation.place(enemy.actor_id, enemy.formation_slot)
    return ManualCombatSession(
        encounter_id="test",
        encounter_name="Test Encounter",
        state=CombatState(
            heroes=heroes,
            enemies={enemy.actor_id: enemy},
            party_formation=party_formation,
            enemy_formation=enemy_formation,
        ),
        initiative=[
            InitiativeEntry(enemy.actor_id, 99),
            InitiativeEntry("a_cutpurse", 98),
        ],
    )


def _prime_reaction_intent(
    session: ManualCombatSession,
    definitions: GameDefinitions,
    *,
    enemy_id: str = "enemy_brute",
    target_id: str = "a_cutpurse",
) -> None:
    skill_id = session.state.enemies[enemy_id].skills[0]
    skill = definitions.skills[skill_id]
    enemy = session.state.actor(enemy_id)
    target = session.state.actor(target_id)
    preview = preview_attack(session.state, enemy_id, skill, target_id)
    session.pending_enemy_intent = EnemyIntent(
        enemy_id=enemy_id,
        enemy_name=enemy.name,
        skill_id=skill_id,
        skill_name=skill.name,
        label=skill.intent_label or skill.name,
        target_id=target_id,
        target_name=target.name,
        threat_level=skill.threat_level,
        obvious_effect=skill.obvious_effect or "Incoming enemy action",
        hit_chance=preview.hit_chance,
        damage_estimate=preview.damage,
        damage_label=preview.damage_label,
    )


def _test_combatant(
    actor_id: str,
    name: str,
    team: Team,
    slot: FormationSlot,
    *,
    class_id: str = "",
    skills: list[str] | None = None,
    hp: int = 10,
    effort: int = 2,
    max_effort: int = 2,
    accuracy: int = 0,
) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=name,
        team=team,
        max_hp=10,
        hp=hp,
        speed=5,
        accuracy=accuracy,
        defense=0,
        damage=0,
        max_effort=max_effort,
        effort=effort,
        skills=skills or ["guard_strike"],
        formation_slot=slot,
        class_id=class_id,
    )


def _reaction_id(session: ManualCombatSession, kind: str) -> str:
    reaction = next(option for option in legal_reaction_options(session) if option.kind == kind)
    return reaction.reaction_id
