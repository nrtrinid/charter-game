from game.combat.combat_state import ActorStatus, Combatant, CombatState, Team
from game.combat.damage import apply_damage, heal_combatant
from game.combat.formation import Formation, FormationSlot


def actor(actor_id: str, hp: int = 5) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id,
        team=Team.HERO,
        max_hp=5,
        hp=hp,
        speed=1,
        accuracy=0,
        defense=0,
        damage=1,
        max_effort=0,
        effort=0,
        skills=["guard_strike"],
        formation_slot=FormationSlot.FRONT_LEFT,
    )


def enemy() -> Combatant:
    return Combatant(
        actor_id="enemy",
        name="enemy",
        team=Team.ENEMY,
        max_hp=5,
        hp=5,
        speed=1,
        accuracy=0,
        defense=0,
        damage=1,
        max_effort=0,
        effort=0,
        skills=["rusted_chop"],
        formation_slot=FormationSlot.FRONT_LEFT,
    )


def state_with(hero: Combatant) -> CombatState:
    return CombatState(
        heroes={hero.actor_id: hero},
        enemies={"enemy": enemy()},
        party_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: hero.actor_id}),
        enemy_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "enemy"}),
    )


def test_hero_at_zero_hp_becomes_downed() -> None:
    hero = actor("hero")
    state = state_with(hero)

    apply_damage(state, "enemy", "hero", 5)

    assert ActorStatus.DOWNED in hero.statuses
    assert hero.hp == 0


def test_downed_hero_cannot_act() -> None:
    hero = actor("hero")
    state = state_with(hero)

    apply_damage(state, "enemy", "hero", 5)

    assert not hero.can_act()


def test_downed_hero_does_not_protect_lane() -> None:
    front = actor("front")
    back = actor("back")
    back.formation_slot = FormationSlot.BACK_LEFT
    state = CombatState(
        heroes={"front": front, "back": back},
        enemies={"enemy": enemy()},
        party_formation=Formation.from_mapping(
            {FormationSlot.FRONT_LEFT: "front", FormationSlot.BACK_LEFT: "back"}
        ),
        enemy_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "enemy"}),
    )

    apply_damage(state, "enemy", "front", 5)

    assert not state.party_formation.is_protected("back", state.heroes)


def test_downed_hero_taking_damage_gains_mortal_wound() -> None:
    hero = actor("hero")
    state = state_with(hero)
    apply_damage(state, "enemy", "hero", 5)

    apply_damage(state, "enemy", "hero", 1)

    assert hero.mortal_wounds == 1


def test_back_to_back_attacks_after_downing_add_mortal_wound_before_recovery() -> None:
    hero = actor("hero")
    state = state_with(hero)

    downing_events = apply_damage(state, "enemy", "hero", 5)
    followup_events = apply_damage(state, "enemy", "hero", 1)

    assert ActorStatus.DOWNED in hero.statuses
    assert hero.hp == 0
    assert hero.mortal_wounds == 1
    assert any(event.event_type.value == "downed" for event in downing_events)
    assert any(
        getattr(event, "status", "") == "mortal_wound"
        for event in followup_events
    )


def test_enemy_death_message_includes_damage_taken() -> None:
    hero = actor("hero")
    state = state_with(hero)

    events = apply_damage(state, "hero", "enemy", 5)

    assert any(
        event.message == "enemy dies: took 5 damage, reduced to 0 HP."
        for event in events
    )


def test_zero_damage_to_downed_hero_does_not_add_mortal_wound() -> None:
    hero = actor("hero")
    state = state_with(hero)
    apply_damage(state, "enemy", "hero", 5)

    apply_damage(state, "enemy", "hero", 0)

    assert hero.mortal_wounds == 0


def test_three_mortal_wounds_kills_permanently() -> None:
    hero = actor("hero")
    state = state_with(hero)
    apply_damage(state, "enemy", "hero", 5)

    apply_damage(state, "enemy", "hero", 1)
    apply_damage(state, "enemy", "hero", 1)
    apply_damage(state, "enemy", "hero", 1)

    assert ActorStatus.DEAD in hero.statuses
    assert ActorStatus.DOWNED not in hero.statuses


def test_healing_above_zero_removes_downed() -> None:
    hero = actor("hero")
    state = state_with(hero)
    apply_damage(state, "enemy", "hero", 5)

    heal_combatant(state, "hero", 2)

    assert hero.hp == 2
    assert ActorStatus.DOWNED not in hero.statuses
