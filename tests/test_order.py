from game.combat.combat_state import CohesionState, Combatant, CombatState, MoraleState, Team
from game.combat.damage import apply_damage
from game.combat.formation import Formation, FormationSlot
from game.combat.morale import apply_horror, is_panic_risk


def hero() -> Combatant:
    return Combatant(
        actor_id="hero",
        name="hero",
        team=Team.HERO,
        max_hp=5,
        hp=5,
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


def state() -> CombatState:
    return CombatState(
        heroes={"hero": hero()},
        enemies={"enemy": enemy()},
        party_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "hero"}),
        enemy_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "enemy"}),
    )


def test_combat_state_has_no_independent_order_meter() -> None:
    combat = state()

    assert not hasattr(combat, "order")


def test_downed_hero_breaks_morale() -> None:
    combat = state()

    apply_damage(combat, "enemy", "hero", 5)

    assert combat.heroes["hero"].morale == MoraleState.BROKEN


def test_death_keeps_morale_broken() -> None:
    combat = state()
    apply_damage(combat, "enemy", "hero", 5)

    apply_damage(combat, "enemy", "hero", 1)
    apply_damage(combat, "enemy", "hero", 1)
    apply_damage(combat, "enemy", "hero", 1)

    assert combat.heroes["hero"].morale == MoraleState.BROKEN


def test_horror_effect_reduces_morale() -> None:
    combat = state()

    apply_horror(combat, 1)

    assert combat.heroes["hero"].morale == MoraleState.SHAKEN


def test_grave_calm_prevents_horror_below_steady() -> None:
    combat = state()
    combat.heroes["hero"].personal_quirk = "grave_calm"

    apply_horror(combat, 2)

    assert combat.heroes["hero"].morale == MoraleState.STEADY


def test_fractured_cohesion_indicates_panic_or_forced_retreat_risk() -> None:
    combat = state()

    combat.heroes["hero"].morale = MoraleState.BROKEN

    assert combat.derive_cohesion() == CohesionState.FRACTURED
    assert is_panic_risk(combat)
