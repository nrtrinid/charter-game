from game.combat.combat_state import Combatant, CombatState, MoraleState, StrainTier, Team
from game.combat.formation import Formation, FormationSlot
from game.combat.turn_order import (
    morale_initiative_modifier,
    roll_initiative,
    roll_initiative_for_actor,
)
from game.core.rng import GameRng


def combatant(actor_id: str, speed: int, team: Team = Team.HERO) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id,
        team=team,
        max_hp=10,
        hp=10,
        speed=speed,
        accuracy=0,
        defense=0,
        damage=1,
        max_effort=0,
        effort=0,
        skills=["guard_strike"],
        formation_slot=FormationSlot.FRONT_LEFT,
    )


def test_initiative_is_speed_plus_1d8() -> None:
    actor = combatant("hero", speed=3)

    assert roll_initiative_for_actor(actor, GameRng(5)) == (
        3 + GameRng(5).randint(1, 8)
    )


def test_seeded_rng_gives_deterministic_initiative() -> None:
    hero = combatant("hero", speed=3)
    enemy = combatant("enemy", speed=2, team=Team.ENEMY)
    state = CombatState(
        heroes={"hero": hero},
        enemies={"enemy": enemy},
        party_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "hero"}),
        enemy_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "enemy"}),
    )

    first = roll_initiative(state, GameRng(9))
    second = roll_initiative(state, GameRng(9))

    assert first == second


def test_morale_state_modifies_initiative() -> None:
    assert morale_initiative_modifier(MoraleState.INSPIRED) == 1
    assert morale_initiative_modifier(MoraleState.STEADY) == 0
    assert morale_initiative_modifier(MoraleState.SHAKEN) == -1
    assert morale_initiative_modifier(MoraleState.BROKEN) == -2


def test_morale_modifies_initiative_without_strain_modifier() -> None:
    inspired_fresh = combatant("hero", speed=3)
    inspired_fresh.morale = MoraleState.INSPIRED
    inspired_fresh.strain = StrainTier.FRESH
    broken_spent = combatant("hero", speed=3)
    broken_spent.morale = MoraleState.BROKEN
    broken_spent.strain = StrainTier.SPENT

    base = 3 + GameRng(11).randint(1, 8)

    assert roll_initiative_for_actor(inspired_fresh, GameRng(11)) == base + 1
    assert roll_initiative_for_actor(broken_spent, GameRng(11)) == base - 2
