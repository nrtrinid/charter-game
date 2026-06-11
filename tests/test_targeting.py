from game.combat.combat_state import Combatant, CombatState, Team
from game.combat.formation import Formation, FormationSlot
from game.combat.targeting import AttackType, cover_penalty, legal_targets


def actor(actor_id: str, team: Team, slot: FormationSlot) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id,
        team=team,
        max_hp=10,
        hp=10,
        speed=1,
        accuracy=0,
        defense=0,
        damage=1,
        max_effort=0,
        effort=0,
        skills=["guard_strike"],
        formation_slot=slot,
    )


def state_with_protected_backliner() -> CombatState:
    hero = actor("hero", Team.HERO, FormationSlot.FRONT_LEFT)
    enemy_front = actor("enemy_front", Team.ENEMY, FormationSlot.FRONT_LEFT)
    enemy_back = actor("enemy_back", Team.ENEMY, FormationSlot.BACK_LEFT)
    return CombatState(
        heroes={"hero": hero},
        enemies={"enemy_front": enemy_front, "enemy_back": enemy_back},
        party_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "hero"}),
        enemy_formation=Formation.from_mapping(
            {
                FormationSlot.FRONT_LEFT: "enemy_front",
                FormationSlot.BACK_LEFT: "enemy_back",
            }
        ),
    )


def test_melee_can_target_frontliners() -> None:
    state = state_with_protected_backliner()

    assert "enemy_front" in legal_targets(state, "hero", AttackType.MELEE)


def test_melee_cannot_target_protected_backliner() -> None:
    state = state_with_protected_backliner()

    assert "enemy_back" not in legal_targets(state, "hero", AttackType.MELEE)


def test_melee_can_target_exposed_backliner() -> None:
    state = state_with_protected_backliner()
    state.enemy_formation.remove("enemy_front")

    assert "enemy_back" in legal_targets(state, "hero", AttackType.MELEE)


def test_reach_can_target_same_lane_protected_backliner() -> None:
    state = state_with_protected_backliner()

    assert "enemy_back" in legal_targets(state, "hero", AttackType.REACH)


def test_reach_cannot_target_opposite_lane_exposed_backliner() -> None:
    hero = actor("hero", Team.HERO, FormationSlot.FRONT_LEFT)
    enemy_back = actor("enemy_back", Team.ENEMY, FormationSlot.BACK_RIGHT)
    state = CombatState(
        heroes={"hero": hero},
        enemies={"enemy_back": enemy_back},
        party_formation=Formation.from_mapping({FormationSlot.FRONT_LEFT: "hero"}),
        enemy_formation=Formation.from_mapping({FormationSlot.BACK_RIGHT: "enemy_back"}),
    )

    assert "enemy_back" not in legal_targets(state, "hero", AttackType.REACH)


def test_ranged_can_target_protected_backliner_with_cover_penalty() -> None:
    state = state_with_protected_backliner()

    assert "enemy_back" in legal_targets(state, "hero", AttackType.RANGED)
    assert cover_penalty(state, "enemy_back", AttackType.RANGED) == -2


def test_magic_can_target_protected_backliner() -> None:
    state = state_with_protected_backliner()

    assert "enemy_back" in legal_targets(state, "hero", AttackType.MAGIC)
