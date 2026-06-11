from game.combat.combat_state import ActorStatus, Combatant, Team
from game.combat.formation import Formation, FormationSlot


def actor(actor_id: str, slot: FormationSlot) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id,
        team=Team.HERO,
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


def test_frontliner_protects_same_lane_backliner() -> None:
    formation = Formation.from_mapping(
        {
            FormationSlot.FRONT_LEFT: "front",
            FormationSlot.BACK_LEFT: "back",
        }
    )
    combatants = {
        "front": actor("front", FormationSlot.FRONT_LEFT),
        "back": actor("back", FormationSlot.BACK_LEFT),
    }

    assert formation.is_protected("back", combatants)


def test_downed_frontliner_does_not_protect() -> None:
    formation = Formation.from_mapping(
        {
            FormationSlot.FRONT_LEFT: "front",
            FormationSlot.BACK_LEFT: "back",
        }
    )
    combatants = {
        "front": actor("front", FormationSlot.FRONT_LEFT),
        "back": actor("back", FormationSlot.BACK_LEFT),
    }
    combatants["front"].statuses.add(ActorStatus.DOWNED)

    assert not formation.is_protected("back", combatants)
    assert formation.is_exposed("back", combatants)


def test_stunned_frontliner_does_not_protect() -> None:
    formation = Formation.from_mapping(
        {
            FormationSlot.FRONT_LEFT: "front",
            FormationSlot.BACK_LEFT: "back",
        }
    )
    combatants = {
        "front": actor("front", FormationSlot.FRONT_LEFT),
        "back": actor("back", FormationSlot.BACK_LEFT),
    }
    combatants["front"].statuses.add(ActorStatus.STUNNED)

    assert not formation.is_protected("back", combatants)


def test_empty_front_slot_exposes_backliner() -> None:
    formation = Formation.from_mapping({FormationSlot.BACK_LEFT: "back"})
    combatants = {"back": actor("back", FormationSlot.BACK_LEFT)}

    assert formation.is_exposed("back", combatants)


def test_movement_swaps_adjacent_allies_correctly() -> None:
    formation = Formation.from_mapping(
        {
            FormationSlot.FRONT_LEFT: "front",
            FormationSlot.BACK_LEFT: "back",
        }
    )

    assert formation.swap_slots(FormationSlot.FRONT_LEFT, FormationSlot.BACK_LEFT)
    assert formation.actor_at(FormationSlot.FRONT_LEFT) == "back"
    assert formation.actor_at(FormationSlot.BACK_LEFT) == "front"


def test_no_default_diagonal_movement() -> None:
    formation = Formation.from_mapping(
        {
            FormationSlot.FRONT_LEFT: "front",
            FormationSlot.BACK_RIGHT: "back",
        }
    )

    assert not formation.swap_slots(FormationSlot.FRONT_LEFT, FormationSlot.BACK_RIGHT)
    assert formation.actor_at(FormationSlot.FRONT_LEFT) == "front"
    assert formation.actor_at(FormationSlot.BACK_RIGHT) == "back"
