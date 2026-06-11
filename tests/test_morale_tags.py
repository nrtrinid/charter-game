from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from game.app.manual_combat import ManualCombatSession, legal_move_slots
from game.combat.actions import use_skill
from game.combat.combat_state import (
    ActorStatus,
    CohesionState,
    Combatant,
    CombatState,
    MoraleState,
    Tag,
    Team,
    apply_marked,
)
from game.combat.damage import apply_damage
from game.combat.damage_range import combatant_damage_range
from game.combat.formation import Formation, FormationSlot
from game.combat.targeting import AttackType, SkillUsableFrom
from game.combat.turn_order import InitiativeEntry
from game.core.rng import GameRng
from game.data.schemas import SkillDefinition
from tests.conftest import get_definitions


@dataclass(frozen=True)
class SkillStub:
    id: str
    name: str
    effort_cost: int
    attack_type: AttackType
    accuracy: int
    damage: int
    tags: list[str]


def test_skill_usable_from_defaults_to_any_position() -> None:
    skill = SkillDefinition.model_validate(
        {
            "id": "test_skill",
            "name": "Test Skill",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "accuracy": 90,
            "damage": 1,
        }
    )

    assert skill.usable_from == SkillUsableFrom.ANY_POSITION


def test_skill_usable_from_accepts_authored_values() -> None:
    front_only = SkillDefinition.model_validate(
        {
            "id": "front_skill",
            "name": "Front Skill",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "usable_from": "front_only",
            "accuracy": 90,
            "damage": 1,
        }
    )

    assert front_only.usable_from == SkillUsableFrom.FRONT_ONLY


def test_skill_usable_from_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        SkillDefinition.model_validate(
            {
                "id": "bad_skill",
                "name": "Bad Skill",
                "category": "basic",
                "effort_cost": 0,
                "attack_type": "melee",
                "usable_from": "sideways",
                "accuracy": 90,
                "damage": 1,
            }
        )


def test_skill_damage_range_defaults_to_fixed_damage() -> None:
    skill = SkillDefinition.model_validate(
        {
            "id": "test_skill",
            "name": "Test Skill",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "accuracy": 90,
            "damage": 3,
        }
    )

    assert skill.damage == 3
    assert skill.damage_min is None
    assert skill.damage_max is None


def test_skill_damage_range_accepts_authored_min_and_max() -> None:
    skill = SkillDefinition.model_validate(
        {
            "id": "test_skill",
            "name": "Test Skill",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "accuracy": 90,
            "damage": 3,
            "damage_min": 2,
            "damage_max": 4,
        }
    )

    assert skill.damage_min == 2
    assert skill.damage_max == 4


def test_skill_damage_range_requires_both_bounds() -> None:
    with pytest.raises(ValidationError):
        SkillDefinition.model_validate(
            {
                "id": "test_skill",
                "name": "Test Skill",
                "category": "basic",
                "effort_cost": 0,
                "attack_type": "melee",
                "accuracy": 90,
                "damage": 3,
                "damage_min": 2,
            }
        )


def test_skill_damage_range_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValidationError):
        SkillDefinition.model_validate(
            {
                "id": "test_skill",
                "name": "Test Skill",
                "category": "basic",
                "effort_cost": 0,
                "attack_type": "melee",
                "accuracy": 90,
                "damage": 3,
                "damage_min": 4,
                "damage_max": 2,
            }
        )


def hero(
    actor_id: str,
    slot: FormationSlot,
    *,
    morale: MoraleState = MoraleState.STEADY,
    tags: set[Tag] | None = None,
    effort: int = 3,
) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id.title(),
        team=Team.HERO,
        max_hp=10,
        hp=10,
        speed=1,
        accuracy=0,
        defense=0,
        damage=0,
        max_effort=3,
        effort=effort,
        skills=["basic_strike", "stand_watch", "frost_jab", "rally", "soak", "spark_line"],
        formation_slot=slot,
        morale=morale,
        tags=set(tags or set()),
    )


def enemy(*, tags: set[Tag] | None = None, hp: int = 10) -> Combatant:
    return Combatant(
        actor_id="enemy",
        name="Enemy",
        team=Team.ENEMY,
        max_hp=10,
        hp=hp,
        speed=1,
        accuracy=0,
        defense=0,
        damage=0,
        max_effort=0,
        effort=0,
        skills=["rusted_chop"],
        formation_slot=FormationSlot.FRONT_LEFT,
        tags=set(tags or set()),
    )


def enemy_combatant(
    actor_id: str,
    slot: FormationSlot,
    *,
    hp: int = 10,
    tags: set[Tag] | None = None,
) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id.title(),
        team=Team.ENEMY,
        max_hp=10,
        hp=hp,
        speed=1,
        accuracy=0,
        defense=0,
        damage=0,
        max_effort=0,
        effort=0,
        skills=["rusted_chop"],
        formation_slot=slot,
        tags=set(tags or set()),
    )


def state(
    heroes: dict[str, Combatant] | None = None,
    enemies: dict[str, Combatant] | None = None,
) -> CombatState:
    heroes = heroes or {"mira": hero("mira", FormationSlot.FRONT_LEFT)}
    enemies = enemies or {"enemy": enemy()}
    party_formation = Formation.empty()
    enemy_formation = Formation.empty()
    for combatant in heroes.values():
        party_formation.place(combatant.actor_id, combatant.formation_slot)
    for combatant in enemies.values():
        enemy_formation.place(combatant.actor_id, combatant.formation_slot)
    return CombatState(
        heroes=heroes,
        enemies=enemies,
        party_formation=party_formation,
        enemy_formation=enemy_formation,
    )


def test_party_derives_strong_unsteady_and_fractured_cohesion() -> None:
    combat = state(
        heroes={
            "mira": hero("mira", FormationSlot.FRONT_LEFT),
            "cade": hero("cade", FormationSlot.FRONT_RIGHT),
            "rowan": hero("rowan", FormationSlot.BACK_LEFT),
            "elya": hero("elya", FormationSlot.BACK_RIGHT, morale=MoraleState.SHAKEN),
        }
    )
    assert combat.derive_cohesion() == CohesionState.UNSTEADY

    combat.heroes["rowan"].morale = MoraleState.SHAKEN
    assert combat.derive_cohesion() == CohesionState.FRACTURED

    combat.heroes["cade"].morale = MoraleState.BROKEN
    assert combat.derive_cohesion() == CohesionState.FRACTURED


def test_downed_hero_breaks_individual_morale() -> None:
    combat = state()

    apply_damage(combat, "enemy", "mira", 10)

    assert ActorStatus.DOWNED in combat.heroes["mira"].statuses
    assert combat.heroes["mira"].morale == MoraleState.BROKEN


def test_rally_raises_ally_morale_by_one_step_to_steady() -> None:
    combat = state(
        heroes={
            "mira": hero("mira", FormationSlot.FRONT_LEFT),
            "elya": hero("elya", FormationSlot.BACK_RIGHT, morale=MoraleState.SHAKEN),
        }
    )
    rally = SkillStub("rally", "Rally", 1, AttackType.MELEE, 100, 0, ["support", "rally"])

    result = use_skill(combat, "mira", rally, "elya", GameRng(1), ignore_target_legality=True)

    assert result.success
    assert combat.heroes["mira"].effort == 2
    assert combat.heroes["elya"].morale == MoraleState.STEADY


def test_inspire_rally_can_raise_ally_morale_to_inspired() -> None:
    combat = state(
        heroes={
            "mira": hero("mira", FormationSlot.FRONT_LEFT),
            "elya": hero("elya", FormationSlot.BACK_RIGHT, morale=MoraleState.STEADY),
        }
    )
    rally = SkillStub(
        "inspiring_word",
        "Inspiring Word",
        1,
        AttackType.MELEE,
        100,
        0,
        ["support", "rally", "inspire"],
    )

    result = use_skill(combat, "mira", rally, "elya", GameRng(1), ignore_target_legality=True)

    assert result.success
    assert combat.heroes["elya"].morale == MoraleState.INSPIRED


def test_guard_applies_guarded_tag_to_ally() -> None:
    combat = state(
        heroes={
            "rowan": hero("rowan", FormationSlot.FRONT_LEFT),
            "elya": hero("elya", FormationSlot.BACK_RIGHT),
        }
    )
    guard = SkillStub(
        "stand_watch",
        "Stand Watch",
        1,
        AttackType.MELEE,
        100,
        0,
        ["support", "guard"],
    )

    result = use_skill(combat, "rowan", guard, "elya", GameRng(1), ignore_target_legality=True)

    assert result.success
    assert Tag.GUARDED in combat.heroes["elya"].tags


def test_shielding_dead_can_guard_enemy_ally() -> None:
    combat = state(
        enemies={
            "bone": enemy_combatant("bone", FormationSlot.FRONT_LEFT),
            "acolyte": enemy_combatant("acolyte", FormationSlot.BACK_LEFT),
        }
    )
    shielding_dead = SkillStub(
        "shielding_dead",
        "Shielding Dead",
        0,
        AttackType.MELEE,
        100,
        0,
        ["enemy", "guard"],
    )

    result = use_skill(combat, "bone", shielding_dead, "acolyte", GameRng(1))

    assert result.success
    assert Tag.GUARDED in combat.enemies["acolyte"].tags


def test_guarded_reduces_next_incoming_damage_and_is_consumed() -> None:
    combat = state(
        heroes={"rowan": hero("rowan", FormationSlot.FRONT_LEFT, tags={Tag.GUARDED})}
    )

    apply_damage(combat, "enemy", "rowan", 5)

    assert combat.heroes["rowan"].hp == 8
    assert Tag.GUARDED not in combat.heroes["rowan"].tags


def test_soak_applies_wet_and_removes_burning() -> None:
    combat = state(enemies={"enemy": enemy(tags={Tag.BURNING})})
    soak = SkillStub("soak", "Soak", 1, AttackType.RANGED, 100, 0, ["soak"])

    result = use_skill(combat, "mira", soak, "enemy", GameRng(1))

    assert result.success
    assert Tag.WET in combat.enemies["enemy"].tags
    assert Tag.BURNING not in combat.enemies["enemy"].tags


def test_frost_jab_freezes_wet_targets_and_marks_dry_targets() -> None:
    frost_jab = SkillStub("frost_jab", "Frost Jab", 1, AttackType.MELEE, 100, 3, ["frost"])
    wet_combat = state(enemies={"enemy": enemy(tags={Tag.WET})})
    dry_combat = state(enemies={"enemy": enemy()})

    wet_result = use_skill(wet_combat, "mira", frost_jab, "enemy", GameRng(1))
    dry_result = use_skill(dry_combat, "mira", frost_jab, "enemy", GameRng(1))

    assert wet_result.success
    assert Tag.FROZEN in wet_combat.enemies["enemy"].tags
    assert Tag.WET not in wet_combat.enemies["enemy"].tags
    assert dry_result.success
    assert Tag.MARKED in dry_combat.enemies["enemy"].tags
    assert dry_combat.enemies["enemy"].tag_turns[Tag.MARKED] == 2


def test_basic_strike_hits_marked_for_bonus_damage_without_consuming() -> None:
    combat = state(enemies={"enemy": enemy(tags={Tag.MARKED})})
    apply_marked(combat.enemies["enemy"])
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, ["basic"])

    result = use_skill(combat, "mira", strike, "enemy", GameRng(1))

    assert result.success
    assert combat.enemies["enemy"].hp == 7
    assert Tag.MARKED in combat.enemies["enemy"].tags
    assert combat.enemies["enemy"].tag_turns[Tag.MARKED] == 2
    assert not any(
        getattr(event, "status", "") == Tag.MARKED.name.lower()
        and not getattr(event, "added", True)
        for event in result.events
    )


def test_fixed_damage_skill_still_deals_exact_damage() -> None:
    combat = state()
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, [])

    result = use_skill(combat, "mira", strike, "enemy", GameRng(1))

    assert result.success
    assert combat.enemies["enemy"].hp == 8


def test_range_damage_skill_rolls_within_configured_bounds() -> None:
    amounts: set[int] = set()
    skill = SkillDefinition.model_validate(
        {
            "id": "ranged_hit",
            "name": "Ranged Hit",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "accuracy": 100,
            "damage": 4,
            "damage_min": 3,
            "damage_max": 4,
        }
    )

    for seed in range(1, 20):
        combat = state()
        result = use_skill(combat, "mira", skill, "enemy", GameRng(seed))
        assert result.success
        amounts.add(10 - combat.enemies["enemy"].hp)

    assert amounts == {3, 4}


def test_seeded_rng_makes_range_damage_deterministic() -> None:
    skill = SkillDefinition.model_validate(
        {
            "id": "ranged_hit",
            "name": "Ranged Hit",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "accuracy": 100,
            "damage": 4,
            "damage_min": 3,
            "damage_max": 4,
        }
    )
    first = state()
    second = state()

    assert use_skill(first, "mira", skill, "enemy", GameRng(7)).success
    assert use_skill(second, "mira", skill, "enemy", GameRng(7)).success

    assert first.enemies["enemy"].hp == second.enemies["enemy"].hp


def test_marked_bonus_applies_after_base_damage_roll() -> None:
    combat = state(enemies={"enemy": enemy(tags={Tag.MARKED})})
    apply_marked(combat.enemies["enemy"])
    skill = SkillDefinition.model_validate(
        {
            "id": "ranged_basic",
            "name": "Ranged Basic",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "accuracy": 100,
            "damage": 3,
            "damage_min": 3,
            "damage_max": 3,
            "tags": ["basic"],
        }
    )

    result = use_skill(combat, "mira", skill, "enemy", GameRng(1))

    assert result.success
    assert combat.enemies["enemy"].hp == 6


def test_guard_reduces_final_range_damage() -> None:
    combat = state(heroes={"mira": hero("mira", FormationSlot.FRONT_LEFT, tags={Tag.GUARDED})})
    skill = SkillDefinition.model_validate(
        {
            "id": "ranged_hit",
            "name": "Ranged Hit",
            "category": "basic",
            "effort_cost": 0,
            "attack_type": "melee",
            "accuracy": 100,
            "damage": 5,
            "damage_min": 5,
            "damage_max": 5,
        }
    )

    result = use_skill(combat, "enemy", skill, "mira", GameRng(1))

    assert result.success
    assert combat.heroes["mira"].hp == 8
    assert Tag.GUARDED not in combat.heroes["mira"].tags


def test_vulnerable_bonus_punishes_marked_or_wounded_targets() -> None:
    marked_combat = state(enemies={"enemy": enemy(tags={Tag.MARKED})})
    apply_marked(marked_combat.enemies["enemy"])
    wounded_combat = state(enemies={"enemy": enemy(hp=8)})
    steady_combat = state(enemies={"enemy": enemy()})
    cheap_shot = SkillStub(
        "cheap_shot",
        "Cheap Shot",
        0,
        AttackType.RANGED,
        100,
        2,
        ["vulnerable_bonus"],
    )

    assert use_skill(marked_combat, "mira", cheap_shot, "enemy", GameRng(1)).success
    assert use_skill(wounded_combat, "mira", cheap_shot, "enemy", GameRng(1)).success
    assert use_skill(steady_combat, "mira", cheap_shot, "enemy", GameRng(1)).success

    assert marked_combat.enemies["enemy"].hp == 6
    assert Tag.MARKED in marked_combat.enemies["enemy"].tags
    assert marked_combat.enemies["enemy"].tag_turns[Tag.MARKED] == 2
    assert wounded_combat.enemies["enemy"].hp == 4
    assert steady_combat.enemies["enemy"].hp == 8


def test_exposed_bonus_punishes_unprotected_backliners() -> None:
    exposed_combat = state(
        enemies={"back": enemy_combatant("back", FormationSlot.BACK_LEFT)}
    )
    protected_combat = state(
        enemies={
            "front": enemy_combatant("front", FormationSlot.FRONT_LEFT),
            "back": enemy_combatant("back", FormationSlot.BACK_LEFT),
        }
    )
    exposed_cut = SkillStub(
        "exposed_cut",
        "On the Mark",
        1,
        AttackType.RANGED,
        100,
        2,
        ["exposed_bonus"],
    )

    assert use_skill(exposed_combat, "mira", exposed_cut, "back", GameRng(1)).success
    assert use_skill(protected_combat, "mira", exposed_cut, "back", GameRng(1)).success

    assert exposed_combat.enemies["back"].hp == 6
    assert protected_combat.enemies["back"].hp == 8


def test_effort_drain_removes_effort_and_heals_the_leech() -> None:
    combat = state(
        heroes={"mira": hero("mira", FormationSlot.FRONT_LEFT, effort=3)},
        enemies={"leech": enemy_combatant("leech", FormationSlot.FRONT_LEFT, hp=4)},
    )
    drain = SkillStub(
        "effort_drain",
        "Effort Drain",
        0,
        AttackType.MAGIC,
        100,
        1,
        ["effort_drain"],
    )

    result = use_skill(combat, "leech", drain, "mira", GameRng(1))

    assert result.success
    assert combat.heroes["mira"].hp == 9
    assert combat.heroes["mira"].effort == 2
    assert combat.enemies["leech"].hp == 5
    assert any(
        event.message == "Mira loses 1 Effort: 3 -> 2 Effort."
        for event in result.events
    )


def test_mark_skill_applies_marked_to_one_target() -> None:
    combat = state()
    mark = SkillStub(
        "mark_the_pattern",
        "Mark the Pattern",
        0,
        AttackType.MAGIC,
        100,
        0,
        ["mark"],
    )

    result = use_skill(combat, "enemy", mark, "mira", GameRng(1))

    assert result.success
    assert Tag.MARKED in combat.heroes["mira"].tags
    assert combat.heroes["mira"].tag_turns[Tag.MARKED] == 2


def test_mark_skill_refreshes_marked_duration() -> None:
    combat = state(heroes={"mira": hero("mira", FormationSlot.FRONT_LEFT, tags={Tag.MARKED})})
    combat.heroes["mira"].tag_turns[Tag.MARKED] = 1
    mark = SkillStub(
        "mark_the_pattern",
        "Mark the Pattern",
        0,
        AttackType.MAGIC,
        100,
        0,
        ["mark"],
    )

    result = use_skill(combat, "enemy", mark, "mira", GameRng(1))

    assert result.success
    assert combat.heroes["mira"].tag_turns[Tag.MARKED] == 2


def test_drag_forward_swaps_backline_target_with_front_lane_hero() -> None:
    combat = state(
        heroes={
            "front": hero("front", FormationSlot.FRONT_LEFT),
            "back": hero("back", FormationSlot.BACK_LEFT),
        }
    )
    drag_forward = SkillStub(
        "drag_forward",
        "Drag Forward",
        0,
        AttackType.RANGED,
        100,
        1,
        ["drag_forward"],
    )

    result = use_skill(combat, "enemy", drag_forward, "back", GameRng(1))

    assert result.success
    assert combat.party_formation.actor_at(FormationSlot.FRONT_LEFT) == "back"
    assert combat.party_formation.actor_at(FormationSlot.BACK_LEFT) == "front"
    assert combat.heroes["back"].formation_slot == FormationSlot.FRONT_LEFT
    assert combat.heroes["front"].formation_slot == FormationSlot.BACK_LEFT
    assert Tag.MARKED in combat.heroes["back"].tags
    assert combat.heroes["back"].tag_turns[Tag.MARKED] == 2


def test_drag_forward_refreshes_existing_mark() -> None:
    combat = state(
        heroes={
            "front": hero("front", FormationSlot.FRONT_LEFT),
            "back": hero("back", FormationSlot.BACK_LEFT),
        }
    )
    apply_marked(combat.heroes["back"])
    combat.heroes["back"].tag_turns[Tag.MARKED] = 1
    drag_forward = SkillStub(
        "drag_forward",
        "Drag Forward",
        0,
        AttackType.RANGED,
        100,
        1,
        ["drag_forward"],
    )

    result = use_skill(combat, "enemy", drag_forward, "back", GameRng(1))

    assert result.success
    assert Tag.MARKED in combat.heroes["back"].tags
    assert combat.heroes["back"].tag_turns[Tag.MARKED] == 2


def test_maw_slam_pays_off_marked_targets() -> None:
    definitions = get_definitions()
    combat = state(heroes={"mira": hero("mira", FormationSlot.FRONT_LEFT, tags={Tag.MARKED})})
    apply_marked(combat.heroes["mira"])
    combat.enemies["enemy"].damage = definitions.enemies["cave_maw_brute"].damage
    maw_slam = definitions.skills["maw_slam"].model_copy(
        update={
            "accuracy": 100,
            "damage": 3,
            "damage_min": 3,
            "damage_max": 3,
        }
    )

    result = use_skill(combat, "enemy", maw_slam, "mira", GameRng(1))

    assert result.success
    assert combat.heroes["mira"].hp == 3
    assert Tag.MARKED in combat.heroes["mira"].tags


def test_shove_back_swaps_front_target_with_backline_ally() -> None:
    combat = state(
        enemies={
            "front": enemy_combatant("front", FormationSlot.FRONT_LEFT),
            "back": enemy_combatant("back", FormationSlot.BACK_LEFT),
        }
    )
    shove = SkillStub(
        "shield_drive",
        "Watchman's Shove",
        1,
        AttackType.REACH,
        100,
        2,
        ["shove_back"],
    )

    result = use_skill(combat, "mira", shove, "front", GameRng(1))

    assert result.success
    assert combat.enemy_formation.actor_at(FormationSlot.BACK_LEFT) == "front"
    assert combat.enemy_formation.actor_at(FormationSlot.FRONT_LEFT) == "back"
    assert combat.enemies["front"].formation_slot == FormationSlot.BACK_LEFT
    assert combat.enemies["back"].formation_slot == FormationSlot.FRONT_LEFT


def test_abyssal_spark_marks_surviving_targets() -> None:
    combat = state()
    spark = SkillStub(
        "abyssal_spark",
        "Inkblack Mark",
        1,
        AttackType.MAGIC,
        100,
        2,
        ["mark"],
    )

    result = use_skill(combat, "mira", spark, "enemy", GameRng(1))

    assert result.success
    assert Tag.MARKED in combat.enemies["enemy"].tags
    assert combat.enemies["enemy"].tag_turns[Tag.MARKED] == 2


def test_spark_line_shocks_wet_targets_for_bonus_damage() -> None:
    combat = state(enemies={"enemy": enemy(tags={Tag.WET})})
    spark = SkillStub("spark_line", "Spark Line", 2, AttackType.MAGIC, 100, 2, ["shock"])

    result = use_skill(combat, "mira", spark, "enemy", GameRng(1))

    assert result.success
    assert combat.enemies["enemy"].hp == 6
    assert Tag.SHOCKED in combat.enemies["enemy"].tags


def test_frozen_heroes_cannot_reposition() -> None:
    combat = state(
        heroes={
            "mira": hero("mira", FormationSlot.FRONT_LEFT, tags={Tag.FROZEN}),
            "elya": hero("elya", FormationSlot.BACK_LEFT),
        }
    )
    session = ManualCombatSession(
        encounter_id="test",
        encounter_name="Test",
        state=combat,
        initiative=[InitiativeEntry(actor_id="mira", initiative=10)],
    )

    assert legal_move_slots(session) == []


def test_v01_enemy_roster_has_one_special_skill_each() -> None:
    definitions = get_definitions()

    assert definitions.skills["guard_strike"].name == "Guard Strike"
    assert definitions.skills["shield_drive"].name == "Watchman's Shove"
    assert definitions.skills["knife_work"].name == "Sleeve Knife"
    assert definitions.skills["staff_jab"].name == "Scribe's Staff"
    assert definitions.skills["exposed_cut"].name == "On the Mark"
    assert definitions.skills["exposed_cut"].description == (
        "Wherever they point, a dagger soon follows."
    )
    assert definitions.skills["exposed_cut"].effect_text == (
        "Throw a knife at one enemy. Deals extra damage to Marked, wounded, "
        "or exposed backline targets."
    )
    assert definitions.skills["bone_saw"].name == "Bone Saw"
    assert definitions.skills["bone_saw"].description == (
        "The best weapon the surgeon had was never meant to be one."
    )
    assert definitions.skills["bone_saw"].effect_text == (
        "Melee attack against one legal enemy using this skill's current targeting rules."
    )
    assert definitions.skills["emergency_stitch"].name == "Emergency Dressing"
    assert definitions.skills["emergency_stitch"].description == (
        "Fast cloth, hard pressure, and just enough time to keep breathing."
    )
    assert definitions.skills["emergency_stitch"].effect_text == (
        "Heal one living ally. Heals more if the ally is Downed or at half HP or lower."
    )
    assert definitions.skills["abyssal_spark"].name == "Inkblack Mark"
    assert definitions.skills["abyssal_spark"].description == (
        "The scribe draws a black sign the Maze seems to recognize."
    )
    assert definitions.skills["abyssal_spark"].effect_text == (
        "Magic attack against one enemy. Marks the target if it survives."
    )
    assert definitions.skills["shield_drive"].damage == 3
    assert "shove_back" in definitions.skills["shield_drive"].tags
    assert definitions.skills["exposed_cut"].damage == 3
    assert "vulnerable_bonus" in definitions.skills["exposed_cut"].tags
    assert "exposed_bonus" in definitions.skills["exposed_cut"].tags
    assert "brink_heal" in definitions.skills["emergency_stitch"].tags
    assert definitions.skills["abyssal_spark"].damage == 3
    assert "mark" in definitions.skills["abyssal_spark"].tags
    assert definitions.enemies["bone_soldier"].skills == ["rusted_chop", "shielding_dead"]
    assert definitions.enemies["skulker"].skills == ["skulker_stone", "cheap_shot"]
    assert definitions.enemies["maze_leech"].skills == ["glass_bite", "effort_drain"]
    assert definitions.enemies["maze_acolyte"].skills == ["black_pulse", "mark_the_pattern"]
    assert definitions.enemies["cave_maw_brute"].skills == ["maw_slam", "drag_forward"]
    assert definitions.enemies["glass_splinter"].skills == ["glass_bite", "splinter_mark"]
    assert definitions.enemies["pattern_ward"].skills == ["ward_pattern", "black_pulse"]
    assert definitions.enemies["breach_stalker"].skills == ["stalker_cut", "stalker_hook"]
    assert "mark" in definitions.skills["splinter_mark"].tags
    assert "guard" in definitions.skills["ward_pattern"].tags
    assert "vulnerable_bonus" in definitions.skills["stalker_cut"].tags
    assert "drag_forward" in definitions.skills["stalker_hook"].tags
    maw = Combatant(
        actor_id="maw",
        name="Maw",
        team=Team.ENEMY,
        max_hp=20,
        hp=20,
        speed=1,
        accuracy=0,
        defense=0,
        damage=definitions.enemies["cave_maw_brute"].damage,
        max_effort=4,
        effort=4,
        skills=definitions.enemies["cave_maw_brute"].skills,
        formation_slot=FormationSlot.FRONT_LEFT,
    )
    assert combatant_damage_range(definitions.skills["drag_forward"], maw) == (3, 4)
    assert combatant_damage_range(definitions.skills["maw_slam"], maw) == (5, 6)
    assert "vulnerable_bonus" in definitions.skills["maw_slam"].tags
    assert definitions.enemies["bandit_cutthroat"].skills == ["bandit_blade", "dirty_finish"]
    assert definitions.enemies["bandit_slinger"].skills == ["sling_stone", "pinning_shot"]
    assert definitions.enemies["bandit_lookout"].skills == ["lookout_poke", "spot_target"]
    assert definitions.enemies["wolf"].skills == ["wolf_bite", "pack_bite"]
    assert definitions.enemies["alpha_wolf"].skills == ["wolf_bite", "howl"]
    assert definitions.skills["maw_slam"].effort_cost == 0
    assert definitions.skills["black_pulse"].effort_cost == 0
    assert definitions.skills["drag_forward"].effort_cost == 1
    assert definitions.skills["mark_the_pattern"].effort_cost == 1
    assert definitions.skills["ward_pattern"].damage == 0
    assert "vulnerable_bonus" in definitions.skills["pinning_shot"].tags
    assert "mark" not in definitions.skills["pinning_shot"].tags
    assert [
        enemy.enemy_id for enemy in definitions.encounters["generated_maze_hunt"].enemies
    ] == ["cave_maw_brute", "pattern_ward"]
    assert [
        enemy.enemy_id for enemy in definitions.encounters["generated_maze_stalker"].enemies
    ] == ["breach_stalker", "pattern_ward", "glass_splinter"]
    wolf_pack = definitions.encounters["wolf_pack"]
    assert [
        (enemy.enemy_id, enemy.actor_id, enemy.formation_slot)
        for enemy in wolf_pack.enemies
    ] == [
        ("wolf", "wolf_1", FormationSlot.FRONT_LEFT),
        ("alpha_wolf", "wolf_2", FormationSlot.BACK_LEFT),
        ("wolf", "wolf_3", FormationSlot.FRONT_RIGHT),
    ]


def test_authored_damage_ranges_follow_hero_and_enemy_direction() -> None:
    definitions = get_definitions()

    for hero_class in definitions.hero_classes.values():
        actor = Combatant(
            actor_id=hero_class.id,
            name=hero_class.name,
            team=Team.HERO,
            max_hp=hero_class.max_hp,
            hp=hero_class.max_hp,
            speed=hero_class.speed,
            accuracy=hero_class.accuracy,
            defense=hero_class.defense,
            damage=hero_class.damage,
            max_effort=hero_class.max_effort,
            effort=hero_class.max_effort,
            skills=hero_class.skills,
            formation_slot=FormationSlot.FRONT_LEFT,
            class_id=hero_class.id,
        )
        for skill_id in hero_class.skills:
            skill = definitions.skills[skill_id]
            old_total = max(0, skill.damage + actor.damage)
            expected = (
                (max(0, old_total - 1), old_total)
                if skill.damage > 0
                else (old_total, old_total)
            )
            assert combatant_damage_range(skill, actor) == expected

    for enemy_definition in definitions.enemies.values():
        actor = Combatant(
            actor_id=enemy_definition.id,
            name=enemy_definition.name,
            team=Team.ENEMY,
            max_hp=enemy_definition.max_hp,
            hp=enemy_definition.max_hp,
            speed=enemy_definition.speed,
            accuracy=enemy_definition.accuracy,
            defense=enemy_definition.defense,
            damage=enemy_definition.damage,
            max_effort=enemy_definition.max_effort,
            effort=enemy_definition.max_effort,
            skills=enemy_definition.skills,
            formation_slot=enemy_definition.formation_slot,
            class_id=enemy_definition.id,
        )
        for skill_id in enemy_definition.skills:
            skill = definitions.skills[skill_id]
            old_total = max(0, skill.damage + actor.damage)
            expected = (
                (old_total, old_total + 1)
                if skill.damage > 0
                else (old_total, old_total)
            )
            assert combatant_damage_range(skill, actor) == expected
