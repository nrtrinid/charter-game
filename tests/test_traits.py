from dataclasses import dataclass

from game.campaign.company import (
    ExpeditionReportState,
    HeroReportSnapshot,
    ReportEventSignal,
    create_new_company,
)
from game.campaign.hero_memory import FreshMemoryState, RecentSignal
from game.campaign.memory import finalize_report_memory, record_report_event_signals
from game.campaign.town import recover_company
from game.combat.actions import use_skill
from game.combat.combat_state import (
    Combatant,
    CombatState,
    LifeState,
    MoraleState,
    StrainMark,
    StrainTier,
    Tag,
    Team,
    apply_marked,
)
from game.combat.formation import Formation, FormationSlot
from game.combat.targeting import AttackType
from game.combat.traits import (
    CONDITION_DRAINED,
    CONDITION_FRAYED,
    CONDITION_WINDED,
    QUIRK_BATTLE_RHYTHM,
    QUIRK_BLOOD_HOT,
    QUIRK_CLEAN_KILL,
    QUIRK_CLOSER,
    QUIRK_DESPERATE_FOCUS,
    QUIRK_GRIM_FINISH,
    QUIRK_HARD_LESSON,
    QUIRK_ICE_NERVES,
    QUIRK_KEEPS_COUNT,
    QUIRK_LAST_ANCHOR,
    QUIRK_MAZE_SIGHTED,
    QUIRK_NO_WASTE,
    QUIRK_PREDATOR,
    QUIRK_RED_WORK,
    QUIRK_STEADY_HAND,
    QUIRK_STEADY_VOICE,
    apply_combat_start_traits,
    has_condition,
)
from game.content.definitions import GameDefinitions
from game.core.events import CombatEffectEvent, MemorySignalEvent
from game.core.rng import GameRng
from game.expedition.maze_director import ScriptedMazeDirectorPolicy, choose_maze_recipe
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


def combatant(
    actor_id: str,
    *,
    team: Team = Team.HERO,
    slot: FormationSlot = FormationSlot.FRONT_LEFT,
    personal_quirk: str | None = None,
    quirks: list[str] | None = None,
    strain: StrainTier = StrainTier.STEADY,
    morale: MoraleState = MoraleState.STEADY,
    strain_marks: set[StrainMark] | None = None,
    tags: set[Tag] | None = None,
    hp: int = 10,
    effort: int = 3,
    defense: int = 0,
    class_id: str = "",
) -> Combatant:
    return Combatant(
        actor_id=actor_id,
        name=actor_id.title(),
        team=team,
        max_hp=10,
        hp=hp,
        speed=1,
        accuracy=0,
        defense=defense,
        damage=0,
        max_effort=3,
        effort=effort,
        morale=morale,
        skills=["basic_strike"],
        formation_slot=slot,
        personal_quirk=personal_quirk,
        quirks=list(quirks or []),
        strain=strain,
        strain_marks=set(strain_marks or set()),
        tags=set(tags or set()),
        class_id=class_id,
    )


def combat_state(
    heroes: dict[str, Combatant],
    enemies: dict[str, Combatant] | None = None,
) -> CombatState:
    enemies = enemies or {
        "enemy": combatant("enemy", team=Team.ENEMY, slot=FormationSlot.FRONT_LEFT)
    }
    party_formation = Formation.empty()
    enemy_formation = Formation.empty()
    for hero in heroes.values():
        party_formation.place(hero.actor_id, hero.formation_slot)
    for enemy in enemies.values():
        enemy_formation.place(enemy.actor_id, enemy.formation_slot)
    return CombatState(
        heroes=heroes,
        enemies=enemies,
        party_formation=party_formation,
        enemy_formation=enemy_formation,
    )


def snapshot(hero_id: str, *, effort: int = 3) -> HeroReportSnapshot:
    return HeroReportSnapshot(
        hero_id=hero_id,
        name=hero_id.title(),
        class_id="watchman",
        hp=10,
        max_hp=10,
        effort=effort,
        max_effort=3,
        mortal_wounds=0,
    )


def test_traits_load_and_starting_classes_have_personal_quirks() -> None:
    definitions = get_definitions()

    assert "hold_the_line" in definitions.personal_quirks
    assert "blood_hot" in definitions.earned_quirks
    assert "grim_finish" in definitions.earned_quirks
    assert "closer" in definitions.earned_quirks
    assert "winded" in definitions.conditions
    assert "spent" in definitions.conditions
    assert "exhausted" not in definitions.conditions
    assert definitions.hero_classes["watchman"].personal_quirk == "hold_the_line"


def test_combat_start_applies_personal_quirk_and_conditions() -> None:
    state = combat_state(
        {
            "rowan": combatant(
                "rowan",
                personal_quirk="hold_the_line",
                strain_marks={StrainMark.DRAINED, StrainMark.BATTERED},
                defense=2,
            ),
            "mira": combatant("mira", slot=FormationSlot.FRONT_RIGHT),
            "cade": combatant("cade", slot=FormationSlot.BACK_LEFT),
            "elya": combatant("elya", slot=FormationSlot.BACK_RIGHT),
        }
    )

    apply_combat_start_traits(state)

    assert state.heroes["rowan"].effort == 2
    assert state.heroes["rowan"].defense == 1
    assert Tag.GUARDED in state.heroes["rowan"].tags


def test_opportunist_and_blood_hot_hooks_apply_during_attack() -> None:
    state = combat_state(
        {
            "cade": combatant(
                "cade",
                personal_quirk="opportunist",
                quirks=[QUIRK_BLOOD_HOT],
                effort=1,
            )
        },
        enemies={
            "enemy": combatant(
                "enemy",
                team=Team.ENEMY,
                tags={Tag.MARKED},
                hp=3,
            )
        },
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, ["basic"])

    result = use_skill(state, "cade", strike, "enemy", GameRng(1))

    assert result.success
    assert not state.enemies["enemy"].is_alive()
    assert state.heroes["cade"].effort == 2
    assert any(
        isinstance(event, CombatEffectEvent)
        and event.actor_id == "cade"
        and event.resource == "effort"
        and event.delta == 1
        and event.before == 1
        and event.after == 2
        and event.source_kind == "quirk"
        and event.source_id == QUIRK_BLOOD_HOT
        for event in result.events
    )


def test_effort_drain_emits_typed_effort_effect() -> None:
    state = combat_state(
        {"cade": combatant("cade", effort=3)},
        enemies={"leech": combatant("leech", team=Team.ENEMY)},
    )
    drain = SkillStub("effort_drain", "Effort Drain", 0, AttackType.MELEE, 100, 0, ["effort_drain"])

    result = use_skill(state, "leech", drain, "cade", GameRng(1))

    assert result.success
    assert state.heroes["cade"].effort == 2
    assert any(
        isinstance(event, CombatEffectEvent)
        and event.actor_id == "cade"
        and event.resource == "effort"
        and event.delta == -1
        and event.before == 3
        and event.after == 2
        and event.source_kind == "skill"
        and event.source_id == "effort_drain"
        for event in result.events
    )


def test_guard_mitigation_emits_typed_combat_effect() -> None:
    state = combat_state(
        {"rowan": combatant("rowan", tags={Tag.GUARDED}, hp=10)},
        enemies={"enemy": combatant("enemy", team=Team.ENEMY)},
    )
    strike = SkillStub("strike", "Strike", 0, AttackType.MELEE, 100, 4, [])

    result = use_skill(state, "enemy", strike, "rowan", GameRng(1))

    assert result.success
    assert any(
        isinstance(event, CombatEffectEvent)
        and event.actor_id == "rowan"
        and event.effect_type == "mitigation"
        and event.delta == -3
        and event.label == "Guard -3"
        for event in result.events
    )


def test_opportunist_bonus_does_not_consume_marked() -> None:
    state = combat_state(
        {
            "cade": combatant(
                "cade",
                personal_quirk="opportunist",
            )
        },
        enemies={
            "enemy": combatant(
                "enemy",
                team=Team.ENEMY,
                tags={Tag.MARKED},
                hp=10,
            )
        },
    )
    apply_marked(state.enemies["enemy"])
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, ["basic"])

    result = use_skill(state, "cade", strike, "enemy", GameRng(1))

    assert result.success
    assert state.enemies["enemy"].hp == 6
    assert Tag.MARKED in state.enemies["enemy"].tags
    assert state.enemies["enemy"].tag_turns[Tag.MARKED] == 2


def test_marked_enemy_kill_emits_marked_execution_memory_signal() -> None:
    state = combat_state(
        {"cade": combatant("cade")},
        enemies={
            "enemy": combatant(
                "enemy",
                team=Team.ENEMY,
                tags={Tag.MARKED},
                hp=3,
            )
        },
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 3, ["basic"])

    result = use_skill(state, "cade", strike, "enemy", GameRng(1))

    assert result.success
    assert any(
        isinstance(event, MemorySignalEvent)
        and event.family_id == "marked_execution"
        and event.hero_id == "cade"
        for event in result.events
    )


def test_clean_kill_steadies_marked_kill_only_up_to_steady() -> None:
    marked_state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_CLEAN_KILL],
                morale=MoraleState.SHAKEN,
            )
        },
        enemies={
            "enemy": combatant("enemy", team=Team.ENEMY, tags={Tag.MARKED}, hp=3)
        },
    )
    unmarked_state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_CLEAN_KILL],
                morale=MoraleState.SHAKEN,
            )
        },
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=3)},
    )
    inspired_state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_CLEAN_KILL],
                morale=MoraleState.STEADY,
            )
        },
        enemies={
            "enemy": combatant("enemy", team=Team.ENEMY, tags={Tag.MARKED}, hp=3)
        },
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 3, ["basic"])

    assert use_skill(marked_state, "cade", strike, "enemy", GameRng(1)).success
    assert use_skill(unmarked_state, "cade", strike, "enemy", GameRng(1)).success
    assert use_skill(inspired_state, "cade", strike, "enemy", GameRng(1)).success

    assert marked_state.heroes["cade"].morale == MoraleState.STEADY
    assert unmarked_state.heroes["cade"].morale == MoraleState.SHAKEN
    assert inspired_state.heroes["cade"].morale == MoraleState.STEADY


def test_predator_rewards_marked_targets_and_penalizes_unmarked_targets() -> None:
    marked_state = combat_state(
        {"cade": combatant("cade", personal_quirk="opportunist", quirks=[QUIRK_PREDATOR])},
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, tags={Tag.MARKED}, hp=10)},
    )
    unmarked_state = combat_state(
        {"cade": combatant("cade", quirks=[QUIRK_PREDATOR])},
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=10)},
    )
    strike = SkillStub("knife_work", "Knife Work", 0, AttackType.MELEE, 100, 2, ["basic"])

    assert use_skill(marked_state, "cade", strike, "enemy", GameRng(1)).success
    assert use_skill(unmarked_state, "cade", strike, "enemy", GameRng(1)).success

    assert marked_state.enemies["enemy"].hp == 5
    assert unmarked_state.enemies["enemy"].hp == 9


def test_killing_blow_memory_signal_includes_context_tags() -> None:
    state = combat_state(
        {
            "cade": combatant(
                "cade",
                morale=MoraleState.STEADY,
                hp=7,
                strain_marks={StrainMark.BATTERED},
            )
        },
        enemies={
            "enemy": combatant("enemy", team=Team.ENEMY, hp=4, class_id="cave_mini_boss"),
        },
    )
    strike = SkillStub("power_strike", "Power Strike", 1, AttackType.MELEE, 100, 4, [])

    result = use_skill(state, "cade", strike, "enemy", GameRng(1))

    assert result.success
    signal = next(
        event
        for event in result.events
        if isinstance(event, MemorySignalEvent) and event.family_id == "killing_blow"
    )
    assert "final_kill" in signal.tags
    assert "steady" in signal.tags
    assert "wounded" in signal.tags
    assert "low_hp" in signal.tags
    assert "effort_kill" in signal.tags
    assert "boss" in signal.tags


def test_grim_finish_bonus_vs_half_hp_target() -> None:
    state = combat_state(
        {"cade": combatant("cade", quirks=[QUIRK_GRIM_FINISH])},
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=5)},
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, ["basic"])

    assert use_skill(state, "cade", strike, "enemy", GameRng(1)).success

    assert state.enemies["enemy"].hp == 2


def test_battle_rhythm_grants_guarded_on_first_kill_only() -> None:
    state = combat_state(
        {"cade": combatant("cade", quirks=[QUIRK_BATTLE_RHYTHM])},
        enemies={
            "enemy_a": combatant("enemy_a", team=Team.ENEMY, hp=2),
            "enemy_b": combatant("enemy_b", team=Team.ENEMY, slot=FormationSlot.FRONT_RIGHT, hp=2),
        },
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, ["basic"])

    assert use_skill(state, "cade", strike, "enemy_a", GameRng(1)).success
    assert Tag.GUARDED in state.heroes["cade"].tags

    state.heroes["cade"].tags.discard(Tag.GUARDED)
    assert use_skill(state, "cade", strike, "enemy_b", GameRng(1)).success
    assert Tag.GUARDED not in state.heroes["cade"].tags


def test_closer_raises_morale_on_final_kill_only() -> None:
    final_state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_CLOSER],
                morale=MoraleState.SHAKEN,
            )
        },
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=3)},
    )
    mid_state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_CLOSER],
                morale=MoraleState.SHAKEN,
            )
        },
        enemies={
            "enemy_a": combatant("enemy_a", team=Team.ENEMY, hp=3),
            "enemy_b": combatant("enemy_b", team=Team.ENEMY, slot=FormationSlot.FRONT_RIGHT, hp=10),
        },
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 3, ["basic"])

    assert use_skill(final_state, "cade", strike, "enemy", GameRng(1)).success
    assert use_skill(mid_state, "cade", strike, "enemy_a", GameRng(1)).success

    assert final_state.heroes["cade"].morale == MoraleState.STEADY
    assert mid_state.heroes["cade"].morale == MoraleState.SHAKEN


def test_no_waste_restores_effort_once_per_combat_on_basic_kill() -> None:
    state = combat_state(
        {"cade": combatant("cade", quirks=[QUIRK_NO_WASTE], effort=1)},
        enemies={
            "enemy_a": combatant("enemy_a", team=Team.ENEMY, hp=2),
            "enemy_b": combatant("enemy_b", team=Team.ENEMY, slot=FormationSlot.FRONT_RIGHT, hp=2),
        },
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, ["basic"])

    assert use_skill(state, "cade", strike, "enemy_a", GameRng(1)).success
    assert state.heroes["cade"].effort == 2

    assert use_skill(state, "cade", strike, "enemy_b", GameRng(1)).success
    assert state.heroes["cade"].effort == 2


def test_steady_hand_raises_morale_once_per_combat() -> None:
    state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_STEADY_HAND],
                morale=MoraleState.STEADY,
            )
        },
        enemies={
            "enemy_a": combatant("enemy_a", team=Team.ENEMY, hp=2),
            "enemy_b": combatant("enemy_b", team=Team.ENEMY, slot=FormationSlot.FRONT_RIGHT, hp=2),
        },
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 2, ["basic"])

    assert use_skill(state, "cade", strike, "enemy_a", GameRng(1)).success
    assert state.heroes["cade"].morale == MoraleState.INSPIRED

    assert use_skill(state, "cade", strike, "enemy_b", GameRng(1)).success
    assert state.heroes["cade"].morale == MoraleState.INSPIRED


def test_red_work_steadies_shaken_killer() -> None:
    state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_RED_WORK],
                morale=MoraleState.SHAKEN,
            )
        },
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=3)},
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 3, ["basic"])

    assert use_skill(state, "cade", strike, "enemy", GameRng(1)).success

    assert state.heroes["cade"].morale == MoraleState.STEADY


def test_hard_lesson_rewards_wounded_first_kill() -> None:
    state = combat_state(
        {
            "cade": combatant(
                "cade",
                quirks=[QUIRK_HARD_LESSON],
                morale=MoraleState.SHAKEN,
                hp=7,
                strain_marks={StrainMark.BATTERED},
            )
        },
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=3)},
    )
    strike = SkillStub("basic_strike", "Basic Strike", 0, AttackType.MELEE, 100, 3, ["basic"])

    assert use_skill(state, "cade", strike, "enemy", GameRng(1)).success

    assert state.heroes["cade"].morale == MoraleState.STEADY


def test_steady_voice_restores_effort_only_when_morale_rises() -> None:
    state = combat_state(
        {
            "mira": combatant("mira", quirks=[QUIRK_STEADY_VOICE], effort=1),
            "elya": combatant("elya", morale=MoraleState.SHAKEN),
        },
    )
    rally = SkillStub("rally", "Rally", 1, AttackType.MELEE, 100, 0, ["support", "rally"])

    first = use_skill(state, "mira", rally, "elya", GameRng(1), ignore_target_legality=True)
    second = use_skill(state, "mira", rally, "elya", GameRng(1), ignore_target_legality=True)

    assert first.success
    assert second.success
    assert state.heroes["elya"].morale == MoraleState.STEADY
    assert state.heroes["mira"].effort == 0


def test_desperate_focus_adds_damage_and_defense_risk_only_while_shaken() -> None:
    shaken_state = combat_state(
        {"mira": combatant("mira", quirks=[QUIRK_DESPERATE_FOCUS], morale=MoraleState.SHAKEN)},
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=10)},
    )
    steady_state = combat_state(
        {"mira": combatant("mira", quirks=[QUIRK_DESPERATE_FOCUS], morale=MoraleState.STEADY)},
        enemies={"enemy": combatant("enemy", team=Team.ENEMY, hp=10)},
    )
    strike = SkillStub("staff_jab", "Staff Jab", 0, AttackType.MELEE, 100, 2, ["weapon"])

    apply_combat_start_traits(shaken_state)
    apply_combat_start_traits(steady_state)
    assert use_skill(shaken_state, "mira", strike, "enemy", GameRng(1)).success
    assert use_skill(steady_state, "mira", strike, "enemy", GameRng(1)).success

    assert shaken_state.heroes["mira"].defense == -1
    assert steady_state.heroes["mira"].defense == 0
    assert shaken_state.enemies["enemy"].hp == 7
    assert steady_state.enemies["enemy"].hp == 8


def test_last_anchor_guards_only_when_fractured_and_steady() -> None:
    fractured = combat_state(
        {
            "rowan": combatant("rowan", quirks=[QUIRK_LAST_ANCHOR], morale=MoraleState.STEADY),
            "mira": combatant("mira", morale=MoraleState.SHAKEN),
            "cade": combatant("cade", morale=MoraleState.SHAKEN),
        }
    )
    not_fractured = combat_state(
        {
            "rowan": combatant("rowan", quirks=[QUIRK_LAST_ANCHOR], morale=MoraleState.STEADY),
            "mira": combatant("mira", morale=MoraleState.STEADY),
        }
    )
    broken_anchor = combat_state(
        {
            "rowan": combatant("rowan", quirks=[QUIRK_LAST_ANCHOR], morale=MoraleState.BROKEN),
            "mira": combatant("mira", morale=MoraleState.SHAKEN),
        }
    )

    apply_combat_start_traits(fractured)
    apply_combat_start_traits(not_fractured)
    apply_combat_start_traits(broken_anchor)

    assert Tag.GUARDED in fractured.heroes["rowan"].tags
    assert Tag.GUARDED not in not_fractured.heroes["rowan"].tags
    assert Tag.GUARDED not in broken_anchor.heroes["rowan"].tags


def test_keeps_count_restores_effort_once_when_ally_is_downed() -> None:
    state = combat_state(
        {
            "mira": combatant("mira", quirks=[QUIRK_KEEPS_COUNT], effort=1),
            "rowan": combatant("rowan", hp=3),
            "elya": combatant("elya", hp=3),
        },
        enemies={"enemy": combatant("enemy", team=Team.ENEMY)},
    )

    first = use_skill(
        state,
        "enemy",
        SkillStub("hit", "Hit", 0, AttackType.MELEE, 100, 3, ["enemy"]),
        "rowan",
        GameRng(1),
        ignore_target_legality=True,
    )
    second = use_skill(
        state,
        "enemy",
        SkillStub("hit", "Hit", 0, AttackType.MELEE, 100, 3, ["enemy"]),
        "elya",
        GameRng(1),
        ignore_target_legality=True,
    )

    assert first.success
    assert second.success
    assert state.heroes["mira"].effort == 2
    assert any(
        getattr(event, "family_id", "") == "ally_downed_witnessed"
        and getattr(event, "hero_id", "") == "mira"
        for event in first.events
    )


def test_ice_nerves_turns_would_be_frozen_into_marked() -> None:
    state = combat_state(
        {"mira": combatant("mira")},
        enemies={
            "enemy": combatant(
                "enemy",
                team=Team.ENEMY,
                quirks=[QUIRK_ICE_NERVES],
                tags={Tag.WET},
            )
        },
    )
    frost = SkillStub("frost_jab", "Frost Jab", 1, AttackType.MELEE, 100, 3, ["frost"])

    result = use_skill(state, "mira", frost, "enemy", GameRng(1))

    assert result.success
    assert Tag.FROZEN not in state.enemies["enemy"].tags
    assert Tag.MARKED in state.enemies["enemy"].tags


def test_report_records_memory_signal_events_with_participant_context() -> None:
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=["hero"],
    )

    record_report_event_signals(
        report,
        [
            MemorySignalEvent(
                message="Hero remembers the kill.",
                hero_id="hero",
                family_id="killing_blow",
                source_summary="Enemy fell.",
            )
        ],
    )

    assert len(report.memory_signals) == 1
    assert report.memory_signals[0].family_id == "killing_blow"
    assert report.memory_signals[0].order == 1


def test_post_expedition_assigns_condition_and_fresh_memory() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[0]
    hero.hp = 0
    hero.life_state = LifeState.DOWNED
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=[hero.hero_id],
        start_hero_states={hero.hero_id: snapshot(hero.hero_id)},
        event_signals=[
            ReportEventSignal(
                kind="killing_blow",
                message="Enemy fell.",
                hero_id=hero.hero_id,
                hero_name=hero.name,
            )
        ],
    )

    finalize_report_memory(company, report, "returned_to_haven")

    assert hero.strain_marks == {StrainMark.BATTERED}
    assert hero.quirks == []
    assert hero.fresh_memories[0].family_id == "killing_blow"
    assert hero.fresh_memories[0].intensity == 1
    assert any("Battered" in moment for moment in report.notable_moments)
    assert any("Killing Blow" in moment for moment in report.notable_moments)


def test_direct_memory_signals_do_not_double_count_with_legacy_event_signals() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[0]
    hero.fresh_memories = [
        FreshMemoryState(
            family_id="killing_blow",
            display_name="Killing Blow",
            intensity=1,
            created_order=1,
            refreshed_order=1,
        )
    ]
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=[hero.hero_id],
        start_hero_states={hero.hero_id: snapshot(hero.hero_id)},
        event_signals=[
            ReportEventSignal(
                kind="killing_blow",
                message="Legacy kill fact.",
                hero_id=hero.hero_id,
                hero_name=hero.name,
            )
        ],
        memory_signals=[
            RecentSignal(
                hero_id=hero.hero_id,
                family_id="killing_blow",
                tags=("kill", "combat"),
                source_summary="Direct kill fact.",
                order=1,
            )
        ],
    )

    finalize_report_memory(company, report, "returned_to_haven")

    assert hero.quirks == []
    assert hero.fresh_memories[0].family_id == "killing_blow"
    assert hero.fresh_memories[0].intensity == 2
    assert hero.career_signals["killing_blow"] == 1


def test_legacy_memory_derivation_runs_when_direct_memory_signals_are_absent() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[0]
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=[hero.hero_id],
        start_hero_states={hero.hero_id: snapshot(hero.hero_id)},
        event_signals=[
            ReportEventSignal(
                kind="killing_blow",
                message="Legacy kill fact.",
                hero_id=hero.hero_id,
                hero_name=hero.name,
            )
        ],
    )

    finalize_report_memory(company, report, "returned_to_haven")

    assert hero.fresh_memories[0].family_id == "killing_blow"
    assert hero.fresh_memories[0].intensity == 1
    assert hero.career_signals["killing_blow"] == 1


def test_routine_memory_families_cap_fresh_intensity_but_keep_career_counts() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[0]
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        outcome="returned_to_haven",
        participant_ids=[hero.hero_id],
        start_hero_states={hero.hero_id: snapshot(hero.hero_id)},
        memory_signals=[
            RecentSignal(
                hero_id=hero.hero_id,
                family_id="relic_greed",
                tags=("loot", "greed"),
                source_summary=f"Loot fact {index}.",
                order=index,
            )
            for index in range(1, 4)
        ],
    )

    finalize_report_memory(company, report, "returned_to_haven")

    assert hero.career_signals["relic_greed"] == 3
    assert hero.career_signals["tag:loot"] == 3
    assert hero.quirks == []
    assert hero.fresh_memories[0].family_id == "relic_greed"
    assert hero.fresh_memories[0].intensity == 1


def test_personal_action_memory_can_stack_to_manifestation_in_one_report() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[0]
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=[hero.hero_id],
        start_hero_states={hero.hero_id: snapshot(hero.hero_id)},
        memory_signals=[
            RecentSignal(
                hero_id=hero.hero_id,
                family_id="killing_blow",
                tags=("kill", "combat"),
                source_summary=f"Kill fact {index}.",
                order=index,
            )
            for index in range(1, 4)
        ],
    )

    finalize_report_memory(company, report, "returned_to_haven")

    assert hero.career_signals["killing_blow"] == 3
    assert hero.fresh_memories == []
    assert hero.quirks[0] in {QUIRK_BLOOD_HOT, QUIRK_GRIM_FINISH, QUIRK_BATTLE_RHYTHM}


def test_post_expedition_manifests_earned_quirk_at_three_intensity() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[0]
    hero.fresh_memories = [
        FreshMemoryState(
            family_id="frost_shock",
            display_name="Frost Shock",
            intensity=2,
            source_summary="The cold stayed.",
            created_order=1,
            refreshed_order=1,
        )
    ]
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=[hero.hero_id],
        start_hero_states={hero.hero_id: snapshot(hero.hero_id)},
        event_signals=[
            ReportEventSignal(
                kind="tag_frozen",
                message="Hero freezes solid.",
                hero_id=hero.hero_id,
                hero_name=hero.name,
            )
        ],
    )

    finalize_report_memory(company, report, "returned_to_haven")

    assert hero.fresh_memories == []
    assert hero.quirks == [QUIRK_ICE_NERVES]
    assert hero.earned_quirk_slots[0].quirk_id == QUIRK_ICE_NERVES
    assert any("Frost Shock" in moment for moment in report.notable_moments)
    assert any("developed Ice Nerves" in moment for moment in report.notable_moments)


def test_condition_escalates_and_reserves_recover_after_expedition() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    active = company.roster[0]
    reserve = company.roster[1]
    active.strain = StrainTier.WORN
    active.strain_marks = {StrainMark.WINDED}
    reserve.strain = StrainTier.WORN
    reserve.strain_marks = {StrainMark.DRAINED}
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=[active.hero_id],
        start_hero_states={active.hero_id: snapshot(active.hero_id, effort=3)},
    )
    active.effort = 0

    finalize_report_memory(company, report, "returned_to_haven")

    assert active.strain_marks == {StrainMark.WINDED, StrainMark.DRAINED}
    assert active.strain == StrainTier.WORN
    assert reserve.strain_marks == set()
    assert reserve.strain == StrainTier.STEADY


def test_spent_and_legacy_exhausted_apply_bundled_condition_behavior() -> None:
    for strain in (StrainTier.SPENT, StrainTier.SPENT):
        state = combat_state(
            {"rowan": combatant("rowan", strain=strain, effort=3)}
        )

        apply_combat_start_traits(state)

        assert state.heroes["rowan"].effort == 2
        assert state.heroes["rowan"].morale.name == "SHAKEN"
        assert has_condition(state.heroes["rowan"], CONDITION_WINDED)
        assert has_condition(state.heroes["rowan"], CONDITION_DRAINED)
        assert has_condition(state.heroes["rowan"], CONDITION_FRAYED)


def test_recovery_ward_clears_conditions() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    company.coin = definitions.town.recovery_cost
    company.roster[0].strain = StrainTier.WORN
    company.roster[0].strain_marks = {StrainMark.BATTERED}

    result = recover_company(company, definitions)

    assert result.success
    assert company.roster[0].strain == StrainTier.STEADY
    assert company.roster[0].strain_marks == set()


def test_maze_sighted_shortens_generated_route_when_possible() -> None:
    definitions: GameDefinitions = get_definitions()
    normal_company = create_new_company(definitions)
    sighted_company = create_new_company(definitions)
    sighted_company.roster[0].quirks = [QUIRK_MAZE_SIGHTED]
    policy = ScriptedMazeDirectorPolicy()

    normal = choose_maze_recipe(
        normal_company,
        source_node_id="maze_breach",
        run_number=1,
        rng=GameRng(5),
        policy=policy,
    )
    sighted = choose_maze_recipe(
        sighted_company,
        source_node_id="maze_breach",
        run_number=1,
        rng=GameRng(5),
        policy=policy,
    )

    assert sighted.route_length == max(3, normal.route_length - 1)
