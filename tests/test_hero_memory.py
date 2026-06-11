from game.campaign.hero_memory import (
    EARNED_QUIRK_LIMIT,
    FRESH_MEMORY_LIMIT,
    FRESH_MEMORY_MAX_INTENSITY,
    MANIFESTABLE_EARNED_QUIRKS,
    STABILITY_LOCKED,
    STABILITY_LOOSE,
    STABILITY_SETTLED,
    EarnedQuirkSlotState,
    FreshMemoryState,
    RecentSignal,
    apply_quirk_manifestation,
    apply_signal_to_career,
    apply_signal_to_fresh_memories,
    choose_weighted_quirk,
    manifest_pending_memories,
    quirk_pool_for_memory,
)
from game.core.rng import GameRng


def signal(family_id: str, order: int, *, score: int = 1) -> RecentSignal:
    return RecentSignal(
        hero_id="hero",
        family_id=family_id,
        score=score,
        source_summary=f"{family_id} happened.",
        order=order,
    )


def memory(
    family_id: str,
    order: int,
    *,
    intensity: int = 1,
    pending: bool = False,
    tags: tuple[str, ...] = (),
) -> FreshMemoryState:
    return FreshMemoryState(
        family_id=family_id,
        display_name=family_id.replace("_", " ").title(),
        tags=tags,
        intensity=intensity,
        created_order=order,
        refreshed_order=order,
        pending_manifestation=pending,
    )


def test_adds_fresh_memory_and_updates_career_counter() -> None:
    update = apply_signal_to_fresh_memories([], signal("killing_blow", 1))
    career = apply_signal_to_career({}, signal("killing_blow", 1, score=2))

    assert len(update.memories) == 1
    assert update.memories[0].family_id == "killing_blow"
    assert update.memories[0].intensity == 1
    assert career["killing_blow"] == 2


def test_duplicate_memory_strengthens_refreshes_and_marks_pending_at_three() -> None:
    first = apply_signal_to_fresh_memories([], signal("marked_execution", 1))
    second = apply_signal_to_fresh_memories(
        list(first.memories),
        signal("marked_execution", 2, score=2),
    )

    strengthened = second.memories[0]
    assert second.strengthened
    assert strengthened.intensity == FRESH_MEMORY_MAX_INTENSITY
    assert strengthened.refreshed_order == 2
    assert strengthened.pending_manifestation


def test_fresh_memory_slots_cap_at_three_and_replace_oldest_non_pending() -> None:
    memories = [
        memory("killing_blow", 1),
        memory("relic_greed", 2),
        memory("frost_shock", 3),
    ]

    update = apply_signal_to_fresh_memories(memories, signal("breach_witness", 4))

    assert len(update.memories) == FRESH_MEMORY_LIMIT
    assert update.replaced_family_id == "killing_blow"
    assert {item.family_id for item in update.memories} == {
        "relic_greed",
        "frost_shock",
        "breach_witness",
    }
    assert update.permanent_memory_summaries


def test_pending_fresh_memory_is_not_overwritten_by_fifo() -> None:
    memories = [
        memory("killing_blow", 1, pending=True),
        memory("relic_greed", 2),
        memory("frost_shock", 3),
    ]

    update = apply_signal_to_fresh_memories(memories, signal("breach_witness", 4))

    assert "killing_blow" in {item.family_id for item in update.memories}
    assert update.replaced_family_id == "relic_greed"


def test_manifestation_fills_empty_slot() -> None:
    fresh = [memory("frost_shock", 1, intensity=3, pending=True)]

    remaining, slots, results = manifest_pending_memories(
        hero_id="hero",
        hero_name="Mara",
        fresh_memories=fresh,
        earned_slots=[],
        career_signals={},
        rng=GameRng(1),
    )

    assert remaining == []
    assert slots == [EarnedQuirkSlotState("ice_nerves", STABILITY_LOOSE, 1)]
    assert results[0].outcome == "added"
    assert results[0].memory_display_name == "Frost Shock"
    assert results[0].quirk_display_name == "Ice Nerves"
    assert results[0].messages == ("Mara developed Ice Nerves from Frost Shock.",)


def test_manifesting_same_quirk_reinforces_loose_to_settled_to_locked() -> None:
    fresh = [memory("frost_shock", 1, intensity=3, pending=True)]

    _remaining, slots, first = manifest_pending_memories(
        hero_id="hero",
        hero_name="Mara",
        fresh_memories=fresh,
        earned_slots=[EarnedQuirkSlotState("ice_nerves", STABILITY_LOOSE, 1)],
        career_signals={},
        rng=GameRng(1),
    )
    _remaining, locked_slots, second = manifest_pending_memories(
        hero_id="hero",
        hero_name="Mara",
        fresh_memories=fresh,
        earned_slots=slots,
        career_signals={},
        rng=GameRng(1),
    )

    assert first[0].outcome == "reinforced"
    assert slots[0].stability == STABILITY_SETTLED
    assert first[0].messages == ("Ice Nerves deepened from another Frost Shock.",)
    assert second[0].outcome == "reinforced"
    assert locked_slots[0].stability == STABILITY_LOCKED
    assert second[0].locked
    assert second[0].messages == (
        "Ice Nerves deepened from another Frost Shock.",
        "Ice Nerves became locked.",
    )


def test_full_slots_replace_oldest_non_locked_and_preserve_locked_slots() -> None:
    earned = [
        EarnedQuirkSlotState("old_loose", STABILITY_LOOSE, 1),
        EarnedQuirkSlotState("old_locked", STABILITY_LOCKED, 2),
        EarnedQuirkSlotState("old_settled", STABILITY_SETTLED, 3),
    ]

    slots, result = apply_quirk_manifestation(
        hero_id="hero",
        hero_name="Mara",
        memory=memory("frost_shock", 5, intensity=3, pending=True),
        quirk_id="ice_nerves",
        earned_slots=earned,
    )

    assert len(slots) == EARNED_QUIRK_LIMIT
    assert "old_locked" in {slot.quirk_id for slot in slots}
    assert "old_loose" not in {slot.quirk_id for slot in slots}
    assert result.outcome == "replaced"
    assert result.replaced_quirk_id == "old_loose"
    assert result.messages == ("Ice Nerves replaced Old Loose from Frost Shock.",)


def test_all_locked_manifestation_becomes_no_slot_change_record() -> None:
    earned = [
        EarnedQuirkSlotState("locked_a", STABILITY_LOCKED, 1),
        EarnedQuirkSlotState("locked_b", STABILITY_LOCKED, 2),
        EarnedQuirkSlotState("locked_c", STABILITY_LOCKED, 3),
    ]

    slots, result = apply_quirk_manifestation(
        hero_id="hero",
        hero_name="Mara",
        memory=memory("frost_shock", 5, intensity=3, pending=True),
        quirk_id="ice_nerves",
        earned_slots=earned,
    )

    assert slots == earned
    assert result.outcome == "all_locked"
    assert result.permanent_memory_summary


def test_weighted_draw_is_deterministic_with_game_rng() -> None:
    pool = (("alpha", 1), ("beta", 3), ("ignored", 0))

    first = choose_weighted_quirk(pool, GameRng(7))
    second = choose_weighted_quirk(pool, GameRng(7))

    assert first == second


def test_quirk_pool_filters_future_hook_quirks() -> None:
    relic_pool = quirk_pool_for_memory(memory("relic_greed", 1, intensity=3))
    maze_pool = quirk_pool_for_memory(memory("maze_thread", 1, intensity=3))
    marked_pool = quirk_pool_for_memory(memory("marked_execution", 1, intensity=3))
    treatment_pool = quirk_pool_for_memory(memory("field_treatment", 1, intensity=3))
    rally_pool = quirk_pool_for_memory(memory("morale_rally", 1, intensity=3))
    shaken_pool = quirk_pool_for_memory(memory("shaken_survival", 1, intensity=3))
    downed_pool = quirk_pool_for_memory(memory("downed_survival", 1, intensity=3))
    broken_pool = quirk_pool_for_memory(memory("broken_survival", 1, intensity=3))
    witnessed_pool = quirk_pool_for_memory(memory("ally_downed_witnessed", 1, intensity=3))

    assert relic_pool == (("gold_fever", 3),)
    assert maze_pool == ()
    assert marked_pool == (("clean_kill", 3), ("predator", 2))
    assert treatment_pool == (("field_medic", 3), ("steady_voice", 1))
    assert rally_pool == (("steady_voice", 1),)
    assert shaken_pool == (("desperate_focus", 1),)
    assert downed_pool == (("last_anchor", 1),)
    assert broken_pool == (("last_anchor", 1),)
    assert witnessed_pool == (("keeps_count", 1),)
    assert "loaded_pockets" not in {quirk_id for quirk_id, _weight in relic_pool}
    assert "thread_sense" not in {quirk_id for quirk_id, _weight in maze_pool}
    assert "bad_geometry" not in {quirk_id for quirk_id, _weight in maze_pool}
    for pool in (
        relic_pool,
        marked_pool,
        treatment_pool,
        rally_pool,
        shaken_pool,
        downed_pool,
        broken_pool,
        witnessed_pool,
    ):
        assert all(quirk_id in MANIFESTABLE_EARNED_QUIRKS for quirk_id, _weight in pool)


def test_future_hook_only_memory_settles_without_manifesting_slot() -> None:
    remaining, slots, results = manifest_pending_memories(
        hero_id="hero",
        hero_name="Mara",
        fresh_memories=[memory("maze_thread", 1, intensity=3, pending=True)],
        earned_slots=[],
        career_signals={},
        rng=GameRng(1),
    )

    assert remaining == []
    assert slots == []
    assert results[0].outcome == "no_eligible_quirk"
    assert results[0].memory_display_name == "Maze Thread"
    assert results[0].messages == ("Maze Thread settled without a new quirk.",)

    _remaining, breach_slots, breach_results = manifest_pending_memories(
        hero_id="hero",
        hero_name="Mara",
        fresh_memories=[memory("breach_witness", 1, intensity=3, pending=True)],
        earned_slots=[],
        career_signals={},
        rng=GameRng(1),
    )

    assert breach_slots == []
    assert breach_results[0].outcome == "no_eligible_quirk"
    assert breach_results[0].memory_display_name == "Breach Witness"


def test_killing_blow_base_pool_includes_basic_quirks() -> None:
    pool = dict(quirk_pool_for_memory(memory("killing_blow", 1, tags=("kill", "combat"))))

    assert pool == {"blood_hot": 2, "grim_finish": 2, "battle_rhythm": 1}


def test_killing_blow_conditional_quirks_require_tags() -> None:
    base = dict(quirk_pool_for_memory(memory("killing_blow", 1, tags=("kill", "combat"))))
    final = dict(
        quirk_pool_for_memory(
            memory("killing_blow", 1, tags=("kill", "combat", "final_kill")),
        )
    )
    basic = dict(
        quirk_pool_for_memory(
            memory("killing_blow", 1, tags=("kill", "combat", "basic")),
        )
    )

    assert "closer" not in base
    assert "no_waste" not in base
    assert final["closer"] == 3
    assert basic["no_waste"] == 3


def test_killing_blow_context_boosts_low_hp_and_effort_kill() -> None:
    pool = dict(
        quirk_pool_for_memory(
            memory(
                "killing_blow",
                1,
                tags=("kill", "combat", "low_hp", "effort_kill", "shaken"),
            )
        )
    )

    assert pool["grim_finish"] == 4
    assert pool["blood_hot"] == 5


def test_killing_blow_advanced_quirks_require_context_tags() -> None:
    steady = dict(
        quirk_pool_for_memory(memory("killing_blow", 1, tags=("kill", "combat", "steady")))
    )
    wounded = dict(
        quirk_pool_for_memory(memory("killing_blow", 1, tags=("kill", "combat", "wounded")))
    )
    pressure = dict(
        quirk_pool_for_memory(
            memory("killing_blow", 1, tags=("kill", "combat", "shaken", "boss")),
        )
    )

    assert steady["steady_hand"] == 1
    assert wounded["hard_lesson"] == 1
    assert pressure["red_work"] == 2


def test_killing_blow_career_boost_applies_with_multi_entry_pool() -> None:
    pool = dict(
        quirk_pool_for_memory(
            memory("killing_blow", 1, tags=("kill", "combat")),
            {"killing_blow": 6},
        )
    )

    assert pool == {"blood_hot": 3, "grim_finish": 3, "battle_rhythm": 2}
