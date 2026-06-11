"""Fresh memory and earned-quirk manifestation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game.core.rng import GameRng

FRESH_MEMORY_LIMIT = 3
FRESH_MEMORY_MAX_INTENSITY = 3
EARNED_QUIRK_LIMIT = 3
MANIFESTABLE_EARNED_QUIRKS = frozenset(
    {
        "blood_hot",
        "grim_finish",
        "battle_rhythm",
        "closer",
        "no_waste",
        "steady_hand",
        "red_work",
        "hard_lesson",
        "gold_fever",
        "ice_nerves",
        "clean_kill",
        "predator",
        "field_medic",
        "steady_voice",
        "desperate_focus",
        "last_anchor",
        "keeps_count",
    }
)

KILLING_BLOW_BASE_POOL: tuple[tuple[str, int], ...] = (
    ("blood_hot", 2),
    ("grim_finish", 2),
    ("battle_rhythm", 1),
)
KILLING_BLOW_CONDITIONAL_POOL: tuple[tuple[str, str, int], ...] = (
    ("closer", "final_kill", 3),
    ("no_waste", "basic", 3),
    ("steady_hand", "steady", 1),
    ("red_work", "shaken", 1),
    ("hard_lesson", "wounded", 1),
)

STABILITY_LOOSE = "loose"
STABILITY_SETTLED = "settled"
STABILITY_LOCKED = "locked"
QUIRK_STABILITIES = {STABILITY_LOOSE, STABILITY_SETTLED, STABILITY_LOCKED}


@dataclass(frozen=True)
class MemoryFamilyDefinition:
    family_id: str
    display_name: str
    tags: tuple[str, ...]
    quirk_pool: tuple[tuple[str, int], ...]
    archival: bool = True


MEMORY_FAMILIES: dict[str, MemoryFamilyDefinition] = {
    "killing_blow": MemoryFamilyDefinition(
        family_id="killing_blow",
        display_name="Killing Blow",
        tags=("kill", "combat"),
        quirk_pool=KILLING_BLOW_BASE_POOL,
    ),
    "marked_execution": MemoryFamilyDefinition(
        family_id="marked_execution",
        display_name="Marked Execution",
        tags=("kill", "marked", "combat"),
        quirk_pool=(("clean_kill", 3), ("predator", 2)),
    ),
    "relic_greed": MemoryFamilyDefinition(
        family_id="relic_greed",
        display_name="Relic Greed",
        tags=("loot", "greed"),
        quirk_pool=(("gold_fever", 3), ("loaded_pockets", 2)),
    ),
    "frost_shock": MemoryFamilyDefinition(
        family_id="frost_shock",
        display_name="Frost Shock",
        tags=("frozen", "shock", "combat"),
        quirk_pool=(("ice_nerves", 1),),
    ),
    "breach_witness": MemoryFamilyDefinition(
        family_id="breach_witness",
        display_name="Breach Witness",
        tags=("breach", "maze"),
        quirk_pool=(("thread_sense", 3), ("bad_geometry", 1)),
    ),
    "maze_thread": MemoryFamilyDefinition(
        family_id="maze_thread",
        display_name="Maze Thread",
        tags=("maze", "route"),
        quirk_pool=(("thread_sense", 2), ("bad_geometry", 2)),
    ),
    "field_treatment": MemoryFamilyDefinition(
        family_id="field_treatment",
        display_name="Field Treatment",
        tags=("support", "healing"),
        quirk_pool=(("field_medic", 3), ("steady_voice", 1)),
    ),
    "morale_rally": MemoryFamilyDefinition(
        family_id="morale_rally",
        display_name="Morale Rally",
        tags=("support", "morale"),
        quirk_pool=(("steady_voice", 1),),
    ),
    "shaken_survival": MemoryFamilyDefinition(
        family_id="shaken_survival",
        display_name="Shaken Survival",
        tags=("morale", "survival"),
        quirk_pool=(("desperate_focus", 1),),
    ),
    "downed_survival": MemoryFamilyDefinition(
        family_id="downed_survival",
        display_name="Downed Survival",
        tags=("downed", "survival"),
        quirk_pool=(("last_anchor", 1),),
    ),
    "broken_survival": MemoryFamilyDefinition(
        family_id="broken_survival",
        display_name="Broken Survival",
        tags=("morale", "survival"),
        quirk_pool=(("last_anchor", 1),),
    ),
    "ally_downed_witnessed": MemoryFamilyDefinition(
        family_id="ally_downed_witnessed",
        display_name="Ally Downed Witnessed",
        tags=("downed", "ally", "morale"),
        quirk_pool=(("keeps_count", 1),),
    ),
}


@dataclass
class RecentSignal:
    hero_id: str
    family_id: str
    score: int = 1
    tags: tuple[str, ...] = ()
    source_summary: str = ""
    node_id: str | None = None
    encounter_id: str | None = None
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "family_id": self.family_id,
            "score": self.score,
            "tags": list(self.tags),
            "source_summary": self.source_summary,
            "node_id": self.node_id,
            "encounter_id": self.encounter_id,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecentSignal:
        return cls(
            hero_id=str(data["hero_id"]),
            family_id=str(data["family_id"]),
            score=int(data.get("score", 1)),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            source_summary=str(data.get("source_summary", "")),
            node_id=None if data.get("node_id") is None else str(data["node_id"]),
            encounter_id=(
                None if data.get("encounter_id") is None else str(data["encounter_id"])
            ),
            order=int(data.get("order", 0)),
        )


@dataclass
class FreshMemoryState:
    family_id: str
    display_name: str
    tags: tuple[str, ...] = ()
    intensity: int = 1
    source_summary: str = ""
    created_order: int = 0
    refreshed_order: int = 0
    pending_manifestation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "display_name": self.display_name,
            "tags": list(self.tags),
            "intensity": self.intensity,
            "source_summary": self.source_summary,
            "created_order": self.created_order,
            "refreshed_order": self.refreshed_order,
            "pending_manifestation": self.pending_manifestation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FreshMemoryState:
        family_id = str(data["family_id"])
        family = memory_family(family_id)
        return cls(
            family_id=family_id,
            display_name=str(data.get("display_name") or family.display_name),
            tags=tuple(str(tag) for tag in data.get("tags", family.tags)),
            intensity=max(1, min(FRESH_MEMORY_MAX_INTENSITY, int(data.get("intensity", 1)))),
            source_summary=str(data.get("source_summary", "")),
            created_order=int(data.get("created_order", 0)),
            refreshed_order=int(data.get("refreshed_order", data.get("created_order", 0))),
            pending_manifestation=bool(data.get("pending_manifestation", False)),
        )


@dataclass
class EarnedQuirkSlotState:
    quirk_id: str
    stability: str = STABILITY_LOOSE
    unlocked_order: int = 0

    def __post_init__(self) -> None:
        if self.stability not in QUIRK_STABILITIES:
            self.stability = STABILITY_LOOSE

    def to_dict(self) -> dict[str, Any]:
        return {
            "quirk_id": self.quirk_id,
            "stability": self.stability,
            "unlocked_order": self.unlocked_order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EarnedQuirkSlotState:
        return cls(
            quirk_id=str(data["quirk_id"]),
            stability=str(data.get("stability", STABILITY_LOOSE)),
            unlocked_order=int(data.get("unlocked_order", 0)),
        )


@dataclass(frozen=True)
class FreshMemoryUpdate:
    memories: tuple[FreshMemoryState, ...]
    permanent_memory_summaries: tuple[str, ...] = ()
    changed_family_id: str = ""
    strengthened: bool = False
    replaced_family_id: str = ""


@dataclass(frozen=True)
class ManifestationResult:
    hero_id: str
    memory_family_id: str
    quirk_id: str
    outcome: str
    messages: tuple[str, ...] = ()
    permanent_memory_summary: str = ""
    memory_display_name: str = ""
    quirk_display_name: str = ""
    replaced_quirk_id: str = ""
    replaced_quirk_display_name: str = ""
    locked: bool = False


def memory_family(family_id: str) -> MemoryFamilyDefinition:
    existing = MEMORY_FAMILIES.get(family_id)
    if existing is not None:
        return existing
    display = family_id.replace("_", " ").title()
    return MemoryFamilyDefinition(
        family_id=family_id,
        display_name=display,
        tags=(),
        quirk_pool=(),
        archival=False,
    )


def fresh_memory_from_signal(signal: RecentSignal) -> FreshMemoryState:
    family = memory_family(signal.family_id)
    intensity = max(1, min(FRESH_MEMORY_MAX_INTENSITY, signal.score))
    return FreshMemoryState(
        family_id=signal.family_id,
        display_name=family.display_name,
        tags=tuple(sorted({*family.tags, *signal.tags})),
        intensity=intensity,
        source_summary=signal.source_summary,
        created_order=signal.order,
        refreshed_order=signal.order,
        pending_manifestation=intensity >= FRESH_MEMORY_MAX_INTENSITY,
    )


def apply_signal_to_career(
    career_signals: dict[str, int],
    signal: RecentSignal,
) -> dict[str, int]:
    score = max(1, signal.score)
    updated = dict(career_signals)
    updated[signal.family_id] = updated.get(signal.family_id, 0) + score
    for tag in signal.tags:
        key = f"tag:{tag}"
        updated[key] = updated.get(key, 0) + score
    return updated


def apply_signal_to_fresh_memories(
    fresh_memories: list[FreshMemoryState],
    signal: RecentSignal,
) -> FreshMemoryUpdate:
    memories = [FreshMemoryState.from_dict(memory.to_dict()) for memory in fresh_memories]
    for index, memory in enumerate(memories):
        if memory.family_id != signal.family_id:
            continue
        family = memory_family(signal.family_id)
        intensity = min(
            FRESH_MEMORY_MAX_INTENSITY,
            memory.intensity + max(1, signal.score),
        )
        memories[index] = FreshMemoryState(
            family_id=memory.family_id,
            display_name=memory.display_name or family.display_name,
            tags=tuple(sorted({*memory.tags, *family.tags, *signal.tags})),
            intensity=intensity,
            source_summary=signal.source_summary or memory.source_summary,
            created_order=memory.created_order,
            refreshed_order=signal.order,
            pending_manifestation=memory.pending_manifestation
            or intensity >= FRESH_MEMORY_MAX_INTENSITY,
        )
        return FreshMemoryUpdate(
            memories=tuple(memories),
            changed_family_id=signal.family_id,
            strengthened=True,
        )

    new_memory = fresh_memory_from_signal(signal)
    if len(memories) < FRESH_MEMORY_LIMIT:
        memories.append(new_memory)
        return FreshMemoryUpdate(
            memories=tuple(memories),
            changed_family_id=signal.family_id,
        )

    replaceable = [
        (index, memory)
        for index, memory in enumerate(memories)
        if not memory.pending_manifestation
    ]
    if not replaceable:
        return FreshMemoryUpdate(memories=tuple(memories), changed_family_id=signal.family_id)

    replace_index, replaced = min(
        replaceable,
        key=lambda item: (item[1].refreshed_order, item[1].created_order),
    )
    memories[replace_index] = new_memory
    permanent: tuple[str, ...] = ()
    if _should_archive(replaced):
        permanent = (_settled_memory_summary(replaced),)
    return FreshMemoryUpdate(
        memories=tuple(memories),
        permanent_memory_summaries=permanent,
        changed_family_id=signal.family_id,
        replaced_family_id=replaced.family_id,
    )


def choose_weighted_quirk(
    weighted_quirks: tuple[tuple[str, int], ...],
    rng: GameRng,
) -> str:
    candidates = tuple((quirk_id, weight) for quirk_id, weight in weighted_quirks if weight > 0)
    if not candidates:
        raise ValueError("Cannot choose a quirk from an empty weighted pool.")
    total = sum(weight for _quirk_id, weight in candidates)
    roll = rng.randint(1, total)
    running = 0
    for quirk_id, weight in candidates:
        running += weight
        if roll <= running:
            return quirk_id
    return candidates[-1][0]


def _killing_blow_quirk_pool(
    memory: FreshMemoryState,
    career_signals: dict[str, int] | None,
) -> tuple[tuple[str, int], ...]:
    tags = set(memory.tags)
    weights: dict[str, int] = {
        quirk_id: weight
        for quirk_id, weight in KILLING_BLOW_BASE_POOL
        if quirk_id in MANIFESTABLE_EARNED_QUIRKS
    }
    for quirk_id, required_tag, weight in KILLING_BLOW_CONDITIONAL_POOL:
        if quirk_id not in MANIFESTABLE_EARNED_QUIRKS:
            continue
        if required_tag not in tags:
            continue
        weights[quirk_id] = weight
    if "fractured" in tags and "red_work" in MANIFESTABLE_EARNED_QUIRKS:
        weights["red_work"] = max(weights.get("red_work", 0), 1)
    if "boss" in tags and "red_work" in MANIFESTABLE_EARNED_QUIRKS:
        weights["red_work"] = weights.get("red_work", 0) + 1
    if "low_hp" in tags:
        weights["grim_finish"] = weights.get("grim_finish", 0) + 2
    if "effort_kill" in tags:
        weights["blood_hot"] = weights.get("blood_hot", 0) + 2
    if "shaken" in tags:
        weights["blood_hot"] = weights.get("blood_hot", 0) + 1
    pool = tuple((quirk_id, weight) for quirk_id, weight in weights.items() if weight > 0)
    if career_signals is None:
        return pool
    career_score = career_signals.get(memory.family_id, 0)
    if career_score < FRESH_MEMORY_MAX_INTENSITY * 2:
        return pool
    return tuple((quirk_id, weight + 1) for quirk_id, weight in pool)


def quirk_pool_for_memory(
    memory: FreshMemoryState,
    career_signals: dict[str, int] | None = None,
) -> tuple[tuple[str, int], ...]:
    if memory.family_id == "killing_blow":
        return _killing_blow_quirk_pool(memory, career_signals)
    pool = tuple(
        (quirk_id, weight)
        for quirk_id, weight in memory_family(memory.family_id).quirk_pool
        if quirk_id in MANIFESTABLE_EARNED_QUIRKS
    )
    if career_signals is None:
        return pool
    career_score = career_signals.get(memory.family_id, 0)
    if career_score < FRESH_MEMORY_MAX_INTENSITY * 2:
        return pool
    return tuple((quirk_id, weight + 1) for quirk_id, weight in pool)


def manifest_pending_memories(
    *,
    hero_id: str,
    hero_name: str,
    fresh_memories: list[FreshMemoryState],
    earned_slots: list[EarnedQuirkSlotState],
    career_signals: dict[str, int],
    rng: GameRng,
) -> tuple[list[FreshMemoryState], list[EarnedQuirkSlotState], list[ManifestationResult]]:
    memories = [FreshMemoryState.from_dict(memory.to_dict()) for memory in fresh_memories]
    slots = [EarnedQuirkSlotState.from_dict(slot.to_dict()) for slot in earned_slots]
    results: list[ManifestationResult] = []
    pending = sorted(
        [memory for memory in memories if memory.pending_manifestation],
        key=lambda memory: (memory.refreshed_order, memory.created_order, memory.family_id),
    )
    for memory in pending:
        pool = quirk_pool_for_memory(memory, career_signals)
        if not pool:
            results.append(
                ManifestationResult(
                    hero_id=hero_id,
                    memory_family_id=memory.family_id,
                    quirk_id="",
                    outcome="no_eligible_quirk",
                    messages=(f"{memory.display_name} settled without a new quirk.",),
                    memory_display_name=memory.display_name,
                    permanent_memory_summary=_manifested_memory_summary(hero_name, memory),
                )
            )
            memories = [existing for existing in memories if existing is not memory]
            continue
        quirk_id = choose_weighted_quirk(pool, rng)
        slots, result = apply_quirk_manifestation(
            hero_id=hero_id,
            hero_name=hero_name,
            memory=memory,
            quirk_id=quirk_id,
            earned_slots=slots,
        )
        results.append(result)
        memories = [existing for existing in memories if existing is not memory]
    return memories, slots, results


def apply_quirk_manifestation(
    *,
    hero_id: str,
    hero_name: str,
    memory: FreshMemoryState,
    quirk_id: str,
    earned_slots: list[EarnedQuirkSlotState],
) -> tuple[list[EarnedQuirkSlotState], ManifestationResult]:
    slots = [EarnedQuirkSlotState.from_dict(slot.to_dict()) for slot in earned_slots]
    for index, slot in enumerate(slots):
        if slot.quirk_id != quirk_id:
            continue
        if slot.stability == STABILITY_LOCKED:
            return slots, ManifestationResult(
                hero_id=hero_id,
                memory_family_id=memory.family_id,
                quirk_id=quirk_id,
                outcome="already_locked",
                messages=(
                    f"{_label(quirk_id)} was already locked from another {memory.display_name}.",
                ),
                permanent_memory_summary=_manifested_memory_summary(hero_name, memory),
                memory_display_name=memory.display_name,
                quirk_display_name=_label(quirk_id),
                locked=True,
            )
        new_stability = (
            STABILITY_SETTLED if slot.stability == STABILITY_LOOSE else STABILITY_LOCKED
        )
        slots[index] = EarnedQuirkSlotState(
            quirk_id=slot.quirk_id,
            stability=new_stability,
            unlocked_order=slot.unlocked_order,
        )
        quirk_name = _label(quirk_id)
        messages = [f"{quirk_name} deepened from another {memory.display_name}."]
        if new_stability == STABILITY_LOCKED:
            messages.append(f"{quirk_name} became locked.")
        return slots, ManifestationResult(
            hero_id=hero_id,
            memory_family_id=memory.family_id,
            quirk_id=quirk_id,
            outcome="reinforced",
            messages=tuple(messages),
            permanent_memory_summary=_manifested_memory_summary(hero_name, memory),
            memory_display_name=memory.display_name,
            quirk_display_name=quirk_name,
            locked=new_stability == STABILITY_LOCKED,
        )

    new_slot = EarnedQuirkSlotState(
        quirk_id=quirk_id,
        stability=STABILITY_LOOSE,
        unlocked_order=_next_slot_order(slots),
    )
    if len(slots) < EARNED_QUIRK_LIMIT:
        slots.append(new_slot)
        quirk_name = _label(quirk_id)
        return slots, ManifestationResult(
            hero_id=hero_id,
            memory_family_id=memory.family_id,
            quirk_id=quirk_id,
            outcome="added",
            messages=(f"{hero_name} developed {quirk_name} from {memory.display_name}.",),
            permanent_memory_summary=_manifested_memory_summary(hero_name, memory),
            memory_display_name=memory.display_name,
            quirk_display_name=quirk_name,
        )

    replaceable = [
        (index, slot)
        for index, slot in enumerate(slots)
        if slot.stability != STABILITY_LOCKED
    ]
    if not replaceable:
        return slots, ManifestationResult(
            hero_id=hero_id,
            memory_family_id=memory.family_id,
            quirk_id=quirk_id,
            outcome="all_locked",
            messages=(
                f"{memory.display_name} settled into {hero_name}'s permanent memories.",
            ),
            permanent_memory_summary=_manifested_memory_summary(hero_name, memory),
            memory_display_name=memory.display_name,
            quirk_display_name=_label(quirk_id),
        )

    replace_index, replaced = min(replaceable, key=lambda item: item[1].unlocked_order)
    slots[replace_index] = new_slot
    quirk_name = _label(quirk_id)
    replaced_name = _label(replaced.quirk_id)
    return slots, ManifestationResult(
        hero_id=hero_id,
        memory_family_id=memory.family_id,
        quirk_id=quirk_id,
        outcome="replaced",
        messages=(
            f"{quirk_name} replaced {replaced_name} from {memory.display_name}.",
        ),
        permanent_memory_summary=_manifested_memory_summary(hero_name, memory),
        memory_display_name=memory.display_name,
        quirk_display_name=quirk_name,
        replaced_quirk_id=replaced.quirk_id,
        replaced_quirk_display_name=replaced_name,
    )


def synthesize_earned_slots(
    flat_quirks: list[str],
    raw_slots: list[EarnedQuirkSlotState] | None = None,
) -> list[EarnedQuirkSlotState]:
    if raw_slots:
        slots = [EarnedQuirkSlotState.from_dict(slot.to_dict()) for slot in raw_slots]
        existing = {slot.quirk_id for slot in slots}
        for quirk_id in flat_quirks:
            if quirk_id not in existing:
                slots.append(
                    EarnedQuirkSlotState(
                        quirk_id=quirk_id,
                        stability=STABILITY_SETTLED,
                        unlocked_order=_next_slot_order(slots),
                    )
                )
        return slots[:EARNED_QUIRK_LIMIT]
    return [
        EarnedQuirkSlotState(
            quirk_id=quirk_id,
            stability=STABILITY_SETTLED,
            unlocked_order=index,
        )
        for index, quirk_id in enumerate(flat_quirks[:EARNED_QUIRK_LIMIT], start=1)
    ]


def flat_quirks_from_slots(slots: list[EarnedQuirkSlotState]) -> list[str]:
    return [slot.quirk_id for slot in slots]


def _should_archive(memory: FreshMemoryState) -> bool:
    return memory.intensity > 1 or memory_family(memory.family_id).archival


def _settled_memory_summary(memory: FreshMemoryState) -> str:
    return f"{memory.display_name} faded into permanent memory."


def _manifested_memory_summary(hero_name: str, memory: FreshMemoryState) -> str:
    return f"{memory.display_name} settled into {hero_name}'s permanent memories."


def _next_slot_order(slots: list[EarnedQuirkSlotState]) -> int:
    return max((slot.unlocked_order for slot in slots), default=0) + 1


def _label(value: str) -> str:
    return value.replace("_", " ").title()
