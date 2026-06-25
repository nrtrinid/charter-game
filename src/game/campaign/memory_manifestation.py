"""Bridge expedition reports to hero_memory manifestation."""

from __future__ import annotations

from collections.abc import Sequence

from game.campaign.company import CompanyState, ExpeditionReportState
from game.campaign.hero_memory import (
    MEMORY_FAMILIES,
    ManifestationResult,
    RecentSignal,
    apply_signal_to_career,
    apply_signal_to_fresh_memories,
    flat_quirks_from_slots,
    manifest_pending_memories,
    memory_family,
)
from game.campaign.hero_state import HeroState
from game.campaign.memory_archive import _add_hero_memory
from game.campaign.memory_signals import _is_routine_fresh_signal, _signal_identity
from game.campaign.memory_util import _label
from game.core.rng import GameRng


def _apply_hero_memory_signals(
    company: CompanyState,
    hero: HeroState,
    report: ExpeditionReportState,
    signals: Sequence[RecentSignal],
    fresh_signal_allowlist: set[tuple[str, str, str | None, str | None, str, int]],
    rng: GameRng,
) -> list[str]:
    moments: list[str] = []
    routine_fresh_applied: set[str] = set()
    for signal in signals:
        hero.career_signals = apply_signal_to_career(hero.career_signals, signal)
        if signal.family_id not in MEMORY_FAMILIES:
            continue
        fresh_signal = signal
        if _is_routine_fresh_signal(signal):
            if signal.family_id in routine_fresh_applied:
                continue
            if _signal_identity(signal) not in fresh_signal_allowlist:
                continue
            routine_fresh_applied.add(signal.family_id)
            fresh_signal = RecentSignal(
                hero_id=signal.hero_id,
                family_id=signal.family_id,
                score=1,
                tags=signal.tags,
                source_summary=signal.source_summary,
                node_id=signal.node_id,
                encounter_id=signal.encounter_id,
                order=signal.order,
            )
        update = apply_signal_to_fresh_memories(hero.fresh_memories, fresh_signal)
        hero.fresh_memories = list(update.memories)
        family = memory_family(signal.family_id)
        if update.strengthened:
            moments.append(f"{hero.name}'s {family.display_name} deepens.")
        elif update.replaced_family_id:
            moments.append(f"{hero.name}'s {family.display_name} takes hold.")
        else:
            moments.append(f"{hero.name}'s {family.display_name} takes hold.")
        for summary in update.permanent_memory_summaries:
            hero_summary = _hero_permanent_memory_summary(hero, summary)
            _add_hero_memory(
                company,
                hero_id=hero.hero_id,
                hero_name=hero.name,
                kind="fresh_memory_settled",
                summary=hero_summary,
                report=report,
                node_id=signal.node_id,
                encounter_id=signal.encounter_id,
            )
            moments.append(hero_summary)

    fresh_memories, earned_slots, results = manifest_pending_memories(
        hero_id=hero.hero_id,
        hero_name=hero.name,
        fresh_memories=hero.fresh_memories,
        earned_slots=hero.earned_quirk_slots,
        career_signals=hero.career_signals,
        rng=rng,
    )
    hero.fresh_memories = fresh_memories
    hero.earned_quirk_slots = earned_slots
    hero.quirks = flat_quirks_from_slots(earned_slots)
    for result in results:
        record_summaries = _manifestation_record_summaries(hero, result)
        moments.extend(record_summaries)
        summaries = record_summaries or (
            (result.permanent_memory_summary,) if result.permanent_memory_summary else ()
        )
        for summary in summaries:
            _add_hero_memory(
                company,
                hero_id=hero.hero_id,
                hero_name=hero.name,
                kind=_manifestation_memory_kind(result.outcome),
                summary=summary,
                report=report,
            )
    return moments
def _hero_permanent_memory_summary(hero: HeroState, summary: str) -> str:
    if "permanent memory" in summary:
        return summary.replace("permanent memory", f"{hero.name}'s permanent memories")
    return f"{summary} ({hero.name})"


def _manifestation_memory_kind(outcome: str) -> str:
    if outcome == "reinforced":
        return "quirk_reinforced"
    if outcome in {"added", "replaced"}:
        return "quirk_manifested"
    return "fresh_memory_settled"


def _manifestation_record_summaries(
    hero: HeroState,
    result: ManifestationResult,
) -> tuple[str, ...]:
    outcome = getattr(result, "outcome", "")
    memory_name = getattr(result, "memory_display_name", "") or _label(
        getattr(result, "memory_family_id", "")
    )
    quirk_name = getattr(result, "quirk_display_name", "") or _label(
        getattr(result, "quirk_id", "")
    )
    if outcome == "added":
        return (f"{hero.name} developed {quirk_name} from repeated {memory_name}.",)
    if outcome == "reinforced":
        lines: tuple[str, ...] = (f"{quirk_name} deepened from another {memory_name}.",)
        if getattr(result, "locked", False):
            lines += (f"{quirk_name} became locked.",)
        return lines
    if outcome == "replaced":
        replaced_name = getattr(result, "replaced_quirk_display_name", "") or _label(
            getattr(result, "replaced_quirk_id", "")
        )
        return (f"{quirk_name} replaced {replaced_name} from {memory_name}.",)
    if outcome == "no_eligible_quirk":
        return (f"{memory_name} settled without a new quirk.",)
    if outcome == "all_locked":
        return (f"{memory_name} settled without a new quirk; all earned slots were locked.",)
    if outcome == "already_locked":
        return (f"{quirk_name} was already locked from another {memory_name}.",)
    return tuple(str(message) for message in getattr(result, "messages", ()) if message)


def _manifestation_seed(report: ExpeditionReportState) -> int:
    seed_text = "|".join(
        (
            report.expedition_id,
            report.dungeon_id or "",
            report.outcome,
            ",".join(report.participant_ids),
            ",".join(report.rooms_entered),
            str(len(report.event_signals)),
            str(len(report.memory_signals)),
        )
    )
    return sum((index + 1) * ord(character) for index, character in enumerate(seed_text))
