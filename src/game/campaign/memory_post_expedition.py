"""Post-expedition strain, morale, and trait application."""

from __future__ import annotations

from game.campaign.company import CompanyState, ExpeditionReportState, HeroReportOutcome
from game.campaign.hero_state import HeroState
from game.campaign.memory_manifestation import (
    _apply_hero_memory_signals,
    _manifestation_seed,
)
from game.campaign.memory_signals import (
    _fresh_signal_allowlist,
    _memory_signals_by_hero,
)
from game.combat.combat_state import LifeState, MoraleState, StrainMark, StrainTier
from game.combat.traits import QUIRK_GOLD_FEVER
from game.core.rng import GameRng


def _apply_post_expedition_traits(
    company: CompanyState,
    report: ExpeditionReportState,
) -> list[str]:
    moments: list[str] = []
    participant_ids = set(report.participant_ids)
    heroes = {hero.hero_id: hero for hero in company.roster}
    memory_signals = _memory_signals_by_hero(report)
    fresh_signal_allowlist = _fresh_signal_allowlist(report)
    manifestation_rng = GameRng(_manifestation_seed(report))

    for hero in company.roster:
        if hero.hero_id not in participant_ids:
            _rest_hero_strain(hero)

    for outcome in report.hero_outcomes:
        outcome_hero = heroes.get(outcome.hero_id)
        if outcome_hero is None or outcome.died:
            continue
        strain_moments = _apply_expedition_strain(outcome_hero, outcome, report)
        moments.extend(strain_moments)

        moments.extend(
            _apply_hero_memory_signals(
                company,
                outcome_hero,
                report,
                memory_signals.get(outcome.hero_id, ()),
                fresh_signal_allowlist,
                manifestation_rng,
            )
        )

        if QUIRK_GOLD_FEVER in outcome_hero.quirks:
            loot_total = sum(report.loot.values())
            if loot_total > 0 and _raise_hero_morale(outcome_hero):
                moments.append(f"{outcome_hero.name}'s Coin Fever brightened after the haul.")
            elif (
                loot_total == 0 and report.outcome != "defeat" and _lower_hero_morale(outcome_hero)
            ):
                moments.append(f"{outcome_hero.name}'s Coin Fever soured without loot.")
    return moments
def _strain_mark_candidate(
    outcome: HeroReportOutcome,
    report: ExpeditionReportState,
) -> StrainMark | None:
    if outcome.downed or outcome.mortal_wounds_delta > 0:
        return StrainMark.BATTERED
    if report.outcome == "defeat" or any(
        signal.kind in {"combat_retreat", "maze_route_collapsed"} for signal in report.event_signals
    ):
        return StrainMark.FRAYED
    start = report.start_hero_states.get(outcome.hero_id)
    end = report.end_hero_states.get(outcome.hero_id)
    if start is not None and end is not None:
        effort_spent = start.effort - end.effort
        if effort_spent >= 2 or end.effort == 0:
            return StrainMark.DRAINED
    if any(signal.kind == "encounter_resolved" for signal in report.event_signals):
        return StrainMark.WINDED
    return None


def _apply_expedition_strain(
    hero: HeroState,
    outcome: HeroReportOutcome,
    report: ExpeditionReportState,
) -> list[str]:
    moments: list[str] = []
    had_encounter = any(signal.kind == "encounter_resolved" for signal in report.event_signals)
    if had_encounter:
        previous = hero.strain
        hero.strain = StrainTier(min(hero.strain.value + 1, StrainTier.SPENT.value))
        if hero.strain != previous and hero.strain.value >= StrainTier.WORN.value:
            moments.append(f"{hero.name}'s Strain rose to {hero.strain.name.title()}.")

    mark = _strain_mark_candidate(outcome, report)
    if mark is not None and mark not in hero.strain_marks:
        hero.strain_marks.add(mark)
        moments.append(f"{hero.name} came back {_trait_label(mark.value)}.")
    return moments


def _rest_hero_strain(hero: HeroState) -> None:
    if hero.strain.value > StrainTier.STEADY.value:
        hero.strain = StrainTier(hero.strain.value - 1)
    if hero.strain.value <= StrainTier.STEADY.value:
        hero.strain_marks.clear()


def _raise_hero_morale(hero: HeroState) -> bool:
    new_value = min(hero.morale.value + 1, MoraleState.INSPIRED.value)
    if new_value == hero.morale.value:
        return False
    hero.morale = MoraleState(new_value)
    return True


def _lower_hero_morale(hero: HeroState) -> bool:
    minimum = (
        MoraleState.STEADY.value
        if hero.personal_quirk == "grave_calm"
        else MoraleState.BROKEN.value
    )
    new_value = max(hero.morale.value - 1, minimum)
    if new_value == hero.morale.value:
        return False
    hero.morale = MoraleState(new_value)
    return True


def _trait_label(trait_id: str) -> str:
    return trait_id.replace("_", " ").title()
def _hero_outcomes(report: ExpeditionReportState) -> list[HeroReportOutcome]:
    outcomes: list[HeroReportOutcome] = []
    for hero_id in report.participant_ids:
        start = report.start_hero_states.get(hero_id)
        end = report.end_hero_states.get(hero_id)
        if start is None or end is None:
            continue
        died = end.life_state == LifeState.DEAD.value
        downed = end.life_state == LifeState.DOWNED.value
        wound_delta = end.mortal_wounds - start.mortal_wounds
        wounded = end.hp < end.max_hp or wound_delta > 0
        if died:
            status = "died"
        elif downed:
            status = "downed"
        elif wounded:
            status = "wounded"
        else:
            status = "survived"
        outcomes.append(
            HeroReportOutcome(
                hero_id=hero_id,
                hero_name=start.name,
                class_id=start.class_id,
                status=status,
                start_hp=start.hp,
                end_hp=end.hp,
                max_hp=end.max_hp,
                start_mortal_wounds=start.mortal_wounds,
                end_mortal_wounds=end.mortal_wounds,
                mortal_wounds_delta=wound_delta,
                died=died,
                downed=downed,
                wounded=wounded,
            )
        )
    return outcomes
