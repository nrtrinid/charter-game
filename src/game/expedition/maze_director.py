"""Maze pressure profiles and early director policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from game.campaign.company import CompanyState, MazeRecipe
from game.campaign.roster import living_roster
from game.combat.traits import QUIRK_MAZE_SIGHTED
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.expedition.generated_maze import (
    GENERATED_MAZE_HUNT_CONTRACT_ID,
    GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
)


@dataclass(frozen=True)
class MazePressureProfile:
    pressure_id: str
    min_route_length: int
    max_route_length: int
    combat_budget: int
    hazard_budget: int
    reward_lure: bool
    include_hunt: bool
    enemy_policy_id: str
    pressure_tags: tuple[str, ...] = ()
    layout_style: str = "winding"
    branch_budget: int = 2
    room_palette: str = "stone"
    encounter_style: str = "standard"


@dataclass(frozen=True)
class MazeDirectorObservation:
    run_number: int
    source_node_id: str
    active_contract_ids: tuple[str, ...]
    completed_contract_ids: tuple[str, ...]
    living_hero_count: int
    wounded_hero_count: int
    supplies: dict[str, int]


class MazeDirectorPolicy(Protocol):
    def choose_profile(
        self,
        observation: MazeDirectorObservation,
        legal_profiles: tuple[MazePressureProfile, ...],
        rng: GameRng,
    ) -> MazePressureProfile:
        """Choose one legal Maze pressure profile."""
        ...


class ScriptedMazeDirectorPolicy:
    def choose_profile(
        self,
        observation: MazeDirectorObservation,
        legal_profiles: tuple[MazePressureProfile, ...],
        rng: GameRng,
    ) -> MazePressureProfile:
        for profile in legal_profiles:
            if profile.include_hunt:
                return profile
        for profile in legal_profiles:
            if profile.pressure_id == "breach_probe":
                return profile
        return legal_profiles[0]


class RandomMazeDirectorPolicy:
    def choose_profile(
        self,
        observation: MazeDirectorObservation,
        legal_profiles: tuple[MazePressureProfile, ...],
        rng: GameRng,
    ) -> MazePressureProfile:
        return rng.choice(legal_profiles)


MAZE_PRESSURE_PROFILES: tuple[MazePressureProfile, ...] = (
    MazePressureProfile(
        pressure_id="breach_probe",
        min_route_length=3,
        max_route_length=5,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("scout", "reward_lure", "overreach"),
        layout_style="winding",
        branch_budget=2,
        room_palette="glass",
        encounter_style="standard",
    ),
    MazePressureProfile(
        pressure_id="long_pressure",
        min_route_length=5,
        max_route_length=5,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("long_route", "combat", "overreach"),
        layout_style="forked",
        branch_budget=3,
        room_palette="market",
        encounter_style="standard",
    ),
    MazePressureProfile(
        pressure_id="tight_probe",
        min_route_length=3,
        max_route_length=4,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("short_route", "low_reward"),
        layout_style="winding",
        branch_budget=1,
        room_palette="stone",
        encounter_style="light",
    ),
    MazePressureProfile(
        pressure_id="marked_hunt",
        min_route_length=4,
        max_route_length=5,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=True,
        enemy_policy_id="basic",
        pressure_tags=("hunt", "boss", "contract"),
        layout_style="dead_end_heavy",
        branch_budget=3,
        room_palette="maw",
        encounter_style="brute",
    ),
)


def build_maze_observation(
    company: CompanyState,
    *,
    source_node_id: str,
    run_number: int,
) -> MazeDirectorObservation:
    living = living_roster(company)
    return MazeDirectorObservation(
        run_number=run_number,
        source_node_id=source_node_id,
        active_contract_ids=tuple(sorted(company.active_contract_ids)),
        completed_contract_ids=tuple(sorted(company.completed_contract_ids)),
        living_hero_count=len(living),
        wounded_hero_count=sum(
            1 for hero in living if hero.hp < hero.max_hp or hero.mortal_wounds > 0
        ),
        supplies=dict(company.supplies),
    )


def legal_maze_pressure_profiles(
    observation: MazeDirectorObservation,
) -> tuple[MazePressureProfile, ...]:
    hunt_active = _hunt_contract_is_active(observation)
    return tuple(
        profile
        for profile in MAZE_PRESSURE_PROFILES
        if not profile.include_hunt or hunt_active
    )


def recipe_from_profile(profile: MazePressureProfile, rng: GameRng) -> MazeRecipe:
    # route_length is the initial spine depth; players can push deeper on demand.
    route_length = rng.randint(profile.min_route_length, profile.max_route_length)
    return MazeRecipe(
        pressure_id=profile.pressure_id,
        route_length=route_length,
        combat_budget=profile.combat_budget,
        hazard_budget=profile.hazard_budget,
        reward_lure=profile.reward_lure,
        include_hunt=profile.include_hunt,
        enemy_policy_id=profile.enemy_policy_id,
        pressure_tags=profile.pressure_tags,
        layout_style=profile.layout_style,
        branch_budget=profile.branch_budget,
        room_palette=profile.room_palette,
        encounter_style=profile.encounter_style,
    )


def choose_maze_recipe(
    company: CompanyState,
    *,
    source_node_id: str,
    run_number: int,
    rng: GameRng,
    definitions: GameDefinitions | None = None,
    policy: MazeDirectorPolicy | None = None,
) -> MazeRecipe:
    observation = build_maze_observation(
        company,
        source_node_id=source_node_id,
        run_number=run_number,
    )
    legal_profiles = legal_maze_pressure_profiles(observation)
    profile = _contract_requested_profile(definitions, observation, legal_profiles)
    if profile is None:
        selected_policy = policy or ScriptedMazeDirectorPolicy()
        profile = selected_policy.choose_profile(observation, legal_profiles, rng)
    recipe = recipe_from_profile(profile, rng)
    if _active_party_has_maze_sighted(company):
        recipe.route_length = max(profile.min_route_length, recipe.route_length - 1)
    return recipe


def _contract_requested_profile(
    definitions: GameDefinitions | None,
    observation: MazeDirectorObservation,
    legal_profiles: tuple[MazePressureProfile, ...],
) -> MazePressureProfile | None:
    if definitions is None:
        return None
    pressure_ids = tuple(
        contract.generated_maze_pressure_id
        for contract in definitions.contracts.values()
        if contract.id in observation.active_contract_ids
        and contract.generated_maze_pressure_id
    )
    for pressure_id in sorted(pressure_ids):
        profile = next(
            (
                candidate
                for candidate in legal_profiles
                if candidate.pressure_id == pressure_id
            ),
            None,
        )
        if profile is not None:
            return profile
    return None


def _hunt_contract_is_active(observation: MazeDirectorObservation) -> bool:
    return any(
        contract_id in observation.active_contract_ids
        for contract_id in (
            GENERATED_MAZE_HUNT_CONTRACT_ID,
            GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
        )
    )


def _active_party_has_maze_sighted(company: CompanyState) -> bool:
    living = {hero.hero_id: hero for hero in living_roster(company)}
    return any(
        hero_id is not None
        and hero_id in living
        and QUIRK_MAZE_SIGHTED in living[hero_id].quirks
        for hero_id in company.active_party_slots.values()
    )
