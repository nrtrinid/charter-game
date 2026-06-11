"""Container for validated authored game definitions."""

from __future__ import annotations

from dataclasses import dataclass

from game.data.schemas import (
    ArtFile,
    ContractDefinition,
    EncounterDefinition,
    EnemiesFile,
    EnemyDefinition,
    ExpeditionDefinition,
    ExpeditionsFile,
    GearDefinition,
    GearFile,
    HeroClassDefinition,
    HeroesFile,
    LocationDefinition,
    LootDefinition,
    LootFile,
    RecruitsFile,
    RumorDefinition,
    SkillDefinition,
    SkillsFile,
    SuppliesFile,
    TownFile,
    TraitDefinition,
    TraitsFile,
    TraitType,
    WorldFile,
)


@dataclass(frozen=True)
class GameDefinitions:
    heroes_file: HeroesFile
    enemies_file: EnemiesFile
    skills_file: SkillsFile
    traits_file: TraitsFile
    recruits_file: RecruitsFile
    expeditions_file: ExpeditionsFile
    gear_file: GearFile
    loot_file: LootFile
    supplies_file: SuppliesFile
    town_file: TownFile
    world_file: WorldFile
    art_file: ArtFile

    @property
    def hero_classes(self) -> dict[str, HeroClassDefinition]:
        return self.heroes_file.classes

    @property
    def enemies(self) -> dict[str, EnemyDefinition]:
        return self.enemies_file.enemies

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        return self.skills_file.skills

    @property
    def traits(self) -> dict[str, TraitDefinition]:
        return self.traits_file.traits

    @property
    def personal_quirks(self) -> dict[str, TraitDefinition]:
        return {
            trait_id: trait
            for trait_id, trait in self.traits.items()
            if trait.type == TraitType.PERSONAL
        }

    @property
    def earned_quirks(self) -> dict[str, TraitDefinition]:
        return {
            trait_id: trait
            for trait_id, trait in self.traits.items()
            if trait.type == TraitType.EARNED
        }

    @property
    def conditions(self) -> dict[str, TraitDefinition]:
        return self.strain_marks

    @property
    def strain_marks(self) -> dict[str, TraitDefinition]:
        return {
            trait_id: trait
            for trait_id, trait in self.traits.items()
            if trait.type in {TraitType.CONDITION, TraitType.STRAIN_MARK}
        }

    @property
    def expeditions(self) -> dict[str, ExpeditionDefinition]:
        return self.expeditions_file.expeditions

    @property
    def gear(self) -> dict[str, GearDefinition]:
        return self.gear_file.gear

    @property
    def loot(self) -> dict[str, LootDefinition]:
        return self.loot_file.loot

    @property
    def encounters(self) -> dict[str, EncounterDefinition]:
        return self.expeditions_file.encounters

    @property
    def recruits(self) -> RecruitsFile:
        return self.recruits_file

    @property
    def supplies(self) -> SuppliesFile:
        return self.supplies_file

    @property
    def town(self) -> TownFile:
        return self.town_file

    @property
    def world(self) -> WorldFile:
        return self.world_file

    @property
    def locations(self) -> dict[str, LocationDefinition]:
        return self.world_file.locations

    @property
    def contracts(self) -> dict[str, ContractDefinition]:
        return self.world_file.contracts

    @property
    def rumors(self) -> dict[str, RumorDefinition]:
        return self.world_file.rumors

    @property
    def art(self) -> ArtFile:
        return self.art_file
