# Data Schemas And Examples

Date: 2026-06-06

Authoring data lives in YAML under `data/` and is validated by Pydantic models in
`src/game/data/schemas.py`. Runtime state is mostly dataclasses in
`src/game/campaign/company.py` and `src/game/combat/combat_state.py`.

## Loaded Definition Container

`GameDefinitions` is the immutable container passed to systems:

```python
@dataclass(frozen=True)
class GameDefinitions:
    heroes_file: HeroesFile
    enemies_file: EnemiesFile
    skills_file: SkillsFile
    traits_file: TraitsFile
    recruits_file: RecruitsFile
    expeditions_file: ExpeditionsFile
    gear_file: GearFile
    supplies_file: SuppliesFile
    town_file: TownFile
    world_file: WorldFile
    art_file: ArtFile
```

It exposes convenience properties such as `hero_classes`, `skills`, `enemies`,
`expeditions`, `encounters`, `town`, `world`, `contracts`, `rumors`, `gear`, and
`art`.

## Hero Class

Schema fields:

- `id`
- `name`
- `max_hp`
- `speed`
- `accuracy`
- `defense`
- `damage`
- `max_effort`
- `skills`
- `personal_quirk`

Example from `data/heroes.yaml`:

```yaml
classes:
  watchman:
    id: watchman
    name: Watchman
    max_hp: 18
    speed: 2
    accuracy: 8
    defense: 2
    damage: 2
    max_effort: 3
    personal_quirk: hold_the_line
    skills:
      - guard_strike
      - shield_drive
```

## Skill

Important fields:

- `id`, `name`, `category`
- `effort_cost`
- `attack_type`: `melee`, `reach`, `ranged`, or `magic`
- `usable_from`: default `any_position`, or positional restrictions such as
  `front_only`
- `accuracy`
- `damage`
- optional visible `damage_min` and `damage_max`
- `tags`
- `description`, `effect_text`
- reaction/intent metadata for combat UI

Example:

```yaml
shield_drive:
  id: shield_drive
  name: Watchman's Shove
  category: special
  effort_cost: 1
  attack_type: reach
  usable_from: front_only
  accuracy: 88
  damage: 3
  damage_min: 2
  damage_max: 3
  tags: [effort, guard, formation, shove_back]
  description: A hard shield shove that can reach down a lane and drive a front enemy backward.
```

## Enemy

Schema fields mirror hero combat stats and add a default `formation_slot` and
tags.

```yaml
enemies:
  cave_maw_brute:
    id: cave_maw_brute
    name: Cave Maw Brute
    max_hp: 20
    speed: 1
    accuracy: 6
    defense: 2
    damage: 2
    max_effort: 4
    skills:
      - maw_slam
      - drag_forward
    formation_slot: FRONT_LEFT
    tags: [boss, beast]
```

## Expedition Node

Important fields:

- `id`, `name`, `node_type`, `text`
- optional UI/narrative fields: `scene_state`, `revisit_text`, `route_hint`,
  `party_hint`, `major_beat`
- map fields: `map_id`, `position`
- progression fields: `encounter`, `next_node`, `exits`, `safe_return`
- rewards/discovery: `reputation_reward`, `coin_reward`, `breach_id`,
  `known_route_unlock`, `loot`, `supply_rewards`, `history`
- story state: `lore_entries`, `flags_set`, `complete_contract`
- room `actions`

Example:

```yaml
- id: old_works_cache
  name: Old Works Cache
  node_type: combat
  text: A cache box sits under webbing and bone charms. Something has been nesting around the lock.
  map_id: shallow_cave
  position: [-2, 1]
  encounter: shallow_cave
  exits: [fungus_chamber]
  actions:
    - id: recover_gate_key
      label: Recover Gate Key
      description: Cut the webbing and search the cache after the fight.
      result_text: A brass cave key drops from the cache wrappings.
      requires_cleared: true
      coin_reward: 3
      loot:
        cave_key: 1
```

## Room Action

Room actions can be once-only and can require cleared combat, inventory, or
supplies. They can grant supplies, loot, reputation, Coin, and revealed exits.

```yaml
- id: brace_loose_stones
  label: Brace Loose Stones
  description: Spend rope to make this crawl safer for retreat.
  result_text: The rope brace holds the worst stones back.
  supply_costs:
    rope: 1
```

## Contract

Contracts live in `data/world.yaml`.

```yaml
blackwood_road_charter:
  id: blackwood_road_charter
  name: Blackwood Road Charter
  act: 1
  location_id: old_road
  expedition_id: opening
  difficulty: 3
  reward_reputation: 2
  coin_reward: 10
  summary: Reopen the road, clear the cave hazard, and bring back proof worth paying for.
  available_at_start: true
  board_order: 0
  tags: [active, overworld, v0_1]
```

Contracts can be posted or locked by completed contracts and known breaches.
Repeatable contracts are identified by the `repeatable` tag.

## Runtime HeroState

Key fields:

- identity: `hero_id`, `name`, `class_id`, `background`, `motive`
- stats: `max_hp`, `hp`, `speed`, `accuracy`, `defense`, `damage`,
  `max_effort`, `effort`
- combat state: `skills`, `formation_slot`, `life_state`, `morale`, `strain`,
  `tags`, `strain_marks`, `mortal_wounds`
- identity growth: `personal_quirk`, `quirks`, `career_signals`,
  `fresh_memories`, `earned_quirk_slots`
- gear: `equipped_gear_id`

## Runtime CompanyState

Key fields:

- identity: `company_id`, `name`
- roster: `roster`, `deceased_heroes`, `active_party_slots`
- resources: `supplies`, `inventory`, `gear_inventory`, `reputation`, `coin`
- progress: `known_breaches`, `known_route_ids`, `known_lore_entries`, `flags`
- contracts: `active_contract_ids`, `completed_contract_ids`,
  `contract_records`
- memory: `expedition_history`, `hero_memories`, `company_timeline`,
  `dungeon_memory`, `world_memory`, `breach_memory`
- town: `town_state`, `recruitment_state`, `purchased_upgrade_ids`
- expedition/session: `active_expedition`, `last_expedition_report`,
  `expedition_reports`
- compatibility: `save_version`

## Result And Events

Engine and app systems usually return:

```python
@dataclass
class Result[T]:
    success: bool
    value: T | None = None
    events: list[GameEvent] = field(default_factory=list)
    error: str | None = None
    hci: HciResultAnalysis | None = None
```

Major event types include combat, movement, damage, Downed/death, expedition
beats, dungeon actions, loot, breach discovery, lore discovery, contract
completion, town services, recruitment, recovery, supplies, active-party changes,
save, and load.
