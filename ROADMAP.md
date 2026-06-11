# Roadmap

## Milestone 0: Repo Scaffold

- Create src-layout Python project.
- Add docs, YAML content, tests, linting, and type-check configuration.

## Milestone 1: Combat Engine

- Implement 2x2 formation.
- Implement targeting, initiative, damage, Downed, Mortal Wounds, death, Effort,
  and Order.
- Cover required deterministic tests.

## Milestone 2: Recruitment and Company State

- Create company state, starting roster, supplies, inventory, reputation, known
  breaches, deceased heroes, expedition history, and town state.
- Preserve recruit backgrounds, known lore, active contracts, completed contracts,
  and story flags in company saves.
- Keep runtime state in dataclasses.

## Milestone 3: Expedition Vertical Slice

- Implement Old Road, Blackwood Forest, Shallow Cave, Cave Mini Boss, breach
  discovery, optional Maze Depth 1, and return to Haven Town.
- Frame the opening as hard Act 1 frontier contract work with Maze leaks.
- Keep authored route data in YAML.

## Milestone 4: Town and Save/Load

- Add basic Haven Town loop.
- Add JSON save/load for company state.
- Exclude mid-combat saves.

## Milestone 5: Polish and Balancing

- Tune class stats, enemies, skill accuracy, rewards, and supplies.
- Improve terminal rendering without moving rules into UI.
- Keep the default difficulty hard through visible risk, scarce recovery, lasting
  injuries, and tempting overreach.

## Milestone 6: Story Scaffolding

- Expand `data/world.yaml` contracts into selectable Act 1 jobs.
- Surface rumors and contract state through town services.
- Add more overworld Maze leaks before full Maze expeditions become normal.
- Keep Pandora's Maze as a working name until the game's identity settles.

## Future Ideas Beyond v0.1

- More breaches and dungeon branches.
- Broader town services.
- Deeper recruitment tables.
- More equipment and loot.
- Stronger expedition choice logic.
- Better recovery and long-term injury handling.
- Act 2 Maze-capital systems: permits, map brokers, relic auctions, rival
  companies, fake safe-route maps, and corpse retrieval contracts.

These ideas are not v0.1 scope unless explicitly requested.
