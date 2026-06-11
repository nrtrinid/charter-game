# Repo Map

Date: 2026-06-06

## Top Level

```text
.
  README.md              Player/developer overview, install/run/test commands.
  DESIGN.md              Primary design authority and exclusions.
  ROADMAP.md             Milestones and future ideas.
  AGENTS.md              Agent workflow, architecture rules, repo memory.
  CHANGELOG.md           Unreleased and versioned changes.
  pyproject.toml         Package metadata, deps, pytest/ruff/mypy config.
  rtk.ps1                Windows repo toolkit for smoke/test/check/all.
  data/                  Authored YAML content.
  src/game/              Python package source.
  tests/                 Deterministic unit, integration, UI, and slice tests.
```

## Source Layout

```text
src/game/
  main.py                CLI entrypoint; default TUI, --cli fallback, --demo.

  app/
    commands.py          Structured command dataclasses and Command union.
    controller.py        Thin controller that dispatches commands to flows.
    flows.py             Town, expedition, dungeon, manual-combat app flows.
    actions.py           Canonical player-facing action metadata.
    views.py             App-facing view models for UI rendering.
    contracts.py         Contract board posting/availability rules.
    hci.py               HCI state capture and result analysis.
    manual_combat.py     Manual combat-lite session coordination.

  core/
    events.py            Structured event dataclasses and EventType enum.
    result.py            Result container with events, errors, and HCI analysis.
    rng.py               Deterministic RNG wrapper.
    ids.py               ID helpers.
    hci.py               HCI result model.

  data/
    schemas.py           Pydantic schemas for authored YAML.
    loaders.py           YAML loading and cross-reference validation.

  content/
    definitions.py       GameDefinitions container for loaded content.

  campaign/
    company.py           CompanyState, HeroState, reports, memory, save shape.
    roster.py            Active/reserve roster helpers and combat sync.
    town.py              Town services: hire, recover, buy, upgrades, formation.
    save_load.py         JSON save/load.
    recruitment.py       Recruit offer generation.
    gear.py              Gear inventory/equipping/effective stats.
    economy.py           Coin helpers.
    reputation.py        Reputation helpers.
    objectives.py        Campaign objective view.
    rewards.py           Contract reward application.
    memory.py            Company/hero memory helpers.
    hero_memory.py       Fresh memory and earned quirk state.

  combat/
    combat_state.py      Combatant, CombatState, morale, strain, tags.
    formation.py         2x2 formation, lanes, protection, movement adjacency.
    targeting.py         Attack type legality and cover penalties.
    actions.py           Skill resolution and combat events.
    damage.py            Damage/healing and Downed/death integration.
    death.py             Downed, Mortal Wound, death helpers.
    turn_order.py        Initiative and turn order.
    morale.py            Morale/cohesion effects.
    traits.py            Trait/condition helpers.
    preview.py           Attack preview data for UI.
    enemy_decision.py    Enemy skill/target selection.
    enemy_learning.py    Enemy decision training/evaluation substrate.
    damage_range.py      Visible damage range helpers.

  expedition/
    node.py              ExpeditionNodeType enum.
    expedition.py        Opening route and breach return/descent helpers.
    dungeon.py           Interactive dungeon state and room transitions.
    generated_maze.py    Runtime generated breach routes.
    maze_director.py     Generated route pressure/recipe selection.
    maze.py              Deterministic Maze stub support.
    cave.py              Cave encounter factories.
    travel.py            World travel helpers.

  ui/
    tui.py               Textual app shell.
    tui_widgets.py       Textual widgets: status, command dock, panels, etc.
    tui_models.py        TUI model helpers.
    screens.py           Rich screen rendering for legacy/dev CLI.
    cli.py               Legacy Rich CLI wrapper/input loop.
    hci_text.py          Human-readable HCI text.
    wounds.py            Wound display helpers.
```

## Data Files

```text
data/
  heroes.yaml            Hero class definitions.
  skills.yaml            Skills, costs, attack types, damage ranges, tags.
  enemies.yaml           Enemy classes and formation defaults.
  recruits.yaml          Starting roster and recruit pool.
  traits.yaml            Personal quirks, earned quirks, strain marks.
  expeditions.yaml       Authored expedition nodes, encounters, room actions.
  world.yaml             Locations, contracts, rumors, difficulty profile.
  town.yaml              Town service costs, roster cap, upgrades.
  supplies.yaml          Starting supplies and supply shop catalog.
  gear.yaml              Gear definitions, costs, effects, unlocks.
  art.yaml               Terminal art assets for TUI/CLI surfaces.
```

## Test Map

```text
tests/
  test_vertical_slice.py        Opening route, breach discovery, rewards, save/load.
  test_save_load.py             Company save shape and migration coverage.
  test_manual_combat.py         Manual combat-lite commands and flow.
  test_town_loop.py             Hiring, recovery, supplies, formation, contracts.
  test_tui.py                   Textual app keyboard/state behavior.
  test_cli.py                   Legacy CLI smoke/fake input behavior.
  test_dungeon.py               Dungeon movement/actions/return behavior.
  test_generated_maze.py        Generated breach route behavior.
  test_targeting.py             Target legality and cover rules.
  test_turn_order.py            Initiative ordering.
  test_death.py                 Downed, Mortal Wounds, death.
  test_traits.py                Traits/strain/quirk behavior.
  test_enemy_decision.py        Enemy choice heuristics.
  test_enemy_learning.py        Enemy learning substrate.
```
