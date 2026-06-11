# The Charter Project Context

Date: 2026-06-06

## Project Goal

The Charter is a hard, text-first expedition company RPG vertical slice. The
technical package is `charter-game`; the repo folder is currently
`dungeon-party-game`.

The player manages a chartered company rather than a chosen hero. Ordinary
recruits enter dangerous places, and the survivors become important through
survival, injuries, decisions, memories, earned quirks, and deaths.

Core fantasy: "Generic bodies go in. Stories come out."

Current working theme: "The Maze does not invade the world. The world invites it
in, one profitable expedition at a time."

## Core Gameplay Loop

1. Start or load a company in Haven.
2. Inspect roster, supplies, gear, ledger, contracts, known places, and reports.
3. Assign an active party of up to four heroes in a 2x2 formation.
4. Spend Coin on recruits, recovery, supplies, gear, or upgrades.
5. Accept or pursue contracts.
6. Travel from Haven into the opening route.
7. Explore authored expedition nodes and generated breach routes.
8. Resolve room actions, hazards, discoveries, and manual combat-lite encounters.
9. Discover breaches, complete contracts, gain Coin/reputation/gear, and return.
10. Preserve wounds, morale, strain marks, memories, route knowledge, contracts,
    and save data for the next expedition.

## Main Systems

- App command layer: UI and tests issue structured dataclass commands from
  `src/game/app/commands.py`.
- App controller: `AppController` routes commands to town, expedition, dungeon,
  combat, save/load, and quit flows.
- Actions/HCI metadata: `ActionProvider` builds numbered player-facing actions,
  labels, aliases, risk metadata, previews, unavailable reasons, and result hints.
- Company model: `CompanyState` owns roster, supplies, inventory, Coin,
  reputation, contracts, breaches, world/dungeon memory, active party slots,
  reports, timeline, and save version.
- Data loading: YAML files in `data/` are validated by Pydantic schemas in
  `src/game/data/schemas.py` and loaded into `GameDefinitions`.
- Combat: deterministic rule modules cover 2x2 formation, targeting, turn order,
  damage, Downed/Mortal Wound/death, morale/cohesion, strain, traits, previews,
  reactions, movement, retreat, and enemy decisions.
- Expedition/dungeon: authored route nodes live in `data/expeditions.yaml`;
  runtime systems track visited/cleared nodes, room actions, known routes,
  generated Maze routes, and breach collapse.
- Town services: recruitment, recovery, supplies, upgrades, gear, active party
  assignment, contract board, roster, memorial, and ledger.
- Save/load: campaign-level JSON at `saves/company.json` by default. Mid-combat
  saves are intentionally excluded.
- UI: Textual fullscreen TUI is the main player-facing frontend. Rich CLI is a
  legacy/dev fallback through `--cli`. Demo mode is deterministic and
  noninteractive.

## Architecture Overview

Data flow is intentionally layered:

1. Authored YAML content is loaded and validated into `GameDefinitions`.
2. UI or tests issue structured commands, not raw engine mutations.
3. `AppController.handle()` captures HCI state, dispatches the command, then
   attaches HCI analysis when a result did not provide one.
4. Flow classes coordinate app-level transitions.
5. Engine modules mutate or query runtime dataclasses and return `Result`
   objects containing structured `GameEvent` records.
6. UI renders view models, actions, and events. UI should not derive combat
   legality, damage, targeting, or major story beats itself.
7. Save/load serializes `CompanyState.to_dict()` and restores through
   `CompanyState.from_dict()` with compatibility migrations for older fields.

## Important Design Constraints

- Keep engine rules separate from UI rendering.
- UI choices become structured commands.
- Engine systems return structured events/results; UI renders those events.
- Keep authored content in YAML whenever practical.
- Prefer deterministic command handling and deterministic tests.
- Route all randomness through `game.core.rng.GameRng`; do not call Python random
  APIs directly outside that module.
- Use dataclasses for runtime state.
- Use Pydantic only for external data validation.
- App commands are the bridge between UI choices and engine behavior.
- Textual is the main frontend; Rich CLI is a legacy/dev fallback.
- Avoid adding new RPG mechanics just because they are common.
- Do not add stress bars, affliction/virtue systems, torch meters, global danger
  timers, MP, cooldowns, hamlet-building, complex crafting, SQL/web backends, ECS,
  procedural editors, or mid-combat saves unless explicitly requested.

## Current Game Scope

The v0.1 opening route is:

Haven East Gate -> Old Road -> Blackwood wilderness paths -> Shallow Cave ->
Cave Mini Boss -> Maze Breach -> optional generated breach route or deterministic
Pandora's Maze Depth 1 stub -> Haven return.

Starting roster classes are Watchman, Cutpurse, Field Surgeon, and Scribe.

The game already includes Coin, reputation, contracts, active party plus reserves,
town services, gear, traits, strain marks, morale/cohesion, hero memories, earned
quirk infrastructure, generated breach routes, save/load compatibility, a default
Textual TUI, and deterministic test coverage.
