# Architecture Decision Records

Date: 2026-06-06

## ADR-001: Use Data-Driven Authored Content

Decision: Author hero classes, skills, enemies, recruits, traits, expeditions,
world state, town services, supplies, gear, and art in YAML where practical.

Reason: This keeps content tweakable without rewriting engine code and supports
fast iteration on balance, route structure, contract state, and UI text.

Consequences: YAML must be validated aggressively. Cross-reference validation in
`src/game/data/loaders.py` matters. Engine code should not hardcode room-specific
logic when a YAML field or generic rule can represent it.

## ADR-002: Separate Runtime State From Authored Definitions

Decision: Load authored definitions into immutable `GameDefinitions`, while
runtime campaign/combat state lives in dataclasses such as `CompanyState`,
`HeroState`, `CombatState`, and `Combatant`.

Reason: Authored content should be stable reference data. Runtime state must be
serializable, mutable during play, and compatible with save/load migrations.

Consequences: Systems generally receive both `company` and `definitions`. New
runtime fields need `to_dict()`/`from_dict()` compatibility work and tests.

## ADR-003: Use Structured Commands Instead Of A Free-Text Parser

Decision: UI and tests issue typed command dataclasses from
`src/game/app/commands.py`. Player-facing affordances are generated as
`ScreenAction` records.

Reason: The game is text-first, but the current experience depends on reliable
menus, focus, numbers, aliases, and deterministic state transitions rather than
ambiguous natural language parsing.

Consequences: Parser-style changes should normally become new `ScreenAction`
metadata and new or existing command dataclasses. Do not hide important state
transitions inside UI input handling.

## ADR-004: Engine Systems Return Structured Events

Decision: Combat, expedition, dungeon, town, save/load, and company systems
return `GameEvent` dataclasses through `Result`.

Reason: Structured events let Textual, CLI, tests, reports, and HCI analysis
share one behavioral source of truth without scraping player-facing prose.

Consequences: Add event types for important new gameplay facts. UI should render
events; it should not infer major beats from message ordering.

## ADR-005: Textual TUI Is The Main Frontend

Decision: `python -m game.main` and `charter-game` launch the Textual TUI by
default. The Rich CLI remains available with `--cli` as a legacy/dev fallback.

Reason: The TUI supports the intended fixed regions, focusable actions, combat
panels, formation boards, command dock, progress strip, art, and richer repeated
play workflow.

Consequences: Polish and new UI features should target Textual first. CLI should
remain useful for debugging and fake-input tests, but it is not an equal polish
target unless explicitly requested.

## ADR-006: Keep Difficulty Visible And Deterministic

Decision: Hardship should come from visible risk, scarce recovery, lasting
injury, formation choices, Coin pressure, and tempting overreach. Randomness is
constrained through `GameRng` and narrow visible damage ranges.

Reason: The game should feel harsh but strategically legible.

Consequences: Avoid hidden spike damage as the default. Show damage ranges and
risk hints where the player needs to make tactical choices. New randomness needs
deterministic tests.
