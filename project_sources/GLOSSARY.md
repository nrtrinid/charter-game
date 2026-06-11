# Glossary

Date: 2026-06-06

Active party: Up to four living roster members assigned to expedition formation
slots.

App command: A structured dataclass from `src/game/app/commands.py` issued by UI
or tests.

Breach: A discovered place where the Maze connects to an overworld dungeon or
route. Current key breach: `shallow_cave_breach`.

Charter: The company's legal work/premise and the game's working identity.

Coin: Spendable company money used for recruiting, recovery, supplies, upgrades,
and gear.

Cohesion: Party-level state derived from hero morale.

Company: The campaign-level protagonist and owner of roster, memory, resources,
contracts, discoveries, and saves.

Contract: A paid objective posted in world data and tracked in company state.

Downed: A life state for heroes at 0 HP. Downed heroes cannot act or protect a
lane.

Dungeon memory: Runtime memory of visited nodes, cleared nodes, completed room
actions, and revealed exits.

Effort: The only special-action resource in v0.1.

Event: A structured dataclass from `src/game/core/events.py` returned by engine
or app systems.

Expedition: A route or dungeon run that can produce combat, discoveries, loot,
contracts, reports, and return state.

Formation: The explicit 2x2 slot arrangement: `BACK_LEFT`, `BACK_RIGHT`,
`FRONT_LEFT`, `FRONT_RIGHT`.

Fresh memory: Recent hero memory signal that can contribute to earned quirks.

GameDefinitions: Loaded and validated authored content container.

Generated breach route: A deterministic temporary Maze route created from a
breach, with room recipes and contract hooks.

Haven: Starting frontier settlement and current town hub.

HeroState: Runtime campaign state for a roster member.

Life state: Alive, Downed, or Dead.

Maze: The impossible exploited place that becomes more connected as people use
it. Working name: Pandora's Maze.

Mortal Wound: Persistent hero wound gained by damage while Downed. Three Mortal
Wounds kill a hero permanently.

Deep Surgery: Haven town service that removes one Mortal Wound from a living hero
for Coin. The hero is marked In Surgery and cannot join the active party until the
next expedition report finalizes.

In Surgery: Town recovery marker applied by Deep Surgery. The hero is benched from
formation assignment until the next expedition cycle completes.

Order: Older design language for party stability. Current code now emphasizes
morale/cohesion, but some docs/tests may still mention Order in combat context.

Recruit: A potential or hired company member with grounded background and motive.

Reserve: Living roster member not currently assigned to active party slots.

Result: Standard return object with `success`, `value`, `events`, `error`, and
optional HCI analysis.

Rich CLI: Legacy/dev fallback frontend launched with `--cli`.

ScreenAction: Canonical UI action metadata, including number, label, value,
aliases, enabled/default state, kind, risk, cost, preview, and result hint.

Strain: Current hero wear tier, replacing older fatigue language.

Strain Mark: Temporary or persistent condition-style mark such as winded, drained,
or frayed.

Tag: Combat tag such as Marked, Stunned, Knocked Down, Frozen, Burning, etc.

Textual TUI: Main fullscreen terminal UI.

World memory: Runtime memory of known locations, visits, discovered nodes,
cleared threats, shortcuts, and consumed rumors.
