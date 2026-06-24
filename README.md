# The Charter

**The Charter** is the working title for a playable v0.1 vertical slice of a
hard, text-first expedition company RPG. The technical project and repo name is
`charter-game`.
The company is the protagonist: ordinary recruits enter dangerous places, and
the survivors become important through play.

Core fantasy: "Generic bodies go in. Stories come out."

This first version favors a complete first gameplay loop, deterministic rules
tests, data-driven content, save/load correctness, and readable terminal UIs over
content volume.

## Install

```bash
uv sync --extra dev
```

If `uv` is unavailable:

```bash
python -m pip install -e ".[dev]"
```

## Run

```bash
uv run python -m game.main
uv run charter-game
```

Fallback after installing with pip:

```bash
python -m game.main
```

The fullscreen Textual interface is the main game surface. The older Rich CLI is
kept as a legacy/dev fallback:

```bash
python -m game.main --cli
```

The basic loop is:

1. Start a charter company in Haven.
2. Inspect the roster, supplies, ledger, active contracts, and known places.
3. Assign an active party, buy supplies, recover wounded survivors, or recruit.
4. Travel from Haven into the opening route.
5. Explore, resolve dungeon choices, and fight manual combat-lite encounters.
6. Discover the Shallow Cave breach.
7. Return to Haven with a report, or enter an early generated Maze route.
8. Preserve roster wounds, morale, traits, world memory, contracts, and save data.

The CLI redraws a compact terminal frame when switching screens. The main menu is
organized into Company, Expedition, Save / Load, Help, and Quit submenus. It also
accepts direct typed aliases:

- `start` / `new`
- `roster` / `r`
- `supplies` / `s`
- `ledger`
- `expedition` / `x`
- `save`
- `load`
- `help` / `?`
- `quit` / `q`

During expedition playback, each prompt advances by scene or encounter instead of
by individual atomic event. Playback controls are shown as numbered options:
Continue, Auto-play Section, and Stop at Safe Menu. Interactive opening-route
combat uses manual combat-lite: the combat screen shows the current actor, Order,
morale/cohesion, party/enemy formations, legal skills, target previews, hit
chance, damage estimate, and recent log. Enter accepts the marked safe default.
After a hero action, a single resolution card bundles the hero action, enemy
response, round changes, morale changes, and encounter outcome. Demo mode keeps
the deterministic automatic combat route.

The Shallow Cave route pauses at the Breach with numbered options to return to
Haven, enter a generated breach route, or descend into the deterministic Maze
Depth 1 stub. Generated breach routes are deterministic per run, can contain
combat, curios, rewards, harder rooms, and marked hunt lairs, and can complete
repeatable scout/hunt contract work before collapsing back to the breach.

Company play is organized through Haven town services:

- Recruiting spends Coin for deterministic recruit offers.
- Recovery spends Coin to restore HP/Effort and clear Downed, while keeping
  Mortal Wounds.
- The quartermaster sells YAML-authored supplies for Coin.
- Formation assigns living roster members into the four active party slots or
  empties a slot.
- The town dashboard separates active party, reserves, wounded/downed counts,
  morale state, supplies, and memorial state.
- World view tracks known locations, travel destinations, rumors, discovered
  nodes, cleared threats, and unlocked shortcuts.

Story scaffolding is data-driven:

- `data/world.yaml` defines the current Maze name, difficulty profile, starting
  settlement, Act 1/future locations, contracts, and rumors.
- `data/recruits.yaml` gives starting recruits and recruit-pool entries grounded
  backgrounds and motives.
- `data/traits.yaml` defines personal quirks, earned quirks, and temporary
  Strain Marks used by morale, combat, and post-expedition memory.
- `data/art.yaml` provides terminal art for heroes, enemies, locations, dungeon
  rooms, and generated Maze palettes.
- Expedition nodes can reveal rumors, set story flags, complete contracts, award
  reputation, and discover breaches.

The Textual TUI uses one save slot at `saves/company.json`. It keeps the same app
commands, town services, travel screens, breach choices, generated Maze routes,
and manual combat-lite rules, but uses fixed zones: status header, main body,
detail pane, recent log, command dock, and footer hints. Those zones are backed
by small Textual widgets such as `StatusHeader`, `CommandDock`, `CombatPanel`,
`FormationBoard`, `DungeonRoomPanel`, `TownDashboardPanel`, and
`ExpeditionProgressStrip`. Use Up/Down to focus commands, Enter to activate,
number keys for visible options, single-key hotkeys where shown, and
Esc/Backspace for Back or Cancel where available. Combat uses a stable command
screen with skill focus, target previews, focused detail text, 2x2 party/enemy
formation boards, portraits/art, grouped resolution output, and morale/cohesion
state. Expedition playback shows a compact progress strip for completed, current,
and pending rooms. Haven screens show service availability, budget state, active
slots, reserves, supplies, and formation context, including a matching 2x2 party
formation board. `--demo --tui` and `--demo --cli` are intentionally rejected so
demo mode stays noninteractive and smoke-test friendly.

The legacy Rich CLI remains available through `--cli` for debugging, fallback
play, and fake-input tests. It should not be treated as an equal polish target
while the Textual TUI is the main game experience.

## Demo Mode

Demo mode runs a deterministic opening route and prints the resulting event log.

```bash
uv run python -m game.main --demo
```

Fallback:

```bash
python -m game.main --demo
```

To save the demo result:

```bash
python -m game.main --demo --save saves/demo.json
```

## Agent toolkit

Agents: read `AGENTS.md` and `docs/AGENT_CONTEXT_MAP.md` before broad exploration.

**Primary:** `.\rtk.ps1 <task>` (uses `.venv` on Windows when present).
**Fallback:** `uv run pytest` / `uv run ruff` / `python -m …` only when `rtk` is
unavailable.

Standard session surface:

```powershell
.\rtk.ps1 help
.\rtk.ps1 preflight
.\rtk.ps1 scout --task "one-line task"
.\rtk.ps1 boundaries
.\rtk.ps1 review-packet
.\rtk.ps1 quick
.\rtk.ps1 all
```

On Unix shells, `./rtk.sh` mirrors the minimum toolkit (`help`, `preflight`,
`scout`, `boundaries`, `review-packet`, `smoke`, `quick`, `test`, `check`).

## Test

Prefer `.\rtk.ps1` so commands use the local `.venv` when present:

```powershell
.\rtk.ps1 smoke
.\rtk.ps1 quick
.\rtk.ps1 tui
.\rtk.ps1 test
.\rtk.ps1 check
.\rtk.ps1 all
```

See **Agent toolkit** above for session commands (`preflight`, `scout`,
`boundaries`, `review-packet`, `help`).

Long TUI and full-suite runs can exceed 5 minutes. If your runner lets you set a
command timeout, use a value above 5 minutes for `.\rtk.ps1 tui`, `.\rtk.ps1 all`,
or broad `pytest` runs so they are not killed while still making normal progress.

Fallback (no rtk):

```bash
uv run pytest
uv run ruff check
uv run mypy src
```

If `uv` is unavailable:

```bash
python -m pytest
python -m ruff check
python -m mypy src
```

## Current v0.1 Scope

- Four starting heroes: Watchman, Cutpurse, Field Surgeon, Scribe.
- Recruit backgrounds and motives for starting roster and recruit offers.
- Personal quirks, earned quirks, Strain tiers/marks, morale, and party
  cohesion tags.
- 2x2 formation with lane protection.
- Melee, reach, ranged, and magic targeting rules.
- Effort as the only special-action resource.
- Downed, Mortal Wound, permanent death, and Order loss rules.
- Structured engine beat events for encounters, rounds, loot, breaches, town
  services, recovery, recruitment, supplies, morale, memory, and active-party
  changes.
- Active party plus reserves; expeditions use the active party slots.
- Act 1 frontier-company framing with an active Blackwood Road Charter.
- Old Road -> Blackwood Forest -> Shallow Cave -> Cave Mini Boss.
- A Maze-leak encounter before the optional full Maze descent.
- Discovery of `shallow_cave_breach`.
- Contract completion, reputation gain, story flags, world memory, shortcuts, and
  Maze rumors.
- Optional generated breach routes with deterministic room recipes, scout/hunt
  contracts, retreat, save/load preservation, and route collapse.
- Optional deterministic Pandora's Maze Depth 1 stub chosen at the Breach.
- Textual fullscreen frontend by default via `python -m game.main`, with fixed
  screen-region widgets, arrow-key command focus, number/hotkey activation,
  focused combat skill/target detail, 2x2 combat and formation boards,
  portraits/art, expedition progress strip, grouped resolution output, Textual
  company naming, save/load confirmations, breach choices, generated Maze
  routing, and town dashboards.
- Legacy Rich CLI fallback via `python -m game.main --cli`, with terminal frame,
  submenu organization, town dashboard, combat view, beat-prompted expedition log,
  save/load, and deterministic demo mode.

## Intentionally Not Included

This version does not include MP, cooldowns, stress bars, afflictions, virtues,
torch/light meters, global danger timers, a broad overworld campaign, hamlet
building, complex crafting, procedural editors, SQL, web backends, or mid-combat
saves.

## Known Limitations

- Combat balance is intentionally simple, though v0.1 is tuned to be harsher than
  a tutorial.
- Manual combat-lite supports exposed hero damage/support skill choices, legal
  target previews, adjacent formation movement, pass/delay, and retreat where
  the current encounter flow allows it. Enemies still resolve automatically, and
  this is not a full tactical combat UI.
- Maze Depth 1 is still a deterministic stub with minimal content; generated
  breach routes are the stronger early repeatable route experiment.
- Traits and morale are foundations with a small authored set, not a deep
  personality simulation yet.
- World travel is narrow and focused on the opening route, not a full map-driven
  campaign.
- Recruitment and town services are foundations, not complete town-building systems.
- Save files are campaign-level JSON only; mid-combat saves are not supported.
- The Textual interface still has no mouse support, full visual snapshot coverage,
  or mid-combat saves.

## License

The Charter is released under the MIT License. See [LICENSE](LICENSE).
