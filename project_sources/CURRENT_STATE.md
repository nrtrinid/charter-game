# Current State

Date: 2026-06-10

Snapshot as of that date. Stale after 2026-06-10 for test counts and release
status — run `.\rtk.ps1 quick` for the current quick-suite count (709 passed on
2026-06-15) or `.\rtk.ps1 all` before handoff. Canonical agent rules live in
`AGENTS.md`, not this file.

## Working

- Python package scaffold with src layout, CLI entrypoint, Textual default TUI,
  Rich CLI fallback, deterministic demo mode, tests, lint, and type-check config.
- Data-driven YAML content and Pydantic validation for core authored data.
- Starting company with four initial classes: Watchman, Cutpurse, Field Surgeon,
  and Scribe.
- Active party slots plus reserves, using a 2x2 formation.
- Formation lane protection, adjacency movement, melee/reach/ranged/magic target
  rules, cover penalty, and attack previews.
- Effort as the special-action resource.
- Downed, Mortal Wounds, permanent death, morale/cohesion, strain tiers/marks,
  combat tags, and turn order.
- Manual combat-lite for hero skill/target choices, movement, delay/pass/retreat
  where implemented, enemy auto-resolution, and grouped result output.
- Town loop with roster, recruitment, recovery, supply purchase, upgrades, gear,
  formation, contracts, ledger, memorial, and reports.
- Coin economy and reputation.
- Contract board with postings gated by completed contracts and known breaches.
- Opening route through Old Road, Blackwood wilderness paths, Shallow Cave, Cave
  Mini Boss, Maze Breach, optional Maze Depth 1 stub, and Haven return.
- Generated breach routes with deterministic seeds, room recipes, scout/hunt
  contract support, collapse/withdraw/retrace behavior, and save/load
  preservation.
- World memory, dungeon memory, breach memory, known routes, known lore, flags,
  expedition history, reports, company timeline, hero memories, fresh memory, and
  earned-quirk infrastructure.
- Save/load for campaign-level JSON with compatibility migration tests.
- Textual TUI fixed regions and widgets for status, command dock, combat,
  formation, dungeon, town, and expedition progress.

## Partial Or Deliberately Limited

- Maze Depth 1 is still a deterministic stub; generated breach routes are the
  stronger repeatable-route experiment.
- Manual combat-lite is not a full tactical combat UI. Enemies are automatic, and
  support skills/movement/retreat are limited to implemented flows.
- World travel is narrow and focused on the v0.1 opening route, not a full map
  campaign.
- Recruitment, gear, upgrades, and town services are foundations rather than deep
  town-building systems.
- Traits, morale, strain marks, hero memories, and earned quirks exist as
  foundations with a small authored set.
- Save files are campaign-level only; mid-combat saves are intentionally not
  supported.
- Textual has no mouse-first workflow and may not yet have exhaustive visual
  snapshot coverage.
- Rich CLI is useful as a fallback/dev surface but should not receive equal polish
  unless explicitly requested.

## Known Limitations

- Combat balance is intentionally simple and harsh.
- The route/content volume is intentionally small.
- Some future Act 1 contracts exist as scaffolded world data but are not broad
  playable routes yet.
- Gameplay, pytest, Ruff, and mypy are v0.1-ready locally. The remaining release
  work is commit/tag/push mechanics and any optional public-repo metadata such as
  CI or LICENSE.

## Near-Term Priorities

1. Keep Textual TUI polish focused on the main repeated workflows: town, route,
   combat, reports, save/load, and formation.
2. Expand generated breach route quality and contract integration before adding a
   broad overworld.
3. Continue hardening save/load compatibility when adding runtime state.
4. Add focused tests with every mechanics change.
5. Tune combat, Coin, supplies, recovery, gear, and contract rewards through
   visible, deterministic rules.
6. Keep story/content additions grounded in Act 1 frontier-company work and Maze
   extraction consequences.
