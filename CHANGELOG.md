# Changelog

Versioning follows Semantic Versioning. During `0.x`, patch releases are for
fixes/docs/tooling, minor releases are for user-facing gameplay/UI/content
changes, and major releases are reserved for explicit compatibility or reset
milestones.

Agents should suggest a release when `Unreleased` contains meaningful completed
work. During release prep, they may choose the next version, update the project
version, and move entries under the chosen version/date; commits, tags, and pushes
still require explicit user approval.

## Unreleased

### Fixed

- 2026-06-11 - Kept generated Maze branches, previews, and seamless forward
  extensions cardinally adjacent so routes no longer create diagonal or
  multi-cell room links.

## [0.1.0] - 2026-06-10

### Added

- 2026-06-07 - Added typed combat effect events and Textual combat callouts for
  Effort drain/restoration, Guard mitigation, and quirk-triggered resource beats.
- 2026-06-06 - Added the dev-only AI lab command suite with enemy package
  health reports, in-memory counterfactual sweeps, authored route envelope
  evaluation, generated Maze route evaluation, and focused regression tests.
- 2026-06-06 - Added fresh-memory and structured earned-quirk manifestation
  infrastructure with report/finalization integration, direct memory signal
  events, starter earned quirks, and save/load compatibility.
- 2026-06-05 - Added backward-compatible visible combat damage ranges, first used
  by Cave Maw Brute Drag Forward and Maw Slam.
- 2026-06-05 - Converted authored hero, enemy, and Field Surgeon healing values
  to narrow visible ranges following the damage range design language.
- 2026-06-05 - Added design reference language for visible combat damage ranges
  and when fixed, tight, high-floor, or wild variance should be used.
- 2026-06-05 - Added `--cli` as the explicit launcher for the legacy Rich CLI
  fallback.
- Added low-token operating guidance to `AGENTS.md` for targeted search, narrow
  file reads, compact command output, and focused tests first.
- Added `rtk.ps1`, a small Windows repo toolkit for smoke tests, focused tests,
  lint/type checks, full verification, and local dev setup through the project
  `.venv`.
- Added a handoff-memory rule: substantial changes should update durable repo
  memory, while trivial edits and transient debugging details should stay out.
- Added an auto-versioning rule so release prep can choose the next SemVer
  version without asking for manual version decisions.
- Added a release-cadence rule so agents can suggest when a release should be
  marked and perform changelog/version prep when asked.
- Tightened the durable-memory rule so substantial repo changes require a
  changelog check before the agent ends its turn.
- 2026-06-05 - Added dated durable-memory guidance to `AGENTS.md` so future
  rolling documentation updates include local-date context.

### Changed

- 2026-06-10 - Cleared the v0.1 local quality gates: full pytest, Ruff, and
  mypy now pass after release documentation, UX, and dev-tool typing cleanup.
- 2026-06-08 - Formation mini board now uses the same slot layout and
  right-facing mini portraits as combat (`Back Left | Front Left` over
  `Back Right | Front Right`); company `[C]` uses this mini portrait board.
- 2026-06-08 - Added combat-style mini portraits to the formation board and
  hero portrait detail panes on formation, assign-hero, roster, and company views.
- 2026-06-08 - Reworked Haven hub, company yard, pack, armory, quartermaster,
  and company overview screens with compact table panels, first-visit hints,
  and focus-detail text for gear and supply purchases.
- 2026-06-07 - Documented that long TUI and full-suite test runs may need
  command timeouts above 5 minutes.
- 2026-06-07 - Refined Textual combat beats with a compact status/effect line
  under damage outcomes and shorter Guard mitigation copy.
- 2026-06-06 - Calibrated AI lab report statuses so route envelope band
  violations warn/fail correctly, no-effect counterfactuals rank neutrally, and
  low-reach Wolf Mark behavior reports as low-use acceptable.
- 2026-06-05 - Replaced overlapping Fatigue/Condition hero wear with Strain
  tiers and Strain Marks, moved Stunned/Knocked Down into combat tags, and kept
  Downed/Dead as special life states with old save compatibility.
- 2026-06-05 - Made the Textual TUI the default launcher for `charter-game` and
  `python -m game.main`, with `--tui` kept as an explicit compatibility alias.
- 2026-06-05 - Reframed `AGENTS.md` so future agents treat the TUI as the main
  player-facing surface and the Rich CLI as a legacy/dev fallback instead of a
  parallel polish target.
- 2026-06-05 - Updated README run instructions and frontend scope language for
  the default TUI flow, legacy `--cli` fallback, and Coin-based town services.
- Polished the contract board so locked and completed postings stay off the
  board, command labels stay focused on contract names, and contract focus text
  carries the objective and charter-only risk note.
- Made dungeon `Interact` the consistent entry point for room actions, keeping
  loot and curios out of the navigation dock, surfacing blocked requirements in
  focus text, and removing completed-only interactions from the command list.
- Cleaned up the dungeon node screen with concise route focus details, flat
  numbered commands, a readable minimap legend, focused exit markers, and small
  interactable hints.
- Replaced visible dungeon route risk labels with concrete route state,
  exceptional warnings, and action costs while preserving internal risk metadata
  for input safety.
- Limited visible dungeon location `!` markers to boss routes while keeping
  internal risk metadata for safe default behavior.
- Made the Black Stone Gate's key and force interactions clear together after
  either route opens the gate.
- Revamped the dungeon map page with the minimap renderer's full graph, node
  memory notes, route dossiers, and inventory/action requirements on location
  nodes, and fixed minimap route numbers when room interactions are present.
- Made pytest quiet by default with `addopts = "-q"` in `pyproject.toml`.
