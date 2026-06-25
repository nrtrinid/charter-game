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

### Added

- 2026-06-15 - Added workspace-root agent pointers (`text-adventure/AGENTS.md`,
  `README.md`, `.cursor/rules/charter-root.mdc`) so sessions route into
  `dungeon-party-game/` before exploration.
- 2026-06-15 - Expanded `AGENTS.md` with documentation hierarchy, dev-tools
  index, and refreshed durable memory for v0.1, AI lab entry, Maze routing, and
  test-count policy; cross-linked `project_sources/CONVENTIONS.md` and stale
  warnings in `CURRENT_STATE.md` / `TESTING.md`.
- 2026-06-15 - Added agent task router (`docs/AGENT_CONTEXT_MAP.md`),
  `.agentignore`, and `prompts/agent_task_prompt_template.md`; linked from
  `AGENTS.md` and `project_sources/00_UPLOAD_INDEX.md`.
- 2026-06-15 - Added `src/game/dev/agent_preflight.py` and
  `src/game/dev/check_engine_boundaries.py` with `.\rtk.ps1 preflight` /
  `.\rtk.ps1 boundaries` for session routing and engine UI-import checks;
  covered by `tests/test_agent_preflight.py`.
- 2026-06-15 - Added 3-stage agent workflow (`prompts/agent_workflow.md`,
  `scout_packet_schema.md`, `drift_check_prompt.md`, `implement_from_plan.md`)
  and `.\rtk.ps1 scout --task` for cheap scout → smart drift → implement loops.
- 2026-06-16 - Added repo-local agent routing (`.cursor/rules/project-root.mdc`,
  `.cursorignore`) and overflow task docs under `docs/agent-tasks/`.
- 2026-06-16 - Added `.\rtk.ps1 help`, `.\rtk.ps1 review-packet`, cross-platform
  `rtk.sh`, and `src/game/dev/agent_review_packet.py` for lightweight handoff
  packets; covered by `tests/test_agent_review_packet.py`.
- 2026-06-16 - CI runs `.\rtk.ps1 boundaries` after lint/type checks.
- 2026-06-23 - Added `docs/STANDARDIZED_AGENT_ERGONOMICS_ROADMAP.md` (cross-repo
  ergonomics contract); linked from `AGENTS.md`; Phase 1 marked complete.
- 2026-06-23 - Aligned `AGENTS.md`, `README.md`, and `docs/AGENT_CONTEXT_MAP.md`
  on the same RTK contract (`help`, `preflight`, `scout`, `boundaries`,
  `review-packet`, `quick`, `all`); ergonomics Phase 2 marked complete.
- 2026-06-23 - Normalized all `docs/AGENT_CONTEXT_MAP.md` task blocks to the
  Phase 3 field schema (`Use when`, plain labels, `NOTES`); ergonomics Phase 3
  marked complete.
- 2026-06-23 - Added `.\rtk.ps1 doctor` (`src/game/dev/agent_doctor.py`) for
  read-only handoff freshness checks; six-part handoff template in `AGENTS.md`
  and `review-packet`; ergonomics Phase 4 marked complete.

### Changed

- 2026-06-24 - Split monolithic `tui_widgets.py` into the `tui_widgets/` package
  (shell, town, dungeon, combat, animation, constants, plus shared `events.py` and
  `formation.py`); barrel `__init__.py` preserves `from game.ui.tui_widgets import …`.
- 2026-06-24 - Extracted TUI screen rendering from `tui.py` into `tui_render/`
  (shell, regional, town, dungeon, combat) mirroring the Slice 2 handler split;
  shared constants/helpers moved to `tui_constants.py`; `CharterApp` keeps thin
  delegates and the screen pipeline.
- 2026-06-23 - Agent docs prefer `.\rtk.ps1` over raw `uv run pytest` / `ruff` /
  `mypy`; `uv` and `python -m` remain documented fallbacks when `rtk` is unavailable.

### Fixed

- 2026-06-23 - Re-encoded `.agentignore` to UTF-8 (no NUL noise in shell tools).
- 2026-06-23 - Preflight porcelain path parsing via `parse_status_path` so modified
  paths under `src/` are not misread as `rc/...`; regression tests in
  `tests/test_agent_preflight.py`.
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
