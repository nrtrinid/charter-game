# Agent Guide

## Token-Saving Operating Mode

- Start with this file before broad exploration.
- Prefer `git diff --name-only`, `rg --files`, and targeted `rg` searches over
  reading whole directories.
- Read exact files and narrow line ranges when possible; summarize large files
  instead of pasting them into context.
- Do not inspect virtualenvs, caches, build output, coverage output, or generated
  artifacts unless the task specifically requires it.
- Run focused tests first, such as `uv run pytest tests/test_cli.py -q` or
  `uv run pytest tests/test_cli.py::test_name -q`; run the full suite only for
  broad changes or final verification.
- On Windows, prefer `.\rtk.ps1 smoke`, `.\rtk.ps1 quick`,
  `.\rtk.ps1 tui`, `.\rtk.ps1 test <path-or-node>`, `.\rtk.ps1 check`,
  and `.\rtk.ps1 all` so commands use the local `.venv` Python 3.13 instead
  of the shell's default Python.
- Long TUI and full-suite runs can legitimately take more than 5 minutes. When
  running `.\rtk.ps1 tui`, `.\rtk.ps1 all`, or broad `pytest` commands, set a
  timeout above 5 minutes up front so the command does not need to be rerun just
  because the harness stopped waiting.
- Summarize command output unless raw output is requested. For long failures,
  keep the traceback, assertion, affected test, and final summary.
- Keep plans and progress updates short unless the task is broad or risky.
- Avoid re-reading files already inspected in the same turn unless they changed.
- Before ending any turn that makes or observes substantial repo changes, check
  whether `CHANGELOG.md` needs an `Unreleased` entry. If meaningful user-facing
  behavior, tooling, architecture, content, tests, or workflows changed, update
  `CHANGELOG.md` in the same turn with the current local date in `YYYY-MM-DD`
  form. If not updating it, explicitly say why in the final response.
- Add stable workflow, architecture, or product-surface lessons here in
  `AGENTS.md` as dated rolling memory. Use a short `YYYY-MM-DD - ...` bullet in
  the Durable Memory section. Do not document trivial edits or transient
  debugging details.
- Use Semantic Versioning for releases and choose the next version automatically
  when asked to prepare or cut a release. While the project is `0.x`, bump patch
  for fixes/docs/tooling, bump minor for user-facing gameplay/UI/content changes,
  and bump major only for an explicit compatibility/reset milestone.
- Suggest marking a release when `CHANGELOG.md` has meaningful `Unreleased`
  entries and the work is ready to hand off, publish, tag, or commit. If the user
  asks for release prep, handoff, publish, or "make it ready", update the project
  version and move `Unreleased` changelog entries under the automatically chosen
  version/date. Do not tag, commit, or push unless explicitly asked.

## Repo Layout

- `src/game/core`: IDs, RNG, events, and result helpers.
- `src/game/combat`: formation, targeting, initiative, actions, damage, death, and Order.
- `src/game/campaign`: company, roster, recruitment, town, save/load, reputation.
- `src/game/expedition`: route, nodes, travel, cave, and Maze stubs.
- `src/game/content`: game definition containers.
- `src/game/data`: Pydantic schemas and YAML loaders.
- `src/game/app`: structured commands and controller.
- `src/game/ui`: Textual TUI frontend, legacy Rich CLI rendering/wrapper, and
  Textual widget classes.
- `data`: authored YAML content.
- `tests`: deterministic rule and vertical-slice tests.

## Commands

Run before considering work complete:

```bash
./rtk.ps1 all
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

## Architectural Rules

- Do not redesign the game unless explicitly asked.
- Keep combat logic separate from UI.
- Do not import ui modules from combat, campaign, expedition, core, content, or data.
- All randomness must go through game.core.rng.
- Do not call Python random APIs directly outside game.core.rng.
- Add or update tests when changing mechanics.
- Keep content minimal.
- Prefer dataclasses for runtime state.
- Use Pydantic only for external data validation.
- Do not add cooldowns, MP, stress bars, torch meters, global timers, overworld
  systems, hamlet-building systems, complex crafting, or extra mechanics unless
  explicitly asked.
- Do not introduce new core mechanics just because they are common in RPGs.
- Keep authored content in YAML when practical.
- Keep balance numbers easy to tweak.
- UI should render engine events/results; engine systems should not print.
- Engine systems should emit explicit beat events for major facts such as
  encounter start/end, round start/end, loot, breach discovery, expedition return,
  town services, recruitment, recovery, supply purchases, and active-party changes.
  UI should not infer these major beats from raw message order.
- The Textual TUI is the main player-facing game surface. `python -m game.main`
  and `charter-game` should launch it by default.
- The Rich CLI is a legacy/dev fallback through `python -m game.main --cli`.
  Keep it useful for debugging, fallback play, and fake-input tests, but do not
  spend feature-parity or polish time on it unless explicitly asked.
- CLI input, prompts, Rich tables, and terminal pacing belong in `src/game/ui`.
- CLI decisions should be presented as numbered option lists. Typed aliases may
  work as shortcuts, but raw `y/n` prompts should not be the primary interface.
- CLI actions should use the app/UI view models when available. Disabled actions
  should render visibly but be rejected by input handling. Enter may select only a
  marked safe default.
- Textual widgets should use app commands and app/UI view models rather than
  importing combat, campaign, expedition, content, or data internals for rendering
  or rules. Native Textual widgets are preferred over wrapping Rich renderers.
- Keep reusable Textual surfaces such as status headers, command docks, formation
  boards, combat panels, progress strips, and town dashboards in UI widget classes
  rather than expanding `CharterApp` with more rendering logic.
- Textual input should support Up/Down focus, Enter activation, number shortcuts,
  shown hotkeys, Esc/Backspace Back/Cancel, visible disabled actions, and fixed
  screen zones for header, body, detail, recent log, command dock, and footer.
- App commands in `src/game/app` should remain the bridge between UI choices and
  engine behavior.
- Manual combat-lite belongs at the app/UI layer. Combat rules stay in
  `src/game/combat`; the CLI may choose hero skills and targets, but enemies,
  targeting legality, damage, death, and turn order remain engine/app coordinated.
- Combat previews such as hit chance, damage estimate, and targeting reason should
  be calculated outside `src/game/ui`; renderers should display those facts, not
  derive them.
- Town service costs and roster cap live in `data/town.yaml`; supply costs remain
  in `data/supplies.yaml`.
- Expeditions should use `CompanyState.active_party_slots`; unassigned living
  roster members are reserves.
- The interactive CLI uses a single save slot at `saves/company.json` unless the
  design asks for multiple slots.
- Preserve noninteractive demo mode for smoke testing.

## Durable Memory

- 2026-06-05 - The Textual TUI is the default game experience. Treat CLI as a
  legacy/dev fallback and avoid polishing it as a second equal frontend unless
  explicitly requested.
- 2026-06-05 - Rolling documentation memory should be dated. Update
  `CHANGELOG.md` and durable `AGENTS.md` lessons during the same turn as
  meaningful product, workflow, architecture, tooling, test, or content changes.

## Source of Truth

Design authority order:

1. `DESIGN.md`
2. `ROADMAP.md`
3. `data/*.yaml`
4. `tests/`
5. `README.md`

## Optional Subagent Workflow

- Use subagents only for bounded exploration, review, test analysis, or architecture
  checks; do not allow parallel write-heavy work that could cause file conflicts.
- If subagents are unavailable, perform the same review checks manually.
- Main implementation decisions remain with the primary agent.
- Apply only high-confidence fixes that improve compliance with the current design.

## Testing Expectations

Mechanics changes need tests. Rule tests should check behavior, not only object
creation. Vertical-slice tests should prove the opening route, breach discovery,
reputation/Coin rewards, return to town, and save/load. Town-loop tests should
cover spending Coin, roster cap, recovery, supplies, active-party assignment, and
save/load compatibility. Manual combat tests should cover legal skills, legal
targets, resolving hero actions, enemy auto-resolution, and combat end. CLI
changes should use fake input/output tests where possible instead of depending on
a real terminal. Textual changes should use `run_test()`/Pilot tests for
keyboard behavior and state transitions.

## Done Criteria

- Project structure exists.
- Docs, YAML content, schemas, loader, CLI, demo mode, save/load, and tests exist.
- Formation, targeting, turn order, death, Order, save/load, and vertical-slice tests pass.
- `pytest`, `ruff`, and `mypy` have been run or the fallback reason is reported.
