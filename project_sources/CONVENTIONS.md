# Coding And Design Conventions

Date: 2026-06-10

Canonical architecture rules: `AGENTS.md` § Architectural Rules. This file
summarizes tooling and handoff conventions; do not fork engine/UI boundaries here.

## Language And Tooling

- Python package using `src/` layout.
- Requires Python `>=3.12,<3.15`.
- Runtime dependencies: Pydantic, PyYAML, Rich, Textual.
- Dev tools: pytest, anyio, hypothesis, ruff, mypy.
- Ruff line length is 100.
- Mypy checks `src` with `check_untyped_defs = true`.

## Verification Commands

Preferred on Windows:

```powershell
.\rtk.ps1 smoke
.\rtk.ps1 quick
.\rtk.ps1 test tests/test_file.py
.\rtk.ps1 check
.\rtk.ps1 all
```

For v0.1 release-readiness, `.\rtk.ps1 test`, `.\rtk.ps1 lint`, and
`.\rtk.ps1 types` all pass locally.

Direct commands:

```bash
uv run pytest
uv run ruff check
uv run mypy src
```

Fallback:

```bash
python -m pytest
python -m ruff check
python -m mypy src
```

## Architecture Rules

- Do not redesign the game unless explicitly asked.
- Keep combat logic separate from UI.
- Do not import `game.ui` modules from combat, campaign, expedition, core,
  content, or data.
- All randomness must go through `game.core.rng.GameRng`.
- Add or update tests when changing mechanics.
- Keep content minimal and data-driven.
- Prefer dataclasses for runtime state.
- Use Pydantic only for external data validation.
- Keep authored content in YAML when practical.
- Keep balance numbers easy to tweak.
- UI should render engine events/results; engine systems should not print.
- Major facts should be explicit structured events, not inferred by UI from raw
  message order.

## UI Conventions

- Textual TUI is the main player-facing surface.
- Rich CLI is a legacy/dev fallback through `python -m game.main --cli`.
- CLI and TUI decisions should be represented as numbered or focused actions
  backed by `ScreenAction` metadata.
- Enter may select only a marked safe default.
- Disabled actions should remain visible when useful but rejected by input
  handling.
- Textual widgets should use app commands and app/view models instead of
  importing engine internals for rendering or rules.
- Reusable Textual surfaces belong in widget classes rather than expanding
  `CharterApp`.

## Gameplay Design Constraints

- Hard should mean strategically harsh rather than randomly unfair.
- Visible risk, scarce recovery, persistent injuries, and tempting overreach are
  preferred over hidden dice spikes.
- Damage ranges should usually be narrow and visible.
- Effort is the only special-action resource in v0.1.
- Avoid adding MP, cooldowns, stress bars, torch timers, full overworld systems,
  hamlet building, complex crafting, SQL/web backends, ECS, procedural editors,
  or mid-combat saves unless explicitly requested.

## Changelog And Memory

- Before ending a meaningful repo-changing turn, check whether `CHANGELOG.md`
  needs an `Unreleased` entry.
- Durable architecture/workflow lessons can be added to `AGENTS.md` with a dated
  `YYYY-MM-DD - ...` bullet.
- Do not document trivial edits or transient debugging details as durable memory.

## Design Authority Order

1. `DESIGN.md`
2. `ROADMAP.md`
3. `data/*.yaml`
4. `tests/`
5. `README.md`
