# Suggested GPT Project Instructions

When helping with this project:

- Assume this is a Python text-first expedition company RPG called The Charter.
- Treat `DESIGN.md` as the primary design authority, followed by `ROADMAP.md`,
  `data/*.yaml`, tests, and then `README.md`.
- Prefer small, testable changes that fit the existing architecture.
- Keep story/content data-driven in YAML when practical.
- Keep engine rules separate from UI rendering.
- Do not import `game.ui` from combat, campaign, expedition, core, content, or
  data modules.
- Use structured app commands from `src/game/app/commands.py` as the interaction
  boundary.
- UI should render `Result` objects, `GameEvent` records, `ScreenAction`
  metadata, and app view models; it should not derive combat legality or major
  story beats itself.
- Route randomness through `game.core.rng.GameRng`.
- Prefer dataclasses for runtime state and Pydantic only for external YAML
  validation.
- Add or update tests for mechanics, save/load shape, command behavior, and UI
  state transitions.
- Use focused tests first; run broader checks for cross-cutting changes.
- Treat Textual as the main player-facing frontend and Rich CLI as a legacy/dev
  fallback unless told otherwise.
- Do not propose large architectural rewrites unless asked.
- Do not add MP, cooldowns, stress bars, torch timers, hamlet building, broad
  overworld systems, SQL/web backends, ECS, complex crafting, procedural editors,
  or mid-combat saves unless explicitly requested.
- When suggesting code, mention the exact file where it belongs.
- When changing runtime save state, include `to_dict()`/`from_dict()` migration
  considerations and save/load tests.
- Keep Act 1 content grounded in hard frontier-company work, exploitation,
  contracts, practical motives, and the Maze becoming worse because people keep
  using it.
