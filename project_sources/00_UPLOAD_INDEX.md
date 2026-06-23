# GPT Project Upload Index

Date: 2026-06-10

This folder is a curated source pack for a ChatGPT/GPT project. Upload these
Markdown files as stable project sources instead of uploading the whole repo.
This snapshot reflects the committed v0.1 documentation state; ignored local
scratch files and generated artifacts are intentionally excluded.

## Recommended Upload Set

Upload all files in this folder:

- `00_UPLOAD_INDEX.md`
- `PROJECT_CONTEXT.md`
- `REPO_MAP.md`
- `DATA_SCHEMAS.md`
- `CONVENTIONS.md`
- `COMMANDS.md`
- `ADRS.md`
- `CURRENT_STATE.md`
- `TESTING.md`
- `GLOSSARY.md`
- `PROJECT_INSTRUCTIONS.md`

`PROJECT_INSTRUCTIONS.md` is best pasted into the project's instructions field.
The other files are best uploaded as project sources.

Live session routing (not part of the upload pack): `docs/AGENT_CONTEXT_MAP.md`,
`prompts/agent_workflow.md`, and `prompts/agent_task_prompt_template.md` in the
repo root tree under `dungeon-party-game/`.

## Source Files To Upload If You Want More Code Context

If the project source budget allows selected code, upload these spine files after
the Markdown docs:

- `README.md`
- `DESIGN.md`
- `ROADMAP.md`
- `AGENTS.md`
- `pyproject.toml`
- `src/game/app/commands.py`
- `src/game/app/controller.py`
- `src/game/app/actions.py`
- `src/game/app/views.py`
- `src/game/core/events.py`
- `src/game/core/result.py`
- `src/game/core/rng.py`
- `src/game/data/schemas.py`
- `src/game/data/loaders.py`
- `src/game/content/definitions.py`
- `src/game/campaign/company.py`
- `src/game/campaign/town.py`
- `src/game/campaign/roster.py`
- `src/game/campaign/save_load.py`
- `src/game/combat/combat_state.py`
- `src/game/combat/formation.py`
- `src/game/combat/targeting.py`
- `src/game/combat/actions.py`
- `src/game/combat/damage.py`
- `src/game/combat/death.py`
- `src/game/combat/turn_order.py`
- `src/game/expedition/dungeon.py`
- `src/game/expedition/expedition.py`
- `src/game/expedition/generated_maze.py`
- `src/game/expedition/maze_director.py`
- `data/world.yaml`
- `data/expeditions.yaml`
- `data/heroes.yaml`
- `data/skills.yaml`
- `data/enemies.yaml`
- `data/recruits.yaml`
- `data/traits.yaml`
- `data/town.yaml`
- `data/supplies.yaml`
- `data/gear.yaml`
- `tests/test_vertical_slice.py`
- `tests/test_save_load.py`
- `tests/test_manual_combat.py`
- `tests/test_town_loop.py`
- `tests/test_targeting.py`
- `tests/test_tui.py`

## Do Not Upload

- `.venv/`
- `node_modules/`
- `build/`, `dist/`, `coverage/`, cache folders
- `saves/` unless you intentionally want a save file inspected
- `.env`, keys, tokens, private data
- large generated artifacts
- lockfiles unless dependency versions become relevant
