# Agent Context Map

Date: 2026-06-16

Task router for agent sessions. **Do not duplicate `AGENTS.md`.**

## How to use

1. Read `AGENTS.md` once per session.
2. Run `.\rtk.ps1 preflight` for git snapshot and task-block guess (or
   `.\rtk.ps1 scout --task "…"` for multi-model Stage 1).
3. Pick the matching block below (or use path triggers).
4. Read only **READ_FIRST** + **LIKELY_FILES** before editing.
5. Run the block **VERIFY** commands; use `.\rtk.ps1 boundaries` when engine
   packages change.
6. Run `.\rtk.ps1 review-packet` before handoff; escalate to `.\rtk.ps1 check`
   + `.\rtk.ps1 all` for cross-cutting work.
7. Run `.\rtk.ps1 help` for the full task list.

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

## Path triggers (first match wins)

| Changed paths | Block |
|---|---|
| `src/game/combat/`, `src/game/app/manual_combat.py` | `combat-rules` |
| `src/game/campaign/`, `data/town.yaml`, `data/gear.yaml` | `campaign-town` |
| `src/game/expedition/`, `data/expeditions.yaml`, `data/world.yaml` | `expedition-dungeon` |
| `src/game/ui/tui*.py`, `src/game/ui/screens.py` | `textual-ui` |
| `src/game/ui/cli.py`, `src/game/ui/hci_text.py`, `src/game/ui/wounds.py` | `cli-legacy` |
| `data/*.yaml`, `src/game/data/` | `content-yaml` |
| `src/game/campaign/save_load.py`, save shape in `company.py` | `save-load` |
| `src/game/dev/`, `docs/dev/` | `ai-lab-balance` |
| `src/game/app/commands.py`, `controller.py`, `flows.py`, `actions.py`, `views.py` | `app-commands` |
| `*.md`, `project_sources/`, `CHANGELOG.md` | `docs-only` |

Copy-paste prompts: `prompts/agent_task_prompt_template.md`.

---

## combat-rules

Detailed: `docs/agent-tasks/combat-rules.md`

- **READ_FIRST**:
  - `AGENTS.md` (Architectural Rules)
  - `project_sources/COMMANDS.md` (Manual combat)
  - `project_sources/ADRS.md` (ADR-002, ADR-003)
- **LIKELY_FILES**:
  - `src/game/combat/`
  - `src/game/app/manual_combat.py`
  - `src/game/app/views.py`
  - `data/skills.yaml`, `data/enemies.yaml`
- **LIKELY_TESTS**:
  - `tests/test_manual_combat.py`
  - `tests/test_targeting.py`
  - `tests/test_formation.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_manual_combat.py tests/test_targeting.py tests/test_formation.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - `.venv/`, caches, `saves/`
- **BOUNDARIES**:
  - Run `.\rtk.ps1 boundaries` when engine packages change (no `game.ui` imports).

---

## campaign-town

- **READ_FIRST**:
  - `AGENTS.md` (town/YAML rules, active party slots, beat events)
  - `project_sources/REPO_MAP.md` (campaign/)
  - `project_sources/DATA_SCHEMAS.md` (town/gear/recruits/supplies)
- **LIKELY_FILES**:
  - `src/game/campaign/`
  - `src/game/app/flows.py`
  - `data/town.yaml`, `data/supplies.yaml`, `data/gear.yaml`, `data/recruits.yaml`
- **LIKELY_TESTS**:
  - `tests/test_town_loop.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_town_loop.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - `.venv/`, caches, `saves/` (unless reproducing a save bug)
- **BOUNDARIES**:
  - Save-shape changes require migration + `tests/test_save_load.py`.

---

## expedition-dungeon

- **READ_FIRST**:
  - `DESIGN.md` (scope)
  - `AGENTS.md` (Architectural Rules)
  - `project_sources/REPO_MAP.md` (expedition/)
- **LIKELY_FILES**:
  - `src/game/expedition/`
  - `src/game/app/flows.py`
  - `data/expeditions.yaml`, `data/world.yaml`
- **LIKELY_TESTS**:
  - `tests/test_dungeon.py`
  - `tests/test_generated_maze.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_dungeon.py tests/test_generated_maze.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - `.venv/`, caches, scratch lab output (`ai_lab_*.txt`)
- **BOUNDARIES**:
  - Keep Maze branches cardinally adjacent; don’t reintroduce diagonal links.

---

## textual-ui

- **READ_FIRST**:
  - `AGENTS.md` (Textual rules + widget layout)
  - `project_sources/CONVENTIONS.md` (UI conventions)
  - `project_sources/COMMANDS.md` (ScreenAction metadata)
- **LIKELY_FILES**:
  - `src/game/ui/tui.py`
  - `src/game/ui/tui_widgets.py`
  - `src/game/ui/tui_models.py`
  - `src/game/ui/screens.py`
  - `src/game/app/views.py`, `src/game/app/actions.py`
- **LIKELY_TESTS**:
  - `tests/test_tui.py`
- **VERIFY**:
  - `.\rtk.ps1 tui`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - `src/game/ui/cli.py` unless the ticket mentions legacy CLI
- **BOUNDARIES**:
  - UI uses app commands/view models; engine rules stay out of `src/game/ui/`.

---

## cli-legacy

- **READ_FIRST**:
  - `AGENTS.md` (CLI is legacy/dev fallback)
  - `project_sources/COMMANDS.md` (main menu + aliases)
- **LIKELY_FILES**:
  - `src/game/ui/cli.py`
  - `src/game/ui/hci_text.py`
  - `src/game/ui/wounds.py`
  - `src/game/ui/screens.py`
- **LIKELY_TESTS**:
  - `tests/test_cli.py`
  - `tests/test_main.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_cli.py tests/test_main.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - `tests/test_tui.py` unless verifying shared view-model behavior
- **BOUNDARIES**:
  - Prefer fake IO tests; avoid terminal-dependent behavior.

---

## content-yaml

- **READ_FIRST**:
  - `project_sources/DATA_SCHEMAS.md`
  - `AGENTS.md` (YAML authoring + balance tweakability)
  - The specific `data/*.yaml` being edited
- **LIKELY_FILES**:
  - `data/*.yaml`
  - `src/game/data/schemas.py`
  - `src/game/data/loaders.py`
- **LIKELY_TESTS**:
  - `tests/test_main.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_main.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - All of `data/` when only one file changed
- **BOUNDARIES**:
  - YAML changes that affect mechanics require focused domain tests.

---

## save-load

- **READ_FIRST**:
  - `AGENTS.md` (save slot + migration)
  - `src/game/campaign/save_load.py`
  - `src/game/campaign/company.py`
- **LIKELY_FILES**:
  - `src/game/campaign/save_load.py`
  - `src/game/campaign/company.py`
- **LIKELY_TESTS**:
  - `tests/test_save_load.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_save_load.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - `saves/*.json` unless reproducing a specific migration case
- **BOUNDARIES**:
  - Migration must be backward-compatible with tests.

---

## ai-lab-balance

Detailed: `docs/agent-tasks/ai-lab-balance.md`

- **READ_FIRST**:
  - `AGENTS.md` (Dev Tools Index)
  - `docs/dev/ai_lab_oracle_sweep_v0_1.md`
  - `project_sources/TESTING.md` (slow/anyio markers)
- **LIKELY_FILES**:
  - `src/game/dev/`
  - `src/game/combat/enemy_decision.py`
- **LIKELY_TESTS**:
  - `tests/test_ai_oracle.py`
  - `tests/test_policy_band_report.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_ai_oracle.py tests/test_policy_band_report.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - `ai_lab_*.txt` scratch output unless the ticket references it
- **BOUNDARIES**:
  - Dev tooling stays out of player-facing flows.

---

## app-commands

- **READ_FIRST**:
  - `project_sources/COMMANDS.md`
  - `project_sources/ADRS.md` (ADR-003)
  - `AGENTS.md` (app command bridge rules)
- **LIKELY_FILES**:
  - `src/game/app/commands.py`
  - `src/game/app/controller.py`
  - `src/game/app/flows.py`
  - `src/game/app/actions.py`
  - `src/game/app/views.py`
- **LIKELY_TESTS**:
  - `tests/test_hci_substrate.py`
- **VERIFY**:
  - `.\rtk.ps1 test tests/test_hci_substrate.py`
  - `.\rtk.ps1 check`
- **DO_NOT_READ**:
  - Unrelated flow tests across the whole suite
- **BOUNDARIES**:
  - UI does not mutate engine state directly; go through app commands + Results.

---

## docs-only

- **READ_FIRST**:
  - `AGENTS.md` (doc hierarchy)
  - `CHANGELOG.md` (if documenting meaningful work)
- **LIKELY_FILES**:
  - The specific `.md` files named in the ticket
  - `project_sources/` (handoff pack)
- **LIKELY_TESTS**:
  - None (unless documented commands/behavior changed)
- **VERIFY**:
  - `.\rtk.ps1 smoke` (optional)
- **DO_NOT_READ**:
  - Entire repo tree for a single-doc fix
- **BOUNDARIES**:
  - Don’t fork architecture rules; cross-link `AGENTS.md`.
