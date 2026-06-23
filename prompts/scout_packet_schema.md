# Scout Packet Schema

Date: 2026-06-15

Stage 1 output only. Return **exactly one** fenced YAML block labeled `SCOUT_PACKET`.
Do not implement code in Stage 1.

## Template

```yaml
# SCOUT_PACKET
task_summary: "<one line restatement of the task>"
block_id: "<primary AGENT_CONTEXT_MAP block, e.g. combat-rules>"
block_ids_extra: []  # optional secondary blocks
block_rationale: "<why this block fits>"

read_first:
  - "<path or doc section, smallest set, ordered>"

likely_files:
  - "<path>"

likely_tests:
  - "<tests/test_*.py>"

verify_commands:
  - ".\\rtk.ps1 check"
  - ".\\rtk.ps1 test <paths from likely_tests>"

boundaries:
  - "<rule from context map / AGENTS that applies>"

do_not_read:
  - "<paths to skip per .agentignore or block>"

smallest_slice: "<one PR-sized implementation step>"

risks:
  - "<save migration, YAML loader, UI import, TUI slow tests, etc.>"

open_questions:
  - "<needs human decision, or empty list>"

preflight_blocks_guess: "<comma-separated blocks from preflight, if any>"
```

## Rules

- `read_first` should be **5 items or fewer** unless the task truly spans areas.
- `likely_files` is a guess, not a mandate — drift check may trim it.
- `verify_commands` must use `.\rtk.ps1`, not raw pytest, unless a single test node.
- If preflight suggested multiple blocks, set `block_id` to the **primary** one and
  list others in `block_ids_extra`.
- Do not paste file contents into the packet — paths and section names only.

## Example (combat tweak)

```yaml
# SCOUT_PACKET
task_summary: "Bandit dirty_finish damage range visible in combat preview"
block_id: combat-rules
block_ids_extra: [content-yaml]
block_rationale: "Skill resolution and preview live in combat; numbers may be in skills.yaml"

read_first:
  - "docs/AGENT_CONTEXT_MAP.md block combat-rules"
  - "src/game/combat/preview.py"
  - "data/skills.yaml (dirty_finish entry only)"

likely_files:
  - "data/skills.yaml"
  - "src/game/combat/preview.py"
  - "tests/test_manual_combat.py"

likely_tests:
  - "tests/test_manual_combat.py"

verify_commands:
  - ".\\rtk.ps1 check"
  - ".\\rtk.ps1 test tests/test_manual_combat.py"

boundaries:
  - "No game.ui imports in combat"
  - "Previews calculated outside src/game/ui"

do_not_read:
  - ".venv/"
  - "tests/test_tui.py"
  - "project_sources/CURRENT_STATE.md"

smallest_slice: "Adjust dirty_finish YAML range + one preview assertion"

risks:
  - "YAML change needs loader smoke if schema fields change"

open_questions: []

preflight_blocks_guess: "combat-rules, content-yaml"
```
