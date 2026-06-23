# combat-rules (overflow)

Use this when the ticket touches combat mechanics, targeting, enemy decision, actions, or manual combat-lite.

## READ_FIRST (3-6)

- `AGENTS.md` (combat/UI separation, RNG, boundaries)
- `project_sources/COMMANDS.md` (manual combat surface)
- `project_sources/ADRS.md` (ADR-002, ADR-003)
- `src/game/combat/` (only the files you'll touch)

## LIKELY_FILES (3-8)

- `src/game/combat/`
- `src/game/app/manual_combat.py`
- `src/game/app/views.py` (combat view models only)
- `src/game/app/actions.py` (only if wiring new actions)
- `data/skills.yaml`, `data/enemies.yaml` (authored kits)

## LIKELY_TESTS (focused)

- `tests/test_manual_combat.py`
- `tests/test_targeting.py`
- `tests/test_formation.py`
- `tests/test_enemy_actions.py`
- `tests/test_enemy_decision.py`

## VERIFY (1-3 commands)

```powershell
.\rtk.ps1 test tests/test_manual_combat.py tests/test_targeting.py tests/test_formation.py
.\rtk.ps1 check
```

## DO_NOT_READ

- `.venv/`, caches, `saves/`
- Full `tests/test_tui.py` unless the ticket crosses into combat UI

## BOUNDARIES

- Engine packages must not import `game.ui` — run `.\rtk.ps1 boundaries` if you touched engine code.
