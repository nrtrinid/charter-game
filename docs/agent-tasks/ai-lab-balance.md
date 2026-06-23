# ai-lab-balance (overflow)

Use this for dev-only audits, oracle sweeps, route lab work, and enemy AI tuning driven by lab metrics.

## READ_FIRST (3-6)

- `AGENTS.md` (Dev Tools Index + workflow rules)
- `docs/dev/ai_lab_oracle_sweep_v0_1.md` (how to interpret oracle sweeps)
- `project_sources/TESTING.md` (slow/anyio marker policy)
- The specific command/module you will run in `src/game/dev/`

## LIKELY_FILES (3-8)

- `src/game/dev/ai_lab.py`
- `src/game/dev/ai_oracle.py`
- `src/game/dev/route_lab.py`
- `src/game/combat/enemy_decision.py`
- `tests/test_ai_oracle.py`, `tests/test_policy_band_report.py`

## LIKELY_TESTS (focused)

- `tests/test_ai_oracle.py`
- `tests/test_policy_band_report.py`
- `tests/test_route_lab.py` (only if touching route lab)
- `tests/test_ai_lab.py` (only if touching ai_lab reporting)

## VERIFY (1-3 commands)

```powershell
.\rtk.ps1 test tests/test_ai_oracle.py tests/test_policy_band_report.py
.\rtk.ps1 check
```

## DO_NOT_READ

- `ai_lab_*.txt` scratch output unless the ticket references a specific run

## BOUNDARIES

- Dev-only tooling stays out of player-facing flows; keep CI cheap.
