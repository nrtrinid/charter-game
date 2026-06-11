# Testing Strategy

Date: 2026-06-10

## Test Philosophy

- Mechanics changes need tests.
- Rule tests should verify behavior, not just object construction.
- Prefer focused tests first, then broader suite for cross-cutting changes.
- Use deterministic RNG seeds.
- UI changes should use fake input/output or Textual `run_test()`/Pilot tests
  where possible.
- Save/load changes need compatibility coverage for missing old fields.

## Core Commands

```powershell
.\rtk.ps1 smoke    # test_main.py only
.\rtk.ps1 quick    # skips @pytest.mark.anyio and @pytest.mark.slow
.\rtk.ps1 tui      # TUI tests only
.\rtk.ps1 test tests/test_file.py
.\rtk.ps1 check
.\rtk.ps1 all      # full pytest + ruff + mypy
```

Current v0.1 release-readiness status: `.\rtk.ps1 test` passes locally
(`785 passed, 1 skipped` on 2026-06-10). `.\rtk.ps1 lint` and
`.\rtk.ps1 types` also pass locally.

Opening-route integration scripts live in `tests/conftest.py` as
`OPENING_DUNGEON_TO_WORKS_CACHE` and `OPENING_POST_COMBAT`. Update those tuples
when the canonical opening critical path changes.

```bash
uv run pytest
uv run ruff check
uv run mypy src
```

## Representative Behavior Tests

Opening vertical slice should prove:

- The deterministic opening route visits expected authored nodes.
- Shallow Cave breach becomes known.
- Shallow Cave known route unlocks.
- contract rewards grant reputation and Coin once.
- expedition history records major completions once.
- route returns to Haven.
- key inventory/rewards are preserved.
- save/load keeps those facts.

Save/load tests should prove:

- Company identity, roster, resources, supplies, inventory, gear, contracts,
  reports, world memory, dungeon memory, breach memory, hero memories, timeline,
  strain marks, earned quirks, and recruitment offers round-trip.
- Missing old fields migrate to safe defaults.
- Legacy status/fatigue/condition fields migrate to life state, tags, strain, and
  strain marks.
- App load failure returns a failed `Result` instead of crashing.

Combat tests should cover:

- Formation lane protection and exposed backliners.
- Melee, reach, ranged, and magic target legality.
- Position restrictions such as `front_only`.
- Damage, healing, Downed, Mortal Wounds, death, and Order/morale consequences.
- Turn order and initiative modifiers.
- Manual combat skill selection, target selection, enemy auto-resolution, delay,
  pass, movement, retreat, and encounter ending.
- Enemy decision heuristics and learning substrate when changed.

Town tests should cover:

- Recruit offer generation and hiring.
- Roster cap and active party assignment.
- Recovery cost/effects.
- Supply purchases and town upgrades.
- Gear purchase/equip/unequip.
- Contract board locked/available/active/completed/repeatable states.
- Save/load compatibility for town state.

Dungeon/generated route tests should cover:

- Starting and resuming an interactive dungeon.
- Moving between revealed/available nodes.
- Room action requirements, costs, rewards, and once-only behavior.
- Safe return to Haven.
- Generated route seed determinism, retrace/withdraw/retreat behavior, room count
  goals, scout/hunt contracts, and route collapse.

## Focused Test Examples

Run one file:

```powershell
.\rtk.ps1 test tests/test_save_load.py
```

Run one test node:

```powershell
.\rtk.ps1 test tests/test_vertical_slice.py::test_deterministic_opening_vertical_slice
```

Direct pytest equivalent:

```bash
uv run pytest tests/test_vertical_slice.py::test_deterministic_opening_vertical_slice -q
```
