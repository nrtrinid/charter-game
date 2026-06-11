# v0.1 Gameplay Readiness Audit

Date: 2026-06-09 (updated 2026-06-10 - docs/test refresh)
Project: The Charter (`charter-game` / `dungeon-party-game`)
Repository: https://github.com/nrtrinid/charter-game
Authority order: `DESIGN.md` → `ROADMAP.md` → `data/*.yaml` → tests → `README.md`

---

## Verdict

**READY TO TAG AFTER RELEASE HYGIENE**

- The v0.1 gameplay loop is implemented end-to-end: Haven prep → opening expedition → manual combat-lite → boss/breach → generated breach routes or Maze Depth 1 stub → safe return → filed report → save/load.
- Engine, controller, TUI, slow harness, and full pytest suites pass locally through `.\rtk.ps1`.
- **Human TUI playthrough verified** (2026-06-09) by maintainer; no gameplay blockers reported.
- The tracked worktree is clean before this documentation refresh; ignored local caches, lab output, save files, and resume scratch remain excluded.
- Remaining gaps are release mechanics, not missing core systems, TUI breakage,
  or local quality gates.
- `.\rtk.ps1 test`, `.\rtk.ps1 lint`, and `.\rtk.ps1 types` all pass locally.

---

## Tested / Inspected

### Commands run

| Command | Result | Notes |
|---------|--------|-------|
| `git status --short --branch` (before docs refresh) | **CLEAN** | `main...origin/main` |
| `.\rtk.ps1 smoke` (`tests/test_main.py`) | **PASS** | 11 passed |
| `.\rtk.ps1 quick` (`pytest -m "not anyio and not slow"`) | **PASS** | 692 passed, 87 deselected |
| `.\rtk.ps1 tui` (`tests/test_tui.py`) | **PASS** | 107 passed |
| `.\rtk.ps1 test -m slow` | **PASS** | 53 passed, 1 skipped, 725 deselected |
| `.\rtk.ps1 test` | **PASS** | 785 passed, 1 skipped |
| `.\rtk.ps1 lint` | **PASS** | Ruff clean |
| `.\rtk.ps1 types` | **PASS** | mypy clean: 83 source files |

### Pytest runs (2026-06-10)

| Suite | Result |
|-------|--------|
| `tests/test_main.py` | PASS (11) |
| Quick non-anyio/non-slow suite | PASS (692) |
| `tests/test_tui.py` | PASS (107) |
| Slow-marked suite | PASS (53 passed, 1 skipped) |
| Full pytest suite | PASS (785 passed, 1 skipped) |

### Files inspected

- `DESIGN.md`, `ROADMAP.md`, `README.md`, `CHANGELOG.md`, `pyproject.toml`, `project_sources/CURRENT_STATE.md`
- `src/game/app/commands.py`, `actions.py`, `flows.py`, `views.py`
- `src/game/expedition/travel.py`, `dungeon.py`, `generated_maze.py`, `maze.py`, `expedition.py`
- `src/game/ui/tui.py`, `tui_widgets.py`
- `data/world.yaml`, `data/expeditions.yaml`
- Key tests: `test_vertical_slice.py`, `test_cli.py`, `test_dungeon.py`, `test_town_loop.py`, `test_generated_maze.py`, `test_missing_carters_contract.py`, `test_save_load.py`, `test_tui.py`

### Manual / TUI playthrough

**VERIFIED** (2026-06-09) — maintainer completed the Human TUI Checklist on the Textual frontend. No blockers reported.

### Repo hygiene inspected

**Do not commit:**

- `_wb.py` — hardcoded local paths
- `ai_lab_*.txt` (11 files) — lab output dumps
- `RESUME_EVIDENCE_BANK.md` — portfolio doc
- `project_sources_upload_pack.zip`
- `.pytest_cache/`, `*.egg-info/`, `saves/*.json` (gitignored)

---

## v0.1 Promise Checklist

| Step | Status | Notes |
|------|--------|-------|
| New company | **PASS** | `create_new_company`; `blackwood_road_charter` active at start |
| Inspect roster / formation / supplies / gear / contracts | **PASS** (gear: **POLISH**) | Gear empty at start; unlocks after contracts |
| Accept or pursue opening contract | **PASS** | Charter active at start; completes at `cave_mini_boss` |
| Start opening expedition | **PASS** | Interactive dungeon + manual combat-lite via TUI |
| Move through Old Road / Blackwood / Shallow Cave | **PASS** (naming: **POLISH**) | Wilderness nodes map to `blackwood_forest` world location |
| Complete at least one manual combat-lite encounter | **PASS** | Move/Pass/Delay/Retreat/heal skills in TUI |
| Reach Cave Mini Boss / Maze Breach | **PASS** | Tested in dungeon + vertical slice |
| Enter or inspect generated breach route / Maze stub | **PASS** | Generated routes = repeat loop; Depth 1 = tutorial stub |
| Withdraw / return to Haven | **PASS** (UX: **POLISH**) | Lands on East Gate arrival brief |
| See report / rewards / wounds / memories / contract state | **PASS** (visibility: **POLISH**) | Full record opt-in via Records Room |
| Save and load | **PASS** | `SAVE_VERSION = 13`; migration tests |
| Resume in sane Haven state | **PASS** | **TUI resume verified by maintainer** |

---

## First-Play Friction Findings

| Severity | Area | Finding | Evidence | Suggested Fix | Files likely involved |
|----------|------|---------|----------|---------------|----------------------|
| BLOCKER | — | None found for v0.1 promise | — | — | — |
| DONE | Documentation | README combat UI limitation synced | `README.md` Known Limitations | Completed in docs refresh | `README.md` |
| DONE | Documentation | DESIGN Manual Combat-Lite synced | `DESIGN.md` Manual Combat-Lite | Completed in docs refresh | `DESIGN.md` |
| DONE | Documentation | DESIGN Coin/reputation split synced | `DESIGN.md` Company And Town Loop | Completed in docs refresh | `DESIGN.md` |
| HIGH | Post-return UX | Lands on East Gate; filed record not auto-shown | `tui.py` `_finish_playback` | Add "View Filed Record" on arrival brief | `tui.py`, `actions.py` |
| HIGH | Post-return UX | Report dock lacks Formation/Memorial/Charter shortcuts | `actions.py` `report_actions` L921–973 | Expand contextual report actions | `actions.py`, `tui.py` |
| MEDIUM | Contracts | Active charter vs board-accept unclear | `test_town_loop.py` | Charter Office copy for first expedition | `tui.py`, `data/world.yaml` |
| MEDIUM | Fast travel | Mark Route required despite YAML `known_route_unlock` | `travel.py`; `test_dungeon.py` | Auto-chart on first cave visit or stronger hint | `dungeon.py`, `travel.py` |
| MEDIUM | Memories | Identity changes buried in Notable Moments | `finalize_report_memory` | Memory section on report or arrival nudge | `views.py`, `tui_widgets.py` |
| MEDIUM | Breach choice | Depth 1 stub vs generated maze framing | `maze_breach` node | Clearer breach screen copy | `tui.py`, `views.py` |
| LOW | Footer | Esc opens system menu | `tui.py` footer | "Esc → System menu" on town screens | `tui.py` |
| POST | World map | Destinations display-only | `views.py`; `tui.py` | Wire to `TravelRegional` when charted | `tui.py`, `actions.py` |

---

## Human TUI Checklist

**Completed 2026-06-09** by maintainer. All items verified on Textual frontend.

1. [x] Launch `uv run python -m game.main`
2. [x] Start new company; Haven hub shows objective, coin, supplies, active charter
3. [x] Inspect roster, formation, supplies, gear, Charter Office
4. [x] Travel Haven → East Gate → Old Road
5. [x] Manual combat-lite encounter (skill + target; Move/Retreat if offered)
6. [x] Reach Cave Mini Boss → Maze Breach
7. [x] Generated maze or Depth 1 stub; withdraw/retrace
8. [x] Safe return; arrival brief shows what changed
9. [x] Records Room → Latest Filed Record
10. [x] Haven hub: Recovery, ledger, contract board
11. [x] Save → quit → load; sane resume
12. [x] Optional: Missing Carters branch after relic turn-in

---

## Fast Travel / World Movement Findings

Fast travel is a **convenience layer**, not exploration replacement: charted hop (Haven ↔ cave mouth, 1 ration), known-route skip (wilderness only), regional walk (full graph when charted). Cave interior unchanged.

**Polish:** interactive path requires explicit Mark Route at cave mouth; world map destinations are display-only (POST-v0.1).

Safe return, generated maze retrace/withdraw/collapse, and regional gating work as designed.

---

## Generated Breach Route Findings

**Strong enough for v0.1 repeat activity.** Scout/hunt/posting contracts, deterministic recipes, mid-run save/load, collapse/retrace/withdraw all tested.

**Polish:** scout "survey" = any room action — clarify in `shallow_cave_breach_scout.summary` (`data/world.yaml`). YAML tweaks beat new systems.

---

## Post-Expedition Town Loop Findings

Engine correctly wires reports, wounds, memories, contracts, and ledger. **Polish:** first return buries payoff (East Gate landing, opt-in filed record, narrow report dock). "Company is protagonist" visible in memorial/ledger/reports with extra navigation.

---

## Gameplay Expansion Recommendation

| Idea | v0.1 blocker? | v0.1 polish? | post-v0.1? | Why |
|------|---------------|--------------|------------|-----|
| Bonds | No | No | **Yes** | DESIGN exclusions |
| More lab tooling | No | No | **Yes** | Dev substrate exceeds v0.1 needs |
| Missing Carters / Carter Wreck | No | **Yes** (surfacing) | No | Already implemented + tested |
| Generated breach YAML polish | No | **Yes** | No | Scout summary, reward tuning |
| Fast travel / movement cleanup | No | **Yes** | Partial | Auto-chart or Mark Route hint |
| More gear / upgrades / recruits | No | No | **Yes** | ROADMAP Milestones 5–6 |
| Deeper Maze Depth 1 | No | No | **Yes** | Stub by design |

---

## Documentation / Release Alignment

| Doc | Issue | Recommended edit |
|-----|-------|------------------|
| `README.md` L230–231 | Stale combat UI limits | Match TUI; note retreat scope |
| `DESIGN.md` L180–187 | Manual Combat-Lite stale | Match TUI |
| `DESIGN.md` L153 | Reputation vs Coin | Coin for town services |
| `CURRENT_STATE.md` | Dated; uncommitted note | Refresh for release |
| `CHANGELOG.md` | All under `Unreleased` | Cut `[0.1.0]` on tag |
| `pyproject.toml` | License field added | MIT license now included for public release |

---

## Minimal Pre-Tag Fix List

### 1. Must fix before v0.1

1. ~~Human TUI playthrough~~ — **done** (2026-06-09).
2. ~~Exclude dev artifacts~~ — ignored local artifacts remain excluded (`_wb.py`, `ai_lab_*.txt`, `RESUME_EVIDENCE_BANK.md`, `project_sources_upload_pack.zip`, caches, saves).
3. ~~Confirm gameplay tests green locally~~ — `.\rtk.ps1 test` passed (785 passed, 1 skipped) on 2026-06-10.
4. Commit this documentation refresh, then tag `v0.1.0` when release mechanics are approved.

### 2. Should fix before v0.1

5. ~~Sync README + DESIGN (combat UI, Coin/reputation).~~
6. Post-return guidance on arrival brief / report dock.
7. Charter Office copy for active charter.
8. Scout contract YAML summary tweak.
9. ~~Cut CHANGELOG to `[0.1.0]`.~~

### 3. Safe after v0.1

10. CI and LICENSE.

---

## Suggested Follow-Up Prompts

1. **Documentation sync:** Update README and DESIGN Known Limitations / Manual Combat-Lite / Company And Town Loop for current TUI (Move, Pass, Delay, Retreat, healing; retreat dungeon-only; Coin not reputation).

2. **Post-return UX:** Add "View Filed Record" to regional arrival brief; expand `report_actions` with Formation and Charter Office when state warrants. Add TUI integration test.

3. **Scout contract clarity:** Update `shallow_cave_breach_scout.summary` in `data/world.yaml` so "survey" means one room action on the generated route.

4. **Auto-chart on first cave visit:** Call `unlock_known_route_for_node` at first `shallow_cave_entrance` visit in interactive dungeon; regression test in `test_dungeon.py`; verify save/load if state changes.

---

## Quality Gate Summary

```
smoke (test_main.py)      PASS  11/11
quick                     PASS  692 passed, 87 deselected
tui                       PASS  107 passed
slow                      PASS  53 passed, 1 skipped, 725 deselected
full pytest               PASS  785 passed, 1 skipped
human TUI checklist       PASS  (maintainer, 2026-06-09)
ruff check src tests      PASS
mypy src                  PASS  83 source files
repository                https://github.com/nrtrinid/charter-game
```

**Gameplay readiness:** shippable; human TUI verification complete.
**Release hygiene:** docs/changelog refreshed; tag `v0.1.0` after committing.
