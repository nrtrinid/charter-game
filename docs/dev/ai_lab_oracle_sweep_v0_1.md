# AI Lab Oracle Sweep v0.1

Date: 2026-06-15 (post bimodal envelope diagnostics)

## Summary verdict

**NEEDS_ROUTE_PRESSURE_TUNING** — do **not** tune monsters yet.

- `opening_pressure_path` is **bimodal / collapsing**: **24% completion** (12/50), **76% noncompletion** (38/50), survivors reach boss at **35.4 HP** (above envelope max 26), but **all-run final HP is only 6.0**. Envelope now reports **`route_bimodal_collapse` FAIL** instead of the misleading standalone "party reaches boss too healthy".
- `opening_critical_path` remains healthy: **0% noncompletion**, **41.4 avg final HP**, policy-band **PASS**; envelope WARN is only "too clean" (completion/boss above band), not collapse.
- Score-only heuristic oracle miss rate is **22-30%** everywhere — expected when auditing learned policy vs heuristic-best.
- Dominant-skill rates stay **16-25%** (PASS); packages (`maw_package`, `bandit_kill_lane`, `wolf_mark`) all **PASS**.
- Generated Maze scout profiles still complete **50/50** with identical WARN score **85** ("completion above target band").

## Next design decision (route, not monsters)

Before YAML or learned-weight edits, decide whether `opening_pressure_path` should:

1. **Soften the route sequence** (reduce pre-boss attrition or maze gate pressure)
2. **Add a pressure valve** (recovery branch, skip option, or attrition cap)
3. **Label as intentional meat-grinder** and retune envelope bands to match that identity

## Commands run (post-fix validation)

| Command | Exit | Notes |
|---------|------|-------|
| `oracle-report --route opening_pressure_path --seeds 50 --hero-policy mixed --preset fresh` | 0 | **FAIL** `route_bimodal_collapse` |
| `oracle-report --route opening_critical_path --seeds 50 --hero-policy mixed --preset fresh` | 0 | Healthy; WARN oracle/attrition only |

## Authored route findings

### opening_critical_path (fresh, mixed, 50 seeds)

| Metric | Value |
|--------|-------|
| Oracle miss rate | **29.8%** (140/470 actions) |
| Avg miss delta | 14.3 |
| Largest miss | `maze_leech_1` `glass_bite` -> oracle `effort_drain` (+29) |
| Dominant skill | `shielding_dead` **22%** |
| Damage to downed | **6.6%** |
| First Downed round avg | **2.5** |
| Mortal wounds/run | **0.62** |
| Deaths/run | **0.10** |
| Noncompletion | **0 (0%)** |
| Pre-boss failures | **0** |
| Boss entry HP | **47.5** |
| Final party HP | **41.4** |
| Envelope | `critical_path` WARN — completion above band; boss too healthy |

**Read:** Critical path is forgiving and policy-robust. Hardness is mostly "too clean" for envelope targets, not unreadable attrition or bimodal collapse.

### opening_pressure_path (fresh, mixed, 50 seeds)

| Metric | Value |
|--------|-------|
| Route completion (learned) | **12/50 (24%)** |
| Noncompletion | **38/50 (76%)** |
| Pre-boss failures | **31** |
| Oracle miss rate | **27.5%** (484/1762) |
| Avg miss delta | 19.2 |
| Largest miss | `maze_acolyte_1` `black_pulse` -> oracle `mark_the_pattern` (+71) |
| Dominant skill | `maw_slam` **16%** |
| Damage to downed | **21.9%** |
| First Downed round avg | **1.9** |
| Mortal wounds/run | **7.72** |
| Deaths/run | **2.50** |
| Boss entry HP (survivors) | **35.4** |
| Final party HP (all runs) | **6.0** |
| Envelope | `optional_pressure_path` **FAIL** — `route_bimodal_collapse` |

**Envelope detail (post-fix):**

```
route_bimodal_collapse: completion 24%; noncompletion 38/50; boss-entry 35.4 among survivors exceeds max 26; final HP 6.0; pre-boss failures 31
```

**Read:** Pressure path is **bimodal**: most runs fail before maze completion, but survivors arrive at boss with high HP; all-run final HP is very low. This is route-sequence pressure, not monster-kit collapse. **Do not tune monsters yet.**

## Diagnosis

### Route pressure issue

**High (primary).** Bimodal collapse — low completion, healthy survivors, low all-run final HP.

### Envelope issue

**Resolved for diagnosis.** `route_bimodal_collapse` tiered FAIL/WARN replaces misleading standalone "boss too healthy" on pressure path.

### Monster kit issue

**Low.** Dominant skill PASS; per-package health PASS. Defer tuning.

### Enemy heuristic issue

**Moderate.** High oracle miss rate reflects learned-vs-heuristic gap. Not the primary bottleneck.

## Code changes from this sweep

- `route_lab.py`: `_detect_bimodal_collapse`, noncompletion metrics, tiered FAIL/WARN scoring
- `ai_oracle.py`: `noncompletion_count` / `pre_boss_failure_count` naming; `route_bimodal_collapse` finding

## Verification

```bash
python -m pytest tests/test_route_lab.py tests/test_ai_lab.py tests/test_ai_oracle.py -q
python -m game.dev.ai_lab oracle-report --route opening_pressure_path --seeds 50 --hero-policy mixed --preset fresh
```

Expect: `route_bimodal_collapse` with `noncompletion 38/50`; no standalone `party reaches boss too healthy`; envelope **FAIL** for pressure path.
