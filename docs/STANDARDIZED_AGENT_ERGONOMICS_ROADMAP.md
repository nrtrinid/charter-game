# Standardized Agent Ergonomics Roadmap

Date: 2026-06-23
Status: Roadmap
Scope: `life-harness`, `text-adventure`, and `ev-tracker`

## Goal

Standardize the agent first-five-minutes experience across all three repos while preserving the lightweight Charter workflow. An agent should be able to enter any repo and quickly answer:

1. Where do I start?
2. What should I read for this task?
3. What should I avoid reading or touching by default?
4. What focused checks match my changed files?
5. What should my final handoff include?

The target is a shared ergonomics contract, not identical implementation. `text-adventure` should stay compact and `rtk.ps1`-first.

## Shared Contract

Each repo should provide these stable surfaces:

- Root `AGENTS.md` or an explicit documented equivalent as the first-read entrypoint.
- `docs/AGENT_CONTEXT_MAP.md` as the task router.
- A read-only preflight command that reports git state, likely task blocks, likely tests, and boundary warnings.
- A doctor or freshness check that catches stale links, dangerous touched files, and command drift.
- A changed-files-to-tests selector or verification helper.
- Default no-read and no-touch guidance for generated, private, large, or dangerous paths.
- A common final response shape for non-trivial work: `Changed / Tests / Docs / Risks / Did not touch / Next safe step`.

## Text Adventure Position

The canonical repo is `text-adventure/dungeon-party-game/`. The outer `text-adventure/` folder is only a workspace wrapper.

Already strong:

- Canonical `AGENTS.md` is concise and useful.
- `docs/AGENT_CONTEXT_MAP.md` routes task areas to read-first files and checks.
- `.\rtk.ps1` gives one local command surface for setup, smoke, quick, test, check, all, preflight, scout, boundaries, and review-packet.
- `prompts/agent_workflow.md` supports scout, drift check, and implementation handoff.
- The durable memory section in `AGENTS.md` captures repo-specific agent lessons.

Known friction (Phase 1 complete 2026-06-23):

- Wrapper `AGENTS.md` and `README.md` plus `.agentignore` are UTF-8 on disk (wrapper files are local-only, outside `charter-game` git).
- Preflight uses `parse_status_path` so porcelain lines are not strip-misaligned (`src/` vs `rc/`).
- There is no separate doctor command yet; some freshness checks are folded into preflight and manual docs discipline.

## Roadmap

### Phase 1 - Repair Known Friction

Fix the small issues that make an otherwise good workflow feel noisy.

Tasks:

- Convert outer wrapper `AGENTS.md` and `README.md` to UTF-8.
- Convert canonical `.agentignore` to UTF-8.
- Fix `src/game/dev/agent_preflight.py` so it does not strip leading porcelain status spaces before slicing paths.
- Add or update `tests/test_agent_preflight.py` to cover a first modified path under `src/`.

Acceptance:

- `Get-Content -Raw` and `rg` show readable Markdown with no NUL noise.
- `.\rtk.ps1 preflight` reports `src/game/...` paths correctly.

**Status:** complete (2026-06-23).

### Phase 2 - Keep The Rtk Contract Canonical

Make `.\rtk.ps1` the documented single local command surface.

Required commands:

```powershell
.\rtk.ps1 help
.\rtk.ps1 preflight
.\rtk.ps1 scout --task "one-line task"
.\rtk.ps1 boundaries
.\rtk.ps1 review-packet
.\rtk.ps1 quick
.\rtk.ps1 all
```

Acceptance:

- `AGENTS.md`, `README.md`, and `docs/AGENT_CONTEXT_MAP.md` all point to the same command surface.
- Any future helper should either become an `rtk.ps1` task or stay clearly optional.

**Status:** complete (2026-06-23).

### Phase 3 - Normalize Context Map Headings

Keep task blocks in a consistent shape:

```md
## <task-name>

Use when:
READ_FIRST:
LIKELY_FILES:
LIKELY_TESTS:
VERIFY:
DO_NOT_READ:
BOUNDARIES:
NOTES:
```

Acceptance:

- New task blocks follow this shape.
- Existing blocks that are touched for other reasons are normalized opportunistically.

**Status:** complete (2026-06-23).

### Phase 4 - Add Lightweight Freshness Checks

Decide whether freshness checks belong in a new `doctor` task or inside preflight.

Minimum checks:

- broken doc links for key agent docs
- dangerous touched files
- stale command references
- boundary-sensitive changes without `.\rtk.ps1 boundaries`

Acceptance:

- Agents have one obvious read-only command for repo freshness before handoff.
- The command warns without making ordinary iteration heavy.

## Local Next Steps

1. Decide whether to add `.\rtk.ps1 doctor` or keep freshness inside `preflight`.
2. Implement Phase 4 minimum freshness checks (broken links, dangerous files, stale commands, missing boundaries run).

## Definition Of Done

Text Adventure is aligned when:

- A new agent reads the wrapper pointer, lands in `dungeon-party-game/`, reads `AGENTS.md`, and runs `.\rtk.ps1 preflight`.
- The context map selects a narrow task block.
- `.\rtk.ps1` provides all standard verification entrypoints.
- No generated, cache, save, scratch, or virtualenv paths are default-read.
- The final handoff names changed files, checks, docs, risks, untouched surfaces, and the next safe step.
