# Drift Check Prompt (Stage 2)

Date: 2026-06-15

Smart model. No code. No implementation.

Gate scope before any implementer runs. Spot-check AGENTS.md and DESIGN.md only
for what the scout packet touches.

## Paste below this line

ROLE: Charter drift checker (Stage 2 of 3). Do not write or edit code.

ORIGINAL TASK:
<paste task>

PREFLIGHT (from .\rtk.ps1 scout):
<paste preflight section from bundle>

SCOUT_PACKET:
<paste scout YAML from Stage 1>

CHECKLIST - reject or revise if any fail:
1. block_id matches the task (docs/AGENT_CONTEXT_MAP.md path triggers)
2. smallest_slice is one PR-sized step, not a redesign
3. read_first is minimal (no full AGENTS.md unless necessary)
4. boundaries cover UI/engine separation, RNG, save/YAML risks if relevant
5. likely_tests match the block VERIFY section
6. No forbidden mechanics (MP, cooldowns, overworld, mid-combat saves)
7. Textual vs CLI scope is correct
8. docs-only tasks are not routed to combat-rules by mistake

OUTPUT - one YAML block labeled DRIFT_VERDICT with fields:
verdict (APPROVED or REVISE), block_id, plan, verify_commands,
boundaries_enforced, revise_notes, risks_accepted, changelog_expected

If REVISE: no plan; only revise_notes for the scout.
If APPROVED: plan must not require reading AGENTS.md cover-to-cover.

## Operator notes

Re-run Stage 1 if revise_notes are large. Only APPROVED goes to Stage 3.
