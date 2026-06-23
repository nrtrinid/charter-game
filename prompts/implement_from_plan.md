# Implement From Plan (Stage 3)

Date: 2026-06-15

**Implement only after Stage 2 `DRIFT_VERDICT: APPROVED`.**

Do not re-scout the repo. Read files listed in the approved plan and
`docs/AGENT_CONTEXT_MAP.md` block for boundaries only.

---

## Paste below this line

```text
ROLE: Charter implementer (Stage 3 of 3).

ORIGINAL TASK:
<paste task>

DRIFT_VERDICT (APPROVED only):
<paste approved DRIFT_VERDICT YAML>

RULES:
- Implement ONLY the plan bullets in DRIFT_VERDICT. No scope expansion.
- If you discover blockers, STOP and report — do not improvise a redesign.
- Minimal diff; match existing patterns in touched files.
- Run verify_commands from DRIFT_VERDICT before finishing.
- Prefer .\rtk.ps1 preflight --verify for final handoff if plan touched multiple areas.

DO NOT READ unless plan requires it:
- Full AGENTS.md (use plan boundaries_enforced instead)
- project_sources/ handoff pack
- .agentignore paths

FINAL RESPONSE must include:
- files changed
- checks run (pass/fail)
- skipped checks and why
- boundary/scope risks
- CHANGELOG.md updated? (yes/no + why — use drift verdict changelog_expected)
```

---

## If drift was REVISE

Do not use this prompt. Return to Stage 1 or fix scout packet, then re-run Stage 2.
