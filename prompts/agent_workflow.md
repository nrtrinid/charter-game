# Agent Workflow (3 stages)

Date: 2026-06-15

Life Harness-style loop for Charter. **Do not repo-contextualize in one shot.**
Run each stage in a separate session (or automation step) with the right model.

```text
Task (one line from you)
  -> Stage 1 SCOUT (cheap model, small context)
  -> Stage 2 DRIFT CHECK (smart model, no code)
  -> Stage 3 IMPLEMENT (from approved plan only)
```

## Stage 0 — Bundle (you or script)

From `dungeon-party-game/`:

```powershell
.\rtk.ps1 scout --task "your one-line task"
```

Copy the printed bundle into Stage 1.

## Stage 1 — Scout (cheap model)

**Prompt file:** use the scout section in `prompts/agent_task_prompt_template.md`
**Output schema:** `prompts/scout_packet_schema.md`

Rules:

- Do **not** implement code
- Do **not** read full `AGENTS.md` unless the context map block requires a section
- Use preflight block guess + `rg` / narrow file reads only
- Return a single `SCOUT_PACKET` block (YAML inside fenced code)

## Stage 2 — Drift check (smart model)

**Prompt file:** `prompts/drift_check_prompt.md`

Inputs:

- Original task
- Preflight stdout (from bundle)
- Scout `SCOUT_PACKET`

Output: `DRIFT_VERDICT` with `APPROVED` or `REVISE`. No code.

If `REVISE`: fix scout packet and re-run Stage 2. Do not implement yet.

## Stage 3 — Implement

**Prompt file:** `prompts/implement_from_plan.md`

Inputs:

- Original task
- `DRIFT_VERDICT` (must be `APPROVED`)
- Approved plan bullets from drift check

Rules:

- Implement **only** the approved plan
- Run `.\rtk.ps1 preflight --verify` before handoff
- Do not expand scope without a new scout + drift cycle

## Quick reference

| Stage | Model tier | Reads | Writes code? |
|-------|------------|-------|--------------|
| 0 bundle | script | git status | no |
| 1 scout | cheap | context map slice | no |
| 2 drift | smart | AGENTS boundaries | no |
| 3 implement | capable | plan + target files | yes |
