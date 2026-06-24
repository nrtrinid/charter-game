# Agent Task Prompt Templates

Date: 2026-06-16

**Preferred:** 3-stage workflow in `prompts/agent_workflow.md` (scout -> drift -> implement).

Run `.\rtk.ps1 scout --task "your task"` to generate the Stage 1 bundle.

---

## Stage 1 - Scout (cheap model)

Do not implement. Output `SCOUT_PACKET` per `prompts/scout_packet_schema.md`.

```text
ROLE: Charter context scout (Stage 1 of 3). Do not write or edit code.

Paste the full output of: .\rtk.ps1 scout --task "<your task>"

Rules:
- Read only docs/AGENT_CONTEXT_MAP.md for the suggested block(s)
- Use rg and narrow file reads; respect .agentignore
- Return one YAML SCOUT_PACKET (see prompts/scout_packet_schema.md)
- Do not read full AGENTS.md unless the block requires a specific section
```

---

## Stage 2 - Drift check (smart model)

See `prompts/drift_check_prompt.md`. No code.

---

## Stage 3 - Implement

See `prompts/implement_from_plan.md`. Only after `DRIFT_VERDICT: APPROVED`.

---

## Legacy single-session templates

Use only when skipping the 3-stage loop.

### Template A - Implement (single session)

```text
Follow AGENTS.md in dungeon-party-game/.

Run .\rtk.ps1 preflight first.

Read docs/AGENT_CONTEXT_MAP.md block: <block-id>

Task:
<scoped task>

Before finishing: .\rtk.ps1 check + focused tests from block VERIFY.

Final response: files changed, checks run, risks, CHANGELOG yes/no.
```

### Template B - Context scout (single session)

Same as Stage 1 but without `.\rtk.ps1 scout` bundle - prefer scout command instead.

---

## Block quick reference

In-repo blocks use the same field names in `docs/AGENT_CONTEXT_MAP.md`.

| Block | Use when |
|---|---|
| `combat-rules` | Skills, targeting, formation, enemy AI, manual combat |
| `campaign-town` | Roster, town services, gear, contracts, Coin |
| `expedition-dungeon` | Routes, nodes, Maze, cave, travel |
| `textual-ui` | Textual screens, widgets, keyboard/focus |
| `cli-legacy` | Rich CLI fallback, `--cli`, fake-input tests |
| `content-yaml` | Authored data, schemas, loaders |
| `save-load` | Save shape, migration, `company.json` |
| `ai-lab-balance` | Dev lab sweeps, oracle, policy-band, enemy training |
| `app-commands` | New commands, flows, ScreenAction metadata |
| `docs-only` | Markdown, changelog, handoff pack |
