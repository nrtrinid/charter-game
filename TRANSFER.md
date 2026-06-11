# Repository publish handoff

Use this when publishing or cloning the v0.1 project snapshot.

## What gets uploaded

- Source, data, tests, docs, and `project_sources/`
- `rtk.ps1` local dev toolkit

Not included (see `.gitignore`):

- `.venv/`, caches, build artifacts
- `saves/*.json`
- `ai_lab_*.txt`, `_wb.py`, `project_sources_upload_pack.zip`
- `RESUME_EVIDENCE_BANK.md`

## 1. Verify locally

From `dungeon-party-game/`:

```powershell
.\rtk.ps1 check
.\rtk.ps1 test
git diff --check
git status --short --ignored
```

For the current v0.1 snapshot, `.\rtk.ps1 test` verifies gameplay and
`.\rtk.ps1 check` verifies the local Ruff/mypy quality gate.

## 2. Publish to GitHub

The intended public repository is:

```text
https://github.com/nrtrinid/charter-game.git
```

For a one-commit public v0.1 history, create a backup branch first, then publish
an orphan `main` built from the verified working tree.

```powershell
git branch backup/pre-v0.1-perfect-commit
git checkout --orphan release/v0.1-root
git add -A
git commit -m "Ship The Charter v0.1"
git branch -M main
git remote set-url origin https://github.com/nrtrinid/charter-game.git
git push -u origin main
```

If preserving the existing local history is preferred, skip the orphan branch
steps and commit normally before pushing.

## 3. Tag v0.1.0

After the pushed repository looks correct:

```powershell
git tag -a v0.1.0 -m "The Charter v0.1.0"
git push origin v0.1.0
```

## 4. Clone elsewhere

```powershell
cd C:\path\to\destination
git clone https://github.com/nrtrinid/charter-game.git charter-game
cd charter-game
```

Then set up the environment:

```powershell
uv sync --extra dev
# or: python -m pip install -e ".[dev]"

.\rtk.ps1 setup
.\rtk.ps1 test
```

## Verify after cloning

```powershell
.\rtk.ps1 check
.\rtk.ps1 test
uv run python -m game.main --help
```
