#!/usr/bin/env bash
set -euo pipefail

task="${1:-smoke}"
shift || true

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python_bin=""
if command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  echo "Missing python/python3 on PATH" >&2
  exit 1
fi

show_help() {
  cat <<'EOF'
rtk.sh - Charter repo toolkit (minimal cross-platform wrapper)

Usage:
  ./rtk.sh <task> [args]

Tasks (minimum set):
  preflight        Agent preflight report
  scout            Print scout bundle (--scout)
  boundaries       Fail if engine packages import game.ui
  smoke            Minimal smoke tests
  quick            Fast pytest subset
  test [args]      Run pytest (args forwarded)
  check            Run ruff check + mypy src
  review-packet    Print lightweight review packet (Markdown)
  help             Show this help
EOF
}

case "$task" in
  help)
    show_help
    ;;
  preflight)
    "$python_bin" -m game.dev.agent_preflight "$@"
    ;;
  scout)
    "$python_bin" -m game.dev.agent_preflight --scout "$@"
    ;;
  boundaries)
    "$python_bin" -m game.dev.check_engine_boundaries "$@"
    ;;
  smoke)
    "$python_bin" -m pytest tests/test_main.py
    ;;
  quick)
    "$python_bin" -m pytest -m "not anyio and not slow" "$@"
    ;;
  test)
    "$python_bin" -m pytest "$@"
    ;;
  check)
    "$python_bin" -m ruff check
    "$python_bin" -m mypy src
    ;;
  review-packet)
    "$python_bin" -m game.dev.agent_review_packet "$@"
    ;;
  *)
    echo "Unknown task: $task" >&2
    echo "Run: ./rtk.sh help" >&2
    exit 2
    ;;
esac

