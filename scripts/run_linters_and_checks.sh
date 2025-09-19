#!/bin/sh
# Run linters (default) or linters+checks without rebuilding the uv env.

set -eu

main() {
  # Parse optional flag
  CHECKS=false
  case "${1:-}" in
    -c|--checks) CHECKS=true ;;
  esac

  # Where your project venv lives (override with ENV_PATH=/some/venv)
  ENV_PATH="${ENV_PATH:-.venv}"

  # Prefer an already-active venv
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    BIN="$VIRTUAL_ENV/bin"
    run() { "$BIN/$@"; }

  # Or prefer a project-local venv if it exists
  elif [ -x "$ENV_PATH/bin/python" ]; then
    BIN="$ENV_PATH/bin"
    run() { "$BIN/$@"; }

  # Otherwise fall back to uv, but do NOT resync/recreate the env
  else
    # Optional interpreter pin: from $PYVER or .python-version if present
    PYVER="${PYVER:-}"
    if [ -z "$PYVER" ] && [ -f .python-version ]; then
      PYVER="$(tr -d ' \n' < .python-version)"
    fi

    export UV_PROJECT_ENVIRONMENT="$ENV_PATH"  # reuse this env path
    UV_ARGS="--frozen --no-sync"               # do not check/sync env
    [ -n "$PYVER" ] && UV_ARGS="$UV_ARGS --python $PYVER"

    run() { uv run $UV_ARGS -- "$@"; }
  fi

  # Linters
  run ruff format

  if [ "$CHECKS" = true ]; then
    run ruff check
    run mypy
    run interrogate -v
    run codespell --check-filenames
    run pytest --cov aind_anatomical_utils
  fi
}

main "$@"
