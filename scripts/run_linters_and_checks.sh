#!/bin/sh
# Run linters (default) or linters+checks without rebuilding the uv env.

set -u

main() {
  # Parse optional flag
  CHECKS=false
  PYTEST_ARGS=""
  SEEN_DASHDASH=false

  # Parse args: -c/--checks anywhere; collect pytest args (after -- or others)
  for arg in "$@"; do
    if [ "$SEEN_DASHDASH" = true ]; then
      PYTEST_ARGS="$PYTEST_ARGS $arg"
      continue
    fi
    case "$arg" in
      -c|--checks) CHECKS=true ;;
      --) SEEN_DASHDASH=true ;;
      *) PYTEST_ARGS="$PYTEST_ARGS $arg" ;;  # pass unknowns to pytest
    esac
  done

  # Where your project venv lives (override with ENV_PATH=/some/venv)
  ENV_PATH="${ENV_PATH:-.venv}"

  # Prefer an already-active venv
  # Choose runner: active venv, local .venv, or uv (no sync)
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
  echo "+ ruff format"
  run ruff format

  if [ "$CHECKS" = true ]; then
    echo "+ ruff check"
    run ruff check
    echo "+ mypy"
    run mypy
    echo "+ interrogate -v"
    run interrogate -v
    echo "+ codespell --check-filenames"
    run codespell --check-filenames
    echo "+ pytest --cov aind_anatomical_utils$([ -n "$PYTEST_ARGS" ] && printf ' -- %s' "$PYTEST_ARGS")"
    # shellcheck disable=SC2086
    run pytest --cov aind_anatomical_utils $PYTEST_ARGS
  else
    echo "(checks skipped; pass -c or --checks to enable)"
  fi
}

main "$@"
