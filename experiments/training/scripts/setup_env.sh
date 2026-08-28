#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Install Python 3.11 and its venv module." >&2
  exit 2
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install --requirement "$REPO_ROOT/requirements.txt"
"$VENV_PYTHON" "$REPO_ROOT/scripts/check_environment.py"

printf '\nEnvironment ready. Activate it with:\n  source "%s/bin/activate"\n' "$VENV_DIR"
