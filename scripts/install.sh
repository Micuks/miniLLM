#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv sync
else
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
  else
    echo "requirements.txt not found; consider installing uv: https://github.com/astral-sh/uv" >&2
    exit 1
  fi
fi


