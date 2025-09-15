#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv run uvicorn miniLLM.service.app:app --host 0.0.0.0 --port 8000
else
  uvicorn miniLLM.service.app:app --host 0.0.0.0 --port 8000
fi


