#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x /opt/homebrew/bin/uv ]]; then UV_BIN=/opt/homebrew/bin/uv; fi
if [[ -z "$UV_BIN" ]]; then echo 'uv is required.' >&2; exit 1; fi
"$UV_BIN" venv --python 3.12.13 .venv
"$UV_BIN" pip install --python .venv/bin/python -r requirements-lock.txt
PYTHONPATH=src .venv/bin/python -m pytest tests -q -p no:cacheprovider
