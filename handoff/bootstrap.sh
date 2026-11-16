#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x /opt/homebrew/bin/uv ]]; then UV_BIN=/opt/homebrew/bin/uv; fi
if [[ -z "$UV_BIN" ]]; then echo 'uv is required; install it before bootstrap.' >&2; exit 1; fi
"$UV_BIN" venv --python 3.12 .venv
"$UV_BIN" pip install --python .venv/bin/python -r handoff/requirements-longitudinal.txt
"$UV_BIN" pip freeze --python .venv/bin/python > handoff/installed-environment.txt
PYTHONPATH=src .venv/bin/python -m pytest tests/test_longitudinal_counts.py tests/test_plate_counts.py tests/test_array_archive.py -q -p no:napari
.venv/bin/python -c 'import torch; print("MPS available:", torch.backends.mps.is_available()); assert torch.backends.mps.is_available()'
