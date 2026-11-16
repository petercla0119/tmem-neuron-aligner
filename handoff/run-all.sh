#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
INPUT_ROOT="${1:-/Users/pmihack/claire/tmem_2026/data/mCherry-mNG-longitudinal}"
OUTPUT_ROOT="${2:-/Users/pmihack/claire/tmem_2026/outputs/mcherry-mng-counts-scale2-v1}"
exec /usr/bin/caffeinate -i .venv/bin/python scripts/run_plate_counts.py --input "$INPUT_ROOT" --output "$OUTPUT_ROOT"
