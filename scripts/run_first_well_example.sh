#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/my_experiment.yaml}
PLATE=${2:-Plate001}
WELL=${3:-A01}
ROI=${4:-Neuron001}
REF_DAY=${5:-Day01}

tmem-align validate-config "$CONFIG"
tmem-align stitch "$CONFIG" --plate "$PLATE" --well "$WELL"
tmem-align register-well "$CONFIG" --plate "$PLATE" --well "$WELL" --reference-day "$REF_DAY"
tmem-align make-roi-stack "$CONFIG" --plate "$PLATE" --well "$WELL" --roi-id "$ROI"
tmem-align quantify "$CONFIG" --plate "$PLATE" --well "$WELL" --roi-id "$ROI"
