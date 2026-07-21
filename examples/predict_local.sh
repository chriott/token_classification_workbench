#!/usr/bin/env bash

set -euo pipefail

python ipi.py predict \
  --model-path outputs/training/baseline \
  --input-file data/prediction_input.csv
